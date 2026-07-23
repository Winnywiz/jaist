"""
Real-conversation COVERAGE: what can dynamic (Method E) attribute that static can't?

No answer key needed. On REAL Method E conversations (loaded from a saved quality_e
run), we take the turns where the RAG actually FAILED and ask each attribution method
to name the cause. We then report COVERAGE per cause — how many failures each method
attributes to Retrieval / Generation / Conversation.

The point (no ground truth required): the STATIC (Xie) method can only ever output
Retrieval or Generation — it is structurally blind to CONVERSATION (coreference)
failures. The DYNAMIC (Method E) method re-probes the live RAG and can identify them.
So this shows a failure CLASS static cannot see, on real data, with no answer key.

Run:
    python -m conv_rag_benchmark.real_convo_coverage --dataset MultiHopRAG --limit 20
    python -m conv_rag_benchmark.real_convo_coverage --dataset MultiHopRAG --limit 8 --offline
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List

from .config import Config, get_logger
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .llm import LLM

from dynamic_vs_static_dx.core import (
    CONVERSATION, GENERATION, RETRIEVAL, ControlledRAG, OfflineLLM,
    abstained, is_correct, set_judge)
from dynamic_vs_static_dx.arms import decompose_subquestions

logger = get_logger("real.coverage")

_PRONOUN = re.compile(r"\b(it|its|they|them|their|he|him|his|she|her|that|this|those|these)\b", re.I)


def _self_contained(llm, question, history):
    """Rewrite pronouns to the named entity using the conversation (ZeQR/CQR)."""
    if not (history and getattr(llm, "available", False)):
        return question
    hist = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
    out = llm.chat_json(
        "Rewrite the QUESTION as a fully self-contained query: replace every pronoun/"
        "reference with the actual entity named in the CONVERSATION. Keep it concise. "
        'Respond JSON: {"query": "..."}',
        f"CONVERSATION:\n{hist}\n\nQUESTION: {question}") or {}
    return str(out.get("query") or "").strip() or question


# --------------------------------------------------------------------------- #
def static_attribute(rag, llm, question, gold) -> str:
    """Xie static: decompose, probe with the RAG's OWN retrieval, attribute by recovery.
    Can only ever return Retrieval or Generation — never Conversation."""
    subs = [q for q, t in decompose_subquestions(llm, question) if t == "core"] or [question]
    for sq in subs[:5]:
        ctx = rag.retrieve(sq)
        if is_correct(gold, rag.answer(sq, ctx, [])):
            return GENERATION                    # gold reachable -> model's fault
    return RETRIEVAL                             # never recovered -> evidence missing


def dynamic_attribute(rag, llm, turn, history) -> str:
    """Method E dynamic: re-probe the live RAG. Can reach Conversation."""
    q, gold, ev, rag_ans = turn["question"], turn["gold"], turn.get("evidence", ""), turn["rag_answer"]
    ctx = [ev] if ev else rag.retrieve(q)

    # P1 — coreference: if the question leans on a pronoun, does NAMING the entity fix it
    # while the pronoun version fails? That is a conversational (coreference) failure.
    if history and _PRONOUN.search(q):
        sc = _self_contained(llm, q, history)
        if sc and sc.lower() != q.lower():
            a_named = rag.answer(sc, ctx, [])
            a_pron = rag.answer(q, ctx, history)
            if is_correct(gold, a_named) and not is_correct(gold, a_pron):
                return CONVERSATION

    # P2 — retrieval: hand it the gold's supporting evidence. If that fixes it -> Retrieval.
    if is_correct(gold, rag.answer(q, ctx, [])):
        return RETRIEVAL

    # P3 — generation: re-ask with strict grounding. If it now abstains though it first
    # answered, the original substance was fabricated -> Generation.
    strict = (q + "\n\nAnswer ONLY from the given passages. If the passages do not state "
              "the answer, reply exactly: I don't know.")
    if abstained(rag.answer(strict, ctx, [])) and not abstained(rag_ans):
        return GENERATION

    # fall back to the same coverage signal the static arm uses (never blind-guess)
    return static_attribute(rag, llm, q, gold)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-conversation attribution coverage: dynamic vs static")
    ap.add_argument("--dataset", default="MultiHopRAG", help="label under output/ (has quality_e.json)")
    ap.add_argument("--file", default=None, help="explicit quality_e*.json path")
    ap.add_argument("--limit", type=int, default=20, help="cap on real failures scored")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)

    config = Config.load(prefer_local_embeddings=False)
    path = args.file or os.path.join(config.output_dir, args.dataset, "quality_e.json")
    if not os.path.exists(path):
        print(f"!! not found: {path}"); return
    data = json.load(open(path, encoding="utf-8"))

    use_llm = config.has_openai and not args.offline
    llm = LLM(model=config.gen_model, config=config) if use_llm else OfflineLLM()
    embedder = Embedder(config=config, llm=llm) if use_llm else None
    if use_llm:
        set_judge(LLM(model=config.judge_model, config=config))

    # rebuild the same RAG so we can re-probe (the saved JSON has answers, not a live RAG)
    ds_name = {"MultiHopRAG": "multihoprag", "MedQA": "medqa", "ArXivCS": "arxivcs"}.get(
        args.dataset, args.dataset.lower())
    seeds = DatasetLoader(ds_name, max_samples=60).load()
    corpus = [c for s in seeds for c in s.context if c and c.strip()]
    rag = ControlledRAG(corpus, config, llm, embedder)
    print(f"# {args.dataset} | real failures from {os.path.basename(path)} | "
          f"{'online' if use_llm else 'OFFLINE'} | corpus={len(corpus)} chunks")

    # collect REAL failure turns (RAG was wrong or hallucinated), with running history
    failures = []
    for c in data.get("conversations", []):
        hist = []
        for t in c.get("turns", []):
            if t.get("outcome") in ("wrong", "hallucinated"):
                failures.append({"turn": t, "history": list(hist)})
            hist += [{"role": "user", "content": t.get("question", "")},
                     {"role": "assistant", "content": t.get("rag_answer", "")}]
    failures = failures[: args.limit]
    print(f"# scoring {len(failures)} real failures\n")

    static_cov = Counter()
    dynamic_cov = Counter()
    rows = []
    for f in failures:
        t = f["turn"]
        s = static_attribute(rag, llm, t["question"], t["gold"])
        d = dynamic_attribute(rag, llm, t, f["history"])
        static_cov[s] += 1
        dynamic_cov[d] += 1
        rows.append({"query_type": t.get("query_type"), "outcome": t.get("outcome"),
                     "static": s, "dynamic": d, "question": t.get("question", "")[:120]})

    n = len(failures) or 1
    cats = [RETRIEVAL, GENERATION, CONVERSATION]
    print(f"{'cause':<16}{'STATIC (Xie)':>16}{'DYNAMIC (Method E)':>22}")
    print(" " + "-" * 52)
    for cat in cats:
        print(f"{cat:<16}{static_cov.get(cat,0):>16}{dynamic_cov.get(cat,0):>22}")
    print(" " + "-" * 52)
    conv_dyn = dynamic_cov.get(CONVERSATION, 0)
    print(f"\n>> CONVERSATION (coreference) failures:")
    print(f"     static  attributes: {static_cov.get(CONVERSATION,0)}  (structurally cannot)")
    print(f"     dynamic attributes: {conv_dyn}  ({round(100*conv_dyn/n)}% of real failures)")
    print(f"   -> {conv_dyn} real failures static is BLIND to, that Method E surfaces.")

    out = os.path.join(os.path.dirname(path), "real_convo_coverage.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"dataset": args.dataset, "n_failures": len(failures),
                   "static_coverage": dict(static_cov), "dynamic_coverage": dict(dynamic_cov),
                   "rows": rows}, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")
    return {"static": dict(static_cov), "dynamic": dict(dynamic_cov)}


if __name__ == "__main__":
    main()
