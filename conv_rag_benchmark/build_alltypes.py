"""
NEW METHOD (prof's idea): a NON-ADAPTIVE, exhaustive prober.

2-turn conversations: turn 0 is a factual seed (asked to the RAG); turn 1 probes the
RAG's answer with ONE question of EVERY type (all 8), instead of Method E's adaptive
"pick one type based on the outcome". The question: does brute-force "ask all types"
match E's smart adaptive selection?

Same #conversations as E (E keeps its own turn count). Scores question quality with the
same metrics as E and saves quality_alltypes.json for comparison.

Run:  python -m conv_rag_benchmark.build_alltypes --convos 20 --rag vector --dataset multihoprag
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .generation.gold_answer_generator import GoldAnswerGenerator
from .generation.query_generator import QUERY_TYPES, QueryGenerator, QueryTurn
from .geval import geval_items, geval_breakdown_by_type
from .graph.graph_builder import GraphBuilder, KnowledgeGraph
from .graph.retriever import GraphRetriever
from .interfaces.rag_interface import build_rag
from .llm import LLM
from .build_e_adaptive import _LABELS
from .generation.adaptive_generator import _GRADE_SYS, OUTCOMES
from .generation.gold_answer_generator import ABSTENTION

ALL_TYPES = list(QUERY_TYPES.keys())          # the 8 types


def _refine_quality(llm, question: str, history) -> str:
    """PRE-SEND QUALITY GATE (same as Method E's): score well-formedness given the
    conversation; if below 4/5, rewrite for clarity WITHOUT changing what it asks and
    WITHOUT naming an entity it intentionally refers to by pronoun."""
    hist = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:]) or "(start)"
    out = llm.chat_json(
        "Rate the QUESTION's well-formedness 1-5 (clear, answerable, self-contained "
        "GIVEN THE CONVERSATION; a pronoun is fine if the referent is clear). If it is "
        "below 4, rewrite it to be clearer WITHOUT changing what it asks and WITHOUT "
        "naming an entity it intentionally refers to by pronoun. "
        'JSON: {"score": 1-5, "improved": "<question, rewritten only if score<4>"}',
        f"CONVERSATION:\n{hist}\nQUESTION: {question}") or {}
    imp = str(out.get("improved") or "").strip()
    try:
        score = int(out.get("score") or 5)
    except (TypeError, ValueError):
        score = 5
    return imp if (score < 4 and imp) else question


def _gold_ok(llm, question: str, gold: str, ev) -> bool:
    """Verify the gold is grounded AND correct (same verifier as Method E)."""
    evid = " ".join(ev.chunks[:5])[:1800]
    out = llm.chat_json(
        "Judge the GOLD answer. Reply true ONLY if it is (a) fully supported by the "
        "EVIDENCE (no facts beyond it) AND (b) a correct answer to the QUESTION. "
        'JSON: {"ok": true/false}',
        f"EVIDENCE: {evid}\nQUESTION: {question}\nGOLD: {gold}") or {}
    return bool(out.get("ok"))


def _grade(judge, question: str, gold: str, rag_answer: str, is_unanswerable: bool) -> str:
    """Grade the RAG's answer into an outcome — same rubric Method E uses, so the
    failure profiles are directly comparable across D/E/F."""
    if not (rag_answer or "").strip():
        return "abstained"
    if judge.available:
        out = judge.chat_json(
            _GRADE_SYS, f"QUESTION: {question}\nGOLD: {gold}\n"
            f"UNANSWERABLE: {is_unanswerable}\nRAG ANSWER: {rag_answer}")
        if out and out.get("label") in OUTCOMES:
            return out["label"]
    low = (rag_answer or "").lower()
    if any(p in low for p in ("i don't know", "cannot answer", "not answerable",
                              "no information", "unable to")):
        return "abstained"
    if is_unanswerable:
        return "hallucinated"
    return "correct" if gold.lower()[:30] in low or low[:30] in gold.lower() else "wrong"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Non-adaptive all-types prober vs E")
    ap.add_argument("--convos", type=int, default=20)
    ap.add_argument("--rag", default="vector",
                    choices=["mock", "vector", "selfrag", "raptor", "graph"])
    ap.add_argument("--quality-gate", action="store_true",
                    help="pre-send refine loop on each question (targets well_formed)")
    ap.add_argument("--strict-gold", action="store_true",
                    help="strict composer + verify gold (targets gold_supported/correct)")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args(argv)

    config = Config.load(dataset=args.dataset, max_samples=max(args.convos, 50),
                         num_conversations=args.convos,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key"); return

    ds_label = _LABELS.get(args.dataset, args.dataset)
    out_dir = os.path.join(config.output_dir, ds_label)
    os.makedirs(out_dir, exist_ok=True)
    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model=config.judge_model, config=config)
    print(f"# {ds_label} | all-types prober | gen {config.gen_model} judge {config.judge_model}")

    seeds = DatasetLoader(config.dataset, max_samples=config.max_samples).load()
    graph_path = os.path.join(out_dir, "graph.json")
    if os.path.exists(graph_path):
        kg = KnowledgeGraph.load(graph_path)
        try:
            kg.chunk_embeddings = np.asarray(embedder.encode(kg.chunks), dtype=float)
        except Exception as _e:
            print(f"  (embed fail {str(_e)[:60]})")
    else:
        chunks = [c for s in seeds for c in s.context if c and c.strip()][:600]
        kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                          graph_mode="typed").build(chunks)
        kg.save(graph_path)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    rag = build_rag(args.rag, kg.chunks, config=config, llm=gen_llm,
                    retriever=retriever, embedder=embedder)
    qgen = QueryGenerator(config=config, llm=gen_llm)
    ggen = GoldAnswerGenerator(config=config, llm=gen_llm, strict=args.strict_gold)

    convos = []
    for i, seed in enumerate(seeds[: args.convos]):
        ctx = " ".join(seed.context[:3]) if seed.context else ""
        ev0 = retriever.retrieve(ctx or seed.question or "topic", k=6)
        seed_q = (seed.question or "").strip()
        if not seed_q:
            out = gen_llm.chat_json(
                'Generate ONE factual question answerable from the EVIDENCE. '
                'JSON: {"question":"..."}', f"EVIDENCE: {ev0.evidence_text(1500)}") or {}
            seed_q = str(out.get("question", "")).strip() or "What is the main topic?"
        seed_turn = QueryTurn(question=seed_q, query_type="Seed", turn_id=0,
                              difficulty=1, capability="", expected_failure="")
        seed_gold = ggen.generate(seed_turn, ev0, [])
        rag_ans0 = rag.answer(seed_q, []).answer or ""
        # probes REACT to the RAG's answer (prof: "from RAG ans"):
        history = [{"role": "user", "content": seed_q},
                   {"role": "assistant", "content": rag_ans0}]
        turns = [{"turn_id": 0, "query_type": "Seed", "question": seed_q,
                  "gold": seed_gold.gold_answer,
                  "evidence": " ".join(ev0.chunks[:5])[:1500], "rag_answer": rag_ans0}]

        for qt in ALL_TYPES:                          # ask ALL 8 types (non-adaptive)
            ev = retriever.retrieve(seed_q, k=8)
            cand = qgen.generate(qt, 1, ev, history)
            if args.quality_gate:                     # pre-send refine (as in E)
                cand.question = _refine_quality(gen_llm, cand.question, history)
            g = ggen.generate(cand, ev, history)
            if args.strict_gold and not cand.is_unanswerable and \
                    not g.gold_answer.startswith(ABSTENTION[:20]) and \
                    not _gold_ok(gen_llm, cand.question, g.gold_answer, ev):
                # one retry on wider evidence; else an explicit abstention gold
                ev = retriever.retrieve(cand.question, k=12)
                g = ggen.generate(cand, ev, history)
                if not g.gold_answer.startswith(ABSTENTION[:20]) and \
                        not _gold_ok(gen_llm, cand.question, g.gold_answer, ev):
                    g.gold_answer = ABSTENTION
            ans = rag.answer(cand.question, history).answer or ""
            outcome = _grade(judge, cand.question, g.gold_answer, ans,
                             qt == "Unanswerable")
            turns.append({"turn_id": 1, "query_type": qt, "question": cand.question,
                          "gold": g.gold_answer, "outcome": outcome,
                          "evidence": " ".join(ev.chunks[:5])[:1500], "rag_answer": ans})
        convos.append({"id": f"conv-{i:03d}", "turns": turns})
        print(f"  conv {i+1}/{args.convos} done", end="\r")

    # ---- score quality with G-Eval (same as E) ----
    items = [{"question": t["question"], "gold": t["gold"], "evidence": t["evidence"],
              "query_type": t["query_type"]}
             for c in convos for t in c["turns"] if (t.get("gold") or "").strip()]
    agg, scored = geval_items(items, model=config.judge_model)
    agg["n"] = len(items)
    by_type = geval_breakdown_by_type(scored)

    # ---- the RAG's FAILURE profile that F found (probe turns only) ----
    from collections import Counter
    probe_turns = [t for c in convos for t in c["turns"] if t.get("turn_id") == 1]
    outcomes = Counter(t["outcome"] for t in probe_turns if t.get("outcome"))
    n_probe = sum(outcomes.values()) or 1
    failure_rate = round((outcomes["wrong"] + outcomes["hallucinated"]) / n_probe, 3)
    fails_by_type = Counter(t["query_type"] for t in probe_turns
                            if t.get("outcome") in ("wrong", "hallucinated"))

    out = os.path.join(out_dir, "quality_alltypes.json")
    json.dump({"method": "all-types (non-adaptive, 2-turn)", "rag": args.rag,
               "quality": agg, "by_query_type": by_type,
               "rag_failure": {"n_turns": n_probe, "outcomes": dict(outcomes),
                               "failure_rate": failure_rate,
                               "failures_by_probe": dict(fails_by_type)},
               "conversations": convos}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n\nall-types quality: well_formed {agg['well_formed']} | "
          f"gold_supported {agg['gold_supported']} | gold_correct {agg['gold_correct']} "
          f"(n={agg['n']})")
    print(f"\n===== what F found about the '{args.rag}' RAG =====")
    print(f"probe turns          : {n_probe}")
    print(f"outcome distribution : {dict(outcomes)}")
    print(f"RAG failure rate     : {failure_rate}  (wrong + hallucinated)")
    print(f"failures by probe    : {dict(fails_by_type)}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
