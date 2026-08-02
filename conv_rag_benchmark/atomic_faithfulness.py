"""
Atomic-claim faithfulness scorer (RAGAS / ARES style) — a STANDALONE prototype.

Re-scores the GOLD answers in a saved run file using decompose-then-entail, so it can
be compared head-to-head against G-Eval's holistic `gold_supported`. Run files live
under result/benchmark_quality/ and are named
``{rag}_{dataset}_t{turns}_c{convos}.json``.

Why: a single 1-5 G-Eval score can't see PARTIAL grounding — a Comparative gold that
states both halves of a comparison when the evidence supports only one half still
scores highish. Decomposing the gold into atomic claims and checking each claim is
entailed by the evidence catches exactly that: score = (# entailed claims) / (# claims).

This file changes NOTHING else. It only READS the saved JSON. Delete it and the
benchmark is unchanged.

Run:
    python -m conv_rag_benchmark.atomic_faithfulness --dataset qasper --n 30
    python -m conv_rag_benchmark.atomic_faithfulness --file result/benchmark_quality/qasper/vector_qasper_t10_c5.json --n 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

from .config import Config
from .llm import LLM
from .generation.gold_answer_generator import ABSTENTION

_DECOMPOSE_SYS = (
    "Break the ANSWER into a list of ATOMIC factual claims — each a single, "
    "self-contained statement asserting one fact. Do not add facts. Do not merge "
    "two facts into one claim. If the answer is a refusal/abstention or contains no "
    'factual assertion, return an empty list. Respond JSON: {"claims": ["...", ...]}'
)

_ENTAIL_SYS = (
    "You perform natural-language inference. For EACH claim, decide whether it is "
    "ENTAILED by (fully supported by, stated in) the EVIDENCE. Judge ONLY against the "
    "evidence, not world knowledge. A claim is entailed only if the evidence states or "
    "directly implies it. "
    'Respond JSON: {"results": [{"claim": "...", "entailed": true/false}, ...]} '
    "with one entry per claim, in order."
)


def _decompose(llm: LLM, gold: str):
    out = llm.chat_json(_DECOMPOSE_SYS, f"ANSWER: {gold}") or {}
    claims = out.get("claims") or []
    return [str(c).strip() for c in claims if str(c).strip()]


def _entail(llm: LLM, claims, evidence: str):
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    out = llm.chat_json(
        _ENTAIL_SYS,
        f"EVIDENCE:\n{evidence[:2500]}\n\nCLAIMS:\n{numbered}") or {}
    res = out.get("results") or []
    # align by position; missing -> treat as not-entailed (conservative)
    flags = []
    for i in range(len(claims)):
        flags.append(bool(res[i].get("entailed")) if i < len(res)
                     and isinstance(res[i], dict) else False)
    return flags


def score_gold(llm: LLM, gold: str, evidence: str):
    """Return (faithfulness in [0,1] or None, n_claims, n_entailed).

    None = no factual claims to ground (abstention / empty) — excluded from the mean,
    same convention G-Eval uses when a criterion is inapplicable."""
    if not gold or gold.strip().startswith(ABSTENTION[:20]):
        return None, 0, 0
    claims = _decompose(llm, gold)
    if not claims:
        return None, 0, 0
    flags = _entail(llm, claims, evidence or "")
    n_ent = sum(1 for f in flags if f)
    return n_ent / len(claims), len(claims), n_ent


def main(argv=None):
    ap = argparse.ArgumentParser(description="Atomic-claim faithfulness vs G-Eval gold_supported")
    ap.add_argument("--dataset", default="qasper",
                    help="dataset folder under result/benchmark_quality/ to search for a run")
    ap.add_argument("--file", default=None,
                    help="explicit path to a run file "
                         "({rag}_{dataset}_t{turns}_c{convos}.json)")
    ap.add_argument("--n", type=int, default=30, help="cap on number of golds scored (cost control)")
    ap.add_argument("--model", default="gpt-4o-mini", help="judge model for decompose+entail")
    args = ap.parse_args(argv)

    config = Config.load()
    path = args.file
    if not path:
        cands = [p for p in sorted(glob.glob(os.path.join(
            config.output_dir, args.dataset, "*_t*_c*.json"))) if not any(x in p for x in ("_docsim", "_summary"))]
        if not cands:
            print(f"!! no run files under {config.output_dir}/{args.dataset} "
                  f"(expected {{rag}}_{{dataset}}_t{{turns}}_c{{convos}}.json)")
            return
        path = cands[0]
    if not os.path.exists(path):
        print(f"!! not found: {path}")
        return
    data = json.load(open(path, encoding="utf-8"))

    llm = LLM(model=args.model, config=config)
    if not llm.available:
        print("!! needs an OpenAI key (RAG-DIVE/.env or OPENAI_API_KEY) to run the judge")
        return
    print(f"# scoring golds from {path}\n# model={args.model} | cap n={args.n}")

    # G-Eval's per-type gold_supported (for the head-to-head), if present
    geval_by_type = {qt: m.get("gold_supported")
                     for qt, m in (data.get("e_by_query_type") or {}).items()}

    per_type = defaultdict(list)      # query_type -> [faithfulness, ...]
    detail = []
    scored = 0
    for c in data.get("conversations", []):
        for t in c.get("turns", []):
            if scored >= args.n:
                break
            f, nc, ne = score_gold(llm, t.get("gold", ""), t.get("evidence", ""))
            scored += 1
            if f is None:
                continue
            per_type[t.get("query_type") or "?"].append(f)
            detail.append({"query_type": t.get("query_type"), "faithfulness": round(f, 3),
                           "n_claims": nc, "n_entailed": ne,
                           "gold": (t.get("gold") or "")[:200]})
        if scored >= args.n:
            break

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    all_f = [f for xs in per_type.values() for f in xs]
    print(f"\n{'query_type':<22}{'n':>4}{'atomic_faith':>14}{'G-Eval gold_sup':>18}")
    print(" " + "-" * 56)
    for qt in sorted(per_type, key=lambda q: mean(per_type[q])):
        af = mean(per_type[qt])
        ge = geval_by_type.get(qt)
        print(f"{qt:<22}{len(per_type[qt]):>4}{str(af):>14}{str(ge):>18}")
    print(" " + "-" * 56)
    print(f"{'OVERALL':<22}{len(all_f):>4}{str(mean(all_f)):>14}")

    out = os.path.join(os.path.dirname(path), "atomic_faithfulness.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"source": path, "model": args.model, "n_scored": scored,
                   "overall": mean(all_f),
                   "by_query_type": {qt: mean(v) for qt, v in per_type.items()},
                   "geval_gold_supported_by_type": geval_by_type,
                   "detail": detail}, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")
    return mean(all_f)


if __name__ == "__main__":
    main()
