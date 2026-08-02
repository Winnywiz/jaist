"""
shared/harness.py — runs the comparison and scores it.

Pipeline:
  1. load a dataset  ->  2. inject controlled failures (shared/setup.py)  ->
  3. keep only failures that actually fired  ->  4. run EVERY method (DYNAMIC + STATIC)
  on the same cases  ->  5. score attribution accuracy vs the injected ground truth  ->
  6. save a JSON to compare/result/.

The methods themselves live in DYNAMIC/method.py (proposed) and STATIC/method.py
(baselines); this file only orchestrates and measures.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

from conv_rag_benchmark.config import Config, get_logger
from conv_rag_benchmark.datasets.loader import DatasetLoader
from conv_rag_benchmark.embeddings import Embedder
from conv_rag_benchmark.llm import LLM

from .setup import (ControlledRAG, OfflineLLM, build_injected_cases, did_fail, set_judge)
from ..STATIC.method import SingleTurn, XieStaticDecomp, XieFollowupOnly
from ..DYNAMIC.method import DynamicFollowup

logger = get_logger("dynamicqa.harness")

# compare/shared/harness.py -> two levels up = the compare/ package dir
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "result", "attribution")


def build_methods(rag: ControlledRAG, llm: LLM):
    """Every method under comparison. STATIC baselines first, PROPOSED dynamic last."""
    return [SingleTurn(),                       # STATIC control
            XieStaticDecomp(rag, llm),          # STATIC (Xie core)
            XieFollowupOnly(rag, llm),          # STATIC (Xie follow-up)
            DynamicFollowup(rag)]               # PROPOSED (dynamic)


def score(cases, preds: Dict[str, List[str]]) -> Dict:
    out = {"n": len(cases), "methods": {}}
    for method, plist in preds.items():
        correct = sum(1 for c, p in zip(cases, plist) if p == c.true_category)
        by_cat = defaultdict(lambda: [0, 0])
        confusion = Counter()
        for c, p in zip(cases, plist):
            by_cat[c.true_category][1] += 1
            if p == c.true_category:
                by_cat[c.true_category][0] += 1
            else:
                confusion[f"{c.true_category}->{p}"] += 1
        out["methods"][method] = {
            "accuracy": round(correct / (len(cases) or 1), 3),
            "by_category": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]}
                            for k, v in sorted(by_cat.items())},
            "top_confusions": dict(confusion.most_common(5)),
        }
    return out


def print_report(report: Dict):
    methods = list(report["methods"])
    cats = sorted({k for m in report["methods"].values() for k in m["by_category"]})
    print("\n" + "=" * 78)
    print(" DYNAMIC vs STATIC — failure-ATTRIBUTION accuracy   (cases: %d)" % report["n"])
    print("=" * 78)
    head = f"{'category':<14}" + "".join(f"{m.split('.')[0]:>16}" for m in methods)
    print(head)
    print(" " + "-" * 76)
    for cat in cats:
        row = f"{cat:<14}"
        for m in methods:
            acc = report["methods"][m]["by_category"].get(cat, {}).get("acc", "-")
            row += f"{str(acc):>16}"
        print(row)
    print(" " + "-" * 76)
    row = f"{'OVERALL':<14}"
    for m in methods:
        row += f"{report['methods'][m]['accuracy']:>16}"
    print(row)
    print("\n legend:", " | ".join(methods))
    base = report["methods"][methods[0]]["accuracy"]
    prop = report["methods"][methods[-1]]["accuracy"]
    print(f"\n proposed(dynamic) - single_turn = {prop - base:+.3f}")
    for m in methods:
        print(f"   {m:<22} confusions: {report['methods'][m]['top_confusions']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="DYNAMIC vs STATIC failure-attribution accuracy")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--n", type=int, default=20, help="seeds (3 injected cases each)")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffles WHICH documents get injected -> independent runs")
    ap.add_argument("--offline", action="store_true",
                    help="no API: plumbing smoke test only, numbers are NOT evidence")
    args = ap.parse_args(argv)

    # pool bigger than n so different seeds draw largely different document subsets
    config = Config.load(dataset=args.dataset, max_samples=max(args.n * 3, 40),
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    os.makedirs(RESULT_DIR, exist_ok=True)

    use_llm = config.has_openai and not args.offline
    llm = LLM(model=config.gen_model, config=config) if use_llm else OfflineLLM()
    embedder = Embedder(config=config, llm=llm) if use_llm else None
    if use_llm:
        set_judge(LLM(model=config.judge_model, config=config))  # semantic correctness
    print(f"# dataset={args.dataset} | n={args.n} | "
          f"LLM={'on' if use_llm else 'OFFLINE (smoke test only)'}")

    seeds = DatasetLoader(args.dataset, max_samples=config.max_samples).load()
    import random as _random
    _random.Random(args.seed).shuffle(seeds)     # seed picks which docs get injected
    corpus = [c for s in seeds for c in s.context if c and c.strip()]
    rag = ControlledRAG(corpus, config, llm, embedder)

    all_cases = build_injected_cases(seeds, rag, args.n, llm)
    # Only score cases where the injected failure ACTUALLY fired (the RAG really
    # failed). A case that didn't fail has no failure to attribute.
    cases = [c for c in all_cases if did_fail(c)]
    fired = Counter(c.true_category for c in cases)
    built = Counter(c.true_category for c in all_cases)
    print("\n# cases that actually FAILED (fired) / built, per category:")
    for cat in sorted(built):
        print(f"#   {cat:<14} {fired.get(cat,0)}/{built[cat]}")
    if not cases:
        print("!! no injected failures fired — nothing to attribute."); return

    methods = build_methods(rag, llm)
    preds = {m.name: [m.predict_category(c) for c in cases] for m in methods}

    report = score(cases, preds)
    print_report(report)
    if not use_llm:
        print("\n  !! OFFLINE: numbers are a plumbing check, NOT evidence. "
              "Re-run without --offline.")

    dump = {"config": {"dataset": args.dataset, "n_seeds": args.n,
                       "n_cases": len(cases), "llm": use_llm},
            "report": report,
            "cases": [{"case_id": c.case_id, "true": c.true_category,
                       "question": c.question, "rag_answer": c.rag_answer,
                       "history": c.history, "antecedent_question": c.antecedent_question,
                       "source_doc_for_question": c.gold_passage,
                       "docs_rag_saw_to_answer": c.given_context,
                       **{m.name: preds[m.name][i] for m in methods}}
                      for i, c in enumerate(cases)]}
    path = os.path.join(RESULT_DIR, f"attribution_{args.dataset}_seed{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(dump, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {path}")
    return report


if __name__ == "__main__":
    main()
