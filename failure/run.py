"""Classify the failed turns in a benchmark conversation log.

Pulls failed turns out of a `*_full.json` log, classifies each along the two taxonomy
axes (RAG failure type + conversational cause), prints a summary, and writes a JSON
report next to the log (or to --out).

Run from the repo root (the folder holding `conv_rag_benchmark/`):

    python -m failure.run --file result/benchmark_quality/mtrag/crag_mtrag_t8_c30_typedgraph_full.json
    python -m failure.run --file <log.json> --include-abstained --limit 50
    python -m failure.run --file <log.json> --offline    # heuristic, no API calls
"""
from __future__ import annotations

import argparse
import json
import os

from conv_rag_benchmark.config import Config
from conv_rag_benchmark.llm import LLM

from .classifier import FailureClassifier, summarize
from .log_loader import load_failed_turns


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify failed turns from a benchmark log.")
    ap.add_argument("--file", required=True, help="path to a benchmark *_full.json log")
    ap.add_argument("--out", default=None, help="output JSON path (default: alongside the log)")
    ap.add_argument("--include-abstained", action="store_true",
                    help="also count answerable turns the RAG abstained on as failures")
    ap.add_argument("--limit", type=int, default=None, help="classify at most N failed turns")
    ap.add_argument("--offline", action="store_true", help="no API: heuristic fallback only")
    args = ap.parse_args(argv)

    turns = load_failed_turns(args.file, include_abstained=args.include_abstained)
    if args.limit:
        turns = turns[:args.limit]
    print(f"# log={os.path.basename(args.file)} | failed turns={len(turns)}")
    if not turns:
        print("!! no failed turns found — nothing to classify."); return

    config = Config.load()
    use_llm = config.has_openai and not args.offline
    llm = LLM(model=config.gen_model, config=config) if use_llm else None
    print(f"# classifier LLM: {'on (' + config.gen_model + ')' if use_llm else 'OFFLINE heuristic'}")

    clf = FailureClassifier(llm)
    results = clf.classify_all(turns)
    report = summarize(results)

    print("\n=== failure classification summary ===")
    print(f" failed turns classified : {report['n']}")
    print(f" by RAG failure type     : {report['by_failure_type']}")
    print(f" by layer                : {report['by_layer']}")
    print(f" by conversational cause : {report['by_conversational_cause']}")
    print(f" by query type           : {report['by_query_type']}")

    out_path = args.out or (os.path.splitext(args.file)[0] + "_failuretypes.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({"source": args.file, "summary": report,
                   "cases": [r.to_dict() for r in results]},
                  fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
