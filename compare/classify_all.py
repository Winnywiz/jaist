"""
compare/classify_all.py — run the failure classifier over every generated conversation log.

Reuses the friend's ``failure/`` package UNMODIFIED (imports its classifier, loader, and
taxonomy). For each ``compare/result/<method>/<rag>/conversation_NNN/conversation.json``
it extracts the failed turns, classifies them, and writes
``conversation_failuretypes.json`` in the same directory — the exact file the attribution
step expects. Runs in a single process (one shared LLM client) so it is far cheaper than
invoking ``failure.run`` once per file, while producing identical output.

At the end it prints an aggregate attribution table per method×rag, so you can see how the
failures break down by taxonomy type/layer across the conversation-generation methods.

Usage (from repo root):
    python -m compare.classify_all                 # classify all logs (skips done)
    python -m compare.classify_all --overwrite     # re-classify everything
    python -m compare.classify_all --offline       # heuristic, no API
    python -m compare.classify_all --glob "compare/result/proposed/*/*/conversation.json"
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
from collections import Counter, defaultdict

from conv_rag_benchmark.config import Config
from conv_rag_benchmark.llm import LLM

from failure.classifier import FailureClassifier, summarize
from failure.log_loader import load_failed_turns


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify all compare/ conversation logs.")
    ap.add_argument("--glob", default="compare/result/**/conversation_*/conversation.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-classify even if a failuretypes.json already exists")
    ap.add_argument("--offline", action="store_true", help="heuristic, no API calls")
    ap.add_argument("--include-abstained", action="store_true",
                    help="also treat answerable turns the RAG abstained on as failures")
    args = ap.parse_args(argv)

    files = sorted(_glob.glob(args.glob, recursive=True))
    if not files:
        print(f"no logs matched {args.glob!r}")
        return

    config = Config.load()
    use_llm = config.has_openai and not args.offline
    llm = LLM(model=config.gen_model, config=config) if use_llm else None
    print(f"# classifier LLM: {'on (' + config.gen_model + ')' if use_llm else 'OFFLINE heuristic'}")
    clf = FailureClassifier(llm)

    # method/rag -> aggregate counters
    agg = defaultdict(lambda: {"convs": 0, "failed": 0,
                               "ftype": Counter(), "layer": Counter(),
                               "cause": Counter()})
    done = skipped = 0
    for f in files:
        # layout: .../<method>/<rag>/<dataset>/conversation_NNN/conversation.json
        parts = f.replace("\\", "/").split("/")
        method, rag, ds = parts[-5], parts[-4], parts[-3]
        key = f"{method} / {rag} / {ds}"
        out_path = os.path.splitext(f)[0] + "_failuretypes.json"
        agg[key]["convs"] += 1

        if os.path.exists(out_path) and not args.overwrite:
            # still fold its counts into the aggregate so the table is complete
            try:
                rep = json.load(open(out_path, encoding="utf-8")).get("summary", {})
                agg[key]["failed"] += rep.get("n", 0)
                agg[key]["ftype"].update(rep.get("by_failure_type", {}))
                agg[key]["layer"].update(rep.get("by_layer", {}))
                agg[key]["cause"].update(rep.get("by_conversational_cause", {}))
                skipped += 1
                continue
            except Exception:
                pass  # unreadable -> reclassify

        turns = load_failed_turns(f, include_abstained=args.include_abstained)
        results = clf.classify_all(turns)
        report = summarize(results)
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"source": f, "summary": report,
                       "cases": [r.to_dict() for r in results]},
                      fw, ensure_ascii=False, indent=2)
        agg[key]["failed"] += report["n"]
        agg[key]["ftype"].update(report["by_failure_type"])
        agg[key]["layer"].update(report["by_layer"])
        agg[key]["cause"].update(report["by_conversational_cause"])
        done += 1
        print(f"  {key:<26} {os.path.basename(os.path.dirname(f))}: "
              f"{report['n']} failed -> {dict(report['by_failure_type'])}")

    print(f"\n# classified {done} logs, skipped {skipped} already-done.")
    print("\n" + "=" * 92)
    print(" AGGREGATE FAILURE ATTRIBUTION per method / rag")
    print("=" * 92)
    print(f"{'method / rag':<26}{'convs':>6}{'failed':>7}  {'by failure_type':<44}{'by layer'}")
    print("-" * 92)
    for k in sorted(agg):
        a = agg[k]
        ft = ", ".join(f"{n}{t[:4]}" for t, n in a["ftype"].most_common())
        ly = ", ".join(f"{n}{t[:3]}" for t, n in a["layer"].most_common())
        print(f"{k:<26}{a['convs']:>6}{a['failed']:>7}  {ft:<44}{ly}")
    if use_llm:
        print(f"\n# classifier usage ({config.gen_model}): {llm.cost_report()}")


if __name__ == "__main__":
    main()
