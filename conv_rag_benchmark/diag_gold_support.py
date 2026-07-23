"""
Diagnostic: WHY does D ground gold better than E? Re-judge gold_supported per turn
and break it down by (a) turn depth and (b) query type, for both methods, so we can
localize E's lower support instead of guessing.

Run:  python -m conv_rag_benchmark.diag_gold_support <E_build.json> <D_build.json>
"""
import argparse
import json
from collections import defaultdict

from .config import Config
from .llm import LLM
from .quality_judge import _JUDGE_SYS

ABST = "Not answerable"


def judged_turns(path, judge):
    d = json.load(open(path, encoding="utf-8"))
    rows = []
    for c in d.get("conversations", []):
        for t in c.get("turns", []):
            gold = t.get("gold") or ""
            if not gold.strip() or gold.startswith(ABST) or t.get("guard_gave_up"):
                continue
            out = judge.chat_json(
                _JUDGE_SYS,
                f"EVIDENCE:\n{str(t.get('evidence',''))[:1600]}\n\n"
                f"QUESTION: {t.get('question','')}\nGOLD: {gold}") or {}
            rows.append({"turn_id": t.get("turn_id"),
                         "query_type": t.get("query_type"),
                         "supported": bool(out.get("gold_supported"))})
    return rows


def summarize(name, rows):
    by_depth = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_depth[r["turn_id"]][1] += 1
        by_depth[r["turn_id"]][0] += r["supported"]
        by_type[r["query_type"]][1] += 1
        by_type[r["query_type"]][0] += r["supported"]
    overall = sum(r["supported"] for r in rows) / len(rows) if rows else 0
    print(f"\n=== {name}: gold_supported = {overall:.3f}  (n={len(rows)}) ===")
    print("  by turn depth:")
    for k in sorted(by_depth, key=lambda x: (x is None, x)):
        s, n = by_depth[k]
        print(f"    turn {k}: {s}/{n} = {s/n:.2f}")
    print("  by query type:")
    for k, (s, n) in sorted(by_type.items()):
        print(f"    {k:<20} {s}/{n} = {s/n:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("e_build")
    ap.add_argument("d_build")
    args = ap.parse_args(argv)
    judge = LLM(model=getattr(Config.load(), "judge_model", None) or "gpt-4o")
    summarize("E (adaptive)", judged_turns(args.e_build, judge))
    summarize("D (static all-types)", judged_turns(args.d_build, judge))


if __name__ == "__main__":
    main()
