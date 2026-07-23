"""
Check the GROUNDING GUARD for type bias (the "roundtrip filter makes datasets
easier" risk, Alberti et al. 2019): the guard retries a turn up to 3 times and,
if no attempt yields a grounded gold, keeps the last candidate with the
ABSTENTION string as its gold. That abstention is the guard's visible footprint,
so the give-up rate per query type / per turn tells us which types the guard
fails on most. A strongly uneven rate means the benchmark is systematically
easier on the surviving types.

Intentionally-Unanswerable turns are excluded: their abstention gold is the
CORRECT answer, not a guard failure.

Run:  python -m conv_rag_benchmark.gate_giveup [file.json ...]
      (no args: scans conv_rag_benchmark/output/*/quality_e*.json)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from .generation.gold_answer_generator import ABSTENTION

OUT = "conv_rag_benchmark/output/gate_giveup.json"


def _gave_up(turn) -> bool:
    # intentionally-Unanswerable turns: abstention gold is CORRECT, not a failure
    # (D builds mark them only via query_type; E builds set is_unanswerable)
    if turn.get("is_unanswerable") or turn.get("query_type") == "Unanswerable":
        return False
    if "guard_gave_up" in turn:          # new builds record the verdict directly
        return bool(turn["guard_gave_up"])
    # old builds: infer from the abstention footprint (misses strict-gold failures
    # whose final gold was substantive text that merely failed verification)
    gold = str(turn.get("gold") or "")
    return gold.strip().startswith(ABSTENTION[:20])


def analyse(path):
    data = json.load(open(path, encoding="utf-8"))
    convs = data.get("conversations") or []
    by_type = defaultdict(lambda: [0, 0])   # qtype -> [gave_up, total]
    by_turn = defaultdict(lambda: [0, 0])   # turn_id -> [gave_up, total]
    for conv in convs:
        for t in conv.get("turns", []):
            if t.get("is_unanswerable"):
                continue
            bad = _gave_up(t)
            for key, table in ((t.get("query_type", "?"), by_type),
                               (t.get("turn_id", "?"), by_turn)):
                table[key][1] += 1
                table[key][0] += int(bad)
    # RAG failure rate per type, with vs without the gave-up turns whose answer
    # key is untrustworthy (exclusion re-check: does the per-type conclusion
    # survive once contaminated turns are dropped?).
    fail_all = defaultdict(lambda: [0, 0])    # qtype -> [failures, total]
    fail_clean = defaultdict(lambda: [0, 0])  # same, gave-up turns excluded
    for conv in convs:
        for t in conv.get("turns", []):
            qt = t.get("query_type", "?")
            bad_outcome = t.get("outcome") in ("wrong", "hallucinated")
            fail_all[qt][1] += 1
            fail_all[qt][0] += int(bad_outcome)
            if not _gave_up(t):
                fail_clean[qt][1] += 1
                fail_clean[qt][0] += int(bad_outcome)

    def _rates(table):
        return {k: {"failures": v[0], "total": v[1],
                    "rate": round(v[0] / v[1], 3) if v[1] else None}
                for k, v in sorted(table.items())}

    return {"n_conversations": len(convs),
            "by_type": {k: {"gave_up": v[0], "total": v[1],
                            "rate": round(v[0] / v[1], 3) if v[1] else None}
                        for k, v in sorted(by_type.items())},
            "by_turn": {str(k): {"gave_up": v[0], "total": v[1],
                                 "rate": round(v[0] / v[1], 3) if v[1] else None}
                        for k, v in sorted(by_turn.items())},
            "failure_by_type_all": _rates(fail_all),
            "failure_by_type_clean": _rates(fail_clean)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help="quality_e*.json build outputs (default: scan output/)")
    args = ap.parse_args(argv)
    files = args.files or sorted(
        glob.glob("conv_rag_benchmark/output/*/quality_e*.json"))

    results = {}
    for f in files:
        try:
            results[f] = analyse(f)
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        r = results[f]
        print(f"\n=== {f}  ({r['n_conversations']} conversations) ===")
        print(f"{'query type':<18}{'gave up':>8}{'total':>7}{'rate':>8}")
        for k, v in r["by_type"].items():
            print(f"{k:<18}{v['gave_up']:>8}{v['total']:>7}"
                  f"{('' if v['rate'] is None else format(v['rate'], '.3f')):>8}")
        print(f"{'turn':<18}{'gave up':>8}{'total':>7}{'rate':>8}")
        for k, v in r["by_turn"].items():
            print(f"turn {k:<13}{v['gave_up']:>8}{v['total']:>7}"
                  f"{('' if v['rate'] is None else format(v['rate'], '.3f')):>8}")
        print(f"{'RAG failure rate':<18}{'all':>8}{'clean':>8}   (clean = gave-up turns excluded)")
        for k in r["failure_by_type_all"]:
            a = r["failure_by_type_all"][k]
            c = r["failure_by_type_clean"].get(k, {})
            fa = "" if a["rate"] is None else format(a["rate"], ".3f")
            fc = "" if c.get("rate") is None else format(c["rate"], ".3f")
            print(f"{k:<18}{fa:>8}{fc:>8}   (n={a['total']}->{c.get('total', 0)})")

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
