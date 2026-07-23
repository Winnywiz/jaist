"""
Confirm the pre-send QUALITY GATE: run E with vs without the gate K times and report
MEAN +/- STD on the quality metrics, so the well-formed gain (and gold-correct cost)
is shown to be real, not a single-run fluke.

Run:  python -u -m conv_rag_benchmark.repeat_qgate --reps 3 --convos 50
"""
import argparse
import json
import os
from statistics import mean, pstdev

from .build_e_adaptive import main as run_e

METRICS = ["well_formed", "gold_supported", "gold_correct"]
OUT = "conv_rag_benchmark/output/qgate_repeated.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--convos", type=int, default=50)
    ap.add_argument("--dataset", default="multihoprag")
    args = ap.parse_args()

    base = ["--convos", str(args.convos), "--turns", "3", "--rag", "vector",
            "--dataset", args.dataset]
    collected = {"no-gate": {m: [] for m in METRICS},
                 "gate": {m: [] for m in METRICS}}

    for cond, extra in [("no-gate", []), ("gate", ["--quality-gate"])]:
        for r in range(args.reps):
            print(f"\n########## {cond} rep {r+1}/{args.reps} ##########", flush=True)
            q = run_e(base + extra)
            if q:
                for m in METRICS:
                    collected[cond][m].append(q.get(m))
            json.dump(collected, open(OUT, "w", encoding="utf-8"), indent=2)
            print(f"  [saved partial -> {OUT}]", flush=True)

    def ms(v):
        v = [x for x in v if x is not None]
        return f"{mean(v):.3f}+-{(pstdev(v) if len(v) > 1 else 0):.3f}" if v else "-"

    print("\n" + "=" * 60)
    print(f" QUALITY GATE confirmation — mean+-std over {args.reps} reps ({args.dataset})")
    print("=" * 60)
    print(f"{'metric':<16}{'no-gate':>16}{'gate':>16}")
    for m in METRICS:
        print(f"{m:<16}{ms(collected['no-gate'][m]):>16}{ms(collected['gate'][m]):>16}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
