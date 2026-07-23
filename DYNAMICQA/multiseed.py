"""
multiseed.py — the review-defensible version of the comparison.

Runs the attribution experiment across several INDEPENDENT seeds (each seed shuffles
which documents get injected, so the runs are not just re-rolls of the same cases) and
reports, per method:

  * per-category accuracy  (Retrieval / Generation / Conversation): mean +- spread
  * MACRO accuracy         (unweighted mean of the 3 categories) — do NOT lead with this;
                            see the warning below.
  * MICRO accuracy         (case-weighted overall) — kept for reference.

!! DO NOT REPORT THE 3-CLASS MACRO AS THE HEADLINE. It averages in Conversation, which
   the STATIC methods cannot emit at all by construction, so they score 0.000 there on
   every seed. Part of the resulting "win" is design, not measurement (77% of the gap on
   multihoprag). Run `python -m DYNAMICQA.fair_macro` for the defensible version:
   macro_shared (Retrieval+Generation only, equal label space) as the accuracy claim, with
   Conversation reported separately as a capability claim.

Why macro: in the single-seed run Retrieval was 18 of 27 cases, so the plain overall was
mostly "did you get Retrieval right". Macro weights each failure TYPE equally, which is
what the research question is actually about.

Run from repo root:
    python -m DYNAMICQA.multiseed --dataset multihoprag --n 20 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev

from .shared.harness import main as run_once, RESULT_DIR

CATS = ["Retrieval", "Generation", "Conversation"]


def _macro(bycat: dict) -> float:
    accs = [bycat[c]["acc"] for c in CATS if c in bycat]
    return round(mean(accs), 3) if accs else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="multi-seed attribution with macro-average + spread")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)

    # method_name -> {"macro":[per seed], "micro":[...], "Retrieval":[...], ...}
    acc = {}
    per_seed_fired = []
    for sd in args.seeds:
        print(f"\n########## SEED {sd} ##########")
        report = run_once(["--dataset", args.dataset, "--n", str(args.n), "--seed", str(sd)])
        if not report:
            print(f"!! seed {sd} produced no report (no failures fired?) — skipped")
            continue
        fired = {c: report["methods"][next(iter(report["methods"]))]["by_category"].get(c, {}).get("n", 0)
                 for c in CATS}
        per_seed_fired.append({"seed": sd, "fired": fired})
        for m, r in report["methods"].items():
            d = acc.setdefault(m, {k: [] for k in ["macro", "micro"] + CATS})
            d["macro"].append(_macro(r["by_category"]))
            d["micro"].append(r["accuracy"])
            for c in CATS:
                if c in r["by_category"]:
                    d[c].append(r["by_category"][c]["acc"])

    if not acc:
        print("no seeds produced results"); return

    def ms(xs):
        if not xs:
            return "   -   "
        m = mean(xs)
        s = pstdev(xs) if len(xs) > 1 else 0.0
        return f"{m:.3f}±{s:.3f}"

    methods = list(acc)
    print("\n" + "=" * 92)
    print(f" MULTI-SEED ATTRIBUTION  |  {args.dataset}  |  seeds={args.seeds}  |  n={args.n}")
    print("=" * 92)
    print("  fired per seed:", per_seed_fired)
    print(f"\n{'metric':<16}" + "".join(f"{m.split('.')[0]:>19}" for m in methods))
    print(" " + "-" * 90)
    for row in CATS + ["micro", "macro"]:
        label = ("MACRO (3-class*)" if row == "macro"
                 else "micro (overall)" if row == "micro" else row)
        line = f"{label:<16}"
        for m in methods:
            line += f"{ms(acc[m][row]):>19}"
        print(line)
    print(" " + "-" * 90)
    print(" legend:", " | ".join(methods))
    print("\n NOTE: mean±sd over", len(args.seeds), "seeds. With only a few seeds this is a")
    print("       rough spread, not a tight 95% CI — add more seeds to narrow it.")
    print("\n * The 3-class MACRO is NOT the headline: it averages in Conversation, which the")
    print("   static methods cannot emit by construction (0.000 on every seed). Run")
    print("   `python -m DYNAMICQA.fair_macro` for the defensible split: macro_shared")
    print("   (Retrieval+Generation) as the accuracy claim, Conversation as a capability.")

    out = {"dataset": args.dataset, "n": args.n, "seeds": args.seeds,
           "fired_per_seed": per_seed_fired,
           "per_method": {m: {k: {"mean": round(mean(v), 3) if v else None,
                                  "sd": round(pstdev(v), 3) if len(v) > 1 else 0.0,
                                  "values": v}
                              for k, v in d.items()}
                          for m, d in acc.items()}}
    path = os.path.join(RESULT_DIR, f"multiseed_{args.dataset}.json")
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(out, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {path}")
    return out


if __name__ == "__main__":
    main()
