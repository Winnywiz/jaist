"""
fair_macro.py — re-analysis that makes the DYNAMIC-vs-STATIC win defensible.

THE PROBLEM THIS FIXES
----------------------
The headline macro averages three categories: Retrieval, Generation, Conversation.
But the STATIC methods cannot emit "Conversation" AT ALL -- by construction. They
never see the RAG's answer, so coreference failures are structurally invisible to
them (see STATIC/method.py). They therefore score exactly 0.000 on that category
on every seed, in every dataset.

Averaging that column into a 3-class macro means part of the "win" is guaranteed by
design, not measured. A reviewer will spot this immediately.

WHAT THIS REPORTS INSTEAD
-------------------------
Three separate quantities, because they answer three different questions:

  macro_shared   Unweighted mean over ONLY the categories every method can emit
                 (Retrieval, Generation). This is the FAIR accuracy comparison:
                 same task, same label space, nobody handicapped.

  conversation   Reported ALONE, as a CAPABILITY result, not an accuracy win:
                 "static = 0.000 by construction; dynamic reaches X". The claim is
                 reachability, not superiority-on-equal-footing.

  macro_3class   The original headline, kept for continuity -- and decomposed, so
                 the share of the gap coming from the structurally-blind category
                 is stated out loud rather than hidden.

Also computes PAIRED per-seed deltas (dynamic - best static). Seeds are shared
across methods, so pairing is more informative than comparing two means.

Reads existing result/multiseed_*.json. No API calls, no re-running.

Usage:
    python -m DYNAMICQA.fair_macro
"""
from __future__ import annotations

import glob
import json
import os
from statistics import mean, pstdev
from typing import Dict, List

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "result", "attribution")

ALL_CATS = ["Retrieval", "Generation", "Conversation"]
# Categories every method under comparison is structurally capable of emitting.
SHARED_CATS = ["Retrieval", "Generation"]
BLIND_CAT = "Conversation"

DYNAMIC_KEY = "4.dynamic_followup"


def _per_seed(method: Dict, cats: List[str]) -> List[float]:
    """Unweighted macro over `cats`, computed per seed (not mean-of-means)."""
    cols = [method[c]["values"] for c in cats if c in method and method[c]["values"]]
    if not cols:
        return []
    n = min(len(c) for c in cols)
    return [mean(col[i] for col in cols) for i in range(n)]


def _fmt(xs: List[float]) -> str:
    if not xs:
        return "     -     "
    s = pstdev(xs) if len(xs) > 1 else 0.0
    return f"{mean(xs):.3f}+/-{s:.3f}"


def analyse(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    pm = d["per_method"]
    ds = d["dataset"]

    rows = {}
    for m, v in pm.items():
        rows[m] = {
            "macro_3class": _per_seed(v, ALL_CATS),
            "macro_shared": _per_seed(v, SHARED_CATS),
            "conversation": v.get(BLIND_CAT, {}).get("values", []),
            "retrieval": v.get("Retrieval", {}).get("values", []),
            "generation": v.get("Generation", {}).get("values", []),
        }

    statics = [m for m in rows if m != DYNAMIC_KEY]
    # Best static = highest mean on the FAIR metric, not the 3-class one.
    best_static = max(statics, key=lambda m: mean(rows[m]["macro_shared"] or [0]))

    dyn, base = rows[DYNAMIC_KEY], rows[best_static]

    def paired(metric: str) -> List[float]:
        a, b = dyn[metric], base[metric]
        n = min(len(a), len(b))
        return [a[i] - b[i] for i in range(n)]

    gap3 = mean(dyn["macro_3class"]) - mean(base["macro_3class"])
    gap_shared = mean(dyn["macro_shared"]) - mean(base["macro_shared"])
    # Conversation's arithmetic contribution to the 3-class gap: it is one of three
    # equally-weighted columns, and static is 0 there, so it contributes conv/3.
    conv_contrib = mean(dyn["conversation"]) / 3.0 if dyn["conversation"] else 0.0

    print("\n" + "=" * 78)
    print(f" {ds}   |  seeds={d['seeds']}  n={d['n']}")
    print("=" * 78)
    print(f"{'method':<24}{'macro_shared (FAIR)':>22}{'macro_3class':>18}{'Conversation':>16}")
    print(" " + "-" * 76)
    for m, v in rows.items():
        star = "  <-- proposed" if m == DYNAMIC_KEY else ""
        print(f"{m:<24}{_fmt(v['macro_shared']):>22}{_fmt(v['macro_3class']):>18}"
              f"{_fmt(v['conversation']):>16}{star}")
    print(" " + "-" * 76)

    print(f"\n vs best static on the FAIR metric ({best_static}):")
    print(f"   macro_shared  gap  = {gap_shared:+.3f}   paired per-seed: "
          f"{[round(x, 3) for x in paired('macro_shared')]}")
    print(f"   macro_3class  gap  = {gap3:+.3f}   paired per-seed: "
          f"{[round(x, 3) for x in paired('macro_3class')]}")
    if gap3:
        print(f"   -> Conversation contributes {conv_contrib:+.3f} of the {gap3:+.3f} "
              f"3-class gap = {100 * conv_contrib / gap3:.0f}% of the headline advantage,")
        print(f"      from a category static CANNOT emit by construction.")
    print(f"\n   Retrieval   dyn {_fmt(dyn['retrieval'])}  vs static {_fmt(base['retrieval'])}"
          f"   paired: {[round(x, 3) for x in paired('retrieval')]}")
    print(f"   Generation  dyn {_fmt(dyn['generation'])}  vs static {_fmt(base['generation'])}"
          f"   paired: {[round(x, 3) for x in paired('generation')]}")
    print(f"\n   Conversation (CAPABILITY, not an accuracy win):")
    print(f"      static  = 0.000 on every seed -- structurally cannot emit this label")
    print(f"      dynamic = {_fmt(dyn['conversation'])} -- reaches it, but "
          f"misses {100 * (1 - mean(dyn['conversation'])):.0f}% of the time")

    return {
        "dataset": ds, "seeds": d["seeds"], "n": d["n"],
        "best_static_on_fair_metric": best_static,
        "per_method": {m: {k: {"mean": round(mean(v), 3) if v else None,
                               "sd": round(pstdev(v), 3) if len(v) > 1 else 0.0,
                               "values": [round(x, 3) for x in v]}
                           for k, v in r.items()} for m, r in rows.items()},
        "gaps_vs_best_static": {
            "macro_shared": {"mean_gap": round(gap_shared, 3),
                             "paired_per_seed": [round(x, 3) for x in paired("macro_shared")]},
            "macro_3class": {"mean_gap": round(gap3, 3),
                             "paired_per_seed": [round(x, 3) for x in paired("macro_3class")]},
            "conversation_share_of_3class_gap": (round(conv_contrib / gap3, 3) if gap3 else None),
        },
    }


def main() -> None:
    paths = sorted(glob.glob(os.path.join(RESULT_DIR, "multiseed_*.json")))
    if not paths:
        print("no multiseed_*.json in result/ -- run DYNAMICQA.multiseed first")
        return
    out = [analyse(p) for p in paths]

    print("\n" + "=" * 78)
    print(" HOW TO REPORT THIS")
    print("=" * 78)
    print(" 1. macro_shared is the accuracy comparison. Same label space, fair fight.")
    print(" 2. Conversation is a CAPABILITY claim, reported separately:")
    print("    static cannot emit it by construction; dynamic reaches it (imperfectly).")
    print(" 3. Do NOT lead with macro_3class: it averages in a column static cannot")
    print("    score on, so part of that gap is design, not measurement.")
    print(" 4. With 3 seeds the +/- is a rough spread, NOT a confidence interval.")

    path = os.path.join(RESULT_DIR, "fair_macro.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"note": "macro_shared = Retrieval+Generation only (categories every "
                           "method can emit). Conversation excluded from the accuracy "
                           "comparison because STATIC cannot emit it by construction; "
                           "reported separately as a capability result.",
                   "datasets": out}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
