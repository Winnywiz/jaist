"""
Per-METHOD consolidated metrics: the building methods D (random-type ablation),
E (adaptive controller), F (exhaustive all-types), each evaluated on every dataset
with the metrics that apply to a benchmark-BUILDING method.

(Attribution accuracy is NOT here — it belongs to the attribution arms, not the
building methods; see consolidate_metrics.py / attribution_repeated_*.json.)

Reads the build files (eval_{D,E,F}_<dataset>.json); question quality is read from
each build's stored 'quality' block, give-up + verbatim are computed offline.

Run:  python -m conv_rag_benchmark.consolidate_by_method
"""
import json
import os

from .probe_difficulty import verbatim_answerable

OUT = "conv_rag_benchmark/output/consolidated_by_method.json"
ABST = "Not answerable"
DATASETS = ["qasper", "arxivcs", "hfdocqa", "mlarxiv"]
METHODS = ["D", "E", "F"]
METHOD_DESC = {"D": "random-type ablation", "E": "adaptive controller",
               "F": "exhaustive all-types"}


def _load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _quality(build, method):
    q = build.get("quality") or {}
    if method == "F":                 # build_alltypes: flat quality dict
        return q if "well_formed" in q else {}
    return q.get("E") or {}           # build_e_adaptive stores the run under 'E'


def _giveup_verbatim(build):
    gv, vb = [0, 0], [0, 0]
    for c in build.get("conversations", []):
        for t in c.get("turns", []):
            if t.get("query_type") == "Unanswerable" or t.get("is_unanswerable"):
                continue
            gold = t.get("gold") or ""
            gv[1] += 1
            # E/D builds carry the explicit flag; F infers from abstention gold
            gu = t.get("guard_gave_up")
            if gu is None:
                gu = gold.strip().startswith(ABST[:20])
            gv[0] += int(bool(gu))
            if gold.strip() and not gold.startswith(ABST):
                vb[1] += 1
                vb[0] += int(verbatim_answerable(gold, t.get("evidence") or ""))
    return (round(gv[0] / gv[1], 3) if gv[1] else None,
            round(vb[0] / vb[1], 3) if vb[1] else None)


def main():
    rows = {}
    for ds in DATASETS:
        for m in METHODS:
            b = _load(f"conv_rag_benchmark/output/eval_{m}_{ds}.json")
            if not b:
                rows[(ds, m)] = None
                continue
            q = _quality(b, m)
            gv, vb = _giveup_verbatim(b)
            rows[(ds, m)] = {
                "well_formed": q.get("well_formed"),
                "gold_supported": q.get("gold_supported"),
                "gold_correct": q.get("gold_correct"),
                "giveup_rate": gv,
                "verbatim_softball_rate": vb,
                "rag_failure_rate": (b.get("rag_failure") or {}).get("failure_rate"),
            }

    json.dump({f"{ds}/{m}": v for (ds, m), v in rows.items()},
              open(OUT, "w", encoding="utf-8"), indent=2)

    def c(v):
        return "   —   " if v is None else f"{v:>7.3f}"

    cols = [("well_formed", "wellform"), ("gold_supported", "gold_sup"),
            ("gold_correct", "gold_cor"), ("giveup_rate", "give-up"),
            ("verbatim_softball_rate", "verbatim"), ("rag_failure_rate", "failrate")]
    print(f"\n{'dataset':<10}{'method':<24}" + "".join(f"{h:>9}" for _, h in cols))
    print("-" * 112)
    for ds in DATASETS:
        for m in METHODS:
            r = rows.get((ds, m))
            label = f"{m} ({METHOD_DESC[m]})"
            if not r:
                print(f"{ds:<10}{label:<24}{'  (build missing)':>20}")
                continue
            line = f"{ds:<10}{label:<24}"
            for key, _ in cols:
                line += f"{c(r[key]):>9}"
            print(line)
        print()
    print("legend: D=random-type ablation, E=adaptive (contribution), F=exhaustive. "
          "give-up & verbatim & failrate = fractions. saved -> " + OUT)


if __name__ == "__main__":
    main()
