"""
Consolidated evaluation report: every metric the project computes, for every
dataset, in ONE table. Reads existing result files (no re-running of judge calls
except the offline probe-difficulty pass), so it's cheap and reproducible.

Columns per dataset:
  question quality   : well_formed / gold_supported / gold_correct   (eval_report.json)
  answer-key trust   : give-up rate                                  (build files)
  probe difficulty   : verbatim-answerable rate                      (build files, offline)
  RAG performance    : failure rate                                  (build files)
  validity control   : doc-requirement F1 gap                        (doc_requirement_by_dataset.json)
  attribution        : dynamic Conversation acc, dynamic overall acc (attribution_repeated_*.json)

Run:  python -m conv_rag_benchmark.consolidate_metrics
"""
import json
import os

from .probe_difficulty import verbatim_answerable

OUT = "conv_rag_benchmark/output/consolidated_metrics.json"
ABST = "Not answerable"

# dataset -> (E build file, attribution repeated file / key)
BUILDS = {
    "qasper":   "conv_rag_benchmark/output/eval_E_qasper.json",
    "arxivcs":  "conv_rag_benchmark/output/eval_E_arxivcs.json",
    "hfdocqa":  "conv_rag_benchmark/output/eval_E_hfdocqa.json",
    "mlarxiv":  "conv_rag_benchmark/output/eval_E_mlarxiv.json",
}
KIND = {"qasper": "with-QA / CS papers", "arxivcs": "docs-only / CS papers",
        "hfdocqa": "with-QA / ML docs", "mlarxiv": "docs-only / ML abstracts"}
ATTR_FILES = ["dynamic_vs_static_dx/output/attribution_repeated_qasper.json",
              "dynamic_vs_static_dx/output/attribution_repeated_hfdocqa_multihoprag.json"]


def _load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def giveup_and_verbatim(build):
    d = _load(build)
    if not d:
        return None, None
    gv = [0, 0]
    vb = [0, 0]
    for c in d.get("conversations", []):
        for t in c.get("turns", []):
            if t.get("query_type") == "Unanswerable" or t.get("is_unanswerable"):
                continue
            gv[1] += 1
            gv[0] += int(bool(t.get("guard_gave_up")))
            gold = t.get("gold") or ""
            if gold.strip() and not gold.startswith(ABST):
                vb[1] += 1
                vb[0] += int(verbatim_answerable(gold, t.get("evidence") or ""))
    return (round(gv[0] / gv[1], 3) if gv[1] else None,
            round(vb[0] / vb[1], 3) if vb[1] else None)


def main():
    eval_rep = _load("conv_rag_benchmark/output/eval_report.json") or {}
    # eval_report keys are file paths; index by dataset basename
    quality_by_ds = {}
    for path, r in eval_rep.items():
        for ds in BUILDS:
            if ds in path:
                quality_by_ds[ds] = r
    docreq = _load("conv_rag_benchmark/output/doc_requirement_by_dataset.json") or {}

    # attribution: dynamic arm Conversation + overall, per dataset
    attr = {}
    for f in ATTR_FILES:
        d = _load(f)
        if not d:
            continue
        for ds, s in (d.get("summary") or {}).items():
            conv = s.get("Conversation", {}).get("4.dynamic_followup")
            over = s.get("overall", {}).get("4.dynamic_followup")
            attr[ds] = {"conv": conv, "overall": over}

    rows = {}
    for ds, build in BUILDS.items():
        gv, vb = giveup_and_verbatim(build)
        q = (quality_by_ds.get(ds) or {}).get("question_quality", {})
        fr = ((quality_by_ds.get(ds) or {}).get("rag_failure") or {}).get("failure_rate")
        # doc-req gap: average the two answerers' f1_gap
        dr = None
        for k, v in docreq.items():
            if ds in k:
                gaps = [m["f1_gap"] for m in v["answerers"].values()]
                dr = round(sum(gaps) / len(gaps), 3)
        rows[ds] = {
            "kind": KIND[ds],
            "well_formed": q.get("well_formed"),
            "gold_supported": q.get("gold_supported"),
            "gold_correct": q.get("gold_correct"),
            "giveup_rate": gv,
            "verbatim_softball_rate": vb,
            "rag_failure_rate": fr,
            "docreq_f1_gap": dr,
            "attr_conversation": (attr.get(ds) or {}).get("conv"),
            "attr_overall": (attr.get(ds) or {}).get("overall"),
        }

    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=2)

    def cell(v):
        if v is None:
            return f"{'—':>8}"
        if isinstance(v, (list, tuple)):
            return f"{v[0]:.2f}±{v[1]:.2f}".rjust(8)
        return f"{v:>8.3f}"

    cols = [("well_formed", "wellform"), ("gold_supported", "gold_sup"),
            ("gold_correct", "gold_cor"), ("giveup_rate", "give-up"),
            ("verbatim_softball_rate", "verbatim"), ("rag_failure_rate", "failrate"),
            ("docreq_f1_gap", "docgap"), ("attr_conversation", "attr_conv"),
            ("attr_overall", "attr_all")]
    print(f"\n{'dataset':<10}{'kind':<22}" + "".join(f"{h:>9}" for _, h in cols))
    for ds, r in rows.items():
        line = f"{ds:<10}{r['kind']:<22}"
        for key, _ in cols:
            line += cell(r[key])[-9:] if False else f"{cell(r[key]):>9}"
        print(line)
    print("\nlegend: give-up/verbatim/failrate = fraction (lower better for give-up & "
          "failrate-inflation); docgap = F1(with docs)-F1(without) (higher=more doc-"
          "dependent); attr_conv/attr_all = dynamic-E attribution acc, mean±std; "
          "— = not applicable (docs-only can't run attribution).")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
