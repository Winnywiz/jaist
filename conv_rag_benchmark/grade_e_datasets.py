"""
Run the full RAG-grading suite (same metrics as grade_def) on Method E's saved
50-conv conversations for MedQA and ArXivCS — filling the gap where grade_def only
covered MultiHopRAG. No regeneration: re-scores the saved quality_e_50conv.json.

Run:  python -m conv_rag_benchmark.grade_e_datasets
"""
import json
import os

from .grade_def import grade

DATASETS = ["MedQA", "ArXivCS", "MultiHopRAG"]


def main():
    base = os.path.join("conv_rag_benchmark", "output")
    rows = {}
    for ds in DATASETS:
        p = os.path.join(base, ds, "quality_e_50conv.json")
        if not os.path.exists(p):
            print(f"  {ds}: missing {p}")
            continue
        print(f"# grading E on {ds} ...", flush=True)
        rows[ds] = grade(p)
        # save per-dataset so it lives next to the other results
        out = os.path.join(base, ds, "grade_e_ragsuite.json")
        json.dump(rows[ds], open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"   saved -> {out}", flush=True)

    metrics = ["n_turns", "correctness", "bertscore_f1", "faithfulness",
               "context_recall_answerable", "context_precision",
               "total_failures", "failure_rate"]
    print(f"\n{'metric':<28}" + "".join(f"{ds:>13}" for ds in rows))
    for k in metrics:
        print(f"{k:<28}" + "".join(f"{str(rows[ds][k]):>13}" for ds in rows))


if __name__ == "__main__":
    main()
