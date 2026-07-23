"""
Grade D / E / F on ArXivCS with the same RAG-grading suite as grade_def (MultiHopRAG).
Reads the saved 50-conv conversations (no regeneration):
  D = quality_e_randomtype.json   (random-type ablation arm = static-style)
  E = quality_e_50conv.json       (adaptive controller)
  F = quality_alltypes.json       (all types every turn)

Run:  python -m conv_rag_benchmark.grade_def_arxiv
"""
import json
import os

from .grade_def import grade

BASE = os.path.join("conv_rag_benchmark", "output", "ArXivCS")
SPECS = [("D", "quality_e_randomtype.json"),
         ("E", "quality_e_50conv.json"),
         ("F", "quality_alltypes.json")]


def main():
    rows = {}
    for m, fn in SPECS:
        p = os.path.join(BASE, fn)
        if not os.path.exists(p):
            print(f"  {m}: missing {p}")
            continue
        print(f"# grading {m} ({fn}) on ArXivCS ...", flush=True)
        rows[m] = grade(p)

    metrics = ["n_turns", "correctness", "bertscore_f1", "faithfulness",
               "context_recall_answerable", "context_precision",
               "total_failures", "failure_rate"]
    print(f"\n{'metric':<28}" + "".join(f"{m:>10}" for m in rows))
    for k in metrics:
        print(f"{k:<28}" + "".join(f"{str(rows[m][k]):>10}" for m in rows))

    out = os.path.join(BASE, "grade_def_metrics.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
