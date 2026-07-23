"""
Build a BLIND human-evaluation spreadsheet from Method D and Method F outputs.

- Mixes D and F questions, shuffles them, HIDES the method (blind) so the human rater
  isn't biased. Adds empty rating columns for the human to fill in.
- Writes a separate answer key mapping each row_id -> method (for un-blinding after).

Run:  python -m conv_rag_benchmark.make_human_eval --label MultiHopRAG
"""
import argparse
import csv
import json
import os
import random


def rows_from(path, method):
    if not os.path.exists(path):
        return []
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for c in d["conversations"]:
        for t in c["turns"]:
            if t.get("query_type") in ("Seed",) or t.get("turn_id") == 0:
                continue
            out.append({
                "method": method,
                "type": t.get("query_type", ""),
                "question": (t.get("question") or "").strip(),
                "gold_answer": (t.get("gold") or "").strip(),
                "evidence": (t.get("evidence") or "").strip()[:600],
                "rag_answer": (t.get("rag_answer") or "").strip(),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="MultiHopRAG")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-method", type=int, default=30,
                    help="how many questions to sample from EACH method (balanced)")
    args = ap.parse_args()
    base = os.path.join("conv_rag_benchmark", "output", args.label)
    rng = random.Random(args.seed)

    e_rows = rows_from(os.path.join(base, "quality_e_50conv.json"), "E")   # the contribution
    d_rows = rows_from(os.path.join(base, "quality_e_randomtype.json"), "D")
    f_rows = rows_from(os.path.join(base, "quality_alltypes.json"), "F")
    # balanced sample: equal number from EACH method (E, D, F) so it's fair + blind
    for r in (e_rows, d_rows, f_rows):
        rng.shuffle(r)
    rows = e_rows[: args.per_method] + d_rows[: args.per_method] + f_rows[: args.per_method]
    rng.shuffle(rows)

    # BLIND eval file (method hidden) + empty rating columns
    eval_path = os.path.join(base, "human_eval_BLIND.csv")
    key_path = os.path.join(base, "human_eval_ANSWERKEY.csv")
    eval_cols = ["row_id", "type", "question", "gold_answer", "evidence", "rag_answer",
                 "well_formed_1to5", "gold_correct_YN", "gold_grounded_YN",
                 "rag_correct_YN", "notes"]
    with open(eval_path, "w", newline="", encoding="utf-8-sig") as fe, \
         open(key_path, "w", newline="", encoding="utf-8-sig") as fk:
        we = csv.DictWriter(fe, fieldnames=eval_cols); we.writeheader()
        wk = csv.writer(fk); wk.writerow(["row_id", "method", "type"])
        for i, r in enumerate(rows, 1):
            we.writerow({"row_id": i, "type": r["type"], "question": r["question"],
                         "gold_answer": r["gold_answer"], "evidence": r["evidence"],
                         "rag_answer": r["rag_answer"], "well_formed_1to5": "",
                         "gold_correct_YN": "", "gold_grounded_YN": "",
                         "rag_correct_YN": "", "notes": ""})
            wk.writerow([i, r["method"], r["type"]])

    n = len(rows)
    from collections import Counter
    counts = Counter(r["method"] for r in rows)
    print(f"# {args.label}: {n} questions, shuffled + blind | by method: {dict(counts)}")
    print(f"Saved eval sheet -> {eval_path}")
    print(f"Saved answer key -> {key_path}")


if __name__ == "__main__":
    main()
