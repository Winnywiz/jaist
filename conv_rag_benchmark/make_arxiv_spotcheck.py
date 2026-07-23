"""
Build a BLIND human spot-check sheet for arxivcs (docs-only) generated golds — the
external validation the docs-only condition currently lacks. Samples substantive
turns, writes the question + gold + evidence and TWO blank columns for the human to
mark. An accompanying answer-key file records the pipeline's own judge verdict so
you can compute human-vs-judge agreement afterwards.

Run:  python -m conv_rag_benchmark.make_arxiv_spotcheck [build.json] [--n 20]
"""
import argparse
import csv
import json
import os
import random

ABST = "Not answerable"
DEFAULT_BUILD = "conv_rag_benchmark/output/eval_E_arxivcs.json"
OUT_DIR = "conv_rag_benchmark/output"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build", nargs="?", default=DEFAULT_BUILD)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d = json.load(open(args.build, encoding="utf-8"))
    items = []
    for c in d.get("conversations", []):
        for t in c.get("turns", []):
            gold = t.get("gold") or ""
            if gold.strip() and not gold.startswith(ABST) and not t.get("guard_gave_up"):
                items.append(t)
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    blind = os.path.join(OUT_DIR, "arxiv_spotcheck_BLIND.csv")
    key = os.path.join(OUT_DIR, "arxiv_spotcheck_ANSWERKEY.csv")
    with open(blind, "w", newline="", encoding="utf-8-sig") as fb, \
         open(key, "w", newline="", encoding="utf-8-sig") as fk:
        wb = csv.writer(fb)
        wk = csv.writer(fk)
        wb.writerow(["id", "question", "gold_answer", "evidence",
                     "supported? (y/n)", "correct? (y/n)"])
        wk.writerow(["id", "query_type"])
        for i, t in enumerate(items):
            wb.writerow([i, t.get("question", ""), t.get("gold", ""),
                         str(t.get("evidence", ""))[:1200], "", ""])
            wk.writerow([i, t.get("query_type", "")])

    print(f"wrote {len(items)} items")
    print(f"  fill in -> {blind}")
    print(f"  (query types hidden in -> {key})")
    print("\nInstructions: for each row, read the EVIDENCE, then mark")
    print("  supported? = is every claim in gold_answer stated in the evidence? (y/n)")
    print("  correct?   = does gold_answer correctly answer the question? (y/n)")
    print("Do NOT look at the answer key while judging. Afterwards, compare your")
    print("y/n marks to the pipeline judge to get human-vs-judge agreement.")


if __name__ == "__main__":
    main()
