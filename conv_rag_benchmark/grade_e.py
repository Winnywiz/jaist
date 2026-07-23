"""
Grade Method E's probed RAG with the same metric suite used in RAG-DIVE / newmethod:

  Correctness (AI judge)      - did the RAG answer match the gold / behave correctly?
  BERTScore-F1 (no AI judge)  - semantic match between RAG answer and gold (roberta-large)
  Faithfulness (AI judge)     - is every claim in the RAG answer supported by the evidence?
  Context recall (answerable) - for answerable turns, does the evidence contain the gold? (proxy)
  Context precision           - is the evidence relevant to the question? (proxy, 1 blob)
  Total failures              - wrong + hallucinated (retrieval+gen)

Reads each dataset's quality_e.json (no regeneration). Run:
    python -m conv_rag_benchmark.grade_e
"""
import json
import os

from openai import OpenAI

from .geval import _load_key

MODEL = "gpt-4o"
client = OpenAI(api_key=_load_key())

_FAITH = ("Given EVIDENCE and an ANSWER, decide if EVERY claim in the ANSWER is supported "
          "by the EVIDENCE (no facts beyond it). If the ANSWER abstains ('I don't know', "
          "'not answerable'), treat it as supported. Reply ONLY 'yes' or 'no'.")
_RECALL = ("Given EVIDENCE and a GOLD answer, can the GOLD be derived from the EVIDENCE? "
           "Reply ONLY 'yes' or 'no'.")
_PREC = ("Given a QUESTION and EVIDENCE, is the EVIDENCE relevant to answering the QUESTION? "
         "Reply ONLY 'yes' or 'no'.")


def _yesno(sys_prompt, user):
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=2,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user}])
        return 1.0 if r.choices[0].message.content.strip().lower().startswith("y") else 0.0
    except Exception:
        return None


def _bertscore(cands, refs):
    import bert_score
    P, R, F1 = bert_score.score(cands, refs, lang="en",
                                rescale_with_baseline=True, verbose=False)
    return round(float(F1.mean()), 3)


def grade(path):
    d = json.load(open(path, encoding="utf-8"))
    turns = [t for c in d["conversations"] for t in c["turns"]]
    n = len(turns)

    # --- Correctness (AI judge, from the live run's outcome) ---
    correct = sum(1 for t in turns
                  if t["outcome"] == "correct"
                  or (t["outcome"] == "abstained" and t.get("is_unanswerable")))
    failures = sum(1 for t in turns if t["outcome"] in ("wrong", "hallucinated"))

    # --- BERTScore-F1 (no AI judge): RAG answer vs gold ---
    pairs = [(t["rag_answer"], t["gold"]) for t in turns
             if (t.get("rag_answer") or "").strip() and (t.get("gold") or "").strip()]
    bert = _bertscore([c for c, _ in pairs], [r for _, r in pairs]) if pairs else None

    # --- Faithfulness / context recall+precision (AI judge) ---
    faith, recall, prec = [], [], []
    for t in turns:
        ev = (t.get("evidence") or "")[:1800]
        ans = t.get("rag_answer") or ""
        gold = t.get("gold") or ""
        q = t.get("question") or ""
        if ans:
            v = _yesno(_FAITH, f"EVIDENCE: {ev}\nANSWER: {ans}")
            if v is not None:
                faith.append(v)
        if not t.get("is_unanswerable"):   # context recall only meaningful when answerable
            v = _yesno(_RECALL, f"EVIDENCE: {ev}\nGOLD: {gold}")
            if v is not None:
                recall.append(v)
        if ev:
            v = _yesno(_PREC, f"QUESTION: {q}\nEVIDENCE: {ev}")
            if v is not None:
                prec.append(v)

    avg = lambda l: round(sum(l) / len(l), 3) if l else None
    return {
        "n_turns": n,
        "correctness": round(correct / n, 3),
        "bertscore_f1": bert,
        "faithfulness": avg(faith),
        "context_recall_answerable": avg(recall),
        "context_precision": avg(prec),
        "total_failures": failures,
        "failure_rate": round(failures / n, 3),
        # question-quality already in the file (G-Eval):
        "question_quality_geval": d["quality"]["E"],
    }


def main():
    base = "conv_rag_benchmark/output"
    rows = {}
    for ds in ("MultiHopRAG", "MedQA"):
        p = os.path.join(base, ds, "quality_e.json")
        if not os.path.exists(p):
            print(f"  {ds}: missing {p}")
            continue
        print(f"# grading {ds} ...")
        rows[ds] = grade(p)

    metrics = ["correctness", "bertscore_f1", "faithfulness",
               "context_recall_answerable", "context_precision",
               "total_failures", "failure_rate"]
    print(f"\n{'metric':<28}" + "".join(f"{ds:>14}" for ds in rows))
    for m in metrics:
        print(f"{m:<28}" + "".join(f"{str(rows[ds][m]):>14}" for ds in rows))
    print(f"\n{'-- question quality (G-Eval) --':<28}")
    for k in ("well_formed", "gold_supported", "gold_correct"):
        print(f"{k:<28}" + "".join(
            f"{str(rows[ds]['question_quality_geval'].get(k)):>14}" for ds in rows))

    json.dump(rows, open(os.path.join(base, "grade_e_metrics.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {base}/grade_e_metrics.json")


if __name__ == "__main__":
    main()
