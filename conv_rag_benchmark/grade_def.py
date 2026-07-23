"""
Grade methods D / E / F with the SAME metric suite as grade_e.py (RAG-DIVE / newmethod):

  Correctness (AI judge)      - outcome-based: correct, or correctly abstained on unanswerable
  BERTScore-F1 (no AI judge)  - semantic match RAG answer vs gold (substantive golds only)
  Faithfulness (AI judge)     - every claim in the RAG answer supported by the evidence?
  Context recall (answerable) - does the evidence contain the gold? (substantive golds only)
  Context precision           - is the evidence relevant to the question?
  Total failures / rate       - wrong + hallucinated

Re-scores the saved n=30 gated-vector conversations (no regeneration). Run:
    python -m conv_rag_benchmark.grade_def
"""
import json
import os

from openai import OpenAI

from .geval import _load_key

MODEL = "gpt-4o"
ABST = "Not answerable"
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


def _unanswerable(t):
    return bool(t.get("is_unanswerable")) or t.get("query_type") == "Unanswerable"


def grade(path):
    d = json.load(open(path, encoding="utf-8"))
    turns = [t for c in d["conversations"] for t in c["turns"]
             if (t.get("outcome") or "").strip()]          # graded turns only
    n = len(turns)

    correct = sum(1 for t in turns
                  if t["outcome"] == "correct"
                  or (t["outcome"] == "abstained" and _unanswerable(t)))
    failures = sum(1 for t in turns if t["outcome"] in ("wrong", "hallucinated"))

    # BERTScore only where BOTH sides are substantive (abstention gold = no reference)
    pairs = [(t["rag_answer"], t["gold"]) for t in turns
             if (t.get("rag_answer") or "").strip()
             and (t.get("gold") or "").strip()
             and not (t.get("gold") or "").startswith(ABST)]
    bert = _bertscore([c for c, _ in pairs], [r for _, r in pairs]) if pairs else None

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
        # recall: answerable turns with a substantive gold only
        if not _unanswerable(t) and gold and not gold.startswith(ABST):
            v = _yesno(_RECALL, f"EVIDENCE: {ev}\nGOLD: {gold}")
            if v is not None:
                recall.append(v)
        if ev:
            v = _yesno(_PREC, f"QUESTION: {q}\nEVIDENCE: {ev}")
            if v is not None:
                prec.append(v)

    avg = lambda l: round(sum(l) / len(l), 3) if l else None
    q = d.get("quality", {})
    return {
        "n_turns": n,
        "correctness": round(correct / n, 3) if n else None,
        "bertscore_f1": bert,
        "faithfulness": avg(faith),
        "context_recall_answerable": avg(recall),
        "n_recall": len(recall),
        "context_precision": avg(prec),
        "total_failures": failures,
        "failure_rate": round(failures / n, 3) if n else None,
        "question_quality_geval": q.get("E") or q,
    }


def main():
    base = os.path.join("conv_rag_benchmark", "output", "MultiHopRAG")
    specs = [("E", "quality_e_strictgold_qgate.json"),
             ("D", "quality_e_randomtype_strictgold_qgate.json"),
             ("F", "quality_alltypes.json")]
    rows = {}
    for m, fn in specs:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            print(f"  {m}: missing {p}")
            continue
        print(f"# grading {m} ({fn}) ...", flush=True)
        rows[m] = grade(p)

    metrics = ["n_turns", "correctness", "bertscore_f1", "faithfulness",
               "context_recall_answerable", "n_recall", "context_precision",
               "total_failures", "failure_rate"]
    print(f"\n{'metric':<28}" + "".join(f"{m:>10}" for m in rows))
    for k in metrics:
        print(f"{k:<28}" + "".join(f"{str(rows[m][k]):>10}" for m in rows))

    out = os.path.join(base, "grade_def_metrics.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
