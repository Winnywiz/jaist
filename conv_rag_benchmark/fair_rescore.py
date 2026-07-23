"""
Re-score saved D and E conversations with FAIRER metrics, to compare strict vs fair:
  #4 well-formed judged WITH the previous turn (conversational), so pronoun follow-ups
     aren't penalised for not being standalone.
  #5 abstention golds ("Not answerable...") are EXCLUDED from grounding/correctness
     (a correct abstention is not an ungrounded answer).

Re-scores the existing conversations (no regeneration). Run:
    python -m conv_rag_benchmark.fair_rescore
"""
import json
import math
import os

from openai import OpenAI

from .geval import _load_key

MODEL = "gpt-4o"
ABSTAIN = "Not answerable"
client = OpenAI(api_key=_load_key())

_WF = ("Rate WELL-FORMEDNESS of the QUESTION 1-5: clear, answerable, and self-contained "
       "GIVEN THE CONVERSATION. A follow-up using a pronoun (it/they) is fine if the "
       "conversation makes the referent clear. Output ONLY the digit 1-5.")
_SUP = ("Rate how well the GOLD is SUPPORTED by (grounded in) the EVIDENCE, 1-5 "
        "(5=every fact in the evidence). Output ONLY the digit 1-5.")
_COR = ("Rate how CORRECTLY the GOLD answers the QUESTION given the EVIDENCE, 1-5. "
        "Output ONLY the digit 1-5.")


def _score(sys_prompt, user):
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": user}],
            max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    except Exception:
        return None
    probs = {}
    for t in r.choices[0].logprobs.content[0].top_logprobs:
        tok = t.token.strip()
        if tok in {"1", "2", "3", "4", "5"}:
            probs[tok] = probs.get(tok, 0.0) + math.exp(t.logprob)
    if not probs:
        return None
    z = sum(probs.values())
    return sum(int(k) * v for k, v in probs.items()) / z / 5.0


def fair_scores(turns):
    wf, sup, cor = [], [], []
    for t in turns:
        q, gold, ev, prev = t["question"], t["gold"], t["evidence"], t.get("prev", "")
        if q.strip():
            s = _score(_WF, f"CONVERSATION (previous turn): {prev}\nQUESTION: {q}")
            if s is not None:
                wf.append(s)
        if gold and gold.startswith(ABSTAIN):
            continue  # #5: correct abstention -> not an ungrounded answer
        s = _score(_SUP, f"EVIDENCE: {ev[:1800]}\nGOLD: {gold}")
        if s is not None:
            sup.append(s)
        s = _score(_COR, f"QUESTION: {q}\nEVIDENCE: {ev[:1500]}\nGOLD: {gold}")
        if s is not None:
            cor.append(s)
    avg = lambda l: round(sum(l) / len(l), 3) if l else None
    return {"well_formed": avg(wf), "gold_supported": avg(sup), "gold_correct": avg(cor),
            "n": len(turns), "n_scored_grounding": len(sup)}


def _d_turns(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for c in d["conversations"]:
        prev = ""
        for t in c["turns"]:
            out.append({"question": t["question"], "gold": t["gold_answer"],
                        "evidence": " ".join(t.get("question_evidence_context", []))[:1800],
                        "prev": prev})
            prev = t["question"]
    return out


def _e_turns(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for c in d["conversations"]:
        prev = ""
        for t in c["turns"]:
            out.append({"question": t["question"], "gold": t["gold"],
                        "evidence": (t.get("evidence", "") or "")[:1800], "prev": prev})
            prev = t["question"]
    return out


def main():
    base = "conv_rag_benchmark/output"
    for ds in ("MultiHopRAG", "MedQA"):
        print(f"\n################ {ds} ################")
        for label, loader, fname in [("D", _d_turns, "benchmark_random.json"),
                                     ("E", _e_turns, "quality_e.json")]:
            p = os.path.join(base, ds, fname)
            if not os.path.exists(p):
                print(f"  {label}: missing {p}")
                continue
            turns = loader(p)
            fair = fair_scores(turns)
            print(f"  {label} FAIR: well_formed {fair['well_formed']} | "
                  f"gold_supported {fair['gold_supported']} | gold_correct {fair['gold_correct']} "
                  f"(grounding n={fair['n_scored_grounding']}/{fair['n']})")


if __name__ == "__main__":
    main()
