"""
Grade Method E's conversational follow-ups with the 5 GRICEAN-MAXIM metrics from
"What Should I Ask?" (follow-up question quality, reference-free):

  Relevance       (Maxim of Relation)  - does the follow-up connect to the prior turn?
  Informativeness (Maxim of Quantity)  - does it seek genuinely NEW information?
  Truthfulness    (Maxim of Quality)   - does it only presuppose what's established?
  Clarity         (Maxim of Manner)    - is it phrased clearly / unambiguously?
  Coherence       (dialogue flow)      - does it follow the conversation naturally?

NOTE (honesty): the original paper operationalises each maxim with a specialised
component (KG out-degree centrality for INFO, DialoGPT perplexity for CLA, BERT
next-utterance for COH, Freebase for TRUTH). Here each maxim is scored by an
LLM-as-judge against the same rubric (probability-weighted 1-5, normalised 0-1),
which is a faithful adaptation, not the exact neural proxies. Reported as such.

Scores ONLY conversational turns (turn_id > 0), since maxims are about follow-ups.
Run:  python -m conv_rag_benchmark.grade_e_gricean
"""
import json
import math
import os

from openai import OpenAI

from .geval import _load_key

MODEL = "gpt-4o"
client = OpenAI(api_key=_load_key())

_MAXIMS = {
    "relevance": ("Maxim of RELATION. Rate 1-5 how much the FOLLOW-UP QUESTION relates "
                  "to the immediately PRIOR turn (its topic/entities). 5 = directly "
                  "builds on it; 1 = unrelated. Output ONLY the digit 1-5."),
    "informativeness": ("Maxim of QUANTITY. Rate 1-5 how much the FOLLOW-UP seeks "
                        "genuinely NEW information not already answered in the PRIOR "
                        "turn. 5 = asks something new; 1 = re-asks what's already known. "
                        "Output ONLY the digit 1-5."),
    "truthfulness": ("Maxim of QUALITY. Rate 1-5 whether the FOLLOW-UP only presupposes "
                     "things ALREADY ESTABLISHED in the conversation (no false or "
                     "unsupported assumptions). 5 = all presuppositions are established; "
                     "1 = assumes false/unmentioned facts. Output ONLY the digit 1-5."),
    "clarity": ("Maxim of MANNER. Rate 1-5 how CLEARLY and unambiguously the FOLLOW-UP "
                "is phrased (a pronoun is fine if its referent is clear from context). "
                "5 = crystal clear; 1 = confusing/ambiguous. Output ONLY the digit 1-5."),
    "coherence": ("Rate 1-5 how naturally the FOLLOW-UP fits the FLOW of the dialogue "
                  "as a next utterance. 5 = sounds like a natural next question; 1 = "
                  "jarring/out of place. Output ONLY the digit 1-5."),
}


def _score(rubric, context, question):
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=1, logprobs=True, top_logprobs=20,
            messages=[{"role": "system", "content": rubric},
                      {"role": "user",
                       "content": f"PRIOR TURN:\n{context}\n\nFOLLOW-UP QUESTION: {question}"}])
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


def grade(path):
    d = json.load(open(path, encoding="utf-8"))
    agg = {m: [] for m in _MAXIMS}
    followup = {m: [] for m in _MAXIMS}
    n_turns = 0
    for c in d["conversations"]:
        prev = None
        for t in c["turns"]:
            if t["turn_id"] > 0 and prev is not None:
                # Grade E's QUESTION quality: use the GOLD answer as prior context
                # (what E designed the follow-up to follow), not the RAG's possibly
                # failed answer — otherwise the RAG's "I don't know" deflates E's score.
                ctx = (f"Q: {prev.get('question','')}\n"
                       f"A: {prev.get('gold') or prev.get('rag_answer','')}")
                n_turns += 1
                for m, rubric in _MAXIMS.items():
                    s = _score(rubric, ctx, t["question"])
                    if s is not None:
                        agg[m].append(s)
                        if t["query_type"] == "Follow-Up":
                            followup[m].append(s)
            prev = t
    avg = lambda l: round(sum(l) / len(l), 3) if l else None
    out = {m: avg(agg[m]) for m in _MAXIMS}
    out["gricean_overall"] = avg([v for l in agg.values() for v in l])
    out["n_conversational_turns"] = n_turns
    out["followup_only"] = {m: avg(followup[m]) for m in _MAXIMS}
    return out


def main():
    base = "conv_rag_benchmark/output"
    rows = {}
    for ds in ("MultiHopRAG", "MedQA", "ArXivCS"):
        p = os.path.join(base, ds, "quality_e.json")
        if not os.path.exists(p):
            print(f"  {ds}: missing {p}")
            continue
        print(f"# grading {ds} (Gricean maxims) ...")
        rows[ds] = grade(p)

    order = ["relevance", "informativeness", "truthfulness", "clarity",
             "coherence", "gricean_overall"]
    print(f"\n{'Gricean maxim':<22}" + "".join(f"{ds:>14}" for ds in rows))
    for m in order:
        print(f"{m:<22}" + "".join(f"{str(rows[ds][m]):>14}" for ds in rows))
    print(f"\n{'(Follow-Up turns only)':<22}")
    for m in order[:-1]:
        print(f"{m:<22}" + "".join(
            f"{str(rows[ds]['followup_only'][m]):>14}" for ds in rows))

    json.dump(rows, open(os.path.join(base, "grade_e_gricean.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {base}/grade_e_gricean.json")


if __name__ == "__main__":
    main()
