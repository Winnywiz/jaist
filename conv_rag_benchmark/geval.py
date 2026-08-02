"""
Real G-Eval scorer (Liu et al., 2023) reusable across metrics.

For each criterion it runs: chain-of-thought evaluation steps + a 1-5 score that is
PROBABILITY-WEIGHTED over the model's token logprobs (the real G-Eval trick), then
normalizes to 0-1. Works for well_formed, gold_supported, gold_correct (and is easy
to extend to more criteria by adding a rubric).

Usage:
    from conv_rag_benchmark.geval import geval_items
    agg, scored = geval_items(items, model="gpt-4o")
where each item = {"question","gold","evidence","query_type"}.
"""
import math
import os
from collections import Counter

from openai import OpenAI

from .llm import LLM
from .atomic_faithfulness import score_gold as _atomic_score_gold

#: Optional 4th criterion: decompose-then-entail faithfulness (RAGAS/ARES-style),
#: NOT a 1-5 logprob rubric. Off by default — pass it in `criteria` to enable.
ATOMIC = "atomic_faithfulness"


def _load_key():
    k = os.getenv("OPENAI_API_KEY", "")
    if k:
        return k.strip().strip('"').strip("'")
    for p in ("RAG-DIVE/.env", ".env", os.path.join(os.path.dirname(__file__), "..", "RAG-DIVE", ".env")):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# Each rubric = (what it measures, the CoT steps, how to build the user content).
RUBRICS = {
    "well_formed": {
        "definition": "how WELL-FORMED the QUESTION is (clear, self-contained, answerable)",
        "steps": ["Check grammar and phrasing.",
                  "Check it is clear and self-contained (understandable on its own).",
                  "Check it is a genuine, answerable question."],
        "content": lambda it: f"QUESTION: {it['question']}",
    },
    "gold_supported": {
        "definition": ("how well the GOLD answer is SUPPORTED by (grounded in) the "
                       "EVIDENCE (5 = every fact is in the evidence, 1 = not in the "
                       "evidence / hallucinated)"),
        "steps": ["Read the evidence.",
                  "Check whether each fact in the gold answer appears in the evidence.",
                  "Rate how fully the gold is grounded in the evidence."],
        "content": lambda it: f"EVIDENCE: {it['evidence'][:1800]}\n\nGOLD ANSWER: {it['gold']}",
    },
    "gold_correct": {
        "definition": ("how CORRECTLY the GOLD answers the QUESTION, taking the "
                       "EVIDENCE as ground truth"),
        "steps": ["Read the question and the evidence.",
                  "Check the gold actually answers the question asked.",
                  "Check the gold is consistent with the evidence.",
                  "Rate the correctness."],
        "content": lambda it: (f"QUESTION: {it['question']}\nEVIDENCE: {it['evidence'][:1500]}"
                               f"\n\nGOLD ANSWER: {it['gold']}"),
    },
}


def _score_one(client, model, rubric, it):
    sys = (f"You evaluate {rubric['definition']} on a 1-5 scale.\nEvaluation steps:\n"
           + "\n".join(f"{i+1}. {s}" for i, s in enumerate(rubric["steps"]))
           + "\nScale: 1 = worst, 5 = best. Output ONLY the single digit (1-5).")
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": rubric["content"](it)}],
            max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    except Exception as e:
        print(f"  geval call failed: {str(e)[:90]}")
        return None
    top = r.choices[0].logprobs.content[0].top_logprobs
    probs = {}
    for tl in top:
        t = tl.token.strip()
        if t in {"1", "2", "3", "4", "5"}:
            probs[t] = probs.get(t, 0.0) + math.exp(tl.logprob)
    if not probs:
        return None
    z = sum(probs.values())
    return sum(int(k) * v for k, v in probs.items()) / z   # 1-5 weighted


def geval_items(items, model="gpt-4o", criteria=None):
    """Return (agg, scored). Scores normalized to 0-1 (raw 1-5 also kept).

    `criteria` may include the optional ATOMIC ('atomic_faithfulness') criterion — a
    decompose-then-entail faithfulness fraction (see atomic_faithfulness.py), NOT a
    1-5 logprob rubric. It is OFF by default; to run it alongside the standard metrics
    pass e.g. ``criteria=list(RUBRICS) + [ATOMIC]``."""
    client = OpenAI(api_key=_load_key())
    criteria = criteria or list(RUBRICS)
    atomic_llm = None                        # built lazily only if ATOMIC is requested
    scored = []
    for it in items:
        rec = {"query_type": it.get("query_type")}
        for crit in criteria:
            if crit == ATOMIC:               # decompose-then-entail, not a logprob rubric
                if atomic_llm is None:
                    atomic_llm = LLM(model=model)
                f, n_claims, n_ent = _atomic_score_gold(
                    atomic_llm, it.get("gold", ""), it.get("evidence", ""))
                rec[crit] = round(f, 3) if f is not None else None
                rec[crit + "_claims"] = n_claims
                rec[crit + "_entailed"] = n_ent
                continue
            s = _score_one(client, model, RUBRICS[crit], it)
            rec[crit] = round(s / 5.0, 3) if s is not None else None
            rec[crit + "_5"] = round(s, 2) if s is not None else None
        scored.append(rec)

    def avg(k):
        vs = [r[k] for r in scored if r.get(k) is not None]
        return round(sum(vs) / len(vs), 3) if vs else None
    agg = {c: avg(c) for c in criteria}
    return agg, scored


def geval_breakdown_by_type(scored, criteria=None):
    criteria = criteria or list(RUBRICS)
    buckets = {}
    for r in scored:
        buckets.setdefault(r.get("query_type") or "unknown", []).append(r)
    out = {}
    for qt, rows in buckets.items():
        k = len(rows)
        out[qt] = {"n": k}
        for c in criteria:
            vs = [r[c] for r in rows if r.get(c) is not None]
            out[qt][c] = round(sum(vs) / len(vs), 3) if vs else None
    return out
