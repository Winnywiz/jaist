"""
Compare a PLAIN LLM-judge vs REAL G-Eval on well-formedness, over the benchmark
questions. Same model (gpt-4o) for both, so the only difference is the METHOD.

  PLAIN  : ask "well_formed? true/false" -> average                (what you have)
  G-EVAL : chain-of-thought steps + 1-5 score, PROBABILITY-WEIGHTED via token
           logprobs (the real G-Eval trick) -> smooth 1-5 score    (Liu et al. 2023)

Reads a saved run file under result/benchmark_quality/, named
``{rag}_{dataset}_t{turns}_c{convos}.json``.

Run: python -m conv_rag_benchmark.geval_compare
     python -m conv_rag_benchmark.geval_compare --file result/benchmark_quality/mtrag/vector_mtrag_t10_c5.json
"""
import argparse
import glob
import json
import math
import os
from collections import Counter

from openai import OpenAI

MODEL = "gpt-4o"


def load_key():
    k = os.getenv("OPENAI_API_KEY", "")
    if k:
        return k.strip().strip('"').strip("'")
    for p in ("RAG-DIVE/.env", ".env"):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


client = OpenAI(api_key=load_key())

PLAIN_SYS = ('Is this question well-formed (clear, self-contained, answerable)? '
            'Reply JSON: {"well_formed": true/false}')

GEVAL_SYS = """You evaluate the WELL-FORMEDNESS of a question on a 1-5 scale.
Evaluation steps:
1. Check grammar and phrasing.
2. Check it is clear and self-contained (understandable on its own).
3. Check it is a genuine, answerable question.
Scale: 1 = very poorly formed, 5 = perfectly well-formed.
Output ONLY the single digit (1-5)."""


def plain(q):
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": PLAIN_SYS},
                               {"role": "user", "content": "QUESTION: " + q}],
        response_format={"type": "json_object"}, max_tokens=20, temperature=0)
    try:
        return 1.0 if json.loads(r.choices[0].message.content).get("well_formed") else 0.0
    except Exception:
        return 0.0


def geval(q):
    """Real G-Eval: probability-weighted 1-5 from token logprobs."""
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": GEVAL_SYS},
                               {"role": "user", "content": "QUESTION: " + q}],
        max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    top = r.choices[0].logprobs.content[0].top_logprobs
    probs = {}
    for tl in top:
        tok = tl.token.strip()
        if tok in {"1", "2", "3", "4", "5"}:
            probs[tok] = probs.get(tok, 0.0) + math.exp(tl.logprob)
    if not probs:
        return None
    Z = sum(probs.values())
    return sum(int(k) * v for k, v in probs.items()) / Z  # weighted average


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plain LLM-judge vs real G-Eval")
    ap.add_argument("--file", default=None,
                    help="run file to compare on (default: first vector run found)")
    ap.add_argument("--out", default="result/benchmark_quality/geval_compare.json")
    args = ap.parse_args(argv)

    path = args.file
    if not path:
        cands = [p for p in sorted(
            glob.glob("result/benchmark_quality/*/vector_*_t*_c*.json"))
            if not any(x in p for x in ("_docsim", "_summary"))]
        if not cands:
            print("no run files found — run the benchmark first, or pass --file")
            return
        path = cands[0]
    print(f"# reading {path}")
    d = json.load(open(path, encoding="utf-8"))
    qs = [t["question"] for c in d["conversations"] for t in c["turns"]]
    print(f"comparing on {len(qs)} questions with {MODEL} ...")

    rows = []
    for q in qs:
        rows.append((q, plain(q), geval(q)))

    p_vals = [p for _, p, _ in rows]
    g_vals = [g for _, _, g in rows if g is not None]
    p_avg = sum(p_vals) / len(p_vals)
    g_avg = sum(g_vals) / len(g_vals)

    print("\n================= PLAIN vs G-EVAL (well-formedness) =================")
    print(f"  PLAIN  (true/false avg)     : {p_avg:.3f}   (0-1)")
    print(f"  G-EVAL (prob-weighted 1-5)  : {g_avg:.2f}    -> /5 = {g_avg/5:.3f}")

    # show the EXTRA INFORMATION G-Eval gives: a spread, not just yes/no
    buckets = Counter()
    for g in g_vals:
        buckets[round(g)] += 1
    print("\n  G-Eval score spread (plain judge can't show this):")
    for s in (1, 2, 3, 4, 5):
        print(f"    {s}/5 : {buckets.get(s,0)}")

    # cases where they DISAGREE most (plain=pass but G-Eval low, or vice versa)
    print("\n  Where they disagree (plain says OK, G-Eval is unsure):")
    dis = sorted([r for r in rows if r[2] is not None and r[1] == 1.0],
                 key=lambda r: r[2])[:4]
    for q, p, g in dis:
        print(f"    plain=PASS  G-Eval={g:.1f}  | {q[:80]}")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"model": MODEL, "n": len(qs), "source": path,
               "plain_avg": p_avg,
               "geval_avg_1to5": g_avg, "geval_avg_norm": g_avg/5,
               "geval_spread": dict(buckets),
               "rows": [{"q": q, "plain": p, "geval": g} for q, p, g in rows]},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
