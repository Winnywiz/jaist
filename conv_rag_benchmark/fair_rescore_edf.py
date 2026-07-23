"""
FAIR rescoring of Methods E / D / F — corrects two scoring artifacts:
  #1 well-formed judged WITH the previous turn (so pronoun follow-ups aren't penalised
     for not being standalone).
  #2 abstention golds ("Not answerable...") excluded from grounding/correctness (a correct
     refusal is not an ungrounded answer).

No regeneration — re-scores the saved conversations. Run:
    python -m conv_rag_benchmark.fair_rescore_edf --label MultiHopRAG
"""
import argparse
import json
import math
import os

from openai import OpenAI

from .geval import _load_key

MODEL = "gpt-4o"
ABST = "Not answerable"
client = OpenAI(api_key=_load_key())
DIG = {"1", "2", "3", "4", "5"}

_WF = ("Rate WELL-FORMEDNESS of the QUESTION 1-5: clear, answerable and self-contained "
       "GIVEN THE CONVERSATION. A follow-up using a pronoun (it/they/he) is fine if the "
       "conversation makes the referent clear. Output ONLY the digit 1-5.")
_SUP = ("Rate how well the GOLD is SUPPORTED by (grounded in) the EVIDENCE, 1-5 "
        "(5 = every fact in the evidence). Output ONLY the digit 1-5.")
_COR = ("Rate how CORRECTLY the GOLD answers the QUESTION given the EVIDENCE, 1-5. "
        "Output ONLY the digit 1-5.")


def _score(sys_p, user):
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "system", "content": sys_p},
                                   {"role": "user", "content": user}],
            max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    except Exception:
        return None
    p = {}
    for t in r.choices[0].logprobs.content[0].top_logprobs:
        tok = t.token.strip()
        if tok in DIG:
            p[tok] = p.get(tok, 0.0) + math.exp(t.logprob)
    if not p:
        return None
    z = sum(p.values())
    return sum(int(k) * v for k, v in p.items()) / z / 5.0


def fair(convos):
    wf, sup, cor = [], [], []
    for c in convos:
        prev = ""
        for t in c["turns"]:
            if t.get("query_type") == "Seed":
                prev = t["question"]; continue
            q, gold, ev = t["question"], t.get("gold", ""), t.get("evidence", "")
            s = _score(_WF, f"CONVERSATION (previous turn): {prev}\nQUESTION: {q}")
            if s is not None:
                wf.append(s)
            prev = q
            if gold.startswith(ABST):            # #2: skip correct abstentions
                continue
            s = _score(_SUP, f"EVIDENCE: {ev[:1800]}\nGOLD: {gold}")
            if s is not None:
                sup.append(s)
            s = _score(_COR, f"QUESTION: {q}\nEVIDENCE: {ev[:1500]}\nGOLD: {gold}")
            if s is not None:
                cor.append(s)
    avg = lambda l: round(sum(l) / len(l), 3) if l else None
    return avg(wf), avg(sup), avg(cor), len(sup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="MultiHopRAG")
    ap.add_argument("--files", default=None,
                    help="override sources, e.g. E:quality_e_strictgold_qgate.json,"
                         "D:quality_e_randomtype_strictgold_qgate.json,F:quality_alltypes.json")
    ap.add_argument("--out", default="fair_rescore_edf.json")
    args = ap.parse_args()
    base = os.path.join("conv_rag_benchmark", "output", args.label)
    specs = [("E", "quality_e_50conv.json"), ("D", "quality_e_randomtype.json"),
             ("F", "quality_alltypes.json")]
    if args.files:
        specs = [tuple(pair.split(":", 1)) for pair in args.files.split(",")]

    print(f"# FAIR rescore (context-aware well-formed + abstentions excluded) — {args.label}\n")
    print(f"{'method':<8}{'well_formed':>13}{'gold_supported':>16}{'gold_correct':>14}")
    rows = {}
    for m, fn in specs:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            print(f"{m:<8} (missing)"); continue
        d = json.load(open(p, encoding="utf-8"))
        w, s, c, n = fair(d["conversations"])
        old = d.get("quality", {}).get("E") or d.get("quality", {})
        rows[m] = {"fair": (w, s, c), "old": old}
        print(f"{m:<8}{str(w):>13}{str(s):>16}{str(c):>14}")

    print("\n(strict/original for comparison)")
    print(f"{'method':<8}{'well_formed':>13}{'gold_supported':>16}{'gold_correct':>14}")
    for m in rows:
        o = rows[m]["old"]
        print(f"{m:<8}{str(o.get('well_formed')):>13}{str(o.get('gold_supported')):>16}"
              f"{str(o.get('gold_correct')):>14}")
    json.dump({m: rows[m]["fair"] for m in rows},
              open(os.path.join(base, args.out), "w"), indent=2)


if __name__ == "__main__":
    main()
