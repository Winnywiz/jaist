"""
HopWeaver-style DOCUMENT-REQUIREMENT (cheating) control for methods D / E / F.

Ask answerer LLMs each benchmark question twice:
  Q-only  : conversation history + question, NO documents
  Q+docs  : conversation history + question + the evidence documents
and score EM / token-F1 against the gold. A large (Q+docs − Q-only) gap means the
questions genuinely REQUIRE the documents (not answerable from parametric memory).

Adaptations from HopWeaver (Section 4, Authentic Reasoning Test):
  * 2 answerer LLMs (gpt-4o-mini, gpt-4o) instead of 4.
  * conversation history given in BOTH conditions (questions are conversational; only
    the documents vary, so the gap isolates document need, not referent need).
  * substantive golds only (abstention golds have no reference answer).

Run:  python -m conv_rag_benchmark.doc_requirement
"""
import json
import os
import re
import string
from collections import Counter

from openai import OpenAI

from .geval import _load_key

ANSWERERS = ["gpt-4o-mini", "gpt-4o"]
ABST = "Not answerable"
client = OpenAI(api_key=_load_key())

_SYS_NODOC = ("Answer the QUESTION using the CONVERSATION for context and your own "
              "knowledge. Give a short, direct answer only.")
_SYS_DOC = ("Answer the QUESTION using the CONVERSATION for context and the DOCUMENTS. "
            "Give a short, direct answer only.")


def _answer(model, sys_p, user):
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=80,
            messages=[{"role": "system", "content": sys_p},
                      {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em(pred, gold):
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def _f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _items(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for c in d["conversations"]:
        hist = []
        for t in c["turns"]:
            q, gold = t.get("question") or "", t.get("gold") or ""
            ev = (t.get("evidence") or "")[:1800]
            if gold.strip() and not gold.startswith(ABST) and \
                    t.get("query_type") not in ("Seed",) and ev:
                out.append({"q": q, "gold": gold, "ev": ev,
                            "hist": "\n".join(hist[-4:])})
            hist.append(f"user: {q}")
            if t.get("rag_answer"):
                hist.append(f"assistant: {t['rag_answer']}")
    return out


DEFAULT_SPECS = [
    ("E-multihoprag", "conv_rag_benchmark/output/MultiHopRAG/quality_e_strictgold_seedfix_rep3.json"),
    ("E-qasper", "conv_rag_benchmark/output/eval_E_qasper.json"),
    ("E-arxivcs", "conv_rag_benchmark/output/eval_E_arxivcs.json"),
]
OUT = os.path.join("conv_rag_benchmark", "output", "doc_requirement_by_dataset.json")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="HopWeaver-style doc-requirement control")
    ap.add_argument("files", nargs="*",
                    help="label=path build files (default: E builds per dataset)")
    args = ap.parse_args(argv)
    specs = [(f.split("=", 1)[0], f.split("=", 1)[1]) if "=" in f
             else (os.path.basename(f).replace(".json", ""), f)
             for f in args.files] or DEFAULT_SPECS
    results = {}
    for m, p in specs:
        if not os.path.exists(p):
            print(f"{m}: missing {p}"); continue
        items = _items(p)
        print(f"# {m}: {len(items)} substantive questions", flush=True)
        per_model = {}
        for model in ANSWERERS:
            em_no, f1_no, em_doc, f1_doc = [], [], [], []
            for i, it in enumerate(items):
                a_no = _answer(model, _SYS_NODOC,
                               f"CONVERSATION:\n{it['hist']}\n\nQUESTION: {it['q']}")
                a_doc = _answer(model, _SYS_DOC,
                                f"CONVERSATION:\n{it['hist']}\n\nDOCUMENTS:\n{it['ev']}"
                                f"\n\nQUESTION: {it['q']}")
                em_no.append(_em(a_no, it["gold"])); f1_no.append(_f1(a_no, it["gold"]))
                em_doc.append(_em(a_doc, it["gold"])); f1_doc.append(_f1(a_doc, it["gold"]))
                if (i + 1) % 20 == 0:
                    print(f"  {m}/{model}: {i+1}/{len(items)}", flush=True)
            avg = lambda l: round(sum(l) / len(l), 3) if l else None
            per_model[model] = {
                "em_q_only": avg(em_no), "f1_q_only": avg(f1_no),
                "em_q_docs": avg(em_doc), "f1_q_docs": avg(f1_doc),
                "f1_gap": round(avg(f1_doc) - avg(f1_no), 3),
                "em_gap": round(avg(em_doc) - avg(em_no), 3)}
        results[m] = {"n": len(items), "answerers": per_model}
        with open(OUT, "w", encoding="utf-8") as fw:     # incremental save
            json.dump(results, fw, ensure_ascii=False, indent=2)

    print(f"\n{'method':<8}{'answerer':<14}{'F1 Q-only':>10}{'F1 Q+docs':>10}{'F1 gap':>8}"
          f"{'EM gap':>8}")
    for m, r in results.items():
        for model, s in r["answerers"].items():
            print(f"{m:<8}{model:<14}{s['f1_q_only']:>10}{s['f1_q_docs']:>10}"
                  f"{s['f1_gap']:>8}{s['em_gap']:>8}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
