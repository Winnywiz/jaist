"""
Full metric suite on the 3 FAIR-comparison methods (all dynamic, only type-selection
differs): E controller / random-dynamic / all-dynamic (F). Reads the saved 10-conv
files; no regeneration.

Run:  python -m conv_rag_benchmark.grade_fair
"""
import json
import os
from collections import Counter

from .grade_def import grade  # RAG-grading suite (correctness/BERT/faith/recall/prec)
from .build_benchmark import doc_required, is_standalone
from .config import Config
from .llm import LLM

BASE = os.path.join("conv_rag_benchmark", "output", "MultiHopRAG")
SPECS = [("E controller", "quality_e.json"),
         ("random-dynamic", "quality_e_randomtype.json"),
         ("all-dynamic (F)", "quality_alltypes.json")]


def _distinct(qs, n):
    g, t = set(), 0
    for q in qs:
        tk = (q or "").lower().split()
        gs = [tuple(tk[i:i + n]) for i in range(len(tk) - n + 1)]
        g.update(gs); t += len(gs)
    return round(len(g) / t, 3) if t else None


def properties(path, gen_llm, judge):
    d = json.load(open(path, encoding="utf-8"))
    turns = [t for c in d["conversations"] for t in c["turns"]]
    qs = [t["question"] for t in turns]
    types = set(t.get("query_type") for t in turns if t.get("query_type") not in ("Seed",))
    docreq = dep = dep_n = 0
    prev = {}
    for c in d["conversations"]:
        pq = ""
        for t in c["turns"]:
            if doc_required(gen_llm, judge, t["question"], t.get("gold", "")):
                docreq += 1
            if t.get("turn_id", 0) > 0:
                dep_n += 1
                if not is_standalone(judge, pq, t["question"]):
                    dep += 1
            pq = t["question"]
    n = len(turns)
    return {
        "doc_required": round(docreq / n, 3) if n else None,
        "dependency": round(dep / dep_n, 3) if dep_n else None,
        "distinct_1": _distinct(qs, 1),
        "distinct_2": _distinct(qs, 2),
        "types": f"{len(types)}/8",
    }


def main():
    config = Config.load(dataset="multihoprag")
    gen_llm = LLM(model=config.gen_model, config=config)
    judge = LLM(model=config.judge_model, config=config)

    rows = {}
    for name, fn in SPECS:
        p = os.path.join(BASE, fn)
        if not os.path.exists(p):
            print(f"  {name}: missing {p}"); continue
        print(f"# grading {name} ...", flush=True)
        rag = grade(p)
        props = properties(p, gen_llm, judge)
        rows[name] = {**rag, **props}

    order = [("correctness", "RAG correctness"), ("bertscore_f1", "BERTScore-F1"),
             ("faithfulness", "Faithfulness"), ("context_recall_answerable", "Context recall"),
             ("context_precision", "Context precision"), ("failure_rate", "Failure rate"),
             ("doc_required", "Doc-required"), ("dependency", "Follow-up dependency"),
             ("types", "Query types covered")]
    print(f"\n{'metric':<22}" + "".join(f"{n:>17}" for n in rows))
    for key, label in order:
        print(f"{label:<22}" + "".join(f"{str(rows[n].get(key)):>17}" for n in rows))
    print(f"{'Distinct-1/2':<22}" +
          "".join(f"{str(rows[n]['distinct_1'])+'/'+str(rows[n]['distinct_2']):>17}" for n in rows))

    json.dump(rows, open(os.path.join(BASE, "grade_fair_metrics.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
