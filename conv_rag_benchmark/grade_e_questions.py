"""
Grade Method E's QUESTION-GENERATION ability with metrics BEYOND well/supported/correct,
reusing D's exact implementations so E and D are measured identically:

  Type diversity        (algorithmic)        - does E use varied question types?
  Doc-required          (MuSiQue/HotpotQA)   - can't be answered without the evidence (anti-cheating)
  Follow-up dependency  (QuAC/CoQA)          - follow-ups genuinely need the prior turn
  Distinct-1 / Distinct-2 (Li 2016, non-LLM) - questions are lexically diverse, not duplicates

Reads each dataset's quality_e.json (no regeneration). Run:
    python -m conv_rag_benchmark.grade_e_questions
"""
import json
import os

from .build_benchmark import doc_required, is_standalone
from .config import Config
from .llm import LLM

_LABELS = {"MultiHopRAG": "multihoprag", "MedQA": "medqa"}


def _distinct_n(questions, n):
    grams, total = set(), 0
    for q in questions:
        toks = (q or "").lower().split()
        gs = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        grams.update(gs)
        total += len(gs)
    return round(len(grams) / total, 3) if total else None


def grade(path, gen_llm, judge):
    d = json.load(open(path, encoding="utf-8"))
    # build ordered items with prev_q (turn 0 is the seed)
    items, questions = [], []
    for c in d["conversations"]:
        prev_q = ""
        for t in c["turns"]:
            items.append({"question": t["question"], "gold": t["gold"],
                          "query_type": t["query_type"], "turn_id": t["turn_id"],
                          "prev_q": prev_q})
            questions.append(t["question"])
            prev_q = t["question"]

    n = len(items)
    all_types = [it["query_type"] for it in items]
    distinct_per_conv = []
    for c in d["conversations"]:
        ts = [t["query_type"] for t in c["turns"]]
        distinct_per_conv.append(len(set(ts)) / len(ts) if ts else 0)

    doc_req = nonstandalone = nonstandalone_n = 0
    for it in items:
        if doc_required(gen_llm, judge, it["question"], it["gold"]):
            doc_req += 1
        if it["turn_id"] > 0:
            nonstandalone_n += 1
            if not is_standalone(judge, it["prev_q"], it["question"]):
                nonstandalone += 1

    return {
        "n_questions": n,
        "type_diversity": {
            "distinct_types_used": len(set(all_types)),
            "avg_distinct_per_conversation":
                round(sum(distinct_per_conv) / len(distinct_per_conv), 3),
        },
        "doc_required_rate": round(doc_req / n, 3),
        "followup_dependency_rate":
            round(nonstandalone / nonstandalone_n, 3) if nonstandalone_n else 0,
        "distinct_1": _distinct_n(questions, 1),
        "distinct_2": _distinct_n(questions, 2),
    }


def main():
    base = "conv_rag_benchmark/output"
    rows = {}
    for ds in ("MultiHopRAG", "MedQA"):
        p = os.path.join(base, ds, "quality_e.json")
        if not os.path.exists(p):
            print(f"  {ds}: missing {p}")
            continue
        config = Config.load(dataset=_LABELS[ds])
        gen_llm = LLM(model=config.gen_model, config=config)
        judge = LLM(model=config.judge_model, config=config)
        print(f"# grading {ds} (gen {config.gen_model} / judge {config.judge_model}) ...")
        rows[ds] = grade(p, gen_llm, judge)

    print(f"\n{'metric':<34}" + "".join(f"{ds:>14}" for ds in rows))
    flat = lambda r: {
        "distinct_types_used": r["type_diversity"]["distinct_types_used"],
        "avg_distinct_per_conv": r["type_diversity"]["avg_distinct_per_conversation"],
        "doc_required_rate": r["doc_required_rate"],
        "followup_dependency_rate": r["followup_dependency_rate"],
        "distinct_1 (lexical)": r["distinct_1"],
        "distinct_2 (lexical)": r["distinct_2"],
    }
    keys = ["distinct_types_used", "avg_distinct_per_conv", "doc_required_rate",
            "followup_dependency_rate", "distinct_1 (lexical)", "distinct_2 (lexical)"]
    flats = {ds: flat(rows[ds]) for ds in rows}
    for k in keys:
        print(f"{k:<34}" + "".join(f"{str(flats[ds][k]):>14}" for ds in rows))

    json.dump(rows, open(os.path.join(base, "grade_e_questions.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {base}/grade_e_questions.json")


if __name__ == "__main__":
    main()
