"""
Grade the benchmark's QUESTION-GENERATION ability with metrics BEYOND
well/supported/correct:

  Type diversity        (algorithmic)        - does it use varied question types?
  Doc-required          (MuSiQue/HotpotQA)   - can't be answered without the evidence (anti-cheating)
  Follow-up dependency  (QuAC/CoQA)          - follow-ups genuinely need the prior turn
  Distinct-1 / Distinct-2 (Li 2016, non-LLM) - questions are lexically diverse, not duplicates

Reads saved run files under result/benchmark_quality/, named
``{rag}_{dataset}_t{turns}_c{convos}.json`` (no regeneration). Run:
    python -m conv_rag_benchmark.grade_questions
    python -m conv_rag_benchmark.grade_questions --pattern 'result/benchmark_quality/*/graph_*.json'
"""
import argparse
import glob
import json
import os

from .config import Config
from .llm import LLM

_LABELS = {"MultiHopRAG": "multihoprag", "MedQA": "medqa"}

# -- anti-cheating / follow-up-dependency probes ---------------------------- #
# (self-contained here so this grader has no cross-script dependency)
_NODOC_SYS = ('Answer the question using ONLY your own knowledge. Do not guess wildly; '
              'if you do not know, answer "UNKNOWN". Respond JSON: {"answer": "..."}')
_MATCH_SYS = ('Does the CANDIDATE answer match the GOLD answer (same fact)? '
              'Respond JSON: {"match": true/false}')
_STANDALONE_SYS = ('Can the QUESTION be fully understood on its own, WITHOUT the previous turn? '
                   'Respond JSON: {"standalone": true/false}')


def doc_required(gen_llm: LLM, judge: LLM, question: str, gold: str) -> bool:
    """True if the question genuinely needs the document (cannot be answered from
    parametric knowledge alone), i.e. NOT cheatable."""
    out = gen_llm.chat_json(_NODOC_SYS, f"QUESTION: {question}") or {}
    cand = str(out.get("answer", "")).strip()
    if not cand or cand.upper() == "UNKNOWN":
        return True
    m = judge.chat_json(_MATCH_SYS, f"GOLD: {gold}\nCANDIDATE: {cand}") or {}
    return not bool(m.get("match"))


def is_standalone(judge: LLM, prev_q: str, question: str) -> bool:
    """True if the question is fully understandable without the previous turn."""
    out = judge.chat_json(_STANDALONE_SYS,
                          f"PREVIOUS TURN: {prev_q}\nQUESTION: {question}") or {}
    return bool(out.get("standalone", True))


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grade generated-question quality")
    ap.add_argument("--pattern", default="result/benchmark_quality/*/vector_*_t*_c*.json",
                    help="glob of run files to grade (default: the vector runs). "
                         "Runs are named {rag}_{dataset}_t{turns}_c{convos}.json")
    ap.add_argument("--out", default="result/benchmark_quality/grade_questions.json")
    args = ap.parse_args(argv)

    paths = [p for p in sorted(glob.glob(args.pattern)) if not any(x in p for x in ("_docsim", "_summary"))]
    if not paths:
        print(f"no run files matched {args.pattern}")
        return
    rows = {}
    for p in paths:
        ds = os.path.basename(os.path.dirname(p))
        key = f"{ds}/{os.path.basename(p)[:-5]}"
        config = Config.load(dataset=_LABELS.get(ds, ds.lower()))
        gen_llm = LLM(model=config.gen_model, config=config)
        judge = LLM(model=config.judge_model, config=config)
        print(f"# grading {key} (gen {config.gen_model} / judge {config.judge_model}) ...")
        rows[key] = grade(p, gen_llm, judge)

    print(f"\n{'metric':<34}" + "".join(f"{ds.split('/')[0]:>16}" for ds in rows))
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
        print(f"{k:<34}" + "".join(f"{str(flats[ds][k]):>16}" for ds in rows))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
