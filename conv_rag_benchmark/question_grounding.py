"""question_grounding.py — how GROUNDED are the generated Multi-Hop / Comparative questions?

Graph accuracy (see graph_quality.py) is only an internal-resource metric. What actually
matters for the thesis is whether the QUESTIONS the generator produces are grounded — i.e.
every declared answer component (entity+dimension for Comparative, each hop for Multi-Hop)
is supported by the evidence the question was authored from. The generator's grounding
guard is supposed to ensure this; this tool MEASURES how well it does.

Per Comparative/Multi-Hop turn: for each `expected_components` slot, ask the judge whether
the turn's evidence supports it. A question is "fully grounded" if ALL its slots are
supported. Reports the per-question and per-slot grounding rate.

Run:
    python -m conv_rag_benchmark.question_grounding \
        --file result/benchmark_quality/qasper/vector_qasper_t9_c8_dynamic_nostrictgold_typedgraph.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional

from .config import Config
from .llm import LLM

_DECOMP = ("Multi-Hop", "Comparative")

_SYS = (
    "You verify whether an EVIDENCE text supports a required answer COMPONENT of a "
    "question. A component is either 'ENTITY :: dimension' (the evidence must state that "
    "dimension FOR that entity) or a reasoning hop 'A->B (relation)' (the evidence must "
    "state that relation between A and B). Reply true only if the evidence actually "
    'supports it. Respond JSON: {"supported": true/false}.'
)


def check_component(llm: LLM, evidence: str, component: str) -> Optional[bool]:
    if not (llm and getattr(llm, "available", False)):
        return None
    out = llm.chat_json(_SYS, f"EVIDENCE:\n{(evidence or '')[:1500]}\n\nCOMPONENT: {component}")
    if not out or "supported" not in out:
        return None
    return bool(out["supported"])


def check_log(path: str, llm: LLM, limit: Optional[int] = None) -> Dict:
    d = json.load(open(path, encoding="utf-8"))
    q_total = q_grounded = slot_total = slot_ok = 0
    by_type = Counter()
    cases: List[Dict] = []
    for conv in d.get("conversations", []):
        for t in conv.get("turns", []):
            comps = t.get("expected_components") or []
            if t.get("query_type") not in _DECOMP or not comps:
                continue
            if limit and q_total >= limit:
                break
            ev = t.get("question_evidence") or t.get("evidence") or ""
            results = [check_component(llm, ev, c) for c in comps]
            results = [r for r in results if r is not None]
            if not results:
                continue
            n_ok = sum(results)
            fully = (n_ok == len(results))
            q_total += 1
            q_grounded += fully
            slot_total += len(results)
            slot_ok += n_ok
            by_type[t["query_type"]] += 1
            cases.append({"query_type": t["query_type"], "question": t.get("question"),
                          "components": comps, "n_supported": n_ok, "n_slots": len(results),
                          "fully_grounded": fully})
    return {
        "file": path,
        "questions_checked": q_total,
        "by_type": dict(by_type),
        "fully_grounded_rate": round(q_grounded / q_total, 3) if q_total else None,
        "slot_support_rate": round(slot_ok / slot_total, 3) if slot_total else None,
        "cases": cases,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure grounding of Multi-Hop/Comparative questions.")
    ap.add_argument("--file", required=True, help="a conversation log JSON")
    ap.add_argument("--limit", type=int, default=None, help="max questions to check")
    args = ap.parse_args(argv)

    config = Config.load()
    llm = LLM(model=config.judge_model, config=config)
    rep = check_log(args.file, llm, limit=args.limit)

    print(f"# {args.file}")
    print(f" Multi-Hop/Comparative questions checked : {rep['questions_checked']}  {rep['by_type']}")
    print(f" FULLY-GROUNDED question rate            : {rep['fully_grounded_rate']}  "
          "(all declared slots supported by evidence)")
    print(f" per-SLOT support rate                   : {rep['slot_support_rate']}")
    print("\n examples:")
    for c in rep["cases"][:8]:
        mark = "OK " if c["fully_grounded"] else "XX "
        print(f"  {mark} [{c['query_type']}] {c['n_supported']}/{c['n_slots']} slots  | {c['question'][:70]}")

    out_path = args.file.replace(".json", "_qgrounding.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(rep, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
