"""
triple_audit.py — audit benchmark triples (question, gold_support, gold_answer).

A stricter, three-dimension quality audit than the holistic G-Eval rubrics. For each
triple it asks an LLM auditor to scrutinise the GOLD data itself (assuming nothing is
correct) along three independent axes:

  faithfulness      every factual claim in the gold ANSWER must be grounded EXCLUSIVELY
                    in the gold SUPPORT — outside knowledge / unsupported leaps are penalised.
                    (≈ gold_supported, but claim-by-claim with quotes.)
  answer_relevancy  does the gold ANSWER directly, fully and concisely answer the QUESTION?
                    Evasive, partial or padded answers are penalised. (≈ gold_correct.)
  support_focus     does the SUPPORT actually contain the evidence the question needs, and
                    what is its signal-to-noise? NEW — no existing metric covers this;
                    catches bloated or evidence-missing context. (≈ RAGAS context relevance.)

Each triple also gets a PASS / NEEDS REVISION / FAIL verdict plus actionable feedback, so
low-scoring items can be fixed rather than just counted.

Reads a saved quality_e*.json (question / evidence / gold per turn). Changes nothing else.

Run:
    python -m conv_rag_benchmark.triple_audit --dataset MultiHopRAG --n 25
    python -m conv_rag_benchmark.triple_audit --file conv_rag_benchmark/output/qasper/quality_e.json --n 25
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from statistics import mean
from typing import Dict, Optional

from .config import Config
from .llm import LLM
from .generation.gold_answer_generator import ABSTENTION

_AUDIT_SYS = (
    "You are an expert RAG dataset auditor. You evaluate a benchmark triple: QUESTION, "
    "GOLD_SUPPORT (reference context) and GOLD_ANSWER (reference answer).\n"
    "Do NOT assume the gold data is correct — scrutinise it for human error, unsupported "
    "leaps of logic, and missing context.\n\n"
    "1. faithfulness (0.0-1.0) — Extract every factual claim in GOLD_ANSWER. For each, verify "
    "it is stated in, or logically inferable EXCLUSIVELY from, GOLD_SUPPORT. Penalise any "
    "outside knowledge, assumption or extrapolation not backed by the support.\n"
    "2. answer_relevancy (0.0-1.0) — Does GOLD_ANSWER directly, fully and concisely answer "
    "QUESTION? Penalise evasive answers, partial answers that miss sub-questions, and padding.\n"
    "3. support_focus (SENTENCE RATIO — count, do not estimate):\n"
    "   a. Count the total distinct sentences in GOLD_SUPPORT -> total_sentences.\n"
    "   b. Extract the exact SUBSET of those sentences that are STRICTLY NECESSARY to infer "
    "or justify the factual claims in GOLD_ANSWER. Quote them verbatim -> necessary_sentences.\n"
    "   c. If GOLD_SUPPORT does not contain the answer at all, necessary_sentences is empty.\n"
    "   Be strict: a sentence is 'necessary' only if removing it would make the answer "
    "underivable. Do NOT include merely topical or contextual sentences.\n\n"
    "Then give a verdict: PASS (usable as-is), NEEDS_REVISION (fixable), FAIL (invalid).\n"
    "Respond ONLY JSON:\n"
    '{"claims":[{"claim":"...","supported":true,"quote":"exact quote from support or empty"}],'
    '"faithfulness":{"score":0.0,"analysis":"..."},'
    '"answer_relevancy":{"score":0.0,"analysis":"..."},'
    '"support_focus":{"total_sentences":0,"necessary_sentences":["..."],"analysis":"..."},'
    '"status":"PASS|NEEDS_REVISION|FAIL","actionable_feedback":"..."}'
)

DIMS = ("faithfulness", "answer_relevancy", "support_focus")


def audit_triple(llm: LLM, question: str, support: str, answer: str,
                 max_support: int = 3000) -> Optional[Dict]:
    """Audit one triple. Returns the parsed audit dict, or None if the call failed."""
    if not (question and answer):
        return None
    out = llm.chat_json(
        _AUDIT_SYS,
        f"QUESTION:\n{question}\n\nGOLD_SUPPORT:\n{(support or '')[:max_support]}"
        f"\n\nGOLD_ANSWER:\n{answer}")
    if not isinstance(out, dict):
        return None
    rec = {"status": str(out.get("status") or "").upper().replace(" ", "_"),
           "actionable_feedback": str(out.get("actionable_feedback") or ""),
           "claims": out.get("claims") or []}
    for d in ("faithfulness", "answer_relevancy"):
        block = out.get(d) or {}
        try:
            rec[d] = max(0.0, min(1.0, float(block.get("score"))))
        except (TypeError, ValueError):
            rec[d] = None
        rec[d + "_analysis"] = str(block.get("analysis") or "")

    # support_focus = |necessary sentences| / |total sentences|.
    # Recomputed HERE from the two counts rather than trusting an LLM-stated ratio
    # (models are unreliable at arithmetic; the counts themselves are what we asked for).
    sf = out.get("support_focus") or {}
    nec = [s for s in (sf.get("necessary_sentences") or []) if str(s).strip()]
    try:
        total = int(sf.get("total_sentences") or 0)
    except (TypeError, ValueError):
        total = 0
    rec["support_total_sentences"] = total
    rec["support_necessary_sentences"] = len(nec)
    rec["support_necessary_quotes"] = nec[:10]
    rec["support_focus"] = (round(min(1.0, len(nec) / total), 3) if total > 0 else None)
    rec["support_focus_analysis"] = str(sf.get("analysis") or "")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit (question, support, answer) benchmark triples")
    ap.add_argument("--dataset", default="MultiHopRAG", help="label under output/")
    ap.add_argument("--file", default=None, help="explicit quality_e*.json path")
    ap.add_argument("--n", type=int, default=25, help="cap on triples audited (cost control)")
    ap.add_argument("--model", default="gpt-4o", help="auditor model (use a STRONG one)")
    ap.add_argument("--skip-unanswerable", action="store_true",
                    help="skip intentionally-unanswerable turns (abstention golds)")
    args = ap.parse_args(argv)

    config = Config.load()
    path = args.file or os.path.join(config.output_dir, args.dataset, "quality_e.json")
    if not os.path.exists(path):
        print(f"!! not found: {path}"); return
    data = json.load(open(path, encoding="utf-8"))

    llm = LLM(model=args.model, config=config)
    if not llm.available:
        print("!! needs an OpenAI key (RAG-DIVE/.env or OPENAI_API_KEY)"); return
    print(f"# auditing triples from {path}\n# auditor={args.model} | cap n={args.n}")

    rows, skipped = [], 0
    for c in data.get("conversations", []):
        for t in c.get("turns", []):
            if len(rows) >= args.n:
                break
            gold = (t.get("gold") or "").strip()
            if args.skip_unanswerable and gold.startswith(ABSTENTION[:20]):
                skipped += 1
                continue
            rec = audit_triple(llm, t.get("question", ""), t.get("evidence", ""), gold)
            if not rec:
                continue
            rec.update({"query_type": t.get("query_type"),
                        "question": (t.get("question") or "")[:200],
                        "gold": gold[:200]})
            rows.append(rec)
        if len(rows) >= args.n:
            break

    if not rows:
        print("!! no triples audited"); return

    def avg(k):
        vs = [r[k] for r in rows if r.get(k) is not None]
        return round(mean(vs), 3) if vs else None

    print(f"\n{'dimension':<20}{'mean':>8}")
    print(" " + "-" * 27)
    for d in DIMS:
        print(f"{d:<20}{str(avg(d)):>8}")
    print(" " + "-" * 27)
    # the raw counts behind support_focus, so the ratio is inspectable
    tot, nec = avg("support_total_sentences"), avg("support_necessary_sentences")
    if tot:
        print(f"  support_focus = necessary/total sentences: "
              f"avg {nec} necessary of {tot} total")
        zero = sum(1 for r in rows if r.get("support_necessary_sentences") == 0)
        print(f"  triples where NO sentence justified the answer: {zero}/{len(rows)}")
    verdicts = Counter(r["status"] for r in rows)
    print(f"\nverdicts (n={len(rows)}"
          + (f", {skipped} unanswerable skipped" if skipped else "") + "):")
    for k, v in verdicts.most_common():
        print(f"   {k or '(none)':<16}{v:>4}  ({100*v/len(rows):.0f}%)")

    # per-type breakdown (which question types produce bad triples)
    by_type: Dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r.get("query_type") or "?", []).append(r)
    print(f"\n{'query_type':<22}{'n':>4}" + "".join(f"{d[:12]:>14}" for d in DIMS))
    print(" " + "-" * 63)
    for qt, rs in sorted(by_type.items(),
                         key=lambda x: mean([r["faithfulness"] for r in x[1]
                                             if r.get("faithfulness") is not None] or [1])):
        line = f"{qt:<22}{len(rs):>4}"
        for d in DIMS:
            vs = [r[d] for r in rs if r.get(d) is not None]
            line += f"{(round(mean(vs), 3) if vs else '-'):>14}"
        print(line)

    # the worst offenders, with the fix instruction
    bad = [r for r in rows if r["status"] in ("FAIL", "NEEDS_REVISION")]
    if bad:
        print(f"\n--- {len(bad)} triple(s) needing attention (showing up to 3) ---")
        for r in bad[:3]:
            print(f"\n  [{r['status']}] {r.get('query_type')} | "
                  f"faith={r['faithfulness']} rel={r['answer_relevancy']} focus={r['support_focus']}")
            print(f"    Q: {r['question'][:110]}")
            print(f"    FIX: {r['actionable_feedback'][:220]}")

    out_path = os.path.join(os.path.dirname(path), "triple_audit.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({"source": path, "auditor": args.model, "n": len(rows),
                   "means": {d: avg(d) for d in DIMS},
                   "verdicts": dict(verdicts), "rows": rows}, fw,
                  ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")
    return {d: avg(d) for d in DIMS}


if __name__ == "__main__":
    main()
