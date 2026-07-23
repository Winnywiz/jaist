"""
Unified Method E — ONE adaptive method, scored on BOTH axes at once.

The proposed method is a single engine: DYNAMIC ADAPTIVE FOLLOW-UP. Given a failing
RAG turn, it generates diagnostic follow-up QUESTIONS conditioned on the RAG's real
answer, and the outcomes of those follow-ups classify the failure CAUSE
(Retrieval / Generation / Conversation).

Because those follow-ups are themselves questions, ONE run yields two numbers:

  1. ATTRIBUTION ACCURACY — did the generated follow-ups pin the RIGHT cause?
     Measured against INJECTED failures (known cause), vs the static Xie baseline.
  2. QUESTION QUALITY — are the generated follow-ups well-formed and grounded?
     Measured with the same G-Eval used for Method E (well_formed / gold_supported /
     gold_correct).

So the dynamic follow-up generator IS Method E, and this file scores that one method
on both question quality and attribution accuracy simultaneously.

Run:
    python -m conv_rag_benchmark.unified_method_e --dataset multihoprag --n 20
    python -m conv_rag_benchmark.unified_method_e --dataset multihoprag --n 8 --offline  # attribution only, no API
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .config import Config, get_logger
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .llm import LLM
from .geval import geval_items, geval_breakdown_by_type, RUBRICS

# reuse the attribution experiment's building blocks (no re-implementation)
from dynamic_vs_static_dx.core import (
    CONVERSATION, GENERATION, RETRIEVAL, ControlledRAG, OfflineLLM,
    abstained, build_injected_cases, is_correct, set_judge)
from dynamic_vs_static_dx.arms import (
    SingleTurn, XieStaticDecomp, XieFollowupOnly, base_attribute, did_fail)

logger = get_logger("unified.methodE")


class MethodE_Attributor:
    """Method E in attribution mode.

    Generates diagnostic FOLLOW-UP questions conditioned on the RAG's answer (P1..P3),
    returns the attributed CAUSE, AND records every follow-up it generated so the same
    questions can be quality-scored. Mirrors dynamic_vs_static_dx.DynamicFollowup so the
    attribution numbers match — but here the generated probes are captured, not discarded."""

    name = "4.dynamic(MethodE)"

    def __init__(self, rag: ControlledRAG):
        self.rag = rag
        self.probes: List[Dict] = []          # every generated follow-up (for quality)

    def _record(self, question: str, evidence: str, gold: str, qtype: str):
        self.probes.append({"question": question, "evidence": (evidence or "")[:1500],
                            "gold": gold or "", "query_type": qtype})

    def predict_category(self, case) -> str:
        if not did_fail(case):
            return "None"
        ctx_text = " ".join(case.given_context)

        # P1 — coreference follow-up: re-ask with the antecedent named. If naming the
        # entity fixes it, the failure was conversational (only a dynamic probe sees this).
        if case.antecedent_question:
            self._record(case.antecedent_question, ctx_text, case.gold_answer, "Follow-Up")
            a1 = self.rag.answer(case.antecedent_question, case.given_context, [])
            if is_correct(case.gold_answer, a1):
                return CONVERSATION

        # P1b — reference-resolution probe: does the RAG name the referent only WITH history?
        if case.history and case.antecedent_question:
            referent = (case.history[-1].get("content") or "").strip()
            probe_q = (f"In the question \"{case.question}\", what specific entity does the "
                       f"pronoun refer to? Reply with ONLY the entity name.")
            if referent:
                self._record(probe_q, ctx_text, referent, "Ambiguous Reference")
                with_h = is_correct(referent, self.rag.answer(probe_q, case.given_context, case.history))
                without_h = is_correct(referent, self.rag.answer(probe_q, case.given_context, []))
                if with_h and not without_h:
                    return CONVERSATION

        # P2 — retrieval follow-up: re-ask given the gold passage. If that fixes it, the
        # evidence was simply missing -> Retrieval.
        if case.gold_passage:
            self._record(case.question, case.gold_passage, case.gold_answer, "Correction")
            a2 = self.rag.answer(case.question, [case.gold_passage], [])
            if is_correct(case.gold_answer, a2):
                return RETRIEVAL

        # P3 — strict-grounding follow-up: re-ask, forbidding ungrounded answers. If the RAG
        # now abstains though it first answered, that answer was fabricated -> Generation.
        strict_q = (case.question + "\n\nAnswer ONLY from the given passages. If the "
                    "passages do not state the answer, reply exactly: I don't know.")
        self._record(strict_q, ctx_text, case.gold_answer, "Clarification")
        a3 = self.rag.answer(strict_q, case.given_context, [])
        if abstained(a3) and not abstained(case.rag_answer):
            return GENERATION

        return base_attribute(case, recovered=False)


def _attribution_report(cases, preds: Dict[str, List[str]]) -> Dict:
    out = {"n": len(cases), "methods": {}}
    for method, plist in preds.items():
        correct = sum(1 for c, p in zip(cases, plist) if p == c.true_category)
        by_cat = defaultdict(lambda: [0, 0])
        for c, p in zip(cases, plist):
            by_cat[c.true_category][1] += 1
            if p == c.true_category:
                by_cat[c.true_category][0] += 1
        out["methods"][method] = {
            "accuracy": round(correct / (len(cases) or 1), 3),
            "by_category": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]}
                            for k, v in sorted(by_cat.items())}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified Method E: quality + attribution in one run")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--n", type=int, default=20, help="seeds (3 injected cases each)")
    ap.add_argument("--offline", action="store_true",
                    help="no API: attribution plumbing only, quality scoring skipped")
    ap.add_argument("--quality-model", default="gpt-4o-mini")
    args = ap.parse_args(argv)

    config = Config.load(dataset=args.dataset, max_samples=max(args.n + 5, 30),
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    out_dir = os.path.join(config.output_dir, "unified")
    os.makedirs(out_dir, exist_ok=True)

    use_llm = config.has_openai and not args.offline
    llm = LLM(model=config.gen_model, config=config) if use_llm else OfflineLLM()
    embedder = Embedder(config=config, llm=llm) if use_llm else None
    if use_llm:
        set_judge(LLM(model=config.judge_model, config=config))
    print(f"# UNIFIED Method E | dataset={args.dataset} | n={args.n} | "
          f"{'online' if use_llm else 'OFFLINE (attribution plumbing only)'}")

    seeds = DatasetLoader(args.dataset, max_samples=config.max_samples).load()
    corpus = [c for s in seeds for c in s.context if c and c.strip()]
    rag = ControlledRAG(corpus, config, llm, embedder)

    all_cases = build_injected_cases(seeds, rag, args.n, llm)
    cases = [c for c in all_cases if did_fail(c)]        # only failures that actually fired
    fired = Counter(c.true_category for c in cases)
    print(f"# injected failures that fired: {dict(fired)}  (total {len(cases)})")
    if not cases:
        print("!! no failures fired — nothing to attribute."); return

    # -- the ONE method (Method E dynamic) + the static baselines --
    method_e = MethodE_Attributor(rag)
    static_arms = [SingleTurn(), XieStaticDecomp(rag, llm), XieFollowupOnly(rag, llm)]

    preds = {a.name: [a.predict_category(c) for c in cases] for a in static_arms}
    preds[method_e.name] = [method_e.predict_category(c) for c in cases]

    # ===== AXIS 1: attribution accuracy (Method E vs static) =====
    attr = _attribution_report(cases, preds)
    print("\n===== ATTRIBUTION ACCURACY (cause classification) =====")
    print(f"{'method':<24}{'overall':>9}   by-category")
    for m, r in attr["methods"].items():
        bycat = {k: v["acc"] for k, v in r["by_category"].items()}
        print(f"{m:<24}{r['accuracy']:>9}   {bycat}")

    # ===== AXIS 2: quality of the follow-up questions Method E generated =====
    quality = e_by_type = None
    if use_llm and method_e.probes:
        print(f"\n# scoring quality of {len(method_e.probes)} Method-E-generated follow-ups ...")
        quality, scored = geval_items(method_e.probes, model=args.quality_model,
                                      criteria=list(RUBRICS))
        e_by_type = geval_breakdown_by_type(scored)
        print("\n===== QUESTION QUALITY of the generated diagnostic follow-ups =====")
        for k in ("well_formed", "gold_supported", "gold_correct"):
            print(f"{k:<16}{quality.get(k)}")
    else:
        print("\n(quality scoring skipped — needs an OpenAI key; run without --offline)")

    dump = {"dataset": args.dataset, "n_cases": len(cases),
            "fired": dict(fired),
            "attribution": attr,
            "question_quality": quality,
            "quality_by_type": e_by_type,
            "generated_followups": method_e.probes}
    path = os.path.join(out_dir, f"unified_{args.dataset}.json")
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(dump, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {path}")
    return dump


if __name__ == "__main__":
    main()
