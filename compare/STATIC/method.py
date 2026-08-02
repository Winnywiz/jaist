"""
STATIC/method.py — THE COMPARISON (BASELINE) METHODS.

Static sub-question generation (Xie et al., NAACL 2025) plus a single-turn control.
"Static" = every probe question is decided UP FRONT and never conditioned on the RAG's
answer. That is the crux of the comparison: static methods cannot react, so they cannot
distinguish a coreference failure and can never output Conversation.

  1. SingleTurn        — attribute from the one failed turn only (no probing at all).
  2. XieStaticDecomp   — decompose the question into ~10 typed sub-questions, probe with
                         the CORE ones, attribute from whether fresh retrieval recovers
                         the gold. Never sees the RAG's answer.
  3. XieFollowupOnly   — same, but probe only with the FOLLOW-UP sub-questions.

All three end in the SAME Retrieval-vs-Generation coverage signal (base_attribute), so
the only thing that varies vs the DYNAMIC method is HOW (and whether) they probe.
"""
from __future__ import annotations

from typing import List, Tuple

from conv_rag_benchmark.llm import LLM

from ..shared.setup import (Case, ControlledRAG, base_attribute, did_fail, is_correct)


def decompose_subquestions(llm: LLM, question: str) -> List[Tuple[str, str]]:
    """Xie 2025: GPT decomposes the question into ~10 typed sub-questions."""
    if not getattr(llm, "available", False):
        return [(question, "core")]                  # offline degenerate fallback
    sys = ("Decompose the complex question into about 10 sub-questions needed to "
           "answer it fully, and classify EACH as one of: core (directly needed), "
           "background (optional context), follow-up (extra detail asked only after "
           "an initial answer). "
           'Respond JSON: {"subquestions":[{"q":"...","type":"core|background|follow-up"}]}')
    out = llm.chat_json(sys, f"QUESTION: {question}")
    subs = (out or {}).get("subquestions") or []
    parsed = [(str(s.get("q", "")).strip(), str(s.get("type", "core")).lower())
              for s in subs if s.get("q")]
    return parsed or [(question, "core")]


class SingleTurn:
    name = "1.single_turn"
    def predict_category(self, case: Case) -> str:
        if not did_fail(case):
            return "None"
        return base_attribute(case, recovered=False)


class _XieStatic:
    """Shared Xie machinery: probe with statically-decomposed sub-questions and
    attribute from whether they recover the gold from FRESH retrieval."""
    keep_type = "core"

    def __init__(self, rag: ControlledRAG, llm: LLM):
        self.rag = rag
        self.llm = llm

    def predict_category(self, case: Case) -> str:
        if not did_fail(case):
            return "None"
        subs = [q for q, t in decompose_subquestions(self.llm, case.question)
                if t == self.keep_type] or [case.question]
        # Probe each sub-question with the RAG's OWN retrieval (decomposition's point).
        recovered = False
        for sq in subs[:6]:
            ctx = self.rag.retrieve(sq)
            ans = self.rag.answer(sq, ctx, [])
            if is_correct(case.gold_answer, ans):
                recovered = True
                break
        # STATIC methods cannot see the RAG's answer, so they never output Conversation.
        return base_attribute(case, recovered)


class XieStaticDecomp(_XieStatic):
    name = "2.xie_static_core"
    keep_type = "core"


class XieFollowupOnly(_XieStatic):
    name = "3.xie_followup_only"
    keep_type = "follow-up"
