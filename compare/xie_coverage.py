"""
Faithful implementation of the Xie et al. (NAACL 2025) sub-question COVERAGE method —
"Do RAG Systems Cover What Matters? Evaluating and Optimizing Responses with Sub-Question
Coverage."

This is the paper's *evaluation/attribution* method (its sections 3.1-3.2), used here as
the "Xie" baseline attributor. Given one (question, RAG answer, retrieved chunks) triple it:

  1. DECOMPOSES the question into ~sub-questions, then CLASSIFIES each as
     core / background / follow-up  (the paper's two-step prompting, Table 5).
  2. Measures COVERAGE with an LLM judge (Table 6): for each sub-question, is it answered
     by the ANSWER? is it covered by the RETRIEVED chunks?
  3. Builds the paper's answered x retrieved 2x2 per type, which IS a retrieval-vs-
     generation attribution:
        not retrieved            -> RETRIEVAL failure (knowledge never retrieved)
        retrieved but not answered -> GENERATION failure (retrieved, not used by the LLM)
        answered                 -> covered (success)
  4. Reports the paper's coverage metrics (M1 answer-coverage, M2 retrieval-coverage), per
     type and overall, with emphasis on CORE sub-questions (the paper's headline signal).

Prompts are quoted from the paper (Tables 5-6); only JSON output-formatting is added.
This module ONLY attributes — it does not modify any RAG or generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from conv_rag_benchmark.config import Config
from conv_rag_benchmark.llm import LLM

_TYPES = ("core", "background", "follow-up")


class XieCoverage:
    """The paper's sub-question coverage evaluator / attributor."""

    # Table 5, step 1 — comprehensive decomposition (paper: "around 20"; we cap lower to
    # bound cost, which does not change the method, only the sub-question budget).
    _DECOMPOSE_SYS = (
        "Decompose the following complex question into a collection of around 15 "
        "sub-questions that you think would be relevant to answer the complex question "
        "fully. Respond as JSON: {\"subquestions\": [\"...\", ...]}"
    )

    # Table 5, step 2 — classify each sub-question (definitions quoted from the paper).
    _CLASSIFY_SYS = (
        "Based on each sub-question's relevance and functional role in answering the "
        "complex question, classify it into three types: core, background, and follow-up.\n"
        "(1) Core: central to the main topic and directly or partially address the complex "
        "question; crucial for its logical reasoning; often involve multiple steps.\n"
        "(2) Background: optional; provide additional context/background that helps clarify "
        "the question; supplementary, not strictly necessary.\n"
        "(3) Follow-up: not needed to answer the question; arise after an initial answer to "
        "seek further clarification/detail; may be out-of-scope.\n"
        "Respond as JSON: {\"classified\": [{\"q\": \"...\", \"type\": \"core|background|follow-up\"}]}"
    )

    # Table 6 — automatic coverage judgment (quoted).
    _COVERAGE_SYS = (
        "You are given a piece of text and a question. Judge if there exists any part of "
        "the given text that can answer the question. If you believe the question can be "
        "answered, identify the text fragment that answers the question; otherwise return "
        "None. Respond as JSON: {\"covered\": true/false, \"fragment\": \"...\"}"
    )

    def __init__(self, config: Optional[Config] = None, llm: Optional[LLM] = None):
        self.config = config or Config.load()
        self.llm = llm or LLM(model=self.config.gen_model, config=self.config)

    # -- step 1+2: two-step decomposition ---------------------------------------- #
    def decompose(self, question: str) -> List[Tuple[str, str]]:
        """Return [(sub_question, type)], type in core/background/follow-up."""
        out = self.llm.chat_json(self._DECOMPOSE_SYS, f"Complex question: {question}")
        subs = [str(s).strip() for s in (out or {}).get("subquestions", []) if str(s).strip()]
        if not subs:
            return []
        listed = "\n".join(f"- {s}" for s in subs)
        out2 = self.llm.chat_json(
            self._CLASSIFY_SYS, f"Complex question: {question}\nSub-questions:\n{listed}")
        cls = {str(c.get("q", "")).strip(): str(c.get("type", "core")).strip().lower()
               for c in (out2 or {}).get("classified", [])}
        result = []
        for s in subs:
            t = cls.get(s, "core")
            result.append((s, t if t in _TYPES else "core"))
        return result

    # -- step 3: coverage judgment (Table 6) ------------------------------------- #
    def _covered(self, text: str, subquestion: str) -> bool:
        if not (text or "").strip():
            return False
        out = self.llm.chat_json(
            self._COVERAGE_SYS,
            f"Piece of text: {text[:3000]}\nQuestion: {subquestion}")
        return bool((out or {}).get("covered"))

    def attribute(self, question: str, answer: str,
                  retrieved_chunks: List[str]) -> Dict:
        """Run the full Xie method on one (question, answer, retrieved chunks) triple.

        For each sub-question: ``answered`` = covered by the ANSWER; ``retrieved`` =
        covered by ANY retrieved chunk (judged over the concatenated retrieved context, so
        the retrieved? signal is one call). Returns per-sub-question coverage, the 2x2
        attribution, and the paper's M1/M2 coverage rates per type + overall."""
        subs = self.decompose(question)
        retrieved_ctx = "\n\n".join(c for c in (retrieved_chunks or []) if c)[:6000]

        records, counts = [], {t: {"n": 0, "answered": 0, "retrieved": 0,
                                    "retrieval_fail": 0, "generation_fail": 0} for t in _TYPES}
        for q, t in subs:
            answered = self._covered(answer, q)
            retrieved = self._covered(retrieved_ctx, q)
            # the paper's diagnostic 2x2 -> retrieval-vs-generation attribution
            layer = None
            if not answered:
                layer = "retrieval" if not retrieved else "generation"
            c = counts[t]
            c["n"] += 1
            c["answered"] += int(answered)
            c["retrieved"] += int(retrieved)
            if layer == "retrieval":
                c["retrieval_fail"] += 1
            elif layer == "generation":
                c["generation_fail"] += 1
            records.append({"subquestion": q, "type": t, "answered": answered,
                            "retrieved": retrieved, "attributed_layer": layer})

        def rate(a, b):
            return round(a / b, 3) if b else None

        per_type = {t: {
            "n": c["n"],
            "answer_coverage_M1": rate(c["answered"], c["n"]),
            "retrieval_coverage_M2": rate(c["retrieved"], c["n"]),
            "retrieval_failures": c["retrieval_fail"],
            "generation_failures": c["generation_fail"],
        } for t, c in counts.items()}

        tot = {k: sum(counts[t][k] for t in _TYPES)
               for k in ("n", "answered", "retrieved", "retrieval_fail", "generation_fail")}
        # headline attribution (paper emphasises CORE): overall layer split of uncovered
        core = counts["core"]
        return {
            "n_subquestions": len(subs),
            "subquestions": records,
            "per_type": per_type,
            "overall": {
                "answer_coverage_M1": rate(tot["answered"], tot["n"]),
                "retrieval_coverage_M2": rate(tot["retrieved"], tot["n"]),
                "retrieval_failures": tot["retrieval_fail"],
                "generation_failures": tot["generation_fail"],
                "attributed_layer": ("retrieval" if tot["retrieval_fail"] > tot["generation_fail"]
                                     else "generation" if tot["generation_fail"] > 0 else "none"),
            },
            "core_attribution": {
                "core_answer_coverage": rate(core["answered"], core["n"]),
                "core_retrieval_failures": core["retrieval_fail"],
                "core_generation_failures": core["generation_fail"],
            },
        }
