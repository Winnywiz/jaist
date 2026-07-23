"""
DYNAMIC/method.py — THE PROPOSED METHOD.

Dynamic follow-up attribution. Given a failed RAG turn, it synthesises follow-up
probes FROM the RAG's actual answer and runs a targeted resolve -> evidence -> strict
protocol. Because it reacts to what the RAG did, it is the ONLY method that can reach
the Conversation (coreference) category — the static methods never see the RAG's answer,
so that category is invisible to them by construction.

Probe order (each step can conclude the cause and stop):
  P1  resolve the reference in the SAME context. If naming the entity alone fixes the
      answer, the failure was conversational (coreference).
  P1b reference-resolution probe: can the RAG name the pronoun's referent only WHEN the
      dropped history is restored? (decouples coreference from answerability for
      multi-hop data where P1's full-recovery test misses real coref failures.)
  P2  supply the gold passage. If that fixes it, the evidence was simply missing ->
      Retrieval.
  P3  strict-grounding probe. Re-ask forbidding ungrounded answers; if the RAG now
      abstains though it first answered confidently, that answer was fabricated ->
      Generation. (Static methods cannot make this call: "absent from what was
      retrieved" and "absent from the corpus" look identical to them.)
  else fall back to the SAME Retrieval-vs-Generation coverage signal the static methods
      use (never blind-guess Generation).
"""
from __future__ import annotations

from ..shared.setup import (CONVERSATION, GENERATION, RETRIEVAL, Case, ControlledRAG,
                            abstained, base_attribute, did_fail, is_correct)


class DynamicFollowup:
    """PROPOSED: follow-ups synthesised FROM the RAG's answer; targeted protocol."""
    name = "4.dynamic_followup"

    def __init__(self, rag: ControlledRAG):
        self.rag = rag

    def predict_category(self, case: Case) -> str:
        if not did_fail(case):
            return "None"
        # P1 (the dynamic advantage): resolve the reference, SAME context. If that
        # alone fixes it, the failure was conversational (coreference), which no
        # static method can see.
        if case.antecedent_question:
            a1 = self.rag.answer(case.antecedent_question, case.given_context, [])
            if is_correct(case.gold_answer, a1):
                return CONVERSATION
        # P1b: REFERENCE-RESOLUTION probe (decouples coreference from answerability).
        # On multi-hop datasets the named question is unanswerable for OTHER reasons,
        # so P1's full-recovery test misses real coref failures. Instead test the thing
        # that actually defines a coreference failure: can the RAG identify the
        # pronoun's REFERENT only WHEN the dropped history is restored? This fires only
        # for Conversation cases (only they carry history + antecedent).
        if case.history and case.antecedent_question:
            referent = (case.history[-1].get("content") or "").strip()
            if referent and self._resolves_referent(case, referent, case.history) \
                    and not self._resolves_referent(case, referent, []):
                return CONVERSATION
        # P2: does supplying the gold passage fix it? -> Retrieval.
        if case.gold_passage:
            a2 = self.rag.answer(case.question, [case.gold_passage], [])
            if is_correct(case.gold_answer, a2):
                return RETRIEVAL
        # P3: STRICT-GROUNDING probe -> Generation. Re-ask with grounding enforced;
        # if the RAG now abstains although it originally gave a substantive answer,
        # that substance was fabricated (overconfident generation).
        if self._strict_abstains(case):
            return GENERATION
        # Otherwise fall back to the SAME retrieval-vs-generation attribution the
        # static methods use (never default blindly to Generation).
        return base_attribute(case, recovered=False)

    def _strict_abstains(self, case: Case) -> bool:
        probe = (case.question + "\n\nAnswer ONLY from the given passages. If the "
                 "passages do not state the answer, reply exactly: I don't know.")
        a3 = self.rag.answer(probe, case.given_context, [])
        return abstained(a3) and not abstained(case.rag_answer)

    def _resolves_referent(self, case: Case, referent: str, history) -> bool:
        """True if the RAG, asked who/what the pronoun refers to, names the referent.
        With history it should resolve; without history the reference is ambiguous."""
        probe = (f"In the question \"{case.question}\", what specific entity does the "
                 f"pronoun refer to? Reply with ONLY the entity name.")
        ans = self.rag.answer(probe, case.given_context, history)
        return is_correct(referent, ans)
