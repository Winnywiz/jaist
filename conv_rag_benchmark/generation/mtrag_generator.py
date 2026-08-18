"""
Replay generator for the mtRAG method — the human-conversation SOURCE.

Where :class:`AdaptiveConversationGenerator` *authors* questions, this generator does
NOT author anything: it takes the real human multi-turn conversations from mtRAG (loaded,
structure-preserving, by :mod:`conv_rag_benchmark.datasets.mtrag_conversations`) and
REPLAYS each human user turn through a target RAG, grading the RAG's answer against the
human agent answer (the gold).

It emits the SAME :class:`AdaptiveConversation` / :class:`AdaptiveTurn` schema as the
generated methods, so the identical conversation-log format and the downstream
``failure/`` classifier both consume it with no changes. The two document sets stay
separate and honest:

  * ``question_generation_documents`` — EMPTY for mtRAG: the questions are human-written,
    there was no automated question-generation retrieval. (Documented, not fabricated.)
  * ``rag_retrieved_documents``       — what the TARGET RAG retrieved to answer, i.e. real
    scores from OUR RAG (the same RAG the generated methods use).
  * mtRAG's own native ELSER contexts (document_id + native score) are preserved under
    ``provenance.native_contexts`` — kept, clearly labelled, never mixed into either set
    above (ELSER scores are not comparable to our cosine scores).

The RAG is fed its OWN prior answers as history (``rag_history``), matching exactly how
the generated methods drive the RAG, so the comparison stays about the conversation
SOURCE and not about how history is threaded.

This module chooses NO corpus and builds NO RAG — the caller passes an already-built
target RAG. That keeps the (still-open) mtRAG corpus decision out of here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..interfaces.rag_interface import RAGInterface
from ..llm import LLM
from .dynamic_generator import (AdaptiveConversation, AdaptiveTurn, OUTCOMES,
                                 _GRADE_SYS)
from ..datasets.mtrag_conversations import MtragConversation


class MtragReplayGenerator:
    """Replay human mtRAG conversations through a target RAG (no question authoring)."""

    def __init__(self, target_rag: RAGInterface, judge: LLM):
        self.target_rag = target_rag
        self.judge = judge

    def _grade(self, question: str, gold: str, rag_answer: str,
               is_unanswerable: bool) -> str:
        """Same grader contract as the adaptive generator (correct/wrong/hallucinated/
        abstained), so outcomes are directly comparable across methods."""
        if not (rag_answer or "").strip():
            return "abstained"
        if self.judge.available:
            out = self.judge.chat_json(
                _GRADE_SYS,
                f"QUESTION: {question}\nGOLD: {gold}\n"
                f"UNANSWERABLE: {is_unanswerable}\nRAG ANSWER: {rag_answer}")
            if out and out.get("label") in OUTCOMES:
                return out["label"]
        low = (rag_answer or "").lower()
        if any(p in low for p in ("i don't know", "cannot answer", "not answerable",
                                  "no information", "unable to")):
            return "abstained"
        if is_unanswerable:
            return "hallucinated"
        return "correct" if gold.lower()[:30] in low or low[:30] in gold.lower() \
            else "wrong"

    def replay(self, conv: MtragConversation,
               conversation_id: Optional[str] = None,
               max_turns: Optional[int] = None) -> AdaptiveConversation:
        """Replay one human conversation, returning the shared-schema log object."""
        cid = conversation_id or conv.conversation_id
        seed_q = conv.turns[0].question if conv.turns else ""
        out = AdaptiveConversation(conversation_id=cid, seed_question=seed_q)
        rag_history: List[Dict] = []

        turns = conv.turns if max_turns is None else conv.turns[:max_turns]
        for t in turns:
            is_unans = t.answerability.strip().upper() == "UNANSWERABLE"
            rag_resp = self.target_rag.answer(t.question, rag_history)
            rag_answer = rag_resp.answer or ""
            outcome = self._grade(t.question, t.answer, rag_answer, is_unans)

            # mtRAG's native retrieved evidence (ELSER) — preserved, clearly labelled,
            # NOT merged into rag_retrieved_documents (which is OUR RAG's cosine retrieval).
            native = [c.to_dict() for c in t.contexts]
            provenance = {
                "content_mode": "mtrag_human",
                "generator_saw_rag_answer": False,   # humans wrote it, not from our RAG
                "mtrag_question_type": t.question_type,
                "mtrag_answerability": t.answerability,
                "mtrag_multi_turn": t.multi_turn,
                "native_contexts": native,           # mtRAG ELSER contexts (score_type=elser)
            }
            out.turns.append(AdaptiveTurn(
                turn_id=t.turn_id,
                query_type=(f"mtrag:{t.question_type}" if t.question_type else "mtrag"),
                question=t.question,
                gold=t.answer,
                # the human evidence the answer was grounded in (mtRAG native text)
                evidence="\n\n".join(c.text for c in t.contexts)[:1500],
                question_evidence="\n\n".join(c.text for c in t.contexts)[:1500],
                rag_answer=rag_answer,
                rag_retrieved_context=list(rag_resp.retrieved_context or []),
                outcome=outcome,
                is_unanswerable=is_unans,
                guard_gave_up=False,
                provenance=provenance,
                # human-authored question -> NO automated question-generation retrieval.
                question_generation_documents=[],
                rag_retrieved_documents=list(rag_resp.retrieved_docs or []),
                generation_source="mtrag_human",
            ))
            rag_history += [{"role": "user", "content": t.question},
                            {"role": "assistant", "content": rag_answer}]
            out.type_sequence.append(t.question_type or "mtrag")
            out.outcome_sequence.append(outcome)
        return out
