"""Conversation-PRESERVING loader for the mtRAG (IBM) human benchmark.

The canonical dataset adapter :func:`conv_rag_benchmark.datasets.loader._load_mtrag`
FLATTENS mtRAG into independent single-turn ``Sample`` rows (one per agent turn) and
drops the conversation grouping, the turn order, and the dialogue history. That is the
right shape for building a corpus of seed questions, but it destroys exactly the
multi-turn structure the ``mtrag`` experiment method needs: to *replay* the human
follow-up questions in order against a target RAG.

This module adds a SEPARATE loader that keeps the human conversation intact. It does not
modify, import, or replace ``_load_mtrag`` — both coexist. The original data file is only
READ, never written.

Each human turn keeps its native retrieval evidence exactly as mtRAG released it: every
context passage carries mtRAG's own ``document_id`` and ELSER retrieval ``score`` (a
sparse-retrieval score, NOT a cosine — see ``score_type``), so the mtRAG arm can report
real doc_id/rank/score just like the generated arms, without recomputing anything.

Nothing here chooses a corpus or runs a RAG — that decision is deliberately deferred.
"""
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Default location of the released conversations file (read-only).
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "mtrag_validation", "data", "conversations.json")


def _as_dict(value) -> Dict:
    """mtRAG stores ``enrichments`` / ``feedback`` / ``query`` as either real dicts or
    their Python ``repr`` string. Parse both safely; never raise."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            out = ast.literal_eval(value)
            return out if isinstance(out, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _first(value):
    """mtRAG enrichment values are often single-element lists (``['Factoid']``)."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value if value is not None else ""


def _to_float(value) -> Optional[float]:
    """mtRAG scores are strings like ``"18.759138"``. Keep None if unparseable —
    never fabricate a score."""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


@dataclass
class MtragContext:
    """One human-retrieved passage for an agent turn, as mtRAG released it."""

    doc_id: str
    rank: int                       # 1-based order within the turn's context list
    score: Optional[float]          # mtRAG's native ELSER score (None if absent)
    score_type: str = "elser"       # sparse-retrieval score; NOT comparable to cosine
    text: str = ""
    title: str = ""

    def to_dict(self) -> Dict:
        return {"doc_id": self.doc_id, "rank": self.rank, "score": self.score,
                "score_type": self.score_type, "text": self.text, "title": self.title}


@dataclass
class MtragTurn:
    """One (human user question -> human agent answer) turn with its evidence."""

    turn_id: int
    question: str                   # the human user utterance
    answer: str                     # the human/curated agent answer (gold)
    question_type: str = ""         # mtRAG enrichment, e.g. Factoid / Comparison
    answerability: str = ""         # mtRAG enrichment (answerable / unanswerable / ...)
    multi_turn: str = ""            # mtRAG enrichment (N/A / Follow-up / ...)
    contexts: List[MtragContext] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"turn_id": self.turn_id, "question": self.question,
                "answer": self.answer, "question_type": self.question_type,
                "answerability": self.answerability, "multi_turn": self.multi_turn,
                "contexts": [c.to_dict() for c in self.contexts]}


@dataclass
class MtragConversation:
    """A full human multi-turn conversation, order preserved."""

    conversation_id: str
    domain: str
    turns: List[MtragTurn] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"conversation_id": self.conversation_id, "domain": self.domain,
                "turns": [t.to_dict() for t in self.turns]}


def load_mtrag_conversations(path: Optional[str] = None,
                             limit: Optional[int] = None) -> List[MtragConversation]:
    """Load mtRAG as full conversations, preserving order/history/evidence.

    Args:
        path: conversations.json (defaults to the bundled ``mtrag_validation`` copy).
        limit: cap the number of conversations returned (None = all).

    Returns:
        A list of :class:`MtragConversation`. Each agent turn is paired with the user
        turn immediately preceding it; the agent's ``original_text`` is preferred when
        the shown text is a canned abstention (matching ``_load_mtrag``'s behaviour) so
        the gold reflects the real attempted answer.
    """
    p = path or DEFAULT_PATH
    with open(p, encoding="utf-8") as fh:
        convos = json.load(fh)

    out: List[MtragConversation] = []
    for ci, conv in enumerate(convos):
        msgs = conv.get("messages") or []
        domain = str(conv.get("domain", ""))
        mc = MtragConversation(conversation_id=f"mtrag-{domain}-{ci:03d}", domain=domain)
        turn_id = 0
        for mi, m in enumerate(msgs):
            if m.get("speaker") != "agent":
                continue
            prev = msgs[mi - 1] if mi > 0 else None
            if not prev or prev.get("speaker") != "user":
                continue
            question = (prev.get("text") or "").strip()
            answer = (m.get("text") or "").strip()
            low = answer.lower()
            if ("don't have the answer" in low or "do not have the answer" in low
                    or "i'm sorry" in low):
                answer = (m.get("original_text") or answer).strip()

            enr = _as_dict(prev.get("enrichments"))
            contexts: List[MtragContext] = []
            for rank, c in enumerate(m.get("contexts") or []):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                contexts.append(MtragContext(
                    doc_id=str(c.get("document_id", "")),
                    rank=rank + 1,
                    score=_to_float(c.get("score")),
                    text=text,
                    title=str(c.get("title", "")),
                ))
            mc.turns.append(MtragTurn(
                turn_id=turn_id, question=question, answer=answer,
                question_type=str(_first(enr.get("Question Type"))),
                answerability=str(_first(enr.get("Answerability"))),
                multi_turn=str(_first(enr.get("Multi-Turn"))),
                contexts=contexts,
            ))
            turn_id += 1
        if mc.turns:
            out.append(mc)
        if limit and len(out) >= limit:
            break
    return out
