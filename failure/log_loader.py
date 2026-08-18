"""Pull failed turns out of a benchmark conversation log (`*_full.json`).

Each turn in a log carries everything the classifier needs: the retrieved context
the RAG saw, the (reconstructed) question, the RAG's answer, and the gold answer.
This module extracts the turns that FAILED and packages them, together with the
prior-turn dialogue history, as :class:`FailedTurn` records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Outcomes that count as a failure by default. "abstained" is a failure only when
#: the turn was actually answerable (handled via ``include_abstained``).
FAILURE_OUTCOMES = ("wrong", "hallucinated")


@dataclass
class FailedTurn:
    """One failed dialogue turn, with the context needed to attribute the failure."""

    conversation_id: str
    turn_id: int
    query_type: str
    question: str
    gold_answer: str
    rag_answer: str
    outcome: str
    is_unanswerable: bool
    evidence: str                                  # gold-supporting passage
    retrieved_context: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)  # prior turns' Q/A
    source_file: str = ""

    def history_text(self) -> str:
        if not self.history:
            return "(no prior turns)"
        return "\n".join(f"  turn {h['turn_id']}: Q: {h['question']}\n"
                         f"            A: {h['answer']}" for h in self.history)

    def context_text(self, max_docs: int = 8, max_chars: int = 500) -> str:
        docs = self.retrieved_context[:max_docs]
        if not docs:
            return "(no retrieved context recorded)"
        return "\n".join(f"  [{i}] {d[:max_chars]}" for i, d in enumerate(docs))


def load_failed_turns(path: str, include_abstained: bool = False) -> List[FailedTurn]:
    """Read a `*_full.json` log and return every failed turn.

    Args:
        path: path to a benchmark `*_full.json` conversation log.
        include_abstained: also treat an *answerable* turn the RAG abstained on as a
            failure (a coverage/knowledge-boundary miss).
    """
    with open(path, "r", encoding="utf-8") as fr:
        data = json.load(fr)

    fail_outcomes = set(FAILURE_OUTCOMES)
    out: List[FailedTurn] = []
    for conv in data.get("conversations", []):
        conv_id = str(conv.get("conversation_id", ""))
        turns = conv.get("turns", [])
        history: List[Dict[str, str]] = []
        for t in turns:
            outcome = t.get("outcome") or ""
            answerable = not t.get("is_unanswerable", False)
            failed = outcome in fail_outcomes or (
                include_abstained and outcome == "abstained" and answerable)
            if failed:
                out.append(FailedTurn(
                    conversation_id=conv_id,
                    turn_id=int(t.get("turn_id", -1)),
                    query_type=t.get("query_type", ""),
                    question=t.get("question", ""),
                    gold_answer=t.get("gold", ""),
                    rag_answer=t.get("rag_answer", ""),
                    outcome=outcome,
                    is_unanswerable=bool(t.get("is_unanswerable", False)),
                    evidence=t.get("evidence", "") or t.get("question_evidence", ""),
                    retrieved_context=list(t.get("rag_retrieved_context", []) or []),
                    history=list(history),
                    source_file=path,
                ))
            # grow the running history AFTER handling this turn
            history.append({"turn_id": t.get("turn_id", -1),
                            "question": t.get("question", ""),
                            "answer": t.get("rag_answer", "")})
    return out
