"""LLM classifier that labels a failed turn along two independent axes.

Axis 1 — RAG FAILURE TYPE: WHAT broke in the pipeline (one of six types, each in a
          retrieval or generation layer). See taxonomy.FAILURE_TYPES.
Axis 2 — CONVERSATIONAL CAUSE: the upstream multi-turn cause that led to the failure,
          if any — or ``not_applicable`` (a pure single-turn RAG issue) / ``uncertain``.

Keeping the two axes separate is deliberate: a conversational failure (e.g. an
unresolved reference) is a *cause* that manifests downstream as a retrieval- or
generation-layer *symptom*. This mirrors the taxonomy's own structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .log_loader import FailedTurn
from .taxonomy import (
    CAUSE_DECISION_KEYS,
    CONVERSATIONAL_CAUSE_KEYS,
    CONVERSATIONAL_CAUSES,
    CONVERSATIONAL_CAUSES_BY_KEY,
    FAILURE_TYPE_KEYS,
    FAILURE_TYPES,
    FAILURE_TYPES_BY_KEY,
    CauseDecision,
)


@dataclass
class ClassifiedFailure:
    conversation_id: str
    turn_id: int
    query_type: str
    outcome: str
    failure_type: str        # one of FAILURE_TYPE_KEYS, or "uncertain"
    layer: str               # retrieval / generation / "" if uncertain
    conversational_cause: str  # a cause key, or not_applicable / uncertain
    rationale: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _taxonomy_prompt() -> str:
    """Render the taxonomy definitions into the classifier's instructions."""
    lines = ["AXIS 1 — RAG FAILURE TYPE (what broke in the pipeline). Choose EXACTLY ONE key:"]
    for d in FAILURE_TYPES:
        lines.append(f"  - {d.key} ({d.name}; layer={d.layer}): {d.description}")
    lines.append("")
    lines.append("AXIS 2 — CONVERSATIONAL CAUSE (the multi-turn cause that led to the failure). "
                 "Choose EXACTLY ONE key:")
    for d in CONVERSATIONAL_CAUSES:
        lines.append(f"  - {d.key} ({d.name}): {d.description} EXCLUSION: {d.exclusion}")
    lines.append(f"  - {CauseDecision.NOT_APPLICABLE.value}: the failure is a pure single-turn "
                 "RAG-pipeline issue with no conversational trigger.")
    lines.append(f"  - {CauseDecision.UNCERTAIN.value}: insufficient evidence to name a "
                 "conversational cause.")
    return "\n".join(lines)


_SYS = (
    "You are an expert failure analyst for a MULTI-TURN Retrieval-Augmented Generation "
    "(RAG) system. You are given ONE failed dialogue turn — the retrieved context the RAG "
    "saw, the user question, the RAG's answer, and the gold answer. Classify the failure "
    "along the two independent axes below.\n\n"
    + _taxonomy_prompt()
    + "\n\nWhen a RETRIEVAL CHECK line is provided, TRUST it for the LAYER: gold ABSENT "
    "from the retrieved context => a retrieval-layer type; gold PRESENT => a generation-layer "
    "type. Do not label a failure 'response_coverage'/'grounding' when the gold was absent "
    "from what the RAG retrieved."
    + "\n\nReply with STRICT JSON only:\n"
    '{"failure_type":"<axis-1 key>","conversational_cause":"<axis-2 key>",'
    '"rationale":"<one short sentence>"}'
)

_STOP = frozenset("the a an of for to in on and or is are was were with by as at from that "
                  "this its their it they he she".split())


def _ctoks(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower())
            if w not in _STOP and len(w) > 1}


def _gold_in_context(gold: str, context: List[str]):
    """True/False if the gold/evidence text is (token-)present in the retrieved context;
    None when there's nothing to check. Coarse but deterministic coverage signal."""
    g = _ctoks(gold)
    if not g or not context:
        return None
    return any(len(g & _ctoks(c)) / len(g) >= 0.5 for c in context)


def _valid_cause(value: str) -> str:
    v = (value or "").strip().lower()
    if v in CONVERSATIONAL_CAUSE_KEYS or v in CAUSE_DECISION_KEYS:
        return v
    return CauseDecision.UNCERTAIN.value


def _valid_type(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in FAILURE_TYPE_KEYS else "uncertain"


class FailureClassifier:
    """Classifies failed turns using the LLM when available, else a lexical fallback."""

    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, turn: FailedTurn) -> ClassifiedFailure:
        if self.llm is not None and getattr(self.llm, "available", False):
            out = self.llm.chat_json(_SYS, self._render(turn)) or {}
            ftype = _valid_type(out.get("failure_type", ""))
            cause = _valid_cause(out.get("conversational_cause", ""))
            rationale = str(out.get("rationale", "")).strip()
        else:
            ftype, cause, rationale = self._fallback(turn)
        layer = FAILURE_TYPES_BY_KEY[ftype].layer.value if ftype in FAILURE_TYPES_BY_KEY else ""
        return ClassifiedFailure(
            conversation_id=turn.conversation_id, turn_id=turn.turn_id,
            query_type=turn.query_type, outcome=turn.outcome,
            failure_type=ftype, layer=layer,
            conversational_cause=cause, rationale=rationale)

    def classify_all(self, turns: List[FailedTurn]) -> List[ClassifiedFailure]:
        return [self.classify(t) for t in turns]

    # ----------------------------------------------------------------------- #
    def _render(self, turn: FailedTurn) -> str:
        return (
            f"DIALOGUE HISTORY (prior turns):\n{turn.history_text()}\n\n"
            f"QUERY TYPE: {turn.query_type}\n"
            f"USER QUESTION (this turn): {turn.question}\n"
            f"GOLD ANSWER: {turn.gold_answer}\n"
            f"RAG ANSWER: {turn.rag_answer}\n"
            f"OUTCOME: {turn.outcome}  |  IS_UNANSWERABLE: {turn.is_unanswerable}\n\n"
            f"RETRIEVED CONTEXT the RAG saw:\n{turn.context_text()}\n\n"
            f"GOLD EVIDENCE (reference):\n{(turn.evidence or '')[:1200]}"
            + self._retrieval_check(turn)
        )

    @staticmethod
    def _retrieval_check(turn: FailedTurn) -> str:
        """Retrieval-aware signal: was the gold evidence actually PRESENT in what the RAG
        retrieved? This is the key cue for the LAYER, and the classifier previously lacked
        it (so it mislabelled retrieval failures as generation). Meaningless for Unanswerable
        turns (gold is an abstention), so skipped there."""
        if turn.is_unanswerable:
            return ""
        present = _gold_in_context(turn.gold_answer, turn.retrieved_context) or \
            _gold_in_context(turn.evidence, turn.retrieved_context)
        if present is None:
            return ""
        if present:
            return ("\n\nRETRIEVAL CHECK: the gold evidence IS PRESENT in the retrieved "
                    "context — the RAG HAD what it needed, so this is most likely a "
                    "GENERATION-layer failure (grounding / response_coverage), NOT retrieval.")
        return ("\n\nRETRIEVAL CHECK: the gold evidence is ABSENT from the retrieved context "
                "— the RAG could not have answered from what it retrieved, so this is most "
                "likely a RETRIEVAL-layer failure (retrieval / context_selection / chunking).")

    def _fallback(self, turn: FailedTurn):
        """Offline heuristic (no LLM): coarse type from the outcome, cause uncertain."""
        if turn.outcome == "hallucinated":
            ftype = "grounding"
        elif turn.is_unanswerable:
            ftype = "knowledge_boundary"
        else:
            ftype = "retrieval"
        return ftype, CauseDecision.UNCERTAIN.value, "offline heuristic (no LLM available)"


def summarize(results: List[ClassifiedFailure]) -> Dict:
    """Aggregate counts by failure type, layer, and conversational cause."""
    from collections import Counter
    ft = Counter(r.failure_type for r in results)
    ly = Counter(r.layer for r in results if r.layer)
    cc = Counter(r.conversational_cause for r in results)
    by_qtype = Counter(r.query_type for r in results)
    return {
        "n": len(results),
        "by_failure_type": dict(ft.most_common()),
        "by_layer": dict(ly.most_common()),
        "by_conversational_cause": dict(cc.most_common()),
        "by_query_type": dict(by_qtype.most_common()),
    }
