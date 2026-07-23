"""
Module 7 — the structured failure taxonomy.

Four categories, each with concrete failure types. The taxonomy is the spine of
the benchmark: the classifier emits these labels, and the report aggregates by
them, so a researcher can see *which capability* of a conversational RAG broke.

    Retrieval      : Missing Retrieval, Wrong Retrieval, Partial Retrieval
    Conversation   : Coreference Failure, History Ignoring, Context Drift,
                     Correction Failure, Topic Shift Failure
    Reasoning      : Multi-Hop Failure, Comparative Failure, Temporal Failure
    Generation     : Hallucination, Overconfident Unknown, Contradiction,
                     Incomplete Answer
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List


class FailureCategory(str, Enum):
    RETRIEVAL = "Retrieval"
    CONVERSATION = "Conversation"
    REASONING = "Reasoning"
    GENERATION = "Generation"


class FailureType(str, Enum):
    # Retrieval
    MISSING_RETRIEVAL = "Missing Retrieval"
    WRONG_RETRIEVAL = "Wrong Retrieval"
    PARTIAL_RETRIEVAL = "Partial Retrieval"
    # Conversation
    COREFERENCE_FAILURE = "Coreference Failure"
    HISTORY_IGNORING = "History Ignoring"
    CONTEXT_DRIFT = "Context Drift"
    CORRECTION_FAILURE = "Correction Failure"
    TOPIC_SHIFT_FAILURE = "Topic Shift Failure"
    # Reasoning
    MULTI_HOP_FAILURE = "Multi-Hop Failure"
    COMPARATIVE_FAILURE = "Comparative Failure"
    TEMPORAL_FAILURE = "Temporal Failure"
    # Generation
    HALLUCINATION = "Hallucination"
    OVERCONFIDENT_UNKNOWN = "Overconfident Unknown"
    CONTRADICTION = "Contradiction"
    INCOMPLETE_ANSWER = "Incomplete Answer"


#: Category -> ordered list of failure types it contains.
FAILURE_TAXONOMY: Dict[FailureCategory, List[FailureType]] = {
    FailureCategory.RETRIEVAL: [
        FailureType.MISSING_RETRIEVAL,
        FailureType.WRONG_RETRIEVAL,
        FailureType.PARTIAL_RETRIEVAL,
    ],
    FailureCategory.CONVERSATION: [
        FailureType.COREFERENCE_FAILURE,
        FailureType.HISTORY_IGNORING,
        FailureType.CONTEXT_DRIFT,
        FailureType.CORRECTION_FAILURE,
        FailureType.TOPIC_SHIFT_FAILURE,
    ],
    FailureCategory.REASONING: [
        FailureType.MULTI_HOP_FAILURE,
        FailureType.COMPARATIVE_FAILURE,
        FailureType.TEMPORAL_FAILURE,
    ],
    FailureCategory.GENERATION: [
        FailureType.HALLUCINATION,
        FailureType.OVERCONFIDENT_UNKNOWN,
        FailureType.CONTRADICTION,
        FailureType.INCOMPLETE_ANSWER,
    ],
}

#: Flat list of every failure-type *value* (string), e.g. for the LLM judge prompt.
ALL_FAILURE_TYPES: List[str] = [t.value for types in FAILURE_TAXONOMY.values()
                                for t in types]

_TYPE_TO_CATEGORY: Dict[str, FailureCategory] = {
    t.value: cat for cat, types in FAILURE_TAXONOMY.items() for t in types
}


def category_of(failure_type: str) -> str:
    """Return the category name for a failure-type string (``"Unknown"`` if absent)."""
    cat = _TYPE_TO_CATEGORY.get(failure_type)
    return cat.value if cat else "Unknown"


def types_in_category(category: FailureCategory) -> List[str]:
    return [t.value for t in FAILURE_TAXONOMY[category]]
