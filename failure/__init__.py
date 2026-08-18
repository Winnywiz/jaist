"""Failure classification: pull failed turns from a benchmark log and label each
along two axes — RAG failure type (retrieval/generation layer) and conversational
cause. See taxonomy.py for the definitions."""
from .taxonomy import (
    FailureLayer, FailureType, ConversationalCause, CauseDecision,
    FAILURE_TYPES, CONVERSATIONAL_CAUSES,
)
from .log_loader import FailedTurn, load_failed_turns
from .classifier import FailureClassifier, ClassifiedFailure, summarize

__all__ = [
    "FailureLayer", "FailureType", "ConversationalCause", "CauseDecision",
    "FAILURE_TYPES", "CONVERSATIONAL_CAUSES",
    "FailedTurn", "load_failed_turns",
    "FailureClassifier", "ClassifiedFailure", "summarize",
]
