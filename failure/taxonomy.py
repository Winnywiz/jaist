"""RAG failure types and possible conversational causes."""
from __future__ import annotations

import dataclasses
import enum


class FailureLayer(str, enum.Enum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"

    def __str__(self) -> str:
        return self.value


class FailureType(str, enum.Enum):
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    CHUNKING = "chunking"
    RETRIEVAL = "retrieval"
    CONTEXT_SELECTION = "context_selection"
    GROUNDING = "grounding"
    RESPONSE_COVERAGE = "response_coverage"

    def __str__(self) -> str:
        return self.value


class ConversationalCause(str, enum.Enum):
    UNRESOLVED_REFERENCE = "unresolved_reference"
    UNRESOLVED_ELLIPSIS = "unresolved_ellipsis"
    TOPIC_INTERFERENCE = "topic_interference"
    CONTEXT_INTERFERENCE = "context_interference"

    def __str__(self) -> str:
        return self.value


class CauseDecision(str, enum.Enum):
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class FailureTypeDefinition:
    failure_type: FailureType
    name: str
    description: str
    layer: FailureLayer

    @property
    def key(self) -> str:
        return self.failure_type.value

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True)
class ConversationalCauseDefinition:
    cause: ConversationalCause
    name: str
    description: str
    exclusion: str

    @property
    def key(self) -> str:
        return self.cause.value

    def __str__(self) -> str:
        return self.name


KNOWLEDGE_BOUNDARY = FailureTypeDefinition(
    FailureType.KNOWLEDGE_BOUNDARY,
    "Knowledge Boundary",
    "The available corpus and verified conversation state cannot support the "
    "complete request, but the system answers as though they can.",
    FailureLayer.RETRIEVAL,
)
CHUNKING = FailureTypeDefinition(
    FailureType.CHUNKING,
    "Chunking",
    "Relevant source information exists, but the indexed chunk structure does "
    "not preserve a usable evidence unit.",
    FailureLayer.RETRIEVAL,
)
RETRIEVAL = FailureTypeDefinition(
    FailureType.RETRIEVAL,
    "Retrieval",
    "A usable relevant indexed chunk exists but is absent from the retrieved "
    "candidate set.",
    FailureLayer.RETRIEVAL,
)
CONTEXT_SELECTION = FailureTypeDefinition(
    FailureType.CONTEXT_SELECTION,
    "Context Selection",
    "Relevant evidence is retrieved but absent from the context delivered to "
    "the generator.",
    FailureLayer.RETRIEVAL,
)
GROUNDING = FailureTypeDefinition(
    FailureType.GROUNDING,
    "Grounding",
    "The response contradicts, fabricates, or overextends beyond the selected "
    "evidence or verified conversation state.",
    FailureLayer.GENERATION,
)
RESPONSE_COVERAGE = FailureTypeDefinition(
    FailureType.RESPONSE_COVERAGE,
    "Response Coverage",
    "The supported response omits an answerable part of the reconstructed user "
    "request.",
    FailureLayer.GENERATION,
)

FAILURE_TYPES = (
    KNOWLEDGE_BOUNDARY,
    CHUNKING,
    RETRIEVAL,
    CONTEXT_SELECTION,
    GROUNDING,
    RESPONSE_COVERAGE,
)
FAILURE_TYPES_BY_KEY = {definition.key: definition for definition in FAILURE_TYPES}
FAILURE_TYPE_KEYS = tuple(definition.key for definition in FAILURE_TYPES)


UNRESOLVED_REFERENCE = ConversationalCauseDefinition(
    ConversationalCause.UNRESOLVED_REFERENCE,
    "Unresolved Reference",
    "The intended referent of a referring expression is not correctly "
    "identified.",
    "Do not assign when no recoverable referent is supplied and the system "
    "appropriately asks for clarification; omitted predicates, arguments, or "
    "scope belong to Unresolved Ellipsis.",
)
UNRESOLVED_ELLIPSIS = ConversationalCauseDefinition(
    ConversationalCause.UNRESOLVED_ELLIPSIS,
    "Unresolved Ellipsis",
    "Omitted information required to interpret the request is not correctly "
    "recovered.",
    "Do not assign solely because a pronoun has the wrong referent; use "
    "Unresolved Reference.",
)
TOPIC_INTERFERENCE = ConversationalCauseDefinition(
    ConversationalCause.TOPIC_INTERFERENCE,
    "Topic Interference",
    "An irrelevant competing topic influences interpretation of the current "
    "request.",
    "Requires evidence of topic competition or a topic shift; general noisy "
    "history belongs to Context Interference.",
)
CONTEXT_INTERFERENCE = ConversationalCauseDefinition(
    ConversationalCause.CONTEXT_INTERFERENCE,
    "Context Interference",
    "Relevant information is obscured by irrelevant or competing supplied "
    "context.",
    "Do not infer hidden memory loss; the interference must be observable in "
    "the supplied query or dialogue history.",
)

CONVERSATIONAL_CAUSES = (
    UNRESOLVED_REFERENCE,
    UNRESOLVED_ELLIPSIS,
    TOPIC_INTERFERENCE,
    CONTEXT_INTERFERENCE,
)
CONVERSATIONAL_CAUSES_BY_KEY = {
    definition.key: definition for definition in CONVERSATIONAL_CAUSES
}
CONVERSATIONAL_CAUSE_KEYS = tuple(
    definition.key for definition in CONVERSATIONAL_CAUSES
)
CAUSE_DECISION_KEYS = tuple(decision.value for decision in CauseDecision)
