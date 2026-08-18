"""
Module 5 — grounded gold-answer generator.

**Critical design point (per the spec):** the gold answer is *not* taken from the
original dataset answer. It is synthesised from the graph evidence retrieved for
the specific (possibly multi-turn, possibly type-twisted) question, so the answer
key always traces back to real corpus text. This is what keeps the benchmark's
answer key from being hallucinated — the failure mode RAG-DIVE suffers from.

Output::

    {
        "gold_answer": str,
        "reasoning_chain": [str, ...],     # human-readable hop-by-hop derivation
        "supporting_nodes": [str, ...],    # RETRIEVED entity ids (candidate support)
        "supporting_edges": [ {source,target,relation}, ... ],  # retrieved edges
        "supporting_chunks": [str, ...],   # retrieved chunks the answer was composed from
    }

The gold answer is ALWAYS composed from ``truth_history`` + retrieved evidence, never
from the RAG-under-test's answer — so the answer key stays identical across the
static and dynamic conditions of the experiment.

For ``Unanswerable`` turns the gold is a canonical abstention string, so the
classifier can reward the RAG for saying "I don't know".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, get_logger
from ..llm import LLM
from ..graph.retriever import RetrievalResult
from .query_generator import QueryTurn

logger = get_logger("generation.gold")

ABSTENTION = "Not answerable from the available knowledge; the system should abstain."


@dataclass
class GoldAnswer:
    """A grounded gold answer with its evidence trail."""

    gold_answer: str
    reasoning_chain: List[str] = field(default_factory=list)
    supporting_nodes: List[str] = field(default_factory=list)
    supporting_edges: List[Dict] = field(default_factory=list)
    supporting_chunks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "gold_answer": self.gold_answer,
            "reasoning_chain": self.reasoning_chain,
            # NOTE: supporting_* are the RETRIEVED evidence trail the answer was
            # composed against (candidate support), not per-claim verified support —
            # they are the top-ranked nodes/edges/chunks handed to the composer.
            "supporting_nodes": self.supporting_nodes,
            "supporting_edges": self.supporting_edges,
            "supporting_chunks": self.supporting_chunks,
        }


class GoldAnswerGenerator:
    """Synthesises gold answers from retrieved graph evidence + history."""

    _SYS = (
        "You are building the ANSWER KEY for a benchmark. Using ONLY the EVIDENCE "
        "and the CONVERSATION, write the correct, concise gold answer to the "
        "QUESTION, plus the step-by-step reasoning chain that derives it from the "
        "evidence. If the evidence does not contain the answer, set "
        '"answerable" to false. Be faithful: never add facts not in the evidence. '
        'Respond as JSON: {"answerable": true/false, "gold_answer": "...", '
        '"reasoning_chain": ["step 1", "step 2", ...]}'
    )

    #: Strict "composer": refuses to assert beyond the evidence. Targets the
    #: dominant gold-support leak — the generator inventing the missing half of a
    #: multi-hop / comparison / sub-topic question instead of abstaining.
    _SYS_STRICT = (
        "You are building the ANSWER KEY for a benchmark. Answer the QUESTION using "
        "ONLY facts EXPLICITLY stated in the EVIDENCE. Do NOT infer, generalize, or "
        "use any outside knowledge.\n"
        "Rules:\n"
        "1. Every claim in gold_answer must be either directly stated in the EVIDENCE, "
        "or logically derivable by CHAINING facts that are each directly stated in the "
        "EVIDENCE (legitimate multi-hop reasoning). You must be able to point to the "
        "exact evidence text for every link in the chain. Never introduce outside "
        "knowledge and never fill a missing link to make the answer sound whole.\n"
        "2. If the question has multiple parts (a comparison, a multi-step chain, an "
        "'and'), EVERY part must be supported by the EVIDENCE. If even one part is "
        "missing, set \"answerable\" to false.\n"
        "3. Prefer a short, literal answer over a complete-sounding one. Never fill "
        "gaps to make the answer sound whole.\n"
        "4. If the EVIDENCE does not fully support an answer, set \"answerable\" to "
        "false rather than answering partially.\n"
        'Respond as JSON: {"answerable": true/false, "gold_answer": "...", '
        '"reasoning_chain": ["evidence-anchored step", ...]}'
    )

    def __init__(self, config: Optional[Config] = None, llm: Optional[LLM] = None,
                 strict: bool = False):
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.strict = strict  # use the strict composer prompt (refuse to over-assert)

    def generate(self, turn: QueryTurn, evidence: RetrievalResult,
                 truth_history: List[Dict]) -> GoldAnswer:
        """Generate the gold answer for a single typed turn.

        THESIS INVARIANT: the conversation context passed here is ``truth_history``
        (user question + GOLD answer), NEVER ``rag_history``. The gold answer must
        never be conditioned on the RAG-under-test's (possibly wrong) reply — that is
        what keeps the answer key clean and identical across the static/dynamic
        conditions. The parameter is named ``truth_history`` so this rule is visible
        at the call site and cannot be satisfied by accidentally passing rag_history.
        """
        # These are the RETRIEVED evidence (top-ranked), i.e. the candidate support
        # the composer was given — NOT a per-claim verified support set.
        retrieved_nodes = [n["id"] for n in evidence.nodes]
        retrieved_edges = evidence.edges

        if turn.is_unanswerable:
            return GoldAnswer(
                gold_answer=ABSTENTION,
                reasoning_chain=["The question asks for information absent from the "
                                 "corpus; the correct behaviour is to abstain."],
                supporting_nodes=retrieved_nodes[:5],
                supporting_edges=retrieved_edges[:5],
                supporting_chunks=evidence.chunks[:3],
            )

        gold = self._llm_gold(turn, evidence, truth_history)
        if gold is None:
            gold = self._fallback_gold(turn, evidence)
        gold.supporting_nodes = retrieved_nodes[:8]
        gold.supporting_edges = retrieved_edges[:8]
        gold.supporting_chunks = evidence.chunks[:4]
        return gold

    # -- backends ----------------------------------------------------------- #
    def _llm_gold(self, turn: QueryTurn, evidence: RetrievalResult,
                  truth_history: List[Dict]) -> Optional[GoldAnswer]:
        if not self.llm.available:
            return None
        hist_text = "\n".join(f"{t['role']}: {t['content']}"
                              for t in truth_history[-6:]) or "(start)"
        user = (
            f"QUESTION: {turn.question}\n"
            f"QUERY TYPE: {turn.query_type}\n"
            f"CONVERSATION:\n{hist_text}\n\n"
            f"EVIDENCE:\n{evidence.evidence_text(2200)}"
        )
        out = self.llm.chat_json(self._SYS_STRICT if self.strict else self._SYS, user)
        if not out:
            return None
        if not out.get("answerable", True):
            return GoldAnswer(
                gold_answer=ABSTENTION,
                reasoning_chain=["Evidence does not support an answer."],
            )
        chain = out.get("reasoning_chain") or []
        if isinstance(chain, str):
            chain = [chain]
        return GoldAnswer(
            gold_answer=str(out.get("gold_answer", "")).strip(),
            reasoning_chain=[str(s) for s in chain],
        )

    def _fallback_gold(self, turn: QueryTurn, evidence: RetrievalResult) -> GoldAnswer:
        """No trustworthy gold could be composed (the LLM composer was unavailable or
        returned nothing).

        Rather than fabricate an answer key from the first sentence of the top chunk —
        which may not answer the question at all and would CONTAMINATE failure
        attribution — we ABSTAIN. Callers treat a gold starting with ``ABSTENTION`` as
        ungrounded: the adaptive generator's grounding guard retries and, if it keeps
        failing, marks the turn ``guard_gave_up`` so it is excluded from per-type
        grading. A missing gold must never masquerade as a real one."""
        return GoldAnswer(
            gold_answer=ABSTENTION,
            reasoning_chain=["Gold-answer composition failed (LLM composer unavailable "
                             "or returned no answer); no trustworthy gold could be "
                             "produced, so this turn abstains rather than guess."],
        )
