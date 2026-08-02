"""
Module 3 — conversation generator.

Builds coherent 3–8 turn conversations from dataset seeds. Each turn is a typed
query (Module 4) grounded in GraphRAG evidence (Module 2) with a grounded gold
answer (Module 5). Turn types are mixed using a small policy that keeps the
dialogue natural:

  * always open with a factual ``Follow-Up``-able seed question,
  * make sure a comparison turn only appears once ≥2 entities are on the table,
  * place ``Correction`` / ``Topic Shift`` mid-conversation,
  * sprinkle one ``Unanswerable`` to probe abstention.

Output::

    {
        "conversation_id": str,
        "turns": [ { question, query_type, turn_id, gold, evidence, ... }, ... ],
        "query_type_sequence": [str, ...],
    }
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, get_logger
from ..datasets.loader import Sample
from ..graph.graph_builder import KnowledgeGraph
from ..graph.retriever import GraphRetriever
from ..llm import LLM
from .query_generator import QUERY_TYPES, QueryGenerator, QueryTurn
from .gold_answer_generator import GoldAnswerGenerator

logger = get_logger("generation.conversation")

#: A reasonable mix that exercises every capability. Index 0 is the seed turn.
_DEFAULT_POLICY = [
    "Follow-Up",
    "Clarification",
    "Comparative",
    "Multi-Hop",
    "Correction",
    "Topic Shift",
    "Ambiguous Reference",
    "Unanswerable",
]


@dataclass
class Conversation:
    """A generated multi-turn conversation with grounded golds."""

    conversation_id: str
    seed_question: str
    turns: List[Dict] = field(default_factory=list)
    query_type_sequence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "seed_question": self.seed_question,
            "query_type_sequence": self.query_type_sequence,
            "turns": self.turns,
        }

    def history_upto(self, turn_id: int) -> List[Dict]:
        """User/assistant message history before ``turn_id`` (for retrieval/gen)."""
        hist: List[Dict] = []
        for t in self.turns:
            if t["turn_id"] >= turn_id:
                break
            hist.append({"role": "user", "content": t["question"]})
            hist.append({"role": "assistant", "content": t["gold"]["gold_answer"]})
        return hist


class ConversationGenerator:
    """Generates coherent typed conversations grounded in a knowledge graph."""

    def __init__(self, kg: KnowledgeGraph, config: Optional[Config] = None,
                 llm: Optional[LLM] = None,
                 retriever: Optional[GraphRetriever] = None,
                 query_gen: Optional[QueryGenerator] = None,
                 gold_gen: Optional[GoldAnswerGenerator] = None):
        self.kg = kg
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.retriever = retriever or GraphRetriever(kg, config=self.config, llm=self.llm)
        self.query_gen = query_gen or QueryGenerator(config=self.config, llm=self.llm)
        self.gold_gen = gold_gen or GoldAnswerGenerator(config=self.config, llm=self.llm)
        self._rng = random.Random(self.config.seed)

    # -- public ------------------------------------------------------------- #
    def generate(self, seed: Sample, conversation_id: str,
                 num_turns: Optional[int] = None) -> Conversation:
        """Generate one conversation seeded from a dataset sample."""
        n = num_turns or self._rng.randint(self.config.min_turns, self.config.max_turns)
        n = max(self.config.min_turns, min(self.config.max_turns, n))
        policy = self._plan_sequence(n)
        logger.info("conversation %s: %d turns %s", conversation_id, n, policy)

        conv = Conversation(conversation_id=conversation_id,
                            seed_question=seed.question)

        for turn_id, qtype in enumerate(policy):
            history = conv.history_upto(turn_id)
            if turn_id == 0:
                # the opening turn is the dataset seed question itself
                turn = self._seed_turn(seed, history)
                evidence = self.retriever.retrieve(seed.question, k=8,
                                                   conversation_history=history)
            else:
                # retrieve around the running conversation, then ask a typed question
                focus = history[-2]["content"] if history else seed.question
                evidence = self.retriever.retrieve(focus, k=8,
                                                   conversation_history=history)
                turn = self.query_gen.generate(qtype, turn_id, evidence, history)

            gold = self.gold_gen.generate(turn, evidence, history)
            conv.turns.append({
                **turn.to_dict(),
                "gold": gold.to_dict(),
                "evidence": {
                    "chunks": evidence.chunks[:5],
                    "node_ids": [nd["id"] for nd in evidence.nodes][:10],
                    "edges": evidence.edges[:10],
                },
            })
            conv.query_type_sequence.append(qtype)
        return conv

    def generate_many(self, seeds: List[Sample]) -> List[Conversation]:
        out = []
        for i, seed in enumerate(seeds[: self.config.num_conversations]):
            out.append(self.generate(seed, conversation_id=f"conv-{i:03d}"))
        return out

    # -- planning ----------------------------------------------------------- #
    def _plan_sequence(self, n: int) -> List[str]:
        """Pick a coherent ordered list of ``n`` query types (turn 0 = seed)."""
        seq = ["Follow-Up"]  # turn 0 placeholder; seed turn overrides text
        pool = [t for t in _DEFAULT_POLICY if t != "Follow-Up"]
        self._rng.shuffle(pool)
        # Comparative / Ambiguous need 2+ entities -> push later
        late = {"Comparative", "Ambiguous Reference", "Multi-Hop"}
        early = [t for t in pool if t not in late]
        later = [t for t in pool if t in late]
        ordered = early + later
        for t in ordered:
            if len(seq) >= n:
                break
            seq.append(t)
        # if still short (n large), cycle through remaining types
        while len(seq) < n:
            seq.append(self._rng.choice(list(QUERY_TYPES)))
        return seq[:n]

    def _seed_turn(self, seed: Sample, history: List[Dict]) -> QueryTurn:
        spec = QUERY_TYPES["Follow-Up"]
        return QueryTurn(
            question=seed.question,
            query_type="Follow-Up",
            turn_id=0,
            difficulty=1,
            capability="Factual Retrieval (seed)",
            expected_failure="Missing Retrieval",
            conversation_history=list(history),
            target_entity=None,
            meta={"seed_id": seed.id},
        )
