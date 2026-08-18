"""
Sub-question decomposition generator — an ADAPTATION of Xie et al. (NAACL 2025).

**Scope note (be precise about what is and isn't from the paper).** Xie et al., "Do RAG
Systems Cover What Matters? Evaluating and Optimizing Responses with Sub-Question
Coverage" (Kaige Xie et al., NAACL 2025), is a sub-question *coverage-evaluation* method,
NOT a multi-turn conversation generator. The paper: decompose the ORIGINAL question into
sub-questions, classify each as **core / background / follow-up**, run the RAG once on the
original question, then for every sub-question judge (A) is it answered by the RAG answer?
and (B) is it covered by the retrieved chunks? — the answered×retrieved 2×2 that IS the
paper's retrieval-vs-generation attribution.

  * The FAITHFUL paper method lives in ``compare/xie_coverage.py`` (``XieCoverage``):
    decomposition + classification + the coverage 2×2, quoting the paper's Tables 5-6.
  * THIS module is a THESIS ADAPTATION: it reuses only Xie's *decomposition* step, then
    replays the sub-questions as the turns of a conversation so the SAME failure classifier
    can consume it alongside the Static/Dynamic methods. Replaying sub-questions as
    conversation turns is NOT part of the Xie paper — do not describe it as "faithful Xie".

Why it sits on the STATIC side of the thesis IV: the decomposition is computed ONCE from
the seed and NEVER conditions on any RAG answer (every sub-question is planned up front) —
the opposite of the proposed DYNAMIC method, which reacts to the RAG's real answer.

Also note Xie's three roles (core/background/follow-up) describe a sub-question's function
relative to the seed; they are NOT the benchmark's 8 query types (Correction/Multi-Hop/…),
which describe generated interaction behaviour — the two taxonomies are different axes and
are never mapped onto each other.

It emits the same ``AdaptiveConversation`` / ``AdaptiveTurn`` schema as every other method,
so the identical failure classifier consumes it unchanged. Both document sets are logged
with real doc_id/rank/score: ``question_generation_documents`` (evidence the sub-question's
gold was authored from) and ``rag_retrieved_documents`` (what the RAG retrieved to answer).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import Config, get_logger
from ..graph.graph_builder import KnowledgeGraph
from ..graph.retriever import GraphRetriever
from ..interfaces.rag_interface import RAGInterface
from ..llm import LLM
from .dynamic_generator import (AdaptiveConversation, AdaptiveTurn, OUTCOMES,
                                 _GRADE_SYS, LATEX_LABEL)
from .gold_answer_generator import GoldAnswerGenerator, ABSTENTION
from .query_generator import QueryTurn

logger = get_logger("generation.subquestion")

#: Paper priority: core first (the paper shows core sub-questions drive retrieval and
#: generation gains), then background, then the optional follow-up ones.
_TYPE_ORDER = {"core": 0, "background": 1, "follow-up": 2}


class XieSubQuestionGenerator:
    """Decompose the seed into typed sub-questions and replay them as a conversation."""

    name = "xie_subquestion"

    #: Faithful to the paper's three sub-question types (core / background / follow-up).
    _DECOMP_SYS = (
        "You decompose a complex QUESTION into the sub-questions a complete answer must "
        "address, following the sub-question-coverage framework. Produce up to 10 "
        "sub-questions and classify EACH into exactly one type:\n"
        "  core       - central to answering the main question (essential).\n"
        "  background - necessary context or supplementary information.\n"
        "  follow-up  - an optional deeper/related aspect, not required for a satisfactory "
        "answer.\n"
        "Each sub-question must be a standalone, answerable question. "
        'Respond as JSON: {"subquestions": [{"q": "...", "type": "core|background|follow-up"}]}'
    )

    def __init__(self, kg: KnowledgeGraph, target_rag: RAGInterface, judge: LLM,
                 config: Optional[Config] = None, gen_llm: Optional[LLM] = None,
                 retriever: Optional[GraphRetriever] = None, strict_gold: bool = False,
                 ground_seed: bool = True):
        self.kg = kg
        self.target_rag = target_rag
        self.judge = judge
        self.config = config or Config.load()
        self.gen_llm = gen_llm or LLM(config=self.config)
        self.retriever = retriever or GraphRetriever(kg, config=self.config, llm=self.gen_llm)
        self.gold_gen = GoldAnswerGenerator(config=self.config, llm=self.gen_llm,
                                            strict=strict_gold)
        # ground_seed=False: the orchestrator already pre-grounded ONE shared seed and
        # passes it verbatim (so Dynamic and Xie start from the IDENTICAL question).
        self.ground_seed = ground_seed

    # -- the paper's decomposition (static: from the seed question only) -------- #
    def _decompose(self, question: str) -> List[Tuple[str, str]]:
        """Return [(subquestion, type)], ordered core -> background -> follow-up."""
        if not self.gen_llm.available:
            return [(question, "core")]
        out = self.gen_llm.chat_json(self._DECOMP_SYS, f"QUESTION: {question}")
        subs = (out or {}).get("subquestions") or []
        parsed = [(str(s.get("q", "")).strip(), str(s.get("type", "core")).strip().lower())
                  for s in subs if s.get("q")]
        parsed = [(q, t if t in _TYPE_ORDER else "core") for q, t in parsed if q]
        # stable sort by paper priority (core first), preserving within-type order
        parsed.sort(key=lambda qt: _TYPE_ORDER.get(qt[1], 0))
        return parsed or [(question, "core")]

    def _grounded_seed(self, seed_q: str) -> str:
        """Ensure the seed question is answerable from the corpus BEFORE decomposing it.

        Mirrors the adaptive generator's seed guard: if the seed's gold comes back as an
        abstention (ungroundable — e.g. a raw dataset question like "Is there an online
        demo?"), rewrite it into a question authored FROM the retrieved corpus evidence,
        so the whole decomposed conversation is about real corpus content (otherwise every
        sub-question inherits the seed's ungroundedness and the RAG just abstains on all)."""
        if not self.gen_llm.available:
            return seed_q
        for attempt in range(3):
            ev = self.retriever.retrieve(seed_q, k=8 if attempt == 0 else 16)
            qturn = QueryTurn(question=seed_q, query_type="Seed", turn_id=0, difficulty=1,
                              capability="", expected_failure="", is_unanswerable=False)
            gold = self.gold_gen.generate(qturn, ev, [])
            if not gold.gold_answer.startswith(ABSTENTION[:20]):
                return seed_q                                # grounded — keep it
            ev_text = LATEX_LABEL.sub(" ", ev.evidence_text(1500))
            out = self.gen_llm.chat_json(
                'Write ONE clear, specific factual question answerable from the TEXT. Do '
                'not mention "text" or "passage"; do not reference tables/sections/figures '
                'by label. Respond JSON: {"question": "..."}', f"TEXT: {ev_text}") or {}
            newq = str(out.get("question") or "").strip()
            if newq:
                seed_q = LATEX_LABEL.sub("", newq).strip()
        return seed_q

    def _grade(self, question: str, gold: str, rag_answer: str,
               is_unanswerable: bool) -> str:
        """Same grader contract as every other method (correct/wrong/hallucinated/abstained)."""
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
        return "correct" if gold.lower()[:30] in low or low[:30] in gold.lower() else "wrong"

    def _answer_turn(self, turn_id: int, question: str, query_type: str,
                     truth_history: List[Dict],
                     rag_history: List[Dict]) -> Tuple[AdaptiveTurn, str]:
        """Retrieve evidence, author the gold, ask the RAG, grade — one turn.
        Returns (turn, gold_text)."""
        evidence = self.retriever.retrieve(question, k=8, conversation_history=truth_history)
        qturn = QueryTurn(question=question, query_type=query_type, turn_id=turn_id,
                          difficulty=2, capability="", expected_failure="",
                          is_unanswerable=False)
        gold = self.gold_gen.generate(qturn, evidence, truth_history)
        gold_text = gold.gold_answer
        # OUR BENCHMARK ADAPTATION (not from the Xie paper): a sub-question the corpus
        # cannot support is treated as unanswerable, so the correct response is to abstain
        # and a confident answer counts as a knowledge-boundary hallucination — the
        # distinction the classifier needs. NOTE: this differs from the Dynamic method,
        # which instead RETRIES to author a groundable follow-up and, failing that, marks
        # the turn guard_gave_up (excluded). So ungroundable turns are handled differently
        # across methods — a comparability caveat, documented in result.md.
        is_unans = gold_text.startswith(ABSTENTION[:20])

        rag_resp = self.target_rag.answer(question, rag_history)
        rag_answer = rag_resp.answer or ""
        outcome = self._grade(question, gold_text, rag_answer, is_unans)

        return AdaptiveTurn(
            turn_id=turn_id, query_type=query_type, question=question,
            gold=gold_text, evidence=" ".join(evidence.chunks[:5])[:1500],
            question_evidence=" ".join(evidence.chunks[:5])[:1500],
            rag_answer=rag_answer,
            rag_retrieved_context=list(rag_resp.retrieved_context or []),
            outcome=outcome, is_unanswerable=is_unans, guard_gave_up=False,
            provenance={
                "content_mode": "xie_static",
                "generator_saw_rag_answer": False,       # decomposition never sees the RAG
                "subquestion_type": query_type.split(":", 1)[-1] if ":" in query_type else "",
                "previous_gold_answer": next((m["content"] for m in reversed(truth_history)
                                              if m.get("role") == "assistant"), ""),
                "previous_rag_answer": next((m["content"] for m in reversed(rag_history)
                                             if m.get("role") == "assistant"), ""),
            },
            question_generation_documents=evidence.scored_documents(),
            rag_retrieved_documents=list(rag_resp.retrieved_docs or []),
            generation_source="xie_static_decomposition",
        ), gold_text

    def generate(self, seed, conversation_id: str, n_turns: int) -> AdaptiveConversation:
        seed_q = (seed.question or "").strip()
        if not seed_q:
            ctx = (seed.context[0][:400] if getattr(seed, "context", None) else "")
            out = self.gen_llm.chat_json(
                'Write ONE clear, specific factual question answerable from the TEXT. '
                'Respond JSON: {"question": "..."}', f"TEXT: {ctx}") or {}
            seed_q = str(out.get("question") or "").strip() or "What is described here?"

        # ground the seed against the corpus before decomposing (see _grounded_seed),
        # unless the orchestrator already pre-grounded a shared seed for all methods.
        if self.ground_seed:
            seed_q = self._grounded_seed(seed_q)

        conv = AdaptiveConversation(conversation_id=conversation_id, seed_question=seed_q)
        truth_history: List[Dict] = []
        rag_history: List[Dict] = []

        # turn 0 — the seed question itself
        seed_turn, seed_gold = self._answer_turn(0, seed_q, "Seed", truth_history, rag_history)
        seed_turn.generation_source = "seed"
        seed_turn.provenance["subquestion_type"] = ""
        conv.turns.append(seed_turn)
        conv.type_sequence.append("Seed")
        conv.outcome_sequence.append(seed_turn.outcome)
        truth_history += [{"role": "user", "content": seed_q},
                          {"role": "assistant", "content": seed_gold}]
        rag_history += [{"role": "user", "content": seed_q},
                        {"role": "assistant", "content": seed_turn.rag_answer}]

        # STATIC decomposition of the seed (computed ONCE, before any RAG answer is used)
        subs = self._decompose(seed_q)
        for i, (subq, stype) in enumerate(subs[: max(0, n_turns - 1)], start=1):
            turn, gold_text = self._answer_turn(i, subq, f"Xie:{stype}",
                                                truth_history, rag_history)
            conv.turns.append(turn)
            conv.type_sequence.append(turn.query_type)
            conv.outcome_sequence.append(turn.outcome)
            truth_history += [{"role": "user", "content": subq},
                              {"role": "assistant", "content": gold_text}]
            rag_history += [{"role": "user", "content": subq},
                            {"role": "assistant", "content": turn.rag_answer}]

        logger.info("xie-subq %s | types=%s | outcomes=%s", conversation_id,
                    conv.type_sequence, conv.outcome_sequence)
        return conv

    def generate_many(self, seeds, n_turns: int) -> List[AdaptiveConversation]:
        out = []
        for i, seed in enumerate(seeds[: self.config.num_conversations]):
            out.append(self.generate(seed, f"conv-{i:03d}", n_turns))
        return out
