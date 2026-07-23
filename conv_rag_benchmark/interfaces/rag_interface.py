"""
Module 6 — the target-RAG interface (the system the benchmark *tests*).

``RAGInterface`` is the abstract contract::

    answer(question: str, history: list[dict]) -> RAGResponse

so any external system — a local pipeline, an HTTP API, a GraphRAG, a vector RAG —
can be dropped in. Three reference implementations ship:

  * :class:`MockRAG`     — deliberately *imperfect*: lexical retrieval + an LLM (or
                           template) generator that, by design, sometimes ignores
                           history / hallucinates, so the failure taxonomy lights up.
  * :class:`VectorRAG`   — embedding (or lexical) retrieve-then-generate over the
                           same corpus chunks.
  * :class:`GraphRAGAdapter` — wraps the benchmark's own GraphRetriever as a
                           (strong) target, useful as an upper-baseline.

A response carries the retrieved context so failures can be split into
retrieval vs generation.
"""
from __future__ import annotations

import abc
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, get_logger
from ..embeddings import Embedder
from ..llm import LLM

logger = get_logger("interfaces.rag")


@dataclass
class RAGResponse:
    """What a target RAG returns for one turn."""

    answer: str
    retrieved_context: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"answer": self.answer, "retrieved_context": self.retrieved_context,
                "meta": self.meta}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class RAGInterface(abc.ABC):
    """Abstract base class for any target RAG under test."""

    name: str = "abstract"

    @abc.abstractmethod
    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        """Answer ``question`` given prior ``history`` (``[{role, content}]``)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook to clear any per-conversation state."""


class _LexicalCorpusRAG(RAGInterface):
    """Shared retrieve-then-generate machinery over a fixed list of chunks."""

    _ANSWER_SYS = (
        "You are a RAG system. Answer the QUESTION using the CONTEXT passages and the "
        "CONVERSATION so far. Give a short, direct answer. Only answer \"I don't know\" "
        "if the context genuinely lacks the information. "
        'Respond as JSON: {"answer": "..."}'
    )

    def __init__(self, chunks: List[str], config: Optional[Config] = None,
                 llm: Optional[LLM] = None, embedder: Optional[Embedder] = None,
                 k: int = 5, use_embeddings: bool = True, use_history: bool = True):
        self.chunks = [c for c in chunks if c and c.strip()]
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.k = k
        self.use_history = use_history
        self._ctoks = [set(_tokens(c)) for c in self.chunks]
        self._idf = self._build_idf()
        self.embedder = embedder if embedder is not None else (
            Embedder(config=self.config, llm=self.llm) if use_embeddings else None)
        self._emb = None
        if self.embedder is not None and self.embedder.available:
            self._emb = self.embedder.encode(self.chunks)

    def _retrieve(self, query: str) -> List[str]:
        if self._emb is not None:
            import numpy as np
            qv = self.embedder.encode([query])
            if qv is not None and len(qv):
                sims = self._emb @ qv[0]
                idx = np.argsort(-sims)[: self.k]
                return [self.chunks[i] for i in idx]
        qt = set(_tokens(query))
        if not qt:
            return self.chunks[: self.k]
        scored = []
        for i, ct in enumerate(self._ctoks):
            shared = qt & ct
            if shared:
                scored.append((sum(self._idf.get(t, 1.0) for t in shared), i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[: self.k]]

    def _generate(self, question: str, context: List[str], history: List[Dict]) -> str:
        ctx = "\n\n".join(f"[{i+1}] {c[:600]}" for i, c in enumerate(context))
        hist = ""
        if self.use_history and history:
            hist = "CONVERSATION:\n" + "\n".join(
                f"{t['role']}: {t['content']}" for t in history[-6:]) + "\n\n"
        if self.llm.available:
            out = self.llm.chat_json(self._ANSWER_SYS,
                                     f"{hist}CONTEXT:\n{ctx}\n\nQUESTION: {question}")
            if out is not None:
                return str(out.get("answer", "")).strip()
        # offline template generator: return the most overlapping sentence
        return self._extractive(question, context)

    @staticmethod
    def _extractive(question: str, context: List[str]) -> str:
        qt = set(_tokens(question))
        best, best_score = "", 0
        for c in context:
            for sent in re.split(r"(?<=[.!?])\s+", c):
                st = set(_tokens(sent))
                score = len(qt & st)
                if score > best_score:
                    best, best_score = sent.strip(), score
        return best or (context[0][:200] if context else "I don't know.")

    def _build_idf(self) -> Dict[str, float]:
        n = len(self.chunks)
        df = Counter()
        for ct in self._ctoks:
            for tok in ct:
                df[tok] += 1
        return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


class VectorRAG(_LexicalCorpusRAG):
    """A faithful retrieve-then-generate vector RAG over the corpus chunks."""

    name = "vector"

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        ctx = self._retrieve(query)
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx,
                           meta={"rag": self.name})


class LongRAG(_LexicalCorpusRAG):
    """LongRAG-style (Jiang et al. 2024): retrieve FEWER but LONGER units instead of
    many short chunks. Consecutive corpus chunks are merged into long units before
    indexing, only the top-1/2 units are retrieved, and the generator sees the FULL
    long context (not truncated to 600 chars). Tests whether coarser retrieval
    granularity improves grounding vs the standard short-chunk VectorRAG."""

    name = "longrag"

    def __init__(self, chunks: List[str], *, group: int = 5,
                 unit_chars: int = 2500, **kw):
        merged: List[str] = []
        clean = [c for c in chunks if c and c.strip()]
        for i in range(0, len(clean), group):
            unit = "\n".join(clean[i:i + group]).strip()
            if unit:
                merged.append(unit)
        self.unit_chars = unit_chars
        kw.setdefault("k", 2)                        # fewer, longer units
        super().__init__(merged, **kw)

    def _generate(self, question: str, context: List[str], history: List[Dict]) -> str:
        ctx = "\n\n".join(f"[{i+1}] {c[:self.unit_chars]}"
                          for i, c in enumerate(context))
        hist = ""
        if self.use_history and history:
            hist = "CONVERSATION:\n" + "\n".join(
                f"{t['role']}: {t['content']}" for t in history[-6:]) + "\n\n"
        if self.llm.available:
            out = self.llm.chat_json(self._ANSWER_SYS,
                                     f"{hist}CONTEXT:\n{ctx}\n\nQUESTION: {question}")
            if out is not None:
                return str(out.get("answer", "")).strip()
        return self._extractive(question, context)

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        ctx = self._retrieve(query)
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx, meta={"rag": self.name})


class MockRAG(_LexicalCorpusRAG):
    """An intentionally *weak* RAG that exhibits realistic failures.

    Knobs (probabilities) inject the very behaviours the taxonomy targets:
      * ``ignore_history_p`` — drops the conversation, breaking coreference,
      * ``hallucinate_p``    — answers even with empty/irrelevant context,
      * ``no_abstain``       — never says "I don't know" (overconfident).
    """

    name = "mock"

    def __init__(self, *args, ignore_history_p: float = 0.4,
                 hallucinate_p: float = 0.25, k: int = 3, **kwargs):
        kwargs.setdefault("use_history", False)  # weak by default
        super().__init__(*args, k=k, **kwargs)
        self.ignore_history_p = ignore_history_p
        self.hallucinate_p = hallucinate_p
        import random
        self._rng = random.Random(self.config.seed)

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        use_hist = history and self._rng.random() > self.ignore_history_p
        query = question
        if use_hist:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        ctx = self._retrieve(query)
        if self._rng.random() < self.hallucinate_p:
            ctx = ctx[:1]  # starve the generator -> more likely to fabricate
        ans = self._generate(question, ctx, history if use_hist else [])
        return RAGResponse(answer=ans, retrieved_context=ctx,
                           meta={"rag": self.name, "used_history": use_hist})


class SelfRAG(_LexicalCorpusRAG):
    """A Self-RAG-style target: generate, *self-critique* groundedness, and
    adaptively re-retrieve once if the first answer is judged unsupported.

    This mimics the reflection / adaptive-retrieval behaviour of Self-RAG
    (Asai et al., 2023) without its trained reflection tokens: the LLM itself
    judges whether its draft is entailed by the context and, if not, the query
    is expanded and retrieval is retried a single time.
    """

    name = "selfrag"

    _CRITIC_SYS = (
        "You are the reflection module of a Self-RAG system. Decide whether the "
        "DRAFT answer is fully supported by the CONTEXT. If it is not, propose a "
        "better search query to find the missing evidence. "
        'Respond as JSON: {"supported": true/false, "requery": "..."}'
    )

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        ctx = self._retrieve(query)
        ans = self._generate(question, ctx, history)
        reflected = False
        if self.llm.available and ans:
            ctxt = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(ctx))
            crit = self.llm.chat_json(
                self._CRITIC_SYS, f"CONTEXT:\n{ctxt}\n\nDRAFT: {ans}")
            if crit is not None and not crit.get("supported", True):
                requery = str(crit.get("requery") or query).strip() or query
                ctx2 = self._retrieve(f"{query} {requery}")
                # union of both retrievals, deduped, keeps the widened evidence
                merged, seen = [], set()
                for c in ctx2 + ctx:
                    if c not in seen:
                        merged.append(c); seen.add(c)
                ctx = merged[: self.k + 2]
                ans = self._generate(question, ctx, history)
                reflected = True
        return RAGResponse(answer=ans, retrieved_context=ctx,
                           meta={"rag": self.name, "reflected": reflected})


class CorrectiveRAG(_LexicalCorpusRAG):
    """A CRAG-style target: *grade* the retrieved set, then CORRECT it before answering.

    Follows Corrective RAG (Yan et al., 2024): a lightweight retrieval evaluator labels
    the retrieved passages ``correct`` / ``ambiguous`` / ``incorrect``. On *correct* the
    context is refined (only the passages judged relevant are kept, dropping noise); on
    *incorrect* the query is rewritten and retrieval retried; on *ambiguous* the refined
    and re-retrieved sets are merged. CRAG's web-search fallback is replaced by a
    rewritten-query re-retrieval over the same corpus, since this harness is offline.

    Contrast with :class:`SelfRAG`: Self-RAG critiques its own *answer* after generating,
    CRAG grades the *evidence* before generating.
    """

    name = "crag"

    _EVAL_SYS = (
        "You are the retrieval evaluator of a Corrective RAG system. For the QUESTION, "
        "decide which numbered PASSAGES are relevant and give an overall verdict: "
        "'correct' = at least one passage clearly helps answer it; 'incorrect' = none are "
        "useful; 'ambiguous' = only partially useful. If not 'correct', propose a better "
        "search query. "
        'Respond as JSON: {"verdict": "correct|ambiguous|incorrect", '
        '"relevant": [1, 2], "requery": "..."}'
    )

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        ctx = self._retrieve(query)
        verdict, corrected = "correct", False
        if self.llm.available and ctx:
            listed = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(ctx))
            ev = self.llm.chat_json(
                self._EVAL_SYS, f"QUESTION: {question}\n\nPASSAGES:\n{listed}") or {}
            verdict = str(ev.get("verdict") or "correct").lower()
            idx = [i for i in (ev.get("relevant") or [])
                   if isinstance(i, int) and 1 <= i <= len(ctx)]
            keep = [ctx[i - 1] for i in idx] or ctx
            if verdict == "correct":
                ctx = keep                      # knowledge refinement: drop the noise
            else:
                requery = str(ev.get("requery") or query).strip() or query
                ctx2 = self._retrieve(f"{query} {requery}")
                base = keep if verdict == "ambiguous" else []
                merged, seen = [], set()
                for c in base + ctx2:
                    if c not in seen:
                        merged.append(c); seen.add(c)
                ctx = merged[: self.k + 2] or ctx
                corrected = True
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx,
                           meta={"rag": self.name, "verdict": verdict,
                                 "corrected": corrected})


class RaptorRAG(_LexicalCorpusRAG):
    """A RAPTOR-style target: retrieve a wide set of leaf chunks, *summarise*
    them into a consolidated higher-level context, then answer from that.

    This approximates RAPTOR's (Sarthi et al., 2024) recursive-summary tree
    with a query-time "collapsed tree": instead of pre-building summary nodes,
    the retrieved leaves are compressed on the fly so the generator reasons over
    an abstraction rather than raw, overlapping passages.
    """

    name = "raptor"

    _SUMMARY_SYS = (
        "You compress retrieved passages into a single dense summary that "
        "preserves every fact relevant to the QUESTION. Do not answer the "
        'question. Respond as JSON: {"summary": "..."}'
    )

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        leaves = self._retrieve(query)
        # widen: pull a couple more than usual so the summary spans abstractions
        extra = [c for c in self._retrieve(question) if c not in leaves]
        leaves = (leaves + extra)[: self.k + 3]
        context = leaves
        if self.llm.available and leaves:
            joined = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(leaves))
            out = self.llm.chat_json(
                self._SUMMARY_SYS, f"QUESTION: {question}\n\nPASSAGES:\n{joined}")
            summ = str((out or {}).get("summary", "")).strip()
            if summ:
                context = [summ]
        ans = self._generate(question, context, history)
        # report the leaf chunks as retrieved context so retrieval failures are visible
        return RAGResponse(answer=ans, retrieved_context=leaves,
                           meta={"rag": self.name, "summarised": context is not leaves})


class GraphRAGAdapter(RAGInterface):
    """Wrap the benchmark's own GraphRetriever as a strong target RAG."""

    name = "graph"

    def __init__(self, retriever, config: Optional[Config] = None,
                 llm: Optional[LLM] = None, k: int = 6):
        self.retriever = retriever
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.k = k

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        ev = self.retriever.retrieve(question, k=self.k, conversation_history=history)
        sys = ("You are a GraphRAG system. Answer the QUESTION from the EVIDENCE and "
               "CONVERSATION. Be concise; say \"I don't know\" if unsupported. "
               'Respond as JSON: {"answer": "..."}')
        hist = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
        if self.llm.available:
            out = self.llm.chat_json(
                sys, f"CONVERSATION:\n{hist}\n\nEVIDENCE:\n{ev.evidence_text()}\n\n"
                     f"QUESTION: {question}")
            ans = str((out or {}).get("answer", "")).strip()
        else:
            ans = _LexicalCorpusRAG._extractive(question, ev.chunks)
        return RAGResponse(answer=ans, retrieved_context=ev.chunks,
                           meta={"rag": self.name})


def build_rag(kind: str, chunks: List[str], config: Config,
              llm: Optional[LLM] = None, retriever=None,
              embedder: Optional[Embedder] = None) -> RAGInterface:
    """Factory: construct a target RAG by name (``mock`` | ``vector`` | ``graph``)."""
    kind = (kind or "mock").lower()
    if kind == "mock":
        return MockRAG(chunks, config=config, llm=llm, embedder=embedder)
    if kind == "vector":
        return VectorRAG(chunks, config=config, llm=llm, embedder=embedder)
    if kind == "selfrag":
        return SelfRAG(chunks, config=config, llm=llm, embedder=embedder)
    if kind == "crag":
        return CorrectiveRAG(chunks, config=config, llm=llm, embedder=embedder)
    if kind == "raptor":
        return RaptorRAG(chunks, config=config, llm=llm, embedder=embedder)
    if kind == "longrag":
        return LongRAG(chunks, config=config, llm=llm, embedder=embedder)
    if kind == "graph":
        if retriever is None:
            raise ValueError("graph RAG requires a retriever")
        return GraphRAGAdapter(retriever, config=config, llm=llm)
    raise ValueError(f"unknown target_rag {kind!r}")
