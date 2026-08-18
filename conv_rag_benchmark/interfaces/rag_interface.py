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

Retrieval score semantics (``RAGResponse.retrieved_docs[i]["score"]``)
---------------------------------------------------------------------
Scores come straight from each backend's ranking step — never recomputed or
fabricated — so their MEANING (and hence comparability) differs by RAG. ``rank`` is
comparable across all of them; the raw ``score`` is not.

  * ``vector`` / ``longrag`` / ``mock`` / ``selfrag`` / ``crag`` / ``raptor`` /
    ``graph`` — ``score_type="cosine"``: inner product of L2-normalised OpenAI
    embeddings (== cosine similarity), when an embedding backend is available.
  * any of the above — ``score_type="idf_overlap"``: the lexical fallback used when
    embeddings are unavailable — sum of idf weights of query/​chunk shared tokens
    (a BM25-like magnitude, NOT comparable to a cosine).
  * ``score=None`` — the backend surfaced no meaningful score (e.g. empty query);
    kept as null rather than invented.

``doc_id`` is the chunk's index in that RAG's own corpus (``self.chunks``), so it is
directly comparable to a retrieval over the SAME corpus. Two caveats: ``raptor``
indexes RAPTOR TREE nodes (leaves + summaries), a superset of the raw chunks; and
``selfrag``/``crag`` re-retrieve and merge, so a doc's ``score`` is the one from the
retrieval that surfaced it and ``rank`` is its position in the final merged set.
"""
from __future__ import annotations

import abc
import math
import os
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
    # ADDITIVE retrieval metadata (kept alongside — not replacing — retrieved_context,
    # which existing consumers still read). Each entry is the FINAL retrieved chunk the
    # RAG answered from, as a record: {doc_id, rank, score, score_type, text}. doc_id is
    # the chunk's index in the RAG's own corpus; rank is 1-based in the order the RAG
    # used; score is the retriever's actual ranking score (None when a backend exposes
    # none — never fabricated); score_type names the metric. See rag_interface docstring
    # for the per-RAG meaning of score.
    retrieved_docs: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"answer": self.answer, "retrieved_context": self.retrieved_context,
                "retrieved_docs": self.retrieved_docs, "meta": self.meta}


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
                 k: int = 5, use_embeddings: bool = True, use_history: bool = True,
                 precomputed_emb=None):
        # NOTE: chunks are filtered for emptiness; when ``precomputed_emb`` is supplied it
        # MUST be row-aligned with the FILTERED chunk list (same order). The caller
        # (large-corpus index path) already passes clean chunks; a length mismatch falls
        # back to a fresh encode so a stale/misaligned cache can never be used silently.
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
        if precomputed_emb is not None and len(precomputed_emb) == len(self.chunks):
            self._emb = precomputed_emb          # reuse the on-disk index (no re-encode)
        elif self.embedder is not None and self.embedder.available:
            self._emb = self.embedder.encode(self.chunks)

    def _retrieve_scored(self, query: str) -> List[Dict]:
        """Retrieve the top-k chunks WITH their ranking metadata (additive).

        Identical ranking/selection to :meth:`_retrieve` — this only stops throwing the
        already-computed score away. Each record is
        ``{doc_id, rank, score, score_type, text}`` where ``doc_id`` is the chunk's index
        in ``self.chunks``, ``rank`` is 1-based, ``score`` is the actual ranking score
        (dense inner product of L2-normalised embeddings == cosine, or the lexical
        idf-overlap sum), and ``score_type`` names it. Scores are never fabricated: when
        neither signal applies (empty query) ``score`` is None."""
        if self._emb is not None:
            import numpy as np
            qv = self.embedder.encode([query])
            if qv is not None and len(qv):
                sims = self._emb @ qv[0]
                idx = np.argsort(-sims)[: self.k]
                return [{"doc_id": int(i), "rank": r + 1,
                         "score": round(float(sims[i]), 6), "score_type": "cosine",
                         "text": self.chunks[i]} for r, i in enumerate(idx)]
        qt = set(_tokens(query))
        if not qt:
            return [{"doc_id": i, "rank": i + 1, "score": None, "score_type": None,
                     "text": self.chunks[i]}
                    for i in range(min(self.k, len(self.chunks)))]
        scored = []
        for i, ct in enumerate(self._ctoks):
            shared = qt & ct
            if shared:
                scored.append((sum(self._idf.get(t, 1.0) for t in shared), i))
        scored.sort(reverse=True)
        return [{"doc_id": int(i), "rank": r + 1, "score": round(float(s), 6),
                 "score_type": "idf_overlap", "text": self.chunks[i]}
                for r, (s, i) in enumerate(scored[: self.k])]

    @staticmethod
    def _reindexed(docs: List[Dict]) -> List[Dict]:
        """Re-number ``rank`` to the final list order (after slicing/merging), leaving
        each doc's captured score untouched. Returned list is a fresh copy."""
        out = []
        for r, d in enumerate(docs):
            e = dict(d)
            e["rank"] = r + 1
            out.append(e)
        return out

    def _retrieve(self, query: str) -> List[str]:
        return [d["text"] for d in self._retrieve_scored(query)]

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
        docs = self._retrieve_scored(query)
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
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
        docs = self._retrieve_scored(query)
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
                           meta={"rag": self.name})


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
        docs = self._retrieve_scored(query)
        if self._rng.random() < self.hallucinate_p:
            docs = docs[:1]  # starve the generator -> more likely to fabricate
        docs = self._reindexed(docs)
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history if use_hist else [])
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
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
        docs = self._retrieve_scored(query)
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)
        reflected = False
        if self.llm.available and ans:
            ctxt = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(ctx))
            crit = self.llm.chat_json(
                self._CRITIC_SYS, f"CONTEXT:\n{ctxt}\n\nDRAFT: {ans}")
            if crit is not None and not crit.get("supported", True):
                requery = str(crit.get("requery") or query).strip() or query
                docs2 = self._retrieve_scored(f"{query} {requery}")
                # union of both retrievals, deduped by text, keeps the widened evidence.
                # Each doc keeps the score from the retrieval that surfaced it (the
                # re-retrieval's docs win ties, matching the ctx2+ctx order below).
                merged, seen = [], set()
                for d in docs2 + docs:
                    if d["text"] not in seen:
                        merged.append(d); seen.add(d["text"])
                docs = self._reindexed(merged[: self.k + 2])
                ctx = [d["text"] for d in docs]
                ans = self._generate(question, ctx, history)
                reflected = True
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
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
        docs = self._retrieve_scored(query)
        verdict, corrected = "correct", False
        if self.llm.available and docs:
            listed = "\n\n".join(f"[{i+1}] {d['text'][:500]}"
                                 for i, d in enumerate(docs))
            ev = self.llm.chat_json(
                self._EVAL_SYS, f"QUESTION: {question}\n\nPASSAGES:\n{listed}") or {}
            verdict = str(ev.get("verdict") or "correct").lower()
            idx = [i for i in (ev.get("relevant") or [])
                   if isinstance(i, int) and 1 <= i <= len(docs)]
            keep = [docs[i - 1] for i in idx] or docs
            if verdict == "correct":
                docs = keep                     # knowledge refinement: drop the noise
            else:
                requery = str(ev.get("requery") or query).strip() or query
                docs2 = self._retrieve_scored(f"{query} {requery}")
                base = keep if verdict == "ambiguous" else []
                merged, seen = [], set()
                for d in base + docs2:
                    if d["text"] not in seen:
                        merged.append(d); seen.add(d["text"])
                docs = merged[: self.k + 2] or docs
                corrected = True
        docs = self._reindexed(docs)
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
                           meta={"rag": self.name, "verdict": verdict,
                                 "corrected": corrected})


class RaptorRAG(_LexicalCorpusRAG):
    """A faithful RAPTOR target (Sarthi et al., ICLR 2024).

    Unlike a plain retriever, RAPTOR pre-builds a **tree**: it recursively clusters the
    chunks and summarises each cluster with an LLM, so the tree holds both raw leaf chunks
    and higher-level summaries. At query time it uses RAPTOR's **collapsed-tree** retrieval:
    all nodes (leaves + every summary) form one pool, and the top-k by cosine similarity are
    retrieved — so a broad question can match a summary and a specific one a raw chunk.

    The tree is built once (see :mod:`raptor_tree`) and cached per dataset. This
    implementation is on the same OpenAI backend as the other RAGs (the authors' package
    pins an incompatible openai version), so the comparison stays controlled.
    """

    name = "raptor"

    def __init__(self, chunks, config=None, llm=None, embedder=None, k=5,
                 use_history=True, tree_nodes=None, cache_path=None):
        # the retriever pool is the RAPTOR TREE (leaves + summaries), not the raw chunks
        from .raptor_tree import build_tree
        base_llm = llm or LLM(config=config or Config.load())
        base_emb = embedder if embedder is not None else Embedder(config=config, llm=base_llm)
        nodes = tree_nodes if tree_nodes is not None else build_tree(
            list(chunks), base_llm, base_emb, cache_path=cache_path)
        super().__init__(nodes, config=config, llm=llm, embedder=embedder,
                         k=k, use_embeddings=True, use_history=use_history)
        self.n_leaves = len(chunks)
        self.n_nodes = len(nodes)

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        docs = self._retrieve_scored(query)         # collapsed-tree: over leaves + summaries
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
                           meta={"rag": self.name, "tree_nodes": self.n_nodes,
                                 "leaves": self.n_leaves})


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
                           retrieved_docs=ev.scored_documents(),
                           meta={"rag": self.name})


class HippoRAG(RAGInterface):
    """A HippoRAG-style target (Gutiérrez et al., 2024).

    HippoRAG mimics the hippocampal index: an OFFLINE knowledge graph built by running
    OpenIE over every passage, and ONLINE retrieval by **Personalized PageRank** over that
    graph. Here the benchmark's typed entity graph (entities + relations extracted per
    chunk — see :class:`GraphBuilder`) IS the OpenIE index. At query time we extract the
    query's entities, seed PPR on the matching graph nodes, and score each passage by the
    PPR mass of the entities it contains, returning the top-k passages. Falls back to dense
    retrieval when no query entity links into the graph.

    Retrieval score is the PPR-derived passage score (``score_type="ppr"``) — a graph
    centrality signal, NOT comparable to cosine. doc_id is the chunk index (into kg.chunks).

    Because it needs the LLM-built OpenIE graph over the WHOLE corpus, it is only feasible
    on small corpora (like the generated-benchmark chunk sets), NOT the 366K-passage mtRAG
    corpora — same constraint as ``graph``/``raptor``."""

    name = "hippo"

    def __init__(self, kg, config: Optional[Config] = None, llm: Optional[LLM] = None,
                 embedder: Optional[Embedder] = None, k: int = 5, use_history: bool = True):
        self.kg = kg
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.embedder = embedder
        self.k = k
        self.use_history = use_history
        # entity name (lowercased) -> node id, for linking query entities into the graph
        self._name2node: Dict[str, str] = {}
        for nid in kg.graph.nodes:
            ent = (kg.graph.nodes[nid].get("entity") or "").strip().lower()
            if ent:
                self._name2node.setdefault(ent, nid)

    def _query_entities(self, question: str) -> List[str]:
        if self.llm.available:
            out = self.llm.chat_json(
                "Extract the key named entities and noun phrases from the QUESTION. "
                'Respond as JSON: {"entities": ["...", ...]}', f"QUESTION: {question}")
            ents = [str(e).strip() for e in (out or {}).get("entities", []) if str(e).strip()]
            if ents:
                return ents
        return re.findall(r"[A-Za-z0-9]{3,}", question)         # fallback: content tokens

    def _seed_nodes(self, entities: List[str]) -> set:
        seeds = set()
        for e in entities:
            el = e.lower().strip()
            if not el:
                continue
            if el in self._name2node:
                seeds.add(self._name2node[el])
                continue
            for name, nid in self._name2node.items():           # loose substring link
                if len(name) >= 3 and (el in name or name in el):
                    seeds.add(nid)
        return seeds

    def _dense_topk(self, query: str):
        """Fallback dense retrieval (query has no graph anchor)."""
        if self.kg.chunk_embeddings is not None and self.embedder is not None \
                and self.embedder.available:
            import numpy as np
            qv = self.embedder.encode([query])
            if qv is not None and len(qv):
                sims = self.kg.chunk_embeddings @ qv[0]
                idx = np.argsort(-sims)[: self.k]
                return [(int(i), round(float(sims[i]), 6), "cosine") for i in idx]
        return [(i, None, None) for i in range(min(self.k, len(self.kg.chunks)))]

    def _generate(self, question: str, context: List[str], history: List[Dict]) -> str:
        ctx = "\n\n".join(f"[{i+1}] {c[:600]}" for i, c in enumerate(context))
        hist = ("CONVERSATION:\n" + "\n".join(f"{t['role']}: {t['content']}"
                                              for t in history[-6:]) + "\n\n") \
            if (self.use_history and history) else ""
        if self.llm.available:
            out = self.llm.chat_json(_LexicalCorpusRAG._ANSWER_SYS,
                                     f"{hist}CONTEXT:\n{ctx}\n\nQUESTION: {question}")
            if out is not None:
                return str(out.get("answer", "")).strip()
        return _LexicalCorpusRAG._extractive(question, context)

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        import networkx as nx
        from collections import defaultdict
        history = history or []
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question
        seeds = self._seed_nodes(self._query_entities(query))

        docs: List[Dict] = []
        if seeds and self.kg.graph.number_of_nodes():
            pers = {n: 1.0 for n in seeds if self.kg.graph.has_node(n)}
            try:
                pr = nx.pagerank(self.kg.graph, personalization=pers) if pers else {}
            except Exception:
                pr = {}
            chunk_scores = defaultdict(float)
            for node, score in pr.items():
                for c in self.kg.graph.nodes[node].get("chunks", []):
                    chunk_scores[c] += score
            ranked = sorted(chunk_scores.items(), key=lambda kv: -kv[1])[: self.k]
            docs = [{"doc_id": int(c), "rank": i + 1, "score": round(float(s), 6),
                     "score_type": "ppr", "text": self.kg.chunks[c]}
                    for i, (c, s) in enumerate(ranked) if 0 <= c < len(self.kg.chunks)]
        if not docs:                                            # dense fallback
            docs = [{"doc_id": i, "rank": r + 1,
                     "score": s, "score_type": st, "text": self.kg.chunks[i]}
                    for r, (i, s, st) in enumerate(self._dense_topk(query))]
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
                           meta={"rag": self.name, "seed_entities": len(seeds)})


def build_rag(kind: str, chunks: List[str], config: Config,
              llm: Optional[LLM] = None, retriever=None,
              embedder: Optional[Embedder] = None,
              cache_dir: Optional[str] = None,
              k: Optional[int] = None,
              precomputed_emb=None) -> RAGInterface:
    """Factory: construct a target RAG by name (``mock`` | ``vector`` | ``graph`` | …).

    ``cache_dir`` (when given) is where RAPTOR caches its pre-built tree per dataset.
    ``k`` (when given) overrides the retrieval budget (number of docs retrieved per
    query); None keeps each backend's own default.
    ``precomputed_emb`` (when given) is a row-aligned corpus embedding matrix from the
    on-disk index, injected into the corpus RAGs so they skip re-encoding a large corpus.
    """
    kind = (kind or "mock").lower()
    kk = {} if k is None else {"k": k}          # retrieval-budget override
    # corpus RAGs accept a precomputed embedding matrix; graph (uses a retriever) and
    # raptor (builds/embeds its own tree) do not.
    ck = dict(kk)
    if precomputed_emb is not None:
        ck["precomputed_emb"] = precomputed_emb
    if kind == "mock":
        return MockRAG(chunks, config=config, llm=llm, embedder=embedder, **ck)
    if kind == "vector":
        return VectorRAG(chunks, config=config, llm=llm, embedder=embedder, **ck)
    if kind == "selfrag":
        return SelfRAG(chunks, config=config, llm=llm, embedder=embedder, **ck)
    if kind == "crag":
        return CorrectiveRAG(chunks, config=config, llm=llm, embedder=embedder, **ck)
    if kind == "raptor":
        cache = os.path.join(cache_dir, "raptor_tree.json") if cache_dir else None
        return RaptorRAG(chunks, config=config, llm=llm, embedder=embedder, cache_path=cache, **kk)
    if kind == "longrag":
        return LongRAG(chunks, config=config, llm=llm, embedder=embedder, **ck)
    if kind == "graph":
        if retriever is None:
            raise ValueError("graph RAG requires a retriever")
        return GraphRAGAdapter(retriever, config=config, llm=llm, **kk)
    if kind == "hippo":
        # HippoRAG runs Personalized PageRank over the typed OpenIE graph; it needs the
        # graph, which the retriever carries (built graph_mode="typed" by the caller).
        if retriever is None or getattr(retriever, "kg", None) is None:
            raise ValueError("hippo RAG requires a retriever built over a typed graph")
        return HippoRAG(retriever.kg, config=config, llm=llm, embedder=embedder, **kk)
    raise ValueError(f"unknown target_rag {kind!r}")
