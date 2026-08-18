"""
Module 2 — GraphRAG hybrid retriever.

Given a question (and optional conversation history) it returns the evidence the
gold-answer generator and the failure classifier need::

    {"nodes": [...], "edges": [...], "chunks": [...]}

Two-stage hybrid retrieval:

  1. **Dense / lexical seed** — score every chunk against the (history-augmented)
     query. Dense cosine when embeddings exist, else IDF token overlap.
  2. **Graph neighborhood expansion** — map the top chunks to the entities that
     occur in them, walk one hop along typed edges, and pull in any *additional*
     chunks attached to those neighbour entities. This is what surfaces the
     bridge facts needed for multi-hop questions.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import Config, get_logger
from ..embeddings import Embedder
from ..llm import LLM
from .graph_builder import KnowledgeGraph

logger = get_logger("graph.retriever")

#: downstream consumers truncate evidence_text() to ~1500-2200 chars, which in
#: practice keeps roughly the first 3 chunks — provenance must land inside this
#: prefix or the gold generator / judge never see it
VISIBLE_PREFIX = 3
#: max relation-support chunks force-inserted into the visible prefix per query
MAX_SUPPORT_INJECT = 2


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


@dataclass
class RetrievalResult:
    """Structured retrieval output."""

    nodes: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    chunk_ids: List[int] = field(default_factory=list)
    # ADDITIVE retrieval metadata (does not change ranking): the relevance score
    # the retriever ranked each surfaced chunk by, aligned index-for-index with
    # ``chunk_ids``/``chunks``. ``score_type`` names what the number means
    # ("cosine" for dense embedding rerank, "idf_overlap" for the lexical fallback).
    # Scores may be empty if scoring was unavailable; never fabricated.
    scores: List[float] = field(default_factory=list)
    score_type: str = ""

    def to_dict(self) -> Dict:
        return {"nodes": self.nodes, "edges": self.edges, "chunks": self.chunks}

    def scored_documents(self) -> List[Dict]:
        """The retrieved chunks as ranked document records (doc_id/rank/score/text).

        ``doc_id`` is the chunk's index in the KnowledgeGraph corpus (``kg.chunks``),
        so it is comparable to any other retrieval over the SAME corpus. ``rank`` is
        1-based in final retrieval order. ``score`` is the actual ranking score (or
        None when unavailable — never invented)."""
        docs: List[Dict] = []
        for rank, cid in enumerate(self.chunk_ids):
            score = self.scores[rank] if rank < len(self.scores) else None
            docs.append({"doc_id": int(cid), "rank": rank + 1,
                         "score": score, "score_type": self.score_type or None,
                         "text": self.chunks[rank] if rank < len(self.chunks) else ""})
        return docs

    def evidence_text(self, max_chars: int = 2400) -> str:
        """Flatten chunks into a single context string for prompting."""
        joined = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(self.chunks))
        return joined[:max_chars]


class GraphRetriever:
    """Hybrid dense + graph-neighborhood retriever over a :class:`KnowledgeGraph`."""

    def __init__(self, kg: KnowledgeGraph, config: Optional[Config] = None,
                 llm: Optional[LLM] = None, embedder: Optional[Embedder] = None):
        self.kg = kg
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.embedder = embedder or Embedder(config=self.config, llm=self.llm)
        self._chunk_tokens = [set(_tokens(c)) for c in kg.chunks]
        self._idf = self._build_idf()

    # -- public ------------------------------------------------------------- #
    def retrieve(self, question: str, k: int = 10,
                 conversation_history: Optional[List[Dict]] = None,
                 expand_hops: int = 1) -> RetrievalResult:
        """Retrieve evidence for ``question`` within an optional conversation.

        Args:
            question: the current turn's question.
            k: number of chunks to return.
            conversation_history: list of ``{"role", "content"}`` turns; their
                text is appended to the query so coreference ("he", "that
                company") still retrieves the right passages.
            expand_hops: graph neighborhood hops (0 = dense only).
        """
        query = self._augment(question, conversation_history)
        seed_ids = self._dense_seed(query, k=k)
        chunk_ids = list(seed_ids)

        # graph expansion -> entities in seed chunks -> neighbours -> their chunks
        seed_nodes = self._nodes_in_chunks(seed_ids)
        frontier = set(seed_nodes)
        all_nodes = set(seed_nodes)
        edges: List[Dict] = []
        for _ in range(max(0, expand_hops)):
            nxt = set()
            for nid in frontier:
                for _, tgt, data in self.kg.graph.out_edges(nid, data=True):
                    edges.append({"source": nid, "target": tgt,
                                  "relation": data.get("relation")})
                    if tgt not in all_nodes:
                        nxt.add(tgt)
                for src, _, data in self.kg.graph.in_edges(nid, data=True):
                    edges.append({"source": src, "target": nid,
                                  "relation": data.get("relation")})
                    if src not in all_nodes:
                        nxt.add(src)
            all_nodes |= nxt
            frontier = nxt
            for nid in nxt:
                for cid in sorted(self.kg.graph.nodes[nid].get("chunks", [])):
                    if cid not in chunk_ids:
                        chunk_ids.append(cid)

        # keep the k most query-relevant chunks overall
        chunk_ids = self._rerank(query, chunk_ids)[:k]
        nodes = [self.kg.node(n) for n in all_nodes if self.kg.graph.has_node(n)]
        # de-duplicate edges
        uniq_edges = {(e["source"], e["target"], e["relation"]): e for e in edges}

        # -- edge provenance guarantee ------------------------------------- #
        # The graph's job here is benchmark construction: it steers WHICH
        # follow-up to ask next (walk a relation to a neighbour). But every
        # relation we advertise to the question generator must arrive WITH the
        # chunk that states it, in the part of the evidence that survives the
        # downstream evidence_text() truncation — otherwise the gold generator
        # and the gold_supported judge grade against evidence that never
        # contains the steering fact. Edges with no co-occurrence chunk at all
        # (extraction artifacts) are dropped entirely: only document-grounded
        # relations may steer generation.
        kept = list(chunk_ids)
        seed_node_set = set(seed_nodes)
        candidates = sorted(
            uniq_edges.values(),
            key=lambda e: -((e["source"] in seed_node_set)
                            + (e["target"] in seed_node_set)))

        # pass 1 — pick support chunks to pull into the visible prefix (edges
        # whose provenance would otherwise be truncated away downstream)
        inject: List[int] = []
        for e in candidates:
            if len(inject) >= MAX_SUPPORT_INJECT:
                break
            support = self._edge_support(e)
            if support and not (set(support) & set(kept[:VISIBLE_PREFIX])):
                best = self._rerank(query, support)[0]
                if best not in inject:
                    inject.append(best)
        if inject:
            kept = [c for c in kept if c not in inject]
            kept = kept[:1] + inject + kept[1:]   # top dense hit stays first
            kept = kept[:k]
        visible = set(kept[:VISIBLE_PREFIX])

        # pass 2 — advertise only edges whose source text is in the visible
        # prefix; everything else (and any relation stated in no chunk) is
        # dropped so generation can only be steered by facts the gold generator
        # and the judge will actually see
        grounded_edges: List[Dict] = []
        for e in candidates:
            sup_visible = [c for c in self._edge_support(e) if c in visible]
            if sup_visible:
                grounded_edges.append({**e, "support": sup_visible})

        chunk_ids = kept
        scores, score_type = self._score_chunks(query, chunk_ids)
        result = RetrievalResult(
            nodes=nodes,
            edges=grounded_edges,
            chunks=[self.kg.chunks[i] for i in chunk_ids],
            chunk_ids=chunk_ids,
            scores=scores,
            score_type=score_type,
        )
        logger.debug("retrieve(k=%d): %d chunks, %d nodes, %d edges",
                     k, len(result.chunks), len(result.nodes), len(result.edges))
        return result

    # -- internals ---------------------------------------------------------- #
    def _edge_support(self, edge: Dict) -> List[int]:
        """Chunks that state the relation: where both endpoint entities occur."""
        g = self.kg.graph
        if not (g.has_node(edge["source"]) and g.has_node(edge["target"])):
            return []
        src = set(g.nodes[edge["source"]].get("chunks", []))
        tgt = set(g.nodes[edge["target"]].get("chunks", []))
        return sorted(src & tgt)

    @staticmethod
    def _augment(question: str, history: Optional[List[Dict]]) -> str:
        if not history:
            return question
        hist = " ".join(t.get("content", "") for t in history[-4:])
        return f"{hist} {question}".strip()

    def _dense_seed(self, query: str, k: int) -> List[int]:
        if self.kg.chunk_embeddings is not None and self.embedder.available:
            qv = self.embedder.encode([query])
            if qv is not None and len(qv):
                sims = self.kg.chunk_embeddings @ qv[0]
                return list(np.argsort(-sims)[:k].astype(int))
        return self._lexical_seed(query, k)

    def _lexical_seed(self, query: str, k: int) -> List[int]:
        qt = set(_tokens(query))
        if not qt:
            return list(range(min(k, len(self.kg.chunks))))
        scored = []
        for i, ct in enumerate(self._chunk_tokens):
            shared = qt & ct
            if shared:
                scored.append((sum(self._idf.get(t, 1.0) for t in shared), i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]

    def _score_chunks(self, query: str, chunk_ids: List[int]):
        """The ranking score of each chunk, in the GIVEN order (does not reorder).

        Uses the SAME formula ``_rerank`` ranks by — dense cosine when embeddings are
        available, else the lexical idf-overlap fallback — so the exposed number IS the
        retriever's own ranking signal, not a recomputed different metric. Returns
        ``(scores, score_type)`` with ``scores`` aligned to ``chunk_ids``."""
        if not chunk_ids:
            return [], ""
        if self.kg.chunk_embeddings is not None and self.embedder.available:
            qv = self.embedder.encode([query])
            if qv is not None and len(qv):
                return ([round(float(self.kg.chunk_embeddings[i] @ qv[0]), 6)
                         for i in chunk_ids], "cosine")
        qt = set(_tokens(query))
        return ([round(float(sum(self._idf.get(t, 1.0)
                                 for t in (qt & self._chunk_tokens[i]))), 6)
                 for i in chunk_ids], "idf_overlap")

    def _rerank(self, query: str, chunk_ids: List[int]) -> List[int]:
        if not chunk_ids:
            return chunk_ids
        if self.kg.chunk_embeddings is not None and self.embedder.available:
            qv = self.embedder.encode([query])
            if qv is not None and len(qv):
                sims = {i: float(self.kg.chunk_embeddings[i] @ qv[0]) for i in chunk_ids}
                return sorted(chunk_ids, key=lambda i: -sims[i])
        qt = set(_tokens(query))
        return sorted(chunk_ids,
                      key=lambda i: -sum(self._idf.get(t, 1.0)
                                         for t in (qt & self._chunk_tokens[i])))

    def _nodes_in_chunks(self, chunk_ids) -> List[str]:
        wanted = set(chunk_ids)
        out = []
        for nid in self.kg.graph.nodes:
            if self.kg.graph.nodes[nid].get("chunks", set()) & wanted:
                out.append(nid)
        return out

    def _build_idf(self) -> Dict[str, float]:
        n = len(self.kg.chunks)
        df = Counter()
        for ct in self._chunk_tokens:
            for tok in ct:
                df[tok] += 1
        return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
