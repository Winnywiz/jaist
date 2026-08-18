"""
Module 1 — GraphRAG knowledge-graph builder.

Turns a flat list of text chunks into a typed entity graph:

  * **nodes** — entities, ``{"id", "entity", "type", "chunks": [chunk_ids]}``
  * **edges** — typed relations, ``{"source", "target", "relation"}``

Extraction strategy (with graceful degradation):

  1. **LLM extraction** (preferred) — one JSON call per chunk asking for entities
     (with types) and relations. Produces *typed* edges (``advisor_of``,
     ``works_for``, ``located_in`` ...), GraphRAG-style.
  2. **Heuristic fallback** — capitalised noun-phrase entities + intra-chunk
     ``co_occurs_with`` edges, so a graph still exists with no API key.

Every node records which source chunks it appears in (``node -> chunks``), which is
what lets the retriever return supporting passages and the gold-answer generator
cite evidence. The graph is a ``networkx.MultiDiGraph`` so multiple typed relations
can connect the same pair of entities.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..config import Config, get_logger
from ..embeddings import Embedder
from ..llm import LLM

logger = get_logger("graph.builder")

#: Fine-grained entity ontology. A richer, more specific taxonomy makes the graph
#: more discriminating, which lets the question generator (a) pick *comparable*
#: same-type entities for Comparative questions, and (b) build cross-type chains
#: for Multi-Hop questions. Extend freely — extraction reads this list at runtime.
ENTITY_TYPES = [
    "Person", "Organization", "Institution", "Location", "Country", "City",
    "Publication", "Book", "Article", "Theory", "Concept", "Event", "Time",
    "Technology", "Dataset", "Field", "Product", "Method", "Award", "Other",
]

_EXTRACT_SYS = (
    "You are an information-extraction engine building a knowledge graph. "
    "From the PASSAGE, extract the salient named entities (each tagged with the MOST "
    "SPECIFIC type from the allowed set) and the factual relations between them. "
    "Prefer the narrowest type that fits (e.g. City over Location, Institution over "
    "Organization, Book or Article over Publication, Theory over Concept). Use "
    "concise canonical entity names. Relation labels must be short snake_case "
    "predicates (e.g. works_for, founded, located_in, developed, authored, "
    "published_in, part_of, born_in, influenced_by, succeeded_by). Extract enough "
    "relations to connect entities into chains (so multi-hop questions are possible), "
    "and be sure to tag comparable same-type entities when the passage mentions "
    "more than one.\n"
    "Allowed entity types: " + ", ".join(ENTITY_TYPES) + ".\n"
    'Respond as JSON: {"entities": [{"name": "...", "type": "<one allowed type>"}], '
    '"relations": [{"source": "...", "target": "...", "relation": "..."}]}'
)

_STOPWORD_CAPS = {"The", "A", "An", "This", "That", "These", "Those", "It",
                  "He", "She", "They", "In", "On", "At", "For", "And", "But",
                  "Of", "To", "By", "As", "With", "From", "After", "Before"}


def slugify(name: str) -> str:
    """Stable node id from an entity name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "entity"


#: Citation / cross-reference pseudo-entities (BIBREF0, TABREF16, SECREF27, FIGREF1).
#: These are LaTeX reference markers, not real entities — they produce meaningless graph
#: edges ("X --authored--> BIBREF0"), so drop them at extraction time.
_JUNK_ENTITY_RE = re.compile(r"^\W*(BIB|TAB|SEC|FIG)REF\d+\W*$", re.I)


def _is_junk_entity(name: str) -> bool:
    n = (name or "").strip()
    return not n or bool(_JUNK_ENTITY_RE.match(n))


@dataclass
class KnowledgeGraph:
    """Container bundling the graph with its source chunks and embeddings.

    Attributes:
        graph: the ``networkx.MultiDiGraph`` (node attrs: ``entity``, ``type``,
            ``chunks``; edge attrs: ``relation``).
        chunks: the source passages (index = chunk id).
        chunk_embeddings: optional (n_chunks, d) array for dense retrieval.
        node_ids: ordered node ids aligned with ``node_embeddings``.
        node_embeddings: optional (n_nodes, d) array.
    """

    graph: nx.MultiDiGraph
    chunks: List[str]
    chunk_embeddings: Optional[np.ndarray] = None
    node_ids: List[str] = field(default_factory=list)
    node_embeddings: Optional[np.ndarray] = None

    # -- convenience -------------------------------------------------------- #
    def node(self, node_id: str) -> Dict:
        data = self.graph.nodes[node_id]
        return {"id": node_id, "entity": data.get("entity"),
                "type": data.get("type"), "chunks": sorted(data.get("chunks", []))}

    def nodes(self) -> List[Dict]:
        return [self.node(n) for n in self.graph.nodes]

    def edges(self) -> List[Dict]:
        # collapse duplicate (source, target, relation) edges extracted from
        # multiple chunks into ONE, with weight = number of supporting chunks
        # (a relation seen in many chunks is stronger evidence).
        agg: Dict = {}
        for u, v, d in self.graph.edges(data=True):
            k = (u, v, d.get("relation"))
            if k in agg:
                agg[k]["weight"] += 1
            else:
                agg[k] = {"source": u, "target": v,
                          "relation": d.get("relation"), "weight": 1}
        return list(agg.values())

    def chunks_for(self, node_id: str) -> List[str]:
        return [self.chunks[i] for i in sorted(self.graph.nodes[node_id].get("chunks", []))
                if 0 <= i < len(self.chunks)]

    def stats(self) -> Dict:
        return {"n_nodes": self.graph.number_of_nodes(),
                "n_edges": self.graph.number_of_edges(),
                "n_chunks": len(self.chunks)}

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str) -> None:
        payload = {
            "nodes": self.nodes(),
            "edges": self.edges(),
            "chunks": self.chunks,
        }
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False, indent=2)
        logger.info("saved graph (%s) to %s", self.stats(), path)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Reconstruct a KnowledgeGraph from a saved graph.json (nodes/edges/chunks).
        Skips the expensive LLM relation-extraction build. chunk_embeddings are NOT
        restored (set them via an embedder afterward for dense retrieval)."""
        with open(path, "r", encoding="utf-8") as fr:
            d = json.load(fr)
        g = nx.MultiDiGraph()
        for n in d.get("nodes", []):
            g.add_node(n["id"], entity=n.get("entity"), type=n.get("type"),
                       chunks=set(n.get("chunks", [])))
        for e in d.get("edges", []):
            g.add_edge(e["source"], e["target"], relation=e.get("relation"))
        kg = cls(graph=g, chunks=d.get("chunks", []))
        logger.info("loaded graph (%s) from %s", kg.stats(), path)
        return kg


class GraphBuilder:
    """Builds a :class:`KnowledgeGraph` from text chunks.

    Args:
        config: active configuration.
        llm: shared LLM wrapper (offline -> heuristic extraction).
        embedder: shared embedder (offline -> no dense vectors, lexical retrieval).
    """

    def __init__(self, config: Optional[Config] = None, llm: Optional[LLM] = None,
                 embedder: Optional[Embedder] = None,
                 graph_mode: Optional[str] = None):
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        self.embedder = embedder or Embedder(config=self.config, llm=self.llm)
        #: 'typed' = LLM typed relations, 'cooccur' = heuristic co-occurrence edges,
        #: 'none' = no graph (dense-only retrieval). Drives the graph ablation.
        self.graph_mode = graph_mode or getattr(self.config, "graph_mode", "typed")

    # -- public ------------------------------------------------------------- #
    def build(self, chunks: List[str], max_chunks: Optional[int] = None) -> KnowledgeGraph:
        """Extract entities/relations from ``chunks`` and assemble the graph."""
        chunks = [c for c in chunks if c and c.strip()]
        if max_chunks:
            chunks = chunks[:max_chunks]

        # 'none' ablation arm: no graph at all -> retriever does dense-only.
        if self.graph_mode == "none":
            logger.info("graph_mode=none: skipping extraction (dense-only retrieval)")
            kg = KnowledgeGraph(graph=nx.MultiDiGraph(), chunks=chunks)
            self._attach_embeddings(kg)
            return kg

        extractor = ("llm-typed" if (self.graph_mode == "typed" and self.llm.available)
                     else "heuristic-cooccur")
        logger.info("building graph from %d chunks (mode=%s, extractor=%s)",
                    len(chunks), self.graph_mode, extractor)

        graph = nx.MultiDiGraph()
        node_chunks: Dict[str, set] = defaultdict(set)

        for cid, chunk in enumerate(chunks):
            entities, relations = self._extract(chunk)
            for ent in entities:
                if _is_junk_entity(ent["name"]):     # drop citation/ref pseudo-entities
                    continue
                nid = slugify(ent["name"])
                node_chunks[nid].add(cid)
                if graph.has_node(nid):
                    # keep the first non-Other type seen
                    if graph.nodes[nid].get("type") in (None, "Other"):
                        graph.nodes[nid]["type"] = ent.get("type", "Other")
                else:
                    graph.add_node(nid, entity=ent["name"], type=ent.get("type", "Other"),
                                   chunks=set())
            for rel in relations:
                if _is_junk_entity(rel["source"]) or _is_junk_entity(rel["target"]):
                    continue                          # drop edges touching a citation label
                s, t = slugify(rel["source"]), slugify(rel["target"])
                if s == t:
                    continue
                # ensure endpoints exist even if not listed as entities
                for nid, name in ((s, rel["source"]), (t, rel["target"])):
                    if not graph.has_node(nid):
                        graph.add_node(nid, entity=name, type="Other", chunks=set())
                    node_chunks[nid].add(cid)
                graph.add_edge(s, t, relation=rel.get("relation", "related_to"),
                               chunk=cid)

        # finalise the node -> chunk index sets
        for nid, cids in node_chunks.items():
            if graph.has_node(nid):
                graph.nodes[nid]["chunks"] = set(cids)

        kg = KnowledgeGraph(graph=graph, chunks=chunks)
        self._attach_embeddings(kg)
        logger.info("graph built: %s", kg.stats())
        return kg

    def extract_local(self, chunks: List[str],
                      cache: Optional[Dict[str, Tuple]] = None) -> Tuple[List[Dict], List[Dict]]:
        """Entities + relations from THESE chunks only — no persistent graph.

        The fair control for "does the graph help?": the caller gets the *same*
        LLM-extracted entities the graph arm sees, but relations only ever hold
        WITHIN a single retrieved chunk. So the graph arm's sole advantage is
        structure that spans chunks, not entity knowledge. Returns evidence-shaped
        ``(nodes, edges)``. ``cache`` is keyed by chunk text to avoid re-extraction.
        """
        cache = cache if cache is not None else {}
        nodes: Dict[str, Dict] = {}
        edges: List[Dict] = []
        for chunk in chunks:
            key = chunk[:200]
            if key not in cache:
                cache[key] = self._extract(chunk)
            entities, relations = cache[key]
            for ent in entities:
                nid = slugify(ent["name"])
                nodes.setdefault(nid, {"id": nid, "entity": ent["name"],
                                       "type": ent.get("type", "Other")})
            for rel in relations:
                if _is_junk_entity(rel["source"]) or _is_junk_entity(rel["target"]):
                    continue                          # drop edges touching a citation label
                s, t = slugify(rel["source"]), slugify(rel["target"])
                if s == t:
                    continue
                for nid, name in ((s, rel["source"]), (t, rel["target"])):
                    nodes.setdefault(nid, {"id": nid, "entity": name, "type": "Other"})
                edges.append({"source": s, "target": t,
                              "relation": rel.get("relation", "related_to")})
        return list(nodes.values()), edges

    # -- extraction --------------------------------------------------------- #
    def _extract(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        # 'cooccur' arm forces proximity edges even when an LLM is available.
        if self.graph_mode == "cooccur":
            return self._heuristic_extract(chunk)
        if self.llm.available:
            out = self.llm.chat_json(_EXTRACT_SYS, f"PASSAGE:\n{chunk[:1500]}")
            if out is not None:
                ents = [e for e in out.get("entities", [])
                        if isinstance(e, dict) and e.get("name")]
                rels = [r for r in out.get("relations", [])
                        if isinstance(r, dict) and r.get("source") and r.get("target")]
                if ents or rels:
                    return ents, rels
        return self._heuristic_extract(chunk)

    @staticmethod
    def _heuristic_extract(chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """Capitalised noun-phrase entities + intra-chunk co-occurrence edges."""
        # Greedy runs of Capitalised Words (allowing internal lowercase 'of/the').
        phrases = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+(?:of|the|de|van|von)?\s*"
                             r"[A-Z][a-zA-Z0-9]+)*)\b", chunk)
        seen, entities = set(), []
        for p in phrases:
            p = p.strip()
            head = p.split()[0]
            if head in _STOPWORD_CAPS or len(p) < 3:
                continue
            if p.lower() in seen:
                continue
            seen.add(p.lower())
            entities.append({"name": p, "type": "Other"})
        # link consecutive entities with a generic co-occurrence relation
        relations = []
        for a, b in zip(entities, entities[1:]):
            relations.append({"source": a["name"], "target": b["name"],
                              "relation": "co_occurs_with"})
        return entities[:12], relations[:12]

    # -- embeddings --------------------------------------------------------- #
    def _attach_embeddings(self, kg: KnowledgeGraph) -> None:
        if not self.embedder.available:
            logger.info("no embedder; retriever will use lexical fallback")
            return
        kg.chunk_embeddings = self.embedder.encode(kg.chunks)
        kg.node_ids = list(kg.graph.nodes)
        node_texts = [kg.graph.nodes[n].get("entity", n) for n in kg.node_ids]
        kg.node_embeddings = self.embedder.encode(node_texts) if node_texts else None
