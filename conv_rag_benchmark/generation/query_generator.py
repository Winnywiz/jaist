"""
Module 4 — typed conversational query generator.

Produces one labelled question per turn::

    {
        "question": str,
        "query_type": str,
        "difficulty": int,        # 1 (simple factual) .. 5 (multi-hop reasoning)
        "capability": str,        # the conversational ability under test
        "expected_failure": str,  # the failure this turn is engineered to expose
    }

The 8 query types map 1:1 onto the conversational capabilities and the failure
taxonomy, so the report can attribute *why* a RAG failed, not just *that* it did.

Each type has an LLM generator (grounded in retrieved graph evidence + history)
and a deterministic template fallback so generation works offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, get_logger
from ..llm import LLM
from ..graph.retriever import RetrievalResult

logger = get_logger("generation.query")

# --------------------------------------------------------------------------- #
# Type registry: query_type -> (capability, expected_failure, base_difficulty)
# --------------------------------------------------------------------------- #
QUERY_TYPES: Dict[str, Dict] = {
    "Follow-Up": {
        "capability": "Coreference Resolution / History Utilization",
        "expected_failure": "Coreference Failure",
        "difficulty": 2,
    },
    "Clarification": {
        "capability": "Context Refinement / Context Preservation",
        "expected_failure": "Incomplete Answer",
        "difficulty": 2,
    },
    "Comparative": {
        "capability": "Multi-Entity Retrieval / Context Fusion",
        "expected_failure": "Comparative Failure",
        "difficulty": 4,
    },
    "Correction": {
        "capability": "History Correction / Context Invalidation",
        "expected_failure": "Correction Failure",
        "difficulty": 3,
    },
    "Topic Shift": {
        "capability": "Context Reset / Avoiding Stale Memory",
        "expected_failure": "Topic Shift Failure",
        "difficulty": 3,
    },
    "Unanswerable": {
        "capability": "Abstention / Hallucination Control",
        "expected_failure": "Overconfident Unknown",
        "difficulty": 3,
    },
    "Multi-Hop": {
        "capability": "Long-range Conversational Reasoning",
        "expected_failure": "Multi-Hop Failure",
        "difficulty": 5,
    },
    "Ambiguous Reference": {
        "capability": "Reference Resolution / Ambiguity Handling",
        "expected_failure": "Coreference Failure",
        "difficulty": 4,
    },
}


@dataclass
class QueryTurn:
    """A generated, typed question for one conversation turn."""

    question: str
    query_type: str
    turn_id: int
    difficulty: int
    capability: str
    expected_failure: str
    conversation_history: List[Dict] = field(default_factory=list)
    # provenance the gold generator / classifier reuse:
    target_entity: Optional[str] = None
    is_unanswerable: bool = False
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "query_type": self.query_type,
            "turn_id": self.turn_id,
            "difficulty": self.difficulty,
            "capability": self.capability,
            "expected_failure": self.expected_failure,
            "conversation_history": self.conversation_history,
            "target_entity": self.target_entity,
            "is_unanswerable": self.is_unanswerable,
        }


class QueryGenerator:
    """Generates a typed question grounded in retrieved evidence and history."""

    _SYS = (
        "You generate ONE conversational question of a specified TYPE for testing a "
        "conversational RAG system. The question must sound natural in the ongoing "
        "dialogue and, unless the type is 'Unanswerable', must be answerable from the "
        "EVIDENCE. Never mention 'documents', 'context' or 'passages'. "
        "TRUTHFULNESS RULE (Gricean Maxim of Quality): only presuppose facts that have "
        "ALREADY been stated in the CONVERSATION SO FAR. Do NOT smuggle facts from the "
        "EVIDENCE into the question as assumptions — e.g. do NOT add descriptive clauses "
        "like 'when he was left heartbroken' or 'the award-winning X' unless that was "
        "already said. Ask ONLY about the single new detail; a bare pronoun referring to "
        "an already-mentioned entity is fine. "
        "GROUNDABILITY RULE: unless the type is 'Unanswerable', the answer to your "
        "question must be a fact EXPLICITLY STATED in the EVIDENCE text — before "
        "writing, locate the exact evidence sentence that answers it. Never ask what "
        "a named article/publication/source 'says' or 'reports' (source names are "
        "metadata, not evidence), and never ask about an entity that the EVIDENCE "
        "does not mention. "
        'Respond as JSON: {"question": "...", "target_entity": "...", '
        '"difficulty": 1-5}'
    )

    _TYPE_INSTRUCTIONS = {
        "Follow-Up": "Ask a follow-up about ONE specific, concrete, checkable detail of "
                     "the most recent entity (a name, number, function, location, part, or "
                     "date), referring to that entity ONLY with a pronoun (he/she/it/they/"
                     "that), never naming it. Do NOT ask open/essay questions about "
                     "'overall' significance or understanding. Keep it SHORT: just the "
                     "pronoun + the single thing you want to know — add NO descriptive "
                     "clause about the entity that wasn't already said. Tests coreference.",
        "Clarification": "Pick ONE concrete detail from the EVIDENCE that the previous "
                         "answer touched on, and ask the assistant to clarify or expand "
                         "THAT specific point (e.g. 'can you explain how X happened?'). "
                         "The clarifying answer must itself be stated in the EVIDENCE. "
                         "Do NOT ask 'what do you mean by ...' about wording the evidence "
                         "cannot resolve. Tests context preservation.",
        "Comparative": "From COMPARABLE SAME-TYPE PAIRS, STRONGLY prefer a pair marked "
                       "with a shared attribute ('X vs Y (shared attribute ...)') and ask "
                       "to compare the two entities on EXACTLY that shared attribute — "
                       "both sides of the comparison are then stated in the evidence. "
                       "Name both entities. NEVER compare on an attribute the evidence "
                       "states for only one side, and do NOT ask open-ended 'how do X "
                       "and Y compare overall' questions. "
                       "Tests multi-entity retrieval and fusion.",
        "Correction": "Correct a wrong entity introduced earlier ('Actually I meant X, "
                      "not Y') and ask about the corrected entity. Tests context replacement.",
        "Topic Shift": "Abruptly switch to a DIFFERENT topic present in the evidence "
                       "('Now tell me about ...'). Tests memory flushing.",
        "Unanswerable": "Ask something about the entity that the evidence CANNOT answer "
                        "and that no corpus could know (e.g. a social-media handle, a "
                        "private opinion). The correct behaviour is to abstain.",
        "Multi-Hop": "Pick ONE REASONING CHAIN (A --r1--> B --r2--> C) and ask a question "
                     "about C phrased in terms of A, WITHOUT naming the middle entity B, so "
                     "the answer can only be reached by traversing A->B->C. The answer is C. "
                     "Do NOT staple two unrelated entities together — the path must connect.",
        "Ambiguous Reference": "Refer to a previously-mentioned entity ONLY by an ambiguous "
                               "pronoun ('it' / 'they' / 'he' / 'she' / 'that company') — never "
                               "by name — but ask about a NEW ATTRIBUTE of it: a specific fact "
                               "STATED IN THE EVIDENCE that has NOT already been given earlier in "
                               "the conversation. The reference must be resolvable only from the "
                               "conversation history (that is the coreference test), and the "
                               "ANSWER must be a NEW fact — never the entity's name, and never a "
                               "fact already stated. Example: after discussing a company's "
                               "revenue, ask 'Who is its CEO?' (pronoun refers back, answer is new).",
    }

    def __init__(self, config: Optional[Config] = None, llm: Optional[LLM] = None,
                 bridge_walk: bool = True):
        self.config = config or Config.load()
        self.llm = llm or LLM(config=self.config)
        # bridge_walk=False reproduces the OLD Multi-Hop behaviour (flat disconnected
        # edge list, "staple" questions) for before/after depth comparison.
        self.bridge_walk = bridge_walk

    def generate(self, query_type: str, turn_id: int,
                 evidence: RetrievalResult, history: List[Dict],
                 entities: Optional[List[str]] = None,
                 feedback: Optional[str] = None,
                 covered: Optional[List[str]] = None) -> QueryTurn:
        """Generate one typed turn.

        Args:
            query_type: one of :data:`QUERY_TYPES`.
            turn_id: 0-based position in the conversation.
            evidence: retrieved graph evidence to ground the question.
            history: prior turns as ``[{"role", "content"}, ...]``.
            entities: candidate entity names (defaults to evidence node entities).
            covered: facts/answers already asked in this conversation, so the generator
                targets a NEW fact instead of recycling one (coverage-guided generation).
        """
        spec = QUERY_TYPES.get(query_type)
        if spec is None:
            raise ValueError(f"unknown query_type {query_type!r}")
        entities = entities or [n.get("entity") for n in evidence.nodes if n.get("entity")]

        turn = self._llm_generate(query_type, evidence, history, entities, feedback, covered)
        if turn is None:
            turn = self._fallback(query_type, history, entities)

        turn.turn_id = turn_id
        turn.capability = spec["capability"]
        turn.expected_failure = spec["expected_failure"]
        if not turn.difficulty:
            turn.difficulty = spec["difficulty"]
        turn.is_unanswerable = (query_type == "Unanswerable")
        turn.conversation_history = list(history)
        return turn

    # -- generation backends ------------------------------------------------ #
    def _llm_generate(self, query_type: str, evidence: RetrievalResult,
                      history: List[Dict], entities: List[str],
                      feedback: Optional[str] = None,
                      covered: Optional[List[str]] = None) -> Optional[QueryTurn]:
        if not self.llm.available:
            return None
        hist_text = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:]) or "(start)"

        # Typed entities + relation triples let the LLM exploit the ontology:
        #   - same-type entities -> Comparative questions,
        #   - cross-type relation chains -> Multi-Hop questions.
        id2name = {n["id"]: n.get("entity", n["id"]) for n in evidence.nodes}
        typed_entities = [f"{n.get('entity')} [{n.get('type', 'Other')}]"
                          for n in evidence.nodes if n.get("entity")][:10]
        rels = []
        for e in evidence.edges[:12]:
            s = id2name.get(e.get("source"), e.get("source"))
            t = id2name.get(e.get("target"), e.get("target"))
            rels.append(f"{s} --[{e.get('relation')}]--> {t}")
        same_type_pairs = self._comparable_pairs(evidence.nodes, evidence.edges)
        # BRIDGE-WALK: connected paths only, and only when we actually want a chain.
        bridge_chains = (self._bridge_paths(evidence.nodes, evidence.edges)
                         if (query_type == "Multi-Hop" and self.bridge_walk) else "")

        user = (
            f"TYPE: {query_type}\n"
            f"TYPE INSTRUCTION: {self._TYPE_INSTRUCTIONS[query_type]}\n"
            f"KNOWN ENTITIES (typed): {', '.join(typed_entities) or '(none)'}\n"
            f"COMPARABLE SAME-TYPE PAIRS: {same_type_pairs or '(none)'}\n"
            + (f"REASONING CHAINS (connected paths — pick ONE, ask about its endpoint "
               f"via its start, and NEVER name the middle entity; the answer is the "
               f"endpoint):\n{bridge_chains}\n" if bridge_chains else "")
            + f"RELATIONS (context):\n"
            + ("\n".join(rels) if rels else "(none)") + "\n"
            f"CONVERSATION SO FAR:\n{hist_text}\n\n"
            f"EVIDENCE:\n{evidence.evidence_text(1800)}"
            + (("\n\nALREADY ANSWERED IN THIS CONVERSATION (do NOT ask anything whose answer "
                "is one of these — pick a DIFFERENT, not-yet-covered fact from the EVIDENCE):\n- "
                + "\n- ".join(str(c)[:120] for c in covered[-8:]))
               if covered else "")
            + (f"\n\nPREVIOUS ATTEMPT REJECTED: {feedback} "
               "Write a DIFFERENT question of the same TYPE that avoids this problem "
               "— anchor it to another fact that IS stated in the EVIDENCE."
               if feedback else "")
        )
        out = self.llm.chat_json(self._SYS, user)
        if not out or not out.get("question"):
            return None
        diff = out.get("difficulty")
        try:
            diff = max(1, min(5, int(diff))) if diff is not None else 0
        except (TypeError, ValueError):
            diff = 0
        te = out.get("target_entity")
        if isinstance(te, (list, tuple)):            # LLM sometimes returns a list
            te = next((str(x).strip() for x in te if str(x).strip()), None)
        elif te is not None and not isinstance(te, str):
            te = str(te).strip()
        return QueryTurn(
            question=str(out["question"]).strip(),
            query_type=query_type,
            turn_id=0,
            difficulty=diff,
            capability="",
            expected_failure="",
            target_entity=(te or (entities[0] if entities else None)),
        )

    #: Relations too vacuous to compare over: they hold between almost any two
    #: entities, so "A vs B in terms of <r>" has no composable gold. Two entities
    #: sharing only one of these share nothing (msrjd --involves--> transformation
    #: vs static scenarios --involves--> time). Excluded from the shared-attribute
    #: path so Comparative reroutes to Multi-Hop instead of freelancing.
    _VACUOUS_RELATIONS = frozenset({
        "related_to", "co_occurs_with", "associated_with",
        "involves", "includes", "contains",
    })

    @staticmethod
    def _bridge_paths(nodes: List[Dict], edges: Optional[List[Dict]] = None,
                      max_paths: int = 3) -> str:
        """Find CONNECTED 2-hop paths A --r1--> B --r2--> C for genuine multi-hop
        ('chain') questions, NOT two entities stapled on a shared relation label.
        The question asks about endpoint C via start A while hiding middle B, so it
        can only be answered by traversing the whole path. A vacuous relation is
        allowed as a hop only if the OTHER hop is contentful (so at least one real
        reasoning step exists). Requires A != B != C and A != C."""
        if not edges:
            return ""
        id2name = {n["id"]: n.get("entity", n["id"]) for n in nodes}
        adj: Dict[str, List] = {}
        for e in edges:
            r, s, t = e.get("relation"), e.get("source"), e.get("target")
            if r and s and t:
                adj.setdefault(s, []).append((r, t))
        paths: List[str] = []
        seen = set()
        for a, outs in adj.items():
            for r1, b in outs:
                if b == a or b not in adj:
                    continue
                for r2, c in adj[b]:
                    if c in (a, b) or (a, c) in seen:
                        continue
                    # at least one hop must be contentful (not both vacuous)
                    if r1 in QueryGenerator._VACUOUS_RELATIONS and \
                            r2 in QueryGenerator._VACUOUS_RELATIONS:
                        continue
                    seen.add((a, c))
                    na, nb, nc = id2name.get(a, a), id2name.get(b, b), id2name.get(c, c)
                    paths.append(f"{na} --[{r1}]--> {nb} --[{r2}]--> {nc}  "
                                 f"(ASK about '{nc}' via '{na}'; do NOT name '{nb}')")
                    if len(paths) >= max_paths:
                        return "\n".join(paths)
        return "\n".join(paths)

    @staticmethod
    def _comparable_pairs(nodes: List[Dict], edges: Optional[List[Dict]] = None,
                          max_pairs: int = 4) -> str:
        """Render comparable entity pairs. Prefer pairs that SHARE a contentful
        relation type (the evidence states the same attribute for BOTH sides, so a
        comparison question over that attribute is composable into a grounded gold);
        fall back to same-type pairs (excluding the generic 'Other')."""
        pairs: List[str] = []
        if edges:
            id2name = {n["id"]: n.get("entity", n["id"]) for n in nodes}
            by_rel: Dict[str, List] = {}
            for e in edges:
                r = e.get("relation")
                s = id2name.get(e.get("source"), e.get("source"))
                t = id2name.get(e.get("target"), e.get("target"))
                if r and s and t and r not in QueryGenerator._VACUOUS_RELATIONS:
                    by_rel.setdefault(r, []).append((s, t))
            for r, st in by_rel.items():
                firsts: List = []
                for s, t in st:                       # distinct sources only
                    if s not in [x[0] for x in firsts]:
                        firsts.append((s, t))
                if len(firsts) >= 2:
                    (s1, t1), (s2, t2) = firsts[0], firsts[1]
                    pairs.append(f"{s1} vs {s2} (shared attribute '{r}': "
                                 f"{s1} --{r}--> {t1}; {s2} --{r}--> {t2})")
                if len(pairs) >= max_pairs:
                    break
        by_type: Dict[str, List[str]] = {}
        for n in nodes:
            t, name = n.get("type", "Other"), n.get("entity")
            if not name or t in (None, "Other"):
                continue
            by_type.setdefault(t, [])
            if name not in by_type[t]:
                by_type[t].append(name)
        for t, names in by_type.items():
            if len(pairs) >= max_pairs:
                break
            if len(names) >= 2:
                pairs.append(f"{names[0]} vs {names[1]} ({t})")
        return "; ".join(pairs)

    def _fallback(self, query_type: str, history: List[Dict],
                  entities: List[str]) -> QueryTurn:
        """Deterministic templates so generation works with no API key."""
        e0 = entities[0] if entities else "the main subject"
        e1 = entities[1] if len(entities) > 1 else e0
        templates = {
            "Follow-Up": "What more can you tell me about it?",
            "Clarification": "Can you explain that more simply?",
            "Comparative": f"How do {e0} and {e1} compare?",
            "Correction": f"Actually, I meant {e1}, not {e0}. What about {e1}?",
            "Topic Shift": f"Now tell me about {e1}.",
            "Unanswerable": f"What is {e0}'s personal TikTok username?",
            "Multi-Hop": f"Based on what we discussed, how is {e0} connected to {e1}?",
            "Ambiguous Reference": "What else is it known for?",
        }
        spec = QUERY_TYPES[query_type]
        return QueryTurn(
            question=templates[query_type],
            query_type=query_type,
            turn_id=0,
            difficulty=spec["difficulty"],
            capability=spec["capability"],
            expected_failure=spec["expected_failure"],
            target_entity=e0,
        )
