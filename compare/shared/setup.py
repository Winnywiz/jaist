"""
shared/setup.py — the experiment scaffolding BOTH methods rely on.

Nothing here is "the method". This file only:
  * defines the three failure categories we inject (Retrieval / Generation / Conversation),
  * INJECTS controlled failures whose true cause is CERTAIN (build_injected_cases),
  * wraps the RAG we probe (ControlledRAG — we control what context it sees),
  * judges answer-correctness (is_correct, via an LLM judge when available),
  * provides the two helpers both methods share (did_fail, base_attribute).

Ground truth is injected: because WE cause each failure, its category is known, so
"attribution accuracy" is objectively measurable. The DYNAMIC/ and STATIC/ methods are
scored on how often they recover that injected category.

Heavy infrastructure (RAG generator, embeddings, LLM client, dataset loading) is reused
from the shared `conv_rag_benchmark` engine — this folder is only the experiment layer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from conv_rag_benchmark.config import Config
from conv_rag_benchmark.embeddings import Embedder
from conv_rag_benchmark.generation.gold_answer_generator import ABSTENTION
from conv_rag_benchmark.interfaces.rag_interface import VectorRAG
from conv_rag_benchmark.llm import LLM

# The categories we inject and score at (Reasoning omitted — see README limitations).
RETRIEVAL = "Retrieval"
GENERATION = "Generation"
CONVERSATION = "Conversation"


class OfflineLLM:
    """Stand-in so the harness runs with zero API calls (deterministic smoke test)."""
    available = False
    def chat_json(self, *a, **k): return None
    def chat(self, *a, **k): return ""
    def cost_report(self): return "offline"


def _tokens(t: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (t or "").lower())


_JUDGE = None              # optional LLM for semantic correctness (set by harness)
_CORRECT_CACHE = {}        # (gold, answer) -> bool, to avoid duplicate judge calls


def set_judge(llm):
    """Use an LLM to judge answer==gold semantically (instead of lexical matching)."""
    global _JUDGE
    _JUDGE = llm


def _lexical_correct(gold: str, answer: str) -> bool:
    g, a = (gold or "").lower().strip(), (answer or "").lower()
    if not g or not a:
        return False
    if g[:30] in a or a[:30] in g:
        return True
    gt = set(_tokens(gold))
    return bool(gt) and len(gt & set(_tokens(answer))) / len(gt) >= 0.6


def is_correct(gold: str, answer: str) -> bool:
    """Semantic correctness via the judge LLM when set, else lenient lexical.

    The lexical check false-fails specific-fact golds phrased differently, which
    breaks both failure-firing detection and the dynamic recovery probes — so a
    semantic judge is used when available."""
    if not (gold and answer):
        return False
    if _JUDGE is not None and getattr(_JUDGE, "available", False):
        key = (gold[:200], answer[:200])
        if key not in _CORRECT_CACHE:
            out = _JUDGE.chat_json(
                'Does the ANSWER convey the same fact as the GOLD? '
                'Reply JSON {"correct": true/false}.',
                f"GOLD: {gold}\nANSWER: {answer}")
            _CORRECT_CACHE[key] = (bool(out.get("correct")) if out
                                   else _lexical_correct(gold, answer))
        return _CORRECT_CACHE[key]
    return _lexical_correct(gold, answer)


_IDK_CACHE = {}            # response -> is_idk bool, to avoid duplicate judge calls

#: MTRAG-style IDK judge prompt. Detects a refusal/abstention robustly (paraphrase,
#: hedging) instead of by keyword, so a real "I don't know" phrased in novel words is
#: not missed — which would otherwise break the P3 fabrication flip and did_fail firing.
_IDK_SYS = (
    "You judge whether a RESPONSE is an \"I don't know\" (IDK) / abstention: it declines "
    "to answer, or states the information is unavailable / not stated / not found / "
    "cannot be determined, INSTEAD of attempting a substantive answer. A response that "
    "asserts any specific factual claim is NOT idk, even if hedged ('I think', "
    "'possibly', 'approximately'). "
    'Reply JSON {"idk": true/false}.'
)


def _lexical_abstained(answer: str) -> bool:
    """Keyword fallback used when no judge LLM is available (offline mode)."""
    low = (answer or "").lower()
    return any(p in low for p in ("i don't know", "i dont know", "cannot answer",
                                  "not answerable", "no information", "unable to",
                                  "not stated", "not mentioned"))


def abstained(answer: str) -> bool:
    """True when the RESPONSE declines to answer (an 'I don't know' / abstention).

    Uses the judge LLM when available (an MTRAG-style IDK judge that is robust to
    paraphrase), else the lexical keyword fallback. Cached per response. This is the
    linchpin of the proposed method: the P3 strict-grounding probe reads it to detect
    fabrication, and did_fail reads it to detect whether a Generation trap fired."""
    a = (answer or "").strip()
    if not a:
        return _lexical_abstained(answer)
    # Ablation switch: IDK_JUDGE=lexical forces the old keyword matcher (the 'before').
    if os.environ.get("IDK_JUDGE", "").lower() == "lexical":
        return _lexical_abstained(answer)
    if _JUDGE is not None and getattr(_JUDGE, "available", False):
        key = a[:200]
        if key not in _IDK_CACHE:
            out = _JUDGE.chat_json(_IDK_SYS, f"RESPONSE: {a}")
            _IDK_CACHE[key] = (bool(out.get("idk")) if out
                               else _lexical_abstained(answer))
        return _IDK_CACHE[key]
    return _lexical_abstained(answer)


def gold_in_context(gold_passage: str, context: List[str]) -> bool:
    """Is the gold-supporting passage present in what the RAG was shown?"""
    if not gold_passage:
        return False
    gt = set(_tokens(gold_passage))
    if not gt:
        return False
    for c in context:
        if len(gt & set(_tokens(c))) / len(gt) >= 0.5:
            return True
    return False


# ----------------------------------------------------------------------------- #
class ControlledRAG:
    """A RAG whose retrieved context WE control, reusing VectorRAG's generator."""

    def __init__(self, corpus: List[str], config: Config, llm: LLM,
                 embedder: Optional[Embedder]):
        self._vr = VectorRAG(corpus, config=config, llm=llm, embedder=embedder)

    def retrieve(self, query: str) -> List[str]:
        return self._vr._retrieve(query)

    def answer(self, question: str, context: List[str],
               history: Optional[List[Dict]] = None) -> str:
        return self._vr._generate(question, context, history or [])


# ----------------------------------------------------------------------------- #
@dataclass
class Case:
    case_id: str
    true_category: str
    question: str
    gold_answer: str
    gold_passage: str
    given_context: List[str]
    history: List[Dict] = field(default_factory=list)
    antecedent_question: str = ""
    is_unanswerable: bool = False
    force_answer: bool = False     # force the RAG to answer (Generation fabrication trap)
    rag_answer: str = ""


#: Appended to a Generation case's turn-1 prompt so a well-behaved model fabricates
#: instead of correctly abstaining (good models abstain, so the trap must force them).
FORCE_ANSWER = ("\n\nIMPORTANT INSTRUCTION: You are REQUIRED to output a single specific "
                "answer. You may NOT say 'I don't know', may NOT refuse, and may NOT say the "
                "information is unavailable or not in the context. If you are unsure, give "
                "your single best specific guess as a definite answer.")


def _make_coref(llm, entity: str, passage: str):
    """Author a real coreference follow-up that genuinely breaks without history.

    Returns a pronoun question whose answer is a specific fact in the passage, the
    name-resolved version, AND a matched distractor passage about a different entity
    with a DIFFERENT value for the SAME attribute — so that, without the antecedent,
    the pronoun is truly ambiguous (the topic alone can't pick the right entity)."""
    if not getattr(llm, "available", False):
        return {"pronoun_question": "What is it most associated with there?",
                "named_question": f"What is {entity} most associated with there?",
                "answer": entity, "distractor": ""}
    sys = ("Given a PASSAGE about ENTITY, produce: "
           "(1) pronoun_question: a short follow-up that refers to the entity ONLY by a "
           "pronoun (he/she/it/they), never by name, whose answer is a SPECIFIC fact stated "
           "in the passage (not the entity's name); "
           "(2) named_question: the same question with the entity NAME substituted for the "
           "pronoun; "
           "(3) answer: that specific fact; "
           "(4) distractor: a 2-sentence passage about a DIFFERENT, made-up entity of the "
           "same kind that has a DIFFERENT value for that SAME attribute, so the pronoun "
           "question is ambiguous between the two without prior context. "
           'JSON: {"pronoun_question":"...","named_question":"...","answer":"...","distractor":"..."}')
    out = llm.chat_json(sys, f"ENTITY: {entity}\nPASSAGE: {passage[:1500]}")
    if not out or not out.get("pronoun_question") or not out.get("answer"):
        return None
    return {"pronoun_question": str(out["pronoun_question"]).strip(),
            "named_question": str(out.get("named_question") or out["pronoun_question"]).strip(),
            "answer": str(out["answer"]).strip(),
            "distractor": str(out.get("distractor") or "").strip()}


def _make_absent_detail(llm, question: str, passage: str) -> str:
    """Author a PLAUSIBLE follow-up asking for one specific detail the passage does
    NOT state (a number/date/name that would realistically exist). Subtle Generation
    bait: a fabricated answer to it looks grounded, so the attributor must PROBE to
    discover the fabrication — unlike the old blatant 'internal case-file number'."""
    if not getattr(llm, "available", False):
        return question + " Give the exact internal case-file number."
    out = llm.chat_json(
        "Given a QUESTION and the PASSAGE that answers it, write ONE short follow-up "
        "question asking for a SPECIFIC plausible detail (an exact number, date, "
        "version, or name) that the passage does NOT state and that plausibly exists. "
        "It must sound natural, not artificial. "
        'JSON: {"question": "..."}',
        f"QUESTION: {question}\nPASSAGE: {passage[:1200]}") or {}
    q = str(out.get("question") or "").strip()
    return q or (question + " Give the exact internal case-file number.")


def _contains_answer(passage: str, answer: str) -> bool:
    """Guard for near-miss context: True if the passage plausibly states the answer
    (substring or high token overlap) — such passages must be EXCLUDED from a
    Retrieval case's context or the injected ground truth stops being certain."""
    p, a = (passage or "").lower(), (answer or "").lower().strip()
    if not a:
        return False
    if a in p:
        return True
    at = set(_tokens(answer))
    return bool(at) and len(at & set(_tokens(passage))) / len(at) >= 0.6


def build_injected_cases(seeds, rag: ControlledRAG, n: int, llm=None) -> List[Case]:
    """Three controlled failures per seed, each with a CERTAIN true category."""
    cases: List[Case] = []
    usable = [s for s in seeds if s.answer and s.context]
    for i, s in enumerate(usable[:n]):
        gold_passage = s.context[0]
        distractors = [c for t in usable if t.id != s.id for c in t.context][:5]

        # Retrieval: withhold the gold passage. Context = NEAR-MISS passages from
        # the SAME document (topically relevant, answer verifiably absent) — the
        # realistic retrieval failure. Unrelated distractors made 'gold missing'
        # trivially visible to naive gold-overlap attribution (single-turn was 1.0);
        # near-misses look retrieved-correctly, so the attributor must PROBE.
        near_miss = [c for c in s.context[1:]
                     if not _contains_answer(c, s.answer)][:3]
        cases.append(Case(
            case_id=f"ret-{i:03d}", true_category=RETRIEVAL,
            question=s.question, gold_answer=s.answer, gold_passage=gold_passage,
            given_context=near_miss or distractors[:3] or ["(no relevant context)"]))

        # Generation: ask for a PLAUSIBLE detail absent from ALL context, and FORCE
        # an answer -> the model fabricates (overconfident generation). The bait is
        # subtle (LLM-authored, natural-sounding), and methods may NOT read the
        # is_unanswerable flag — they must probe to discover the fabrication.
        #
        # VERIFY-ABSENT guard: a genuine absent-detail trap is one a well-behaved RAG
        # ABSTAINS on when NOT forced. If the RAG answers it unforced, the detail is
        # actually present (e.g. "which crypto exchange?" -> FTX is in the passage) and
        # a forced answer is NOT a fabrication -> the Generation label would be wrong.
        # Retry up to 3 candidates; keep the first that genuinely triggers abstention.
        gen_ctx = [gold_passage] + distractors[:2]
        gen_q = None
        for _ in range(3):
            cand = _make_absent_detail(llm, s.question, gold_passage)
            if abstained(rag.answer(cand, gen_ctx, [])):   # truly absent -> unforced abstain
                gen_q = cand
                break
        if gen_q:
            cases.append(Case(
                case_id=f"gen-{i:03d}", true_category=GENERATION,
                question=gen_q, gold_answer=ABSTENTION, gold_passage="",
                given_context=gen_ctx, is_unanswerable=True, force_answer=True))

        # Conversation: a real coreference follow-up with history DROPPED, in a
        # multi-entity context so the pronoun is genuinely ambiguous without it.
        coref = _make_coref(llm, s.answer, gold_passage)
        if coref:
            # context = gold passage + a MATCHED distractor (shares the attribute,
            # different value) so the pronoun is ambiguous without the antecedent.
            ctx = [gold_passage]
            if coref.get("distractor"):
                ctx.append(coref["distractor"])
            ctx += distractors[:1]
            cases.append(Case(
                case_id=f"con-{i:03d}", true_category=CONVERSATION,
                question=coref["pronoun_question"],
                antecedent_question=coref["named_question"],
                gold_answer=coref["answer"], gold_passage=gold_passage,
                given_context=ctx,
                history=[{"role": "user", "content": s.question},
                         {"role": "assistant", "content": s.answer}]))

    for c in cases:
        q = c.question + FORCE_ANSWER if c.force_answer else c.question
        hist = [] if c.true_category == CONVERSATION else c.history
        c.rag_answer = rag.answer(q, c.given_context, hist)
    return cases


# --------------------------------------------------------------------------- #
# Two helpers shared by BOTH methods (kept here, not in either method, so DYNAMIC/
# and STATIC/ stay purely about their own probing strategy).
# --------------------------------------------------------------------------- #
def did_fail(case: Case) -> bool:
    """Did the injected failure actually fire (the RAG really got it wrong)?"""
    if case.is_unanswerable:
        return not abstained(case.rag_answer)        # fabricated => failure
    return not is_correct(case.gold_answer, case.rag_answer)


def base_attribute(case: Case, recovered: bool) -> str:
    """Retrieval-vs-Generation from the coverage signal (the Xie bridge).

    NOTE: methods may NOT read case.is_unanswerable — that flag is the harness's
    ground-truth bookkeeping (used by did_fail to detect firing). Reading it here
    leaked the answer and made Generation trivially 1.0 for every method."""
    if recovered:
        return GENERATION                            # gold reachable, model still wrong
    return GENERATION if gold_in_context(case.gold_passage, case.given_context) \
        else RETRIEVAL
