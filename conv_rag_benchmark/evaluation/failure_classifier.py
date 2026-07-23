"""
Module 8 — failure classifier (LLM-as-a-judge, multi-label).

Given one evaluated turn, decide whether the RAG succeeded and, if not, attach one
or more failure labels from the taxonomy. The classifier combines:

  * **a deterministic retrieval-attribution pass** — was the gold-supporting
    evidence actually in the RAG's retrieved context? This cleanly separates
    *retrieval* failures from *generation* failures and is computable offline.
  * **an LLM judge pass** — grades correctness and assigns conversation/reasoning/
    generation labels (the subjective ones), with a confidence and an explanation.

When no LLM is available it degrades to a rule-based classifier driven by the
turn's ``query_type`` / ``expected_failure`` plus the retrieval attribution, so the
pipeline still produces labelled output.

Input dict (one turn)::

    {question, query_type, gold_answer, retrieved_evidence, rag_answer,
     is_unanswerable, expected_failure, history}

Output :class:`Diagnosis`::

    {success: bool, failure_types: [str], confidence: float, explanation: str}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, get_logger
from ..generation.gold_answer_generator import ABSTENTION
from ..llm import LLM
from .failure_taxonomy import ALL_FAILURE_TYPES, FailureType, category_of

#: A turn is effectively unanswerable whenever its gold is the abstention sentinel,
#: regardless of its query_type. (query_generator only flags type=="Unanswerable",
#: so e.g. a Topic-Shift turn that lands on "abstain" would otherwise be mis-graded
#: as a wrong answer when the RAG correctly says "I don't know".)
def _gold_is_abstention(gold: str) -> bool:
    return bool(gold) and gold.strip().startswith(ABSTENTION[:20])

logger = get_logger("evaluation.classifier")

_ABSTAIN_PAT = re.compile(
    r"\b(i\s+don'?t\s+know|cannot\s+be\s+determined|not\s+(stated|mentioned|"
    r"available|enough|specified)|no\s+information|unable\s+to|insufficient)\b",
    re.IGNORECASE,
)


@dataclass
class Diagnosis:
    """Result of classifying one turn."""

    success: bool
    failure_types: List[str] = field(default_factory=list)
    confidence: float = 0.5
    explanation: str = ""
    correct: Optional[bool] = None
    gold_retrieved: Optional[bool] = None
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "failure_types": self.failure_types,
            "categories": self.categories,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "correct": self.correct,
            "gold_retrieved": self.gold_retrieved,
        }


class FailureClassifier:
    """Diagnoses a turn into success + multi-label failure types."""

    _JUDGE_SYS = (
        "You are an expert evaluator of conversational RAG systems. You are given a "
        "QUESTION (with its query TYPE and the CONVERSATION), the grounded GOLD "
        "answer with its EVIDENCE, and the RAG's ANSWER plus what it RETRIEVED. "
        "Decide if the RAG answer is correct, and assign zero or more FAILURE LABELS "
        "from the allowed list describing WHY it is wrong. Multiple labels are "
        "allowed. Be precise and only label failures you can justify from the inputs.\n"
        "Allowed failure labels: {labels}.\n"
        'Respond as JSON: {{"correct": true/false, "failure_types": ["..."], '
        '"confidence": 0.0-1.0, "explanation": "one sentence"}}'
    )

    def __init__(self, config: Optional[Config] = None, llm: Optional[LLM] = None):
        self.config = config or Config.load()
        # the JUDGE uses the (separate) judge model to avoid self-evaluation bias
        self.llm = llm or LLM(model=self.config.judge_model, config=self.config)

    # -- public ------------------------------------------------------------- #
    def classify(self, turn: Dict) -> Diagnosis:
        """Classify one evaluated turn (see module docstring for the schema)."""
        gold = (turn.get("gold_answer") or "").strip()
        rag = (turn.get("rag_answer") or "").strip()
        context = turn.get("retrieved_evidence") or turn.get("retrieved_context") or []
        # derive unanswerability from the gold too, not just the type flag, so a
        # correct abstention is never counted as a failure (see _gold_is_abstention)
        is_unanswerable = bool(turn.get("is_unanswerable")) or _gold_is_abstention(gold)
        turn["is_unanswerable"] = is_unanswerable  # propagate to the rule path

        # 1. retrieval attribution (deterministic, always available)
        gold_retrieved = self._gold_in_context(gold, context, is_unanswerable)

        # 2. correctness + labels
        if self.llm.available:
            diag = self._llm_classify(turn, gold_retrieved)
        else:
            diag = self._rule_classify(turn, gold, rag, gold_retrieved)

        diag.gold_retrieved = gold_retrieved
        diag.categories = sorted({category_of(f) for f in diag.failure_types})
        return diag

    def classify_all(self, turns: List[Dict]) -> List[Diagnosis]:
        return [self.classify(t) for t in turns]

    # -- LLM path ----------------------------------------------------------- #
    def _llm_classify(self, turn: Dict, gold_retrieved: bool) -> Diagnosis:
        ctx = turn.get("retrieved_evidence") or turn.get("retrieved_context") or []
        ctx_text = "\n".join(f"[{i+1}] {str(c)[:400]}" for i, c in enumerate(ctx))
        hist = turn.get("history") or []
        hist_text = "\n".join(f"{t['role']}: {t['content']}" for t in hist[-6:])
        user = (
            f"QUERY TYPE: {turn.get('query_type')}\n"
            f"EXPECTED FAILURE (the type this turn probes): {turn.get('expected_failure')}\n"
            f"CONVERSATION:\n{hist_text}\n\n"
            f"QUESTION: {turn.get('question')}\n"
            f"GOLD ANSWER: {turn.get('gold_answer')}\n"
            f"GOLD EVIDENCE: {turn.get('gold_evidence', '')[:800]}\n\n"
            f"RAG ANSWER: {turn.get('rag_answer')}\n"
            f"RAG RETRIEVED:\n{ctx_text}\n\n"
            f"(Note: gold-supporting evidence was "
            f"{'PRESENT' if gold_retrieved else 'ABSENT'} in what the RAG retrieved.)"
        )
        out = self.llm.chat_json(
            self._JUDGE_SYS.format(labels=", ".join(ALL_FAILURE_TYPES)), user)
        if out is None:
            return self._rule_classify(turn, turn.get("gold_answer", ""),
                                       turn.get("rag_answer", ""), gold_retrieved)
        correct = bool(out.get("correct", False))
        labels = [l for l in (out.get("failure_types") or []) if l in ALL_FAILURE_TYPES]
        if not correct and not labels:
            # ensure a wrong answer always carries at least a retrieval-attributed label
            labels = [self._retrieval_label(gold_retrieved)]
        labels = self._dedup(labels)
        return Diagnosis(
            success=correct and not labels,
            failure_types=[] if correct else labels,
            confidence=float(out.get("confidence", 0.6) or 0.6),
            explanation=str(out.get("explanation", "")).strip(),
            correct=correct,
        )

    # -- offline rule path -------------------------------------------------- #
    def _rule_classify(self, turn: Dict, gold: str, rag: str,
                       gold_retrieved: bool) -> Diagnosis:
        is_unanswerable = bool(turn.get("is_unanswerable"))
        abstained = bool(_ABSTAIN_PAT.search(rag))

        if is_unanswerable:
            # success = the RAG abstained; failure = it answered confidently
            if abstained:
                return Diagnosis(True, [], 0.7, "Correctly abstained on an "
                                 "unanswerable question.", correct=True)
            return Diagnosis(False, [FailureType.OVERCONFIDENT_UNKNOWN.value,
                                     FailureType.HALLUCINATION.value], 0.7,
                             "Fabricated an answer to an unanswerable question.",
                             correct=False)

        correct = self._lexical_correct(gold, rag)
        if correct:
            return Diagnosis(True, [], 0.6, "Answer matches the gold.", correct=True)

        # wrong answer: attribute. Start from retrieval, then add the type's expected.
        labels = [self._retrieval_label(gold_retrieved)]
        expected = turn.get("expected_failure")
        if expected in ALL_FAILURE_TYPES and expected not in labels:
            labels.append(expected)
        if abstained and gold_retrieved:
            labels = [FailureType.INCOMPLETE_ANSWER.value]
        return Diagnosis(False, self._dedup(labels), 0.5,
                         "Answer does not match the gold; "
                         f"gold evidence was {'present' if gold_retrieved else 'absent'} "
                         "in retrieval.", correct=False)

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _retrieval_label(gold_retrieved: bool) -> str:
        return (FailureType.HALLUCINATION.value if gold_retrieved
                else FailureType.MISSING_RETRIEVAL.value)

    def _gold_in_context(self, gold: str, context: List[str],
                         is_unanswerable: bool) -> bool:
        if is_unanswerable:
            return True  # nothing to retrieve; treat as satisfied
        if not gold or not context:
            return False
        if self.llm.available:
            ctx = "\n".join(f"[{i+1}] {str(c)[:400]}" for i, c in enumerate(context))
            out = self.llm.chat_json(
                "Given CONTEXT passages and a GOLD answer, is the information needed "
                'to produce the GOLD present in the CONTEXT? JSON: {"present": true/false}',
                f"CONTEXT:\n{ctx}\n\nGOLD: {gold}")
            if out is not None:
                return bool(out.get("present", False))
        # lexical fallback: meaningful token overlap of the gold with any chunk
        gt = set(re.findall(r"[a-z0-9]+", gold.lower()))
        gt = {t for t in gt if len(t) > 2}
        if not gt:
            return False
        for c in context:
            ct = set(re.findall(r"[a-z0-9]+", str(c).lower()))
            if len(gt & ct) / len(gt) >= 0.5:
                return True
        return False

    @staticmethod
    def _lexical_correct(gold: str, rag: str) -> bool:
        if not rag:
            return False
        g, a = gold.lower().strip(), rag.lower().strip()
        if g == a or (len(g) > 3 and (g in a or a in g)):
            return True
        gt = {t for t in re.findall(r"[a-z0-9]+", g) if len(t) > 2}
        at = set(re.findall(r"[a-z0-9]+", a))
        return bool(gt) and len(gt & at) / len(gt) >= 0.6

    @staticmethod
    def _dedup(labels: List[str]) -> List[str]:
        seen, out = set(), []
        for l in labels:
            if l and l not in seen:
                seen.add(l)
                out.append(l)
        return out
