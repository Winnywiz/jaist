"""Rule-based failure attribution for DECOMPOSABLE question types.

Comparative and Multi-Hop questions declare `expected_components` — the answer slots a
complete correct answer must cover (e.g. ['A accuracy','A cost','B accuracy','B cost']).
For a FAILED turn of these types we can attribute the failure by RULE instead of asking
the LLM classifier:

  - some required slots are MISSING from the answer  -> response_coverage (an answerable
    part of the request was omitted)
  - all slots are present but the turn still failed   -> grounding (the values are wrong /
    fabricated, i.e. the answer covers the slots but gets them wrong)

This is objective and reproducible (no LLM judgment), which is exactly why decomposable
question types are valuable: the answer's STRUCTURE reveals the failure. Coverage is
checked by token overlap — a coarse but deterministic rule.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

_STOP = frozenset("the a an of for to in on and or is are was were with by as at from "
                  "that this its their it they he she".split())


def _tokens(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower())
            if w not in _STOP}


def _sentence_token_sets(text: str) -> List[set]:
    """Split into sentences and tokenize each — coverage is checked PER SENTENCE so a
    slot's entity and its dimension must co-occur (not just appear somewhere separately)."""
    return [_tokens(s) for s in re.split(r"[.!?\n;]+", text or "") if s.strip()]


def slot_covered(slot: str, sentence_tokens: List[set],
                 ent_thresh: float = 1.0, dim_thresh: float = 0.5,
                 plain_thresh: float = 0.6) -> bool:
    """Covered iff SOME single sentence satisfies the slot.

    Structured slot 'ENTITY :: dimension' (Comparative): the sentence must contain the
    ENTITY (gate — all entity tokens present) AND enough of the dimension tokens. Gating
    on the entity stops a dimension mentioned for a DIFFERENT entity from counting.

    Plain slot (Multi-Hop hops, distinctive names): sentence token overlap >= plain_thresh.
    """
    if "::" in slot:
        ent, dim = slot.split("::", 1)
        et, dt = _tokens(ent), _tokens(dim)
        if not et:
            return False
        for sent in sentence_tokens:
            ent_ok = len(et & sent) / len(et) >= ent_thresh
            dim_ok = (not dt) or len(dt & sent) / len(dt) >= dim_thresh
            if ent_ok and dim_ok:
                return True
        return False
    st = _tokens(slot)
    return bool(st) and any(len(st & sent) / len(st) >= plain_thresh
                            for sent in sentence_tokens)


def attribute_by_slots(components: List[str], answer: str) -> Dict:
    """Rule-based attribution from slot coverage. Returns the covered/missing slots and
    a rule label. Caller should only apply this to FAILED Comparative/Multi-Hop turns."""
    sents = _sentence_token_sets(answer)
    covered = [(s, slot_covered(s, sents)) for s in components]
    missing = [s for s, ok in covered if not ok]
    n_cov = sum(1 for _, ok in covered if ok)
    label = "response_coverage" if missing else "grounding"
    return {
        "rule_label": label,
        "n_slots": len(components),
        "n_covered": n_cov,
        "missing_slots": missing,
        "coverage": round(n_cov / len(components), 2) if components else None,
    }


def report_log(path: str) -> Dict:
    """Run slot-based attribution over every failed Comparative/Multi-Hop turn in a log."""
    d = json.load(open(path, encoding="utf-8"))
    cases: List[Dict] = []
    for conv in d.get("conversations", []):
        for t in conv.get("turns", []):
            comps = t.get("expected_components") or []
            if t.get("query_type") in ("Comparative", "Multi-Hop") \
                    and t.get("outcome") in ("wrong", "hallucinated") and comps:
                res = attribute_by_slots(comps, t.get("rag_answer", ""))
                cases.append({"conversation_id": conv.get("conversation_id"),
                              "turn_id": t.get("turn_id"),
                              "query_type": t.get("query_type"),
                              "question": t.get("question"),
                              "expected_components": comps,
                              "rag_answer": t.get("rag_answer", "")[:200],
                              **res})
    from collections import Counter
    return {"n": len(cases),
            "by_rule_label": dict(Counter(c["rule_label"] for c in cases)),
            "cases": cases}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rule-based slot attribution for "
                                             "Comparative / Multi-Hop failures.")
    ap.add_argument("--file", required=True, help="a benchmark *_full/conversation log JSON")
    a = ap.parse_args()
    rep = report_log(a.file)
    print(f"# {a.file}")
    print(f" decomposable failures with slots : {rep['n']}")
    print(f" by RULE label                    : {rep['by_rule_label']}")
    for c in rep["cases"][:8]:
        print(f"\n[{c['query_type']}] coverage={c['coverage']} -> {c['rule_label']}")
        print(f"  Q       : {c['question'][:90]}")
        print(f"  slots   : {c['expected_components']}")
        print(f"  missing : {c['missing_slots']}")
        print(f"  answer  : {c['rag_answer'][:90]}")
