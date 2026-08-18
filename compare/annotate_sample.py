"""
compare/annotate_sample.py — build the human-annotation review file (the DEPENDENT
VARIABLE stage).

The failure classifier already assigned a PREDICTED failure type to every failed turn
(the ``*_failuretypes.json`` files). To measure attribution *accuracy* — the thesis DV —
we need a human GROUND-TRUTH label for a sample of those turns, then compare predicted
vs true (see ``compare/annotate_score.py``).

This script samples failed turns (``outcome`` in {wrong, hallucinated}), STRATIFIED by
method so each method is represented, joins each turn with everything a human needs to
judge it — question, gold, RAG answer, and BOTH document sets (question-generation vs
RAG-retrieved, with real doc_id/rank/score) — and writes a single review file:

    compare/result/main/annotation/review_todo.json

The annotator opens that file and fills ``true_failure_type`` for each case (leaving the
rest untouched). ``true_layer`` is DERIVED from the type by the scorer, so you only pick
the type. The taxonomy legend is embedded under ``_guide`` so definitions are inline.

Nothing here modifies any conversation, classifier output, or the ``failure/`` package —
it only READS them and writes a new review file.

Usage:
    python -m compare.annotate_sample                 # 25 failed turns per method, seed 42
    python -m compare.annotate_sample --per-method 40 --seed 7
    python -m compare.annotate_sample --root compare/result/main --doc-chars 400
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional

import failure.taxonomy as tax

FAILED_OUTCOMES = {"wrong", "hallucinated"}


def _guide_block() -> Dict:
    """The six-type taxonomy legend, pulled live from failure/taxonomy.py so the review
    file always matches the classifier's actual label space."""
    types = []
    for key, defn in tax.FAILURE_TYPES_BY_KEY.items():
        layer = getattr(defn, "layer", None)
        types.append({
            "failure_type": key,
            "layer": getattr(layer, "value", str(layer)),
            "definition": (getattr(defn, "description", None)
                           or getattr(defn, "definition", "") or "").strip(),
        })
    return {
        "how_to_annotate": (
            "For each case in 'cases', read question / gold / rag_answer and the two "
            "document sets, decide which ONE failure_type actually occurred, and write it "
            "into 'true_failure_type'. Use exactly one of the keys listed in "
            "'failure_types' below. If the turn is NOT actually a failure (the classifier "
            "over-fired and the RAG answer is fine), write 'not_a_failure'. Leave every "
            "other field unchanged. Optionally add 'annotator_notes'. Then run "
            "`python -m compare.annotate_score`."
        ),
        "tip_retrieval_vs_generation": (
            "The key discriminator: compare rag_retrieved_documents against the gold. If "
            "the chunk that supports the gold is ABSENT from what the RAG retrieved, it is "
            "a retrieval-layer failure (retrieval / chunking / context_selection / "
            "knowledge_boundary). If the supporting chunk IS present but the answer is "
            "still wrong/incomplete, it is a generation-layer failure (grounding / "
            "response_coverage)."
        ),
        "failure_types": types,
        "special_values": ["not_a_failure"],
    }


def _trim_docs(docs: Optional[List[Dict]], doc_chars: int) -> List[Dict]:
    """Keep the real metadata (doc_id/rank/score/score_type) but truncate long text so the
    review file stays readable."""
    out = []
    for d in (docs or []):
        text = (d.get("text") or "")
        out.append({
            "doc_id": d.get("doc_id"),
            "rank": d.get("rank"),
            "score": d.get("score"),
            "score_type": d.get("score_type"),
            "text": text[:doc_chars] + ("…" if len(text) > doc_chars else ""),
        })
    return out


def _parse_path(conv_path: str, root: str):
    """(method, rag, dataset, conversation_dir) from a result path. Mirrors classify_all:
    <root>/<method>/<rag>/<dataset>/conversation_NNN/conversation.json."""
    rel = os.path.relpath(conv_path, root).replace("\\", "/").split("/")
    # rel = [method, rag, dataset, conversation_NNN, conversation.json]
    if len(rel) < 5:
        return None
    return rel[0], rel[1], rel[2], rel[3]


def collect_cases(root: str, doc_chars: int) -> List[Dict]:
    """Every failed turn under root, joined with its classifier prediction + evidence."""
    records: List[Dict] = []
    for conv_path in glob.glob(os.path.join(root, "**", "conversation_*",
                                            "conversation.json"), recursive=True):
        ft_path = conv_path[:-len(".json")] + "_failuretypes.json"
        # actual name is conversation_failuretypes.json (sibling), not conversation_....
        ft_path = os.path.join(os.path.dirname(conv_path),
                               "conversation_failuretypes.json")
        parsed = _parse_path(conv_path, root)
        if not parsed:
            continue
        method, rag, dataset, conv_dir = parsed
        try:
            conv = json.load(open(conv_path, encoding="utf-8"))
        except Exception:
            continue
        # predicted labels keyed by turn_id (may be absent if not classified yet)
        pred_by_turn: Dict[int, Dict] = {}
        if os.path.exists(ft_path):
            try:
                ft = json.load(open(ft_path, encoding="utf-8"))
                for c in ft.get("cases", []):
                    pred_by_turn[c.get("turn_id")] = c
            except Exception:
                pass
        for conversation in conv.get("conversations", []):
            for turn in conversation.get("turns", []):
                if turn.get("outcome") not in FAILED_OUTCOMES:
                    continue
                tid = turn.get("turn_id")
                pred = pred_by_turn.get(tid, {})
                records.append({
                    "id": f"{method}|{rag}|{dataset}|{conv_dir}|turn_{tid}",
                    "method": method, "rag": rag, "dataset": dataset,
                    "conversation": conv_dir, "turn_id": tid,
                    "query_type": turn.get("query_type"),
                    "outcome": turn.get("outcome"),
                    "is_unanswerable": turn.get("is_unanswerable", False),
                    "question": turn.get("question"),
                    "gold": turn.get("gold"),
                    "rag_answer": turn.get("rag_answer"),
                    "question_generation_documents": _trim_docs(
                        turn.get("question_generation_documents"), doc_chars),
                    "rag_retrieved_documents": _trim_docs(
                        turn.get("rag_retrieved_documents"), doc_chars),
                    "classifier_predicted": {
                        "failure_type": pred.get("failure_type"),
                        "layer": pred.get("layer"),
                        "conversational_cause": pred.get("conversational_cause"),
                        "rationale": pred.get("rationale"),
                    },
                    # ---- FILL THIS (leave "" until you decide) ----
                    "true_failure_type": "",
                    "annotator_notes": "",
                })
    return records


def stratified_sample(records: List[Dict], per_method: int, seed: int) -> List[Dict]:
    """Up to `per_method` failed turns per method, chosen deterministically by `seed`."""
    by_method: Dict[str, List[Dict]] = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)
    rng = random.Random(seed)
    out: List[Dict] = []
    for method in sorted(by_method):
        pool = by_method[method]
        rng.shuffle(pool)
        out.extend(pool[:per_method])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the human-annotation review file.")
    ap.add_argument("--root", default=os.path.join("compare", "result", "main"),
                    help="results root to scan (default compare/result/main)")
    ap.add_argument("--per-method", type=int, default=25,
                    help="max failed turns to sample per method (default 25)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    ap.add_argument("--doc-chars", type=int, default=400,
                    help="truncate each document's text to this many chars (default 400)")
    ap.add_argument("--out", default=None,
                    help="output review file (default <root>/annotation/review_todo.json)")
    ap.add_argument("--merge", action="store_true",
                    help="if the output already exists, PRESERVE any true_failure_type you "
                         "already filled (re-sample only adds new cases; never clobbers labels)")
    args = ap.parse_args(argv)

    out_path = args.out or os.path.join(args.root, "annotation", "review_todo.json")

    all_records = collect_cases(args.root, args.doc_chars)
    if not all_records:
        print(f"No failed turns found under {args.root}. "
              "(Did the runs + classifier complete?)")
        return
    sample = stratified_sample(all_records, args.per_method, args.seed)

    # preserve already-entered labels on re-run
    if args.merge and os.path.exists(out_path):
        try:
            prev = json.load(open(out_path, encoding="utf-8"))
            filled = {c["id"]: c for c in prev.get("cases", [])
                      if (c.get("true_failure_type") or "").strip()}
            for c in sample:
                if c["id"] in filled:
                    c["true_failure_type"] = filled[c["id"]]["true_failure_type"]
                    c["annotator_notes"] = filled[c["id"]].get("annotator_notes", "")
            kept = sum(1 for c in sample if (c.get("true_failure_type") or "").strip())
            print(f"  merged: preserved {kept} existing label(s)")
        except Exception as exc:
            print(f"  (merge skipped: {exc})")

    payload = {"_guide": _guide_block(),
               "meta": {"root": args.root, "per_method": args.per_method,
                        "seed": args.seed, "total_failed_turns": len(all_records),
                        "sampled": len(sample)},
               "cases": sample}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)

    # per-method counts so you know the coverage
    counts: Dict[str, int] = {}
    for c in sample:
        counts[c["method"]] = counts.get(c["method"], 0) + 1
    print(f"\nwrote {len(sample)} cases to {out_path}")
    print(f"  (from {len(all_records)} total failed turns)")
    for m in sorted(counts):
        print(f"    {m:16} {counts[m]:3d} cases to annotate")
    print("\nNext: open the file, fill each 'true_failure_type', then run:")
    print("    python -m compare.annotate_score")


if __name__ == "__main__":
    main()
