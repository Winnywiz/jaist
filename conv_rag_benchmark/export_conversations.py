"""
export_conversations.py — human-readable per-turn dump of a generated benchmark.

For every turn it lays out the full provenance chain, so you can trace exactly how each
question/answer was produced and what the tested RAG did with it:

  question_type            the adaptive type chosen for this turn
  question                 the generated question
  doc_to_generate_question the retrieved passage the QUESTION was authored from
                           (turn field `question_evidence`)
  gold                     the reference answer
  doc_for_gold             the evidence the GOLD was composed against (`evidence`)
  rag_answer               what the RAG-under-test replied
  doc_for_rag_answer       the passages the RAG ITSELF retrieved to answer
                           (`rag_retrieved_context`; empty for benchmarks generated
                           before this field was added)
  outcome                  correct / wrong / abstained (the grade)

Writes both a readable .md and a compact .json next to the source file.

Run:
    python -m conv_rag_benchmark.export_conversations \
        --file result/benchmark_quality/qasper/quality_e_strictgold_nonegraph.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List


def _clip(x, n=600):
    if isinstance(x, list):
        x = "\n---\n".join(str(c) for c in x)
    x = (x or "").strip().replace("\r", " ")
    return x[:n] + (" …[truncated]" if len(x) > n else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Readable per-turn export of a generated benchmark")
    ap.add_argument("--file", required=True, help="a quality_e*.json to export")
    ap.add_argument("--maxchars", type=int, default=600, help="clip long passages to N chars")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip embedding similarity (lexical only, no API calls)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"!! not found: {args.file}"); return
    data = json.load(open(args.file, encoding="utf-8"))
    convos = data.get("conversations", [])
    base = os.path.splitext(args.file)[0]

    # set up retrieval-alignment similarity (embeddings + lexical) if RAG docs exist
    from .retrieval_alignment import _build_idf, turn_similarities
    all_turns = [t for c in convos for t in c.get("turns", [])]
    have_docs = any(t.get("rag_retrieved_context") for t in all_turns)
    idf, embedder = {}, None
    if have_docs:
        idf = _build_idf([d for t in all_turns for d in (t.get("rag_retrieved_context") or [])]
                         + [t.get("evidence", "") for t in all_turns])
        if not args.no_embed:
            from .config import Config
            from .embeddings import Embedder
            from .llm import LLM
            cfg = Config.load()
            emb = Embedder(config=cfg, llm=LLM(model=cfg.gen_model, config=cfg))
            embedder = emb if emb.available else None

    def _pct(v):
        return f"{100*v:.0f}%" if isinstance(v, (int, float)) else "n/a"

    rows: List[dict] = []
    md = [f"# Conversation export — {os.path.basename(args.file)}",
          f"\n{len(convos)} conversations, "
          f"{sum(len(c.get('turns', [])) for c in convos)} turns.\n"]
    has_rag_docs = False

    for ci, c in enumerate(convos):
        md.append(f"\n---\n\n## Conversation {ci}\n")
        for t in c.get("turns", []):
            rag_docs = t.get("rag_retrieved_context") or []
            has_rag_docs = has_rag_docs or bool(rag_docs)
            sims = turn_similarities(t, idf, embedder) if rag_docs else None
            row = {
                "conversation": ci,
                "turn": t.get("turn_id"),
                "question_type": t.get("query_type"),
                "question": t.get("question", ""),
                "doc_to_generate_question": t.get("question_evidence", ""),
                "gold": t.get("gold", ""),
                "doc_for_gold": t.get("evidence", ""),
                "rag_answer": t.get("rag_answer", ""),
                "doc_for_rag_answer": rag_docs,
                "outcome": t.get("outcome"),
                "retrieval_alignment": sims,
            }
            rows.append(row)
            md += [
                f"\n### Turn {row['turn']} — `{row['question_type']}`  ·  outcome: **{row['outcome']}**",
                f"\n**Q:** {row['question']}",
                f"\n**Doc retrieved to GENERATE the question:**\n> {_clip(row['doc_to_generate_question'], args.maxchars)}",
                f"\n**Gold answer:** {row['gold']}",
                f"\n**Doc the GOLD was composed from:**\n> {_clip(row['doc_for_gold'], args.maxchars)}",
                f"\n**RAG answer:** {row['rag_answer']}",
                f"\n**Doc the RAG retrieved to ANSWER:**\n> {_clip(row['doc_for_rag_answer'], args.maxchars) or '(not recorded in this run)'}",
            ]
            if sims:
                q, g, a = sims["q_vs_ragdoc"], sims["gold_vs_ragdoc"], sims["ans_vs_ragdoc"]
                md += [
                    "\n**Retrieval alignment of the RAG's docs** (embed / lexical):",
                    f"\n- question ↔ retrieved doc : **{_pct(q['embed'])}** / {_pct(q['lex'])}   (is retrieval relevant to the query?)",
                    f"\n- gold doc ↔ retrieved doc : **{_pct(g['embed'])}** / {_pct(g['lex'])}   rank {g['rank']}  ·  got the right doc: **{g['got_gold']}**",
                    f"\n- RAG answer ↔ retrieved doc : **{_pct(a['embed'])}** / {_pct(a['lex'])}   (is the answer grounded in what it retrieved?)",
                ]
            md.append("")

    md_path, json_path = base + "_export.md", base + "_export.json"
    open(md_path, "w", encoding="utf-8").write("\n".join(md))
    json.dump({"source": args.file, "n_turns": len(rows),
               "rag_docs_recorded": has_rag_docs, "turns": rows},
              open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"exported {len(rows)} turns")
    print(f"  readable -> {md_path}")
    print(f"  json     -> {json_path}")
    if not has_rag_docs:
        print("\n  NOTE: 'doc_for_rag_answer' is empty — this benchmark was generated BEFORE\n"
              "  the RAG-retrieved-context field was added. Regenerate to populate it.")
    return json_path


if __name__ == "__main__":
    main()
