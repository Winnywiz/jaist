"""
retrieval_alignment.py — did the RAG retrieve the doc the answer actually lives in?

For every turn it compares the passages the RAG-UNDER-TEST retrieved (``rag_retrieved_context``)
against the GOLD evidence (the passage the question/gold was authored from) and reports, as
percentages, how well the RAG's retrieval aligned with the source of truth:

  best_embed_sim   max cosine similarity (OpenAI embeddings) between the gold evidence and
                   any RAG-retrieved passage  -> semantic alignment
  best_lex_sim     max lexical overlap (BM25-flavoured token overlap) -> surface alignment
  best_rank        the position (1 = top) of that best-matching passage in the RAG's list
  gold_retrieved   True if best_embed_sim >= threshold (the RAG surfaced the right doc)

Aggregated over the conversation you get, in effect, a retrieval-quality score for the
tested RAG: mean alignment %, and recall = fraction of turns where it retrieved the gold doc.

Needs a benchmark generated AFTER `rag_retrieved_context` was added (regenerate if the field
is empty). Embedding sims need an OpenAI key; lexical sims always work.

Run:
    python -m conv_rag_benchmark.retrieval_alignment \
        --file result/benchmark_quality/MultiHopRAG/quality_e_strictgold_nonegraph.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from statistics import mean
from typing import Dict, List

from .config import Config
from .embeddings import Embedder
from .llm import LLM

_TOK = re.compile(r"[a-z0-9]+")
EMBED_MATCH = 0.55          # cosine >= this => the gold doc was effectively retrieved


def _tokens(s: str) -> List[str]:
    return _TOK.findall((s or "").lower())


def _lexical_sim(a: str, b: str, idf: Dict[str, float]) -> float:
    """IDF-weighted overlap coefficient (BM25-flavoured), in [0,1]."""
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    if not ta or not tb:
        return 0.0
    shared = set(ta) & set(tb)
    num = sum(idf.get(t, 1.0) for t in shared)
    den = min(sum(idf.get(t, 1.0) for t in ta), sum(idf.get(t, 1.0) for t in tb))
    return round(num / den, 3) if den else 0.0


def _build_idf(docs: List[str]) -> Dict[str, float]:
    n = len(docs) or 1
    df: Counter = Counter()
    for d in docs:
        for t in set(_tokens(d)):
            df[t] += 1
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}


def _cos(a, b) -> float:
    na = (sum(a * a) ** 0.5); nb = (sum(b * b) ** 0.5)
    return float(a @ b / (na * nb + 1e-9))


def turn_similarities(turn: Dict, idf: Dict[str, float], embedder=None) -> Dict:
    """All per-turn retrieval-alignment numbers, reused by the metric and the export.

    Against the docs the RAG-under-test retrieved (``rag_retrieved_context``), best sim of:
      q_vs_ragdoc     the QUESTION  -> is the RAG's retrieval relevant to the query?
      gold_vs_ragdoc  the GOLD doc  -> did the RAG surface the right source? (+rank, +got_gold)
      ans_vs_ragdoc   the RAG ANSWER-> is the answer grounded in what the RAG retrieved?
    Each as {embed, lex} percentages in [0,1]. Returns None if no RAG docs recorded.
    """
    docs = turn.get("rag_retrieved_context") or []
    if not docs:
        return None
    q = turn.get("question", "") or ""
    gold = turn.get("evidence", "") or turn.get("question_evidence", "") or ""
    ans = turn.get("rag_answer", "") or ""

    lex = {"q": [_lexical_sim(q, d, idf) for d in docs],
           "gold": [_lexical_sim(gold, d, idf) for d in docs],
           "ans": [_lexical_sim(ans, d, idf) for d in docs]}
    emb = {"q": None, "gold": None, "ans": None}
    if embedder is not None:
        vecs = embedder.encode([q, gold, ans] + docs)
        if vecs is not None and len(vecs) == len(docs) + 3:
            dv = vecs[3:]
            for i, key in enumerate(("q", "gold", "ans")):
                if (vecs[i] is not None) and any(vecs[i]):
                    emb[key] = [round(_cos(vecs[i], dv[j]), 3) for j in range(len(docs))]

    def pack(key):
        e = emb[key]; l = lex[key]
        best_l = max(l) if l else 0.0
        best_e = max(e) if e else None
        return {"embed": best_e, "lex": round(best_l, 3),
                "rank": ((e.index(best_e) if e else l.index(best_l)) + 1) if (e or l) else None}

    out = {"q_vs_ragdoc": pack("q"), "gold_vs_ragdoc": pack("gold"),
           "ans_vs_ragdoc": pack("ans"), "n_rag_docs": len(docs)}
    ge = out["gold_vs_ragdoc"]["embed"]
    out["gold_vs_ragdoc"]["got_gold"] = (ge is not None and ge >= EMBED_MATCH)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="RAG retrieval alignment vs gold evidence, per turn")
    ap.add_argument("--file", required=True)
    ap.add_argument("--no-embed", action="store_true", help="lexical only (no API calls)")
    ap.add_argument("--show", type=int, default=12, help="how many per-turn rows to print")
    args = ap.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"!! not found: {args.file}"); return
    data = json.load(open(args.file, encoding="utf-8"))
    turns = [t for c in data.get("conversations", []) for t in c.get("turns", [])]
    have = [t for t in turns if t.get("rag_retrieved_context")]
    if not have:
        print("!! no turn has 'rag_retrieved_context' — regenerate the benchmark first "
              "(this field was added recently)."); return

    # local IDF over every passage present, for the lexical measure
    alldocs = [d for t in have for d in t["rag_retrieved_context"]] + \
              [t.get("evidence", "") for t in have]
    idf = _build_idf(alldocs)

    config = Config.load()
    embedder = None
    if not args.no_embed:
        llm = LLM(model=config.gen_model, config=config)
        embedder = Embedder(config=config, llm=llm)
        if not embedder.available:
            print("# no embeddings available -> lexical only"); embedder = None

    rows = []
    for t in have:
        gold = t.get("evidence", "") or t.get("question_evidence", "")
        docs = t["rag_retrieved_context"]
        lex = [_lexical_sim(gold, d, idf) for d in docs]
        emb = None
        if embedder is not None and gold.strip():
            vecs = embedder.encode([gold] + docs)
            if vecs is not None and len(vecs) == len(docs) + 1:
                g = vecs[0]
                emb = [round(float(g @ vecs[i + 1] /
                        ((sum(g*g)**0.5)*(sum(vecs[i+1]*vecs[i+1])**0.5) + 1e-9)), 3)
                       for i in range(len(docs))]
        best_lex = max(lex) if lex else 0.0
        best_emb = max(emb) if emb else None
        best_rank = (emb.index(best_emb) if emb else lex.index(best_lex)) + 1
        rows.append({
            "query_type": t.get("query_type"),
            "question": (t.get("question") or "")[:90],
            "n_rag_docs": len(docs),
            "best_embed_sim": best_emb,
            "best_lex_sim": best_lex,
            "best_rank": best_rank,
            "gold_retrieved": (best_emb is not None and best_emb >= EMBED_MATCH),
            "outcome": t.get("outcome"),
        })

    # ---- per-turn table ----
    print(f"# {os.path.basename(args.file)} | {len(rows)} turns with RAG retrieval recorded\n")
    print(f"{'type':<20}{'embed%':>8}{'lex%':>7}{'rank':>6}{'got_gold':>10}  question")
    print("-" * 100)
    for r in rows[: args.show]:
        e = f"{100*r['best_embed_sim']:.0f}" if r['best_embed_sim'] is not None else " - "
        print(f"{str(r['query_type']):<20}{e:>8}{100*r['best_lex_sim']:>6.0f}%"
              f"{r['best_rank']:>6}{str(r['gold_retrieved']):>10}  {r['question']}")
    if len(rows) > args.show:
        print(f"   … {len(rows)-args.show} more")

    # ---- aggregate ----
    embs = [r["best_embed_sim"] for r in rows if r["best_embed_sim"] is not None]
    lexs = [r["best_lex_sim"] for r in rows]
    recall = mean(1.0 if r["gold_retrieved"] else 0.0 for r in rows) if embs else None
    print("\n" + "=" * 60)
    print(" RETRIEVAL ALIGNMENT (RAG-retrieved docs vs gold evidence)")
    print("=" * 60)
    if embs:
        print(f"  mean best embedding similarity : {100*mean(embs):.1f}%")
        print(f"  gold-doc recall (sim>={EMBED_MATCH}) : {100*recall:.0f}%  "
              f"({sum(r['gold_retrieved'] for r in rows)}/{len(rows)} turns)")
    print(f"  mean best lexical  similarity  : {100*mean(lexs):.1f}%")
    # does poor retrieval align with wrong answers?
    bad = [r for r in rows if r["outcome"] in ("wrong", "hallucinated")]
    if bad and embs:
        be = [r["best_embed_sim"] for r in bad if r["best_embed_sim"] is not None]
        if be:
            print(f"  mean embed-sim on FAILED turns : {100*mean(be):.1f}%  "
                  f"(lower => failures track poor retrieval)")

    out = os.path.splitext(args.file)[0] + "_retrieval_alignment.json"
    json.dump({"source": args.file, "n_turns": len(rows),
               "mean_best_embed_sim": round(mean(embs), 3) if embs else None,
               "mean_best_lex_sim": round(mean(lexs), 3),
               "gold_doc_recall": round(recall, 3) if recall is not None else None,
               "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
