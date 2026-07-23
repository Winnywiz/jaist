"""
question_doc_similarity.py — per turn, the TWO evidences that actually exist, each as a
LIST of docs scored by cosine similarity to the question / gold answer / RAG answer:

  gen_evidence : the benchmark's evidence — the chunks used to WRITE & verify the gold
                 (stored in the turn's `gen_evidence` list if present, else split from the
                 flattened `evidence` string).
  rag_evidence : what the RAG-under-test actually RETRIEVED to answer (the turn's
                 `rag_retrieved_context`).

Each doc reports:
  {"doc": "<chunk>", "sim_to_question": .., "sim_to_gold_answer": .., "sim_to_rag_answer": ..}

sim = cosine similarity of OpenAI embeddings (the same signal dense retrieval ranks by).
Showing BOTH lists is what lets you attribute a failure: if the answer chunk is in
gen_evidence but NOT in rag_evidence -> retrieval miss; if it's in BOTH but the RAG still
answered wrong -> generation miss.

No re-retrieval is done — everything is scored from what the run actually stored, so the
two lists are unambiguous (unlike the old version, which re-retrieved a fresh list).

Run (from the package root):
    python -m conv_rag_benchmark.question_doc_similarity --dataset qasper \
        --file result/benchmark_quality/qasper/quality_e_strictgold_nonegraph_t10.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from statistics import mean

import numpy as np

from .config import Config
from .embeddings import Embedder
from .llm import LLM


def _cos(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return round(float(a @ b) / (na * nb + 1e-9), 3)


def _as_docs(turn, list_field: str, str_field: str, k: int):
    """Return up to k evidence chunks: prefer a stored LIST field, else split the
    flattened string field into sentence-ish chunks."""
    docs = turn.get(list_field)
    if isinstance(docs, list) and docs:
        return [d for d in docs if isinstance(d, str) and d.strip()][:k]
    blob = turn.get(str_field) or ""
    if not blob.strip():
        return []
    # split the joined evidence string back into readable chunks
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z$])", blob)
    return [p.strip() for p in parts if len(p.strip()) > 25][:k]


def _score(docs, qv, gv, av, embedder, clip):
    if not docs:
        return []
    dvs = embedder.encode(docs)
    out = []
    for i, d in enumerate(docs):
        out.append({
            "doc": d[:clip] + ("…" if len(d) > clip else ""),
            "sim_to_question": _cos(qv, dvs[i]),
            "sim_to_gold_answer": _cos(gv, dvs[i]),
            "sim_to_rag_answer": _cos(av, dvs[i]),
        })
    out.sort(key=lambda e: -e["sim_to_question"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="per-turn gen_evidence + rag_evidence, each scored")
    ap.add_argument("--dataset", required=True, help="loader name (for Config only), e.g. qasper")
    ap.add_argument("--file", required=True, help="the quality_e*.json to annotate")
    ap.add_argument("--k", type=int, default=5, help="max docs to score per evidence list")
    ap.add_argument("--clip", type=int, default=220, help="chars of each doc to keep in output")
    ap.add_argument("--show", type=int, default=3, help="how many turns to print")
    args = ap.parse_args(argv)
    try:                       # Windows console is cp1252; avoid UnicodeEncodeError on prints
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not os.path.exists(args.file):
        print(f"!! not found: {args.file}"); return
    data = json.load(open(args.file, encoding="utf-8"))

    config = Config.load(dataset=args.dataset, max_samples=5, prefer_local_embeddings=False)
    llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=llm)
    if not embedder.available:
        print("!! embeddings unavailable — needs an OpenAI key"); return
    print(f"# {args.file}: scoring stored gen_evidence + rag_evidence (no re-retrieval)\n")

    out_turns = []
    overlap_scores = []
    for c in data.get("conversations", []):
        for t in c.get("turns", []):
            q = (t.get("question") or "").strip()
            ans = (t.get("rag_answer") or "").strip()
            gold = (t.get("gold") or "").strip()
            if not q:
                continue
            qv, av, gv = (embedder.encode([q])[0],
                          embedder.encode([ans or q])[0],
                          embedder.encode([gold or q])[0])
            gen_docs = _as_docs(t, "gen_evidence", "evidence", args.k)
            rag_docs = _as_docs(t, "rag_retrieved_context", "rag_retrieved_context", args.k)
            gen_ev = _score(gen_docs, qv, gv, av, embedder, args.clip)
            rag_ev = _score(rag_docs, qv, gv, av, embedder, args.clip)
            # simple gen<->rag overlap: best cosine of each gen doc to any rag doc
            ov = None
            if gen_docs and rag_docs:
                gvs, rvs = embedder.encode(gen_docs), embedder.encode(rag_docs)
                ov = round(mean(max(_cos(g, r) for r in rvs) for g in gvs), 3)
                overlap_scores.append(ov)
            out_turns.append({"query_type": t.get("query_type"), "outcome": t.get("outcome"),
                              "question": q, "gold_answer": gold, "rag_answer": ans,
                              "gen_rag_overlap": ov,
                              "gen_evidence": gen_ev, "rag_evidence": rag_ev})

    # save FIRST, so a console print error can never lose the results
    out_path = os.path.splitext(args.file)[0] + "_doc_similarity.json"
    json.dump({"source": args.file, "dataset": args.dataset, "k": args.k,
               "n_turns": len(out_turns), "turns": out_turns},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved -> {out_path}\n")

    for r in out_turns[: args.show]:
        print(f"[{r['query_type']}] ({r['outcome']})  gen<->rag overlap={r['gen_rag_overlap']}")
        print(f"  question:    {r['question'][:88]}")
        print(f"  gold_answer: {r['gold_answer'][:88]}")
        for label, key in [("GEN evidence (wrote the gold)", "gen_evidence"),
                           ("RAG evidence (system retrieved)", "rag_evidence")]:
            print(f"  {label}:")
            for e in r[key]:
                print(f"    - q:{100*e['sim_to_question']:>3.0f}%  gold:{100*e['sim_to_gold_answer']:>3.0f}%  "
                      f"ans:{100*e['sim_to_rag_answer']:>3.0f}%  | {e['doc'][:62]}")
        print()

    if overlap_scores:
        print("=" * 56)
        print(f" mean gen<->rag evidence overlap : {100*mean(overlap_scores):.1f}%")
        print("=" * 56)


if __name__ == "__main__":
    main()
