"""
recheck_evidence.py — is a "zero-evidence" triple really broken, or just truncated?

The generator authors a gold from up to k=8-16 retrieved chunks, but only
``" ".join(chunks[:5])[:1500]`` is SAVED on the turn. So an audit that reads the saved
field can call a gold "unsupported" when the supporting sentence was simply cut off.

This tool separates the two explanations. For each triple the audit flagged with ZERO
necessary sentences, it:

  1. re-retrieves evidence for the question from the full corpus (no 1500-char cut), and
  2. re-runs the same audit against that fuller evidence.

Interpretation:
  supported now  -> TRUNCATION ARTIFACT. The gold was grounded; the saved record was lossy.
  still zero     -> REAL DEFECT. The claim is not in the corpus (parametric-knowledge leak)
                    or the gold is vacuous.

Run (from the package root):
    python -m conv_rag_benchmark.recheck_evidence --dataset mlarxiv \
        --quality result/benchmark_quality/mlarxiv/quality_e_randomtype_strictgold.json \
        --audit   result/benchmark_quality/mlarxiv/triple_audit_ratio_n50.json
"""
from __future__ import annotations

import argparse
import json
import os

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .llm import LLM
from .triple_audit import audit_triple


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-audit zero-evidence triples against full evidence")
    ap.add_argument("--dataset", required=True, help="loader name, e.g. mlarxiv / qasper")
    ap.add_argument("--quality", required=True, help="the quality_e*.json the audit came from")
    ap.add_argument("--audit", required=True, help="the triple_audit_*.json with the results")
    ap.add_argument("--k", type=int, default=16, help="chunks to re-retrieve (no truncation)")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args(argv)

    audit = json.load(open(args.audit, encoding="utf-8"))
    # the triples worth re-checking: no necessary sentence found, and NOT intentionally
    # unanswerable (those legitimately have no supporting evidence)
    targets = [r for r in audit["rows"]
               if r.get("support_necessary_sentences") == 0
               and r.get("query_type") != "Unanswerable"]
    if not targets:
        print("no real zero-evidence triples to re-check"); return
    print(f"# re-checking {len(targets)} zero-evidence triples from {os.path.basename(args.audit)}")

    # map question -> full gold (the audit stores a truncated copy)
    qual = json.load(open(args.quality, encoding="utf-8"))
    golds = {(t.get("question") or "").strip(): t.get("gold", "")
             for c in qual.get("conversations", []) for t in c.get("turns", [])}

    config = Config.load(dataset=args.dataset, max_samples=60, prefer_local_embeddings=False)
    llm = LLM(model=config.gen_model, config=config)
    if not llm.available:
        print("!! needs an OpenAI key"); return
    auditor = LLM(model=args.model, config=config)

    # rebuild the corpus + a dense retriever, exactly as generation had available
    seeds = DatasetLoader(args.dataset, max_samples=config.max_samples).load()
    corpus = [c for s in seeds for c in s.context if c and c.strip()]
    from .interfaces.rag_interface import VectorRAG
    vr = VectorRAG(corpus, config=config, llm=llm,
                   embedder=Embedder(config=config, llm=llm))
    print(f"# corpus rebuilt: {len(corpus)} chunks | re-retrieving k={args.k}\n")

    fixed = still_broken = 0
    rows = []
    for r in targets:
        q = r["question"].strip()
        gold = golds.get(q) or r.get("gold", "")
        # full evidence: no [:5] slice, no 1500-char cut
        chunks = vr._retrieve(q)[: args.k]
        full_ev = "\n\n".join(chunks)
        re_audit = audit_triple(auditor, q, full_ev, gold, max_support=12000)
        nec = (re_audit or {}).get("support_necessary_sentences")
        ok = bool(nec)
        fixed += ok
        still_broken += (not ok)
        rows.append({"question": q[:160], "query_type": r.get("query_type"),
                     "saved_evidence_chars": None,
                     "full_evidence_chars": len(full_ev),
                     "necessary_before": 0, "necessary_after": nec,
                     "faithfulness_after": (re_audit or {}).get("faithfulness"),
                     "verdict": "TRUNCATION_ARTIFACT" if ok else "REAL_DEFECT"})
        print(f"  [{'FIXED ' if ok else 'STILL0'}] {r.get('query_type'):<20} "
              f"nec {0} -> {nec}   {q[:60]}")

    n = len(targets)
    print("\n" + "=" * 70)
    print(f" {args.dataset}: {n} zero-evidence triples re-checked against FULL evidence")
    print("=" * 70)
    print(f"  supported once untruncated (TRUNCATION ARTIFACT): {fixed}/{n} "
          f"({100*fixed/n:.0f}%)")
    print(f"  still unsupported          (REAL DEFECT)        : {still_broken}/{n} "
          f"({100*still_broken/n:.0f}%)")

    out = os.path.join(os.path.dirname(args.audit), "recheck_evidence.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"dataset": args.dataset, "n_rechecked": n,
                   "truncation_artifact": fixed, "real_defect": still_broken,
                   "rows": rows}, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
