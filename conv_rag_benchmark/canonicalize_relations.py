"""
Relation-label canonicalization for the typed graph, plus the pair-supply metric.

Problem: the LLM extractor names the same attribute differently across chunks
('works_for' vs 'employed_by'), so the shared-attribute pair mining that grounds
Comparative questions misses real pairs. This script clusters synonymous relation
labels with ONE LLM call, rewrites the edges, and reports how many comparable
pairs (same relation, two distinct source entities) exist before vs after.

Writes graph_canon.json next to the input; does NOT overwrite graph.json.

Run:  python -m conv_rag_benchmark.canonicalize_relations <graph.json ...>
"""
import argparse
import json
import os
from collections import defaultdict

from .config import Config
from .llm import LLM


def pair_supply(edges):
    """Number of relations stated for >= 2 distinct source entities (the raw
    material for grounded Comparative questions), and the pair count."""
    src_by_rel = defaultdict(set)
    for e in edges:
        if e.get("relation") and e.get("source"):
            src_by_rel[e["relation"]].add(e["source"])
    rels = {r: s for r, s in src_by_rel.items() if len(s) >= 2}
    return len(rels), sum(len(s) * (len(s) - 1) // 2 for s in rels.values())


def canonical_map(llm, labels):
    out = llm.chat_json(
        "Group these relation labels into clusters of SYNONYMS (same real-world "
        "attribute). For each cluster pick ONE canonical label. Only merge true "
        "synonyms; leave distinct attributes separate. "
        'JSON: {"mapping": {"<label>": "<canonical>", ...}} (include every label)',
        "LABELS: " + ", ".join(sorted(labels))) or {}
    m = out.get("mapping") or {}
    return {k: str(v).strip() for k, v in m.items() if v and str(v).strip()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    args = ap.parse_args(argv)
    cfg = Config.load()
    llm = LLM(config=cfg)

    for gp in args.graphs:
        g = json.load(open(gp, encoding="utf-8"))
        edges = g.get("edges", [])
        labels = {e.get("relation") for e in edges if e.get("relation")}
        r0, p0 = pair_supply(edges)
        mapping = canonical_map(llm, labels) if llm.available else {}
        merged = sum(1 for k, v in mapping.items() if k != v)
        for e in edges:
            r = e.get("relation")
            if r in mapping:
                e["relation"] = mapping[r]
        r1, p1 = pair_supply(edges)
        out = os.path.join(os.path.dirname(gp), "graph_canon.json")
        json.dump(g, open(out, "w", encoding="utf-8"))
        print(f"{gp}: {len(labels)} labels, {merged} merged | comparable relations "
              f"{r0} -> {r1} | comparable pairs {p0} -> {p1} | saved {out}")


if __name__ == "__main__":
    main()
