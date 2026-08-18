"""graph_quality.py — measure how ACCURATE a typed knowledge graph is.

The graph is built by an LLM extracting (entity --relation--> entity) triples from
chunks. `graph_stats.pct_typed_edges` only tells you the relations are *specific*, not
whether they are *correct*. This tool samples edges and checks, per edge, whether the
source chunk actually SUPPORTS the relation — an LLM-judged grounding check — giving a
concrete "graph accuracy" number.

Method per sampled edge (source --relation--> target):
  1. find a chunk both entities appear in (intersection of their node.chunks),
  2. ask the judge LLM whether that chunk's text supports the triple,
  3. tally supported / not-supported / unverifiable (no shared chunk).

Run:
    python -m conv_rag_benchmark.graph_quality \
        --graph result/benchmark_quality/qasper/graph.json --n 30
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional

from .config import Config
from .llm import LLM

_SYS = (
    "You verify whether a TEXT supports a factual CLAIM written as a triple "
    "(entity --relation--> entity). Reply true ONLY if the text explicitly states or "
    "clearly implies that exact relation between those two entities. If the text merely "
    "mentions the entities without that relation, reply false. "
    'Respond JSON: {"supported": true/false}.'
)


def verify_edge(llm: LLM, chunk: str, source: str, relation: str, target: str) -> Optional[bool]:
    if not (llm and getattr(llm, "available", False)):
        return None
    out = llm.chat_json(
        _SYS,
        f'TEXT:\n{chunk[:1500]}\n\nCLAIM: "{source}" --{relation}--> "{target}"')
    if not out or "supported" not in out:
        return None
    return bool(out["supported"])


def check_graph(graph_path: str, n: int, llm: LLM, seed: int = 0) -> Dict:
    g = json.load(open(graph_path, encoding="utf-8"))
    chunks: List[str] = g.get("chunks", [])
    by_id = {node["id"]: node for node in g.get("nodes", [])}
    edges = list(g.get("edges", []))
    random.Random(seed).shuffle(edges)

    supported = notsupported = unverifiable = 0
    cases: List[Dict] = []
    for e in edges:
        if supported + notsupported >= n:      # stop once we've VERIFIED n edges
            break
        s_node, t_node = by_id.get(e.get("source")), by_id.get(e.get("target"))
        if not s_node or not t_node:
            unverifiable += 1
            continue
        shared = set(s_node.get("chunks", [])) & set(t_node.get("chunks", []))
        shared = [c for c in shared if 0 <= c < len(chunks)]
        if not shared:                          # edge spans chunks -> can't check from one
            unverifiable += 1
            continue
        chunk = chunks[sorted(shared)[0]]
        ok = verify_edge(llm, chunk, s_node["entity"], e.get("relation", ""), t_node["entity"])
        if ok is None:
            unverifiable += 1
            continue
        supported += ok
        notsupported += (not ok)
        cases.append({"triple": f'{s_node["entity"]} --{e.get("relation")}--> {t_node["entity"]}',
                      "supported": ok, "chunk": chunk[:200]})

    total = supported + notsupported
    return {
        "graph": graph_path,
        "n_verified": total,
        "n_unverifiable": unverifiable,
        "accuracy": round(supported / total, 3) if total else None,
        "supported": supported,
        "not_supported": notsupported,
        "cases": cases,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure typed-graph accuracy (edge grounding).")
    ap.add_argument("--graph", required=True, help="path to a graph.json")
    ap.add_argument("--n", type=int, default=30, help="number of edges to VERIFY")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    config = Config.load()
    llm = LLM(model=config.judge_model, config=config)
    rep = check_graph(args.graph, args.n, llm, seed=args.seed)

    print(f"# graph: {rep['graph']}")
    print(f" edges verified      : {rep['n_verified']}")
    print(f" edges unverifiable  : {rep['n_unverifiable']} (no single shared chunk)")
    print(f" GRAPH ACCURACY      : {rep['accuracy']}  "
          f"({rep['supported']} supported / {rep['not_supported']} not)")
    print("\n examples:")
    for c in rep["cases"][:8]:
        mark = "OK " if c["supported"] else "XX "
        print(f"  {mark} {c['triple']}")

    out_path = args.graph.replace(".json", "_quality.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(rep, fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
