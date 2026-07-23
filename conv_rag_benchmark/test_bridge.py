"""
A/B test: does the BRIDGE WALK raise multi-hop question depth?

  OLD  = similarity retrieval (current) -> Multi-Hop question
  NEW  = 2-hop entity-bridge path        -> Multi-Hop question

Judges each question's reasoning depth 1-3 with an INDEPENDENT gpt-4o judge
(1=single fact, 2=detail within a topic, 3=multi-hop across entities).

Run: python -m conv_rag_benchmark.test_bridge --n 15
"""
from __future__ import annotations

import argparse
import random

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .generation.query_generator import QueryGenerator
from .graph.graph_builder import GraphBuilder
from .graph.retriever import GraphRetriever
from .llm import LLM
from .build_benchmark import bridge_path_evidence

_DEPTH_SYS = (
    "Rate the reasoning DEPTH needed to answer this question, 1-3: "
    "1 = single fact lookup; 2 = a specific detail or combining within one topic; "
    "3 = multi-hop: requires connecting facts about DIFFERENT entities or events. "
    'Respond JSON: {"depth": 1|2|3}')


def depth_of(judge, q):
    out = judge.chat_json(_DEPTH_SYS, f"QUESTION: {q}") or {}
    try:
        return int(out.get("depth", 0)) or None
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args(argv)

    config = Config.load(dataset="multihoprag", max_samples=50,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model="gpt-4o", config=config)          # independent depth judge
    print(f"# gen: {config.gen_model} | depth-judge: gpt-4o")

    seeds = DatasetLoader("multihoprag", max_samples=50).load()
    chunks = [c for s in seeds for c in s.context if c and c.strip()][:600]
    kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                      graph_mode="typed").build(chunks)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    qg = QueryGenerator(config=config, llm=gen_llm)
    rng = random.Random(0)

    old_qs, new_qs = [], []
    for _ in range(args.n):
        seed = rng.choice(seeds)
        ev = retriever.retrieve(seed.question, k=8)
        t = qg.generate("Multi-Hop", 1, ev, [])
        if t and t.question:
            old_qs.append(t.question)
        br = bridge_path_evidence(kg, rng)
        if br is not None:
            t2 = qg.generate("Multi-Hop", 1, br, [])
            if t2 and t2.question:
                new_qs.append(t2.question)

    def stats(qs):
        ds = [d for d in (depth_of(judge, q) for q in qs) if d]
        avg = round(sum(ds) / len(ds), 2) if ds else 0
        d3 = sum(1 for d in ds if d >= 3)
        return avg, d3, len(ds)

    oa, o3, on = stats(old_qs)
    na, n3, nn = stats(new_qs)
    print("\n================= MULTI-HOP DEPTH A/B (judge=gpt-4o) =================")
    print(f"  OLD  similarity walk : avg depth {oa}   depth-3 {o3}/{on}  ({round(o3/on,2) if on else 0})")
    print(f"  NEW  BRIDGE walk     : avg depth {na}   depth-3 {n3}/{nn}  ({round(n3/nn,2) if nn else 0})")
    print("\n  NEW bridge-walk examples:")
    for q in new_qs[:4]:
        print(f"    - {q}")
    print(f"\njudge cost: {judge.cost_report()}")


if __name__ == "__main__":
    main()
