"""
Ablation: does the outcome-driven CONTROLLER actually add value, or is
just "being dynamic" enough?

Runs two live, dynamic loops on the SAME graph, RAG and seeds — the only thing
that differs is how the next question TYPE is chosen:

  * controller     : type chosen by the outcome of the RAG's last answer (_next_type)
  * dynamic-random : same loop, but the type is chosen at RANDOM (outcome ignored)

If the controller elicits MORE real failures (wrong + hallucinated) than random,
then the controller — not mere adaptivity — is what targets the RAG's weaknesses.

Run:
    python -m conv_rag_benchmark.run_ablation --convos 10 --turns 6 --rag vector
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from .config import Config
from .connectors import connect_rag, connect_dataset
from .embeddings import Embedder
from .generation.dynamic_generator import AdaptiveConversationGenerator
from .graph.graph_builder import GraphBuilder
from .graph.retriever import GraphRetriever
from .llm import LLM
from .quality_judge import judge_items


def _profile(convos, judge):
    items, turns = [], []
    for c in convos:
        for t in c.turns:
            items.append({"question": t.question, "gold": t.gold,
                          "evidence": t.evidence, "query_type": t.query_type})
            turns.append(t)
    quality, _ = judge_items(judge, items)
    outcomes = Counter(t.outcome for t in turns)
    n = len(turns) or 1
    return {
        "n": len(turns),
        "well_formed": quality.get("well_formed"),
        "gold_supported": quality.get("gold_supported"),
        "gold_correct": quality.get("gold_correct"),
        "outcomes": dict(outcomes),
        "failure_rate": round((outcomes["wrong"] + outcomes["hallucinated"]) / n, 3),
        "failures_by_probe": dict(Counter(
            t.query_type for t in turns if t.outcome in ("wrong", "hallucinated"))),
        "types_used": dict(Counter(t.query_type for t in turns)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="E controller vs dynamic-random ablation")
    ap.add_argument("--convos", type=int, default=10)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--d-chunks", type=int, default=600)
    ap.add_argument("--rag", default="vector", choices=["mock", "vector", "graph"])
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args(argv)

    config = Config.load(dataset="multihoprag", max_samples=max(args.convos, 50),
                         num_conversations=args.convos,
                         min_turns=args.turns, max_turns=args.turns,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key"); return

    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model=config.judge_model, config=config)
    print(f"# gen: {config.gen_model} | judge: {config.judge_model} | RAG: {args.rag}")

    seeds = connect_dataset(config.dataset, max_samples=config.max_samples)
    chunks = [c for s in seeds for c in s.context if c and c.strip()][:args.d_chunks]
    print(f"# building typed graph over {len(chunks)} chunks")
    kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                      graph_mode="typed").build(chunks)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    target_rag = connect_rag(args.rag, kg.chunks, config=config, llm=gen_llm,
                             retriever=retriever, embedder=embedder)
    conv_seeds = seeds[: config.num_conversations]

    results = {}
    for policy in ("controller", "random"):
        print(f"\n# running policy = {policy} ...")
        gen = AdaptiveConversationGenerator(kg, target_rag, judge, config=config,
                                            gen_llm=gen_llm, retriever=retriever,
                                            type_policy=policy)
        convos = gen.generate_many(conv_seeds, args.turns)
        results[policy] = {"profile": _profile(convos, judge),
                           "conversations": [c.to_dict() for c in convos]}

    # D (static) from cache, for reference
    prev = os.path.join(config.output_dir, "quality_bcd.json")
    d = json.load(open(prev, encoding="utf-8")).get("scores", {}).get("D", {}) \
        if os.path.exists(prev) else {}

    ec, er = results["controller"]["profile"], results["random"]["profile"]
    print("\n" + "=" * 70)
    print(" ABLATION: E controller  vs  Dynamic-random  vs  D (static, cached)")
    print("=" * 70)
    print(f"{'metric':<18}{'E controller':>14}{'Dyn-random':>14}{'D (cached)':>13}")
    for k, dk in [("well_formed", "well_formed"), ("gold_supported", "gold_supported"),
                  ("gold_correct", "gold_correct")]:
        print(f"{k:<18}{str(ec[k]):>14}{str(er[k]):>14}{str(d.get(dk,'-')):>13}")
    print("-" * 70)
    print(f"{'turns':<18}{ec['n']:>14}{er['n']:>14}{str(d.get('n','-')):>13}")
    print(f"{'failure_rate':<18}{ec['failure_rate']:>14}{er['failure_rate']:>14}"
          f"{'n/a':>13}")
    print(f"\n  outcomes  E-controller: {ec['outcomes']}")
    print(f"  outcomes  Dyn-random  : {er['outcomes']}")
    print(f"\n  failures caught  E-controller: {ec['failures_by_probe']}")
    print(f"  failures caught  Dyn-random  : {er['failures_by_probe']}")
    delta = round(ec["failure_rate"] - er["failure_rate"], 3)
    print(f"\n  >>> controller - random failure_rate = {delta:+.3f}  "
          f"({'controller catches MORE failures' if delta > 0 else 'no controller advantage'})")

    # same self-describing convention as run_benchmark, tagged as the ablation
    out = os.path.join(
        config.output_dir,
        f"ablation_{args.rag}_{config.dataset}_t{args.turns}_c{args.convos}.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"controller": results["controller"]["profile"],
                   "random": results["random"]["profile"],
                   "d_cached": d,
                   "controller_convos": results["controller"]["conversations"],
                   "random_convos": results["random"]["conversations"]},
                  fw, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
