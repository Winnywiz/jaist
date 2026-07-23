"""
Controlled graph ablation — "which graph is better?", everything else held fixed.

Loads ONE MultiHopRAG document subset, then runs three arms that differ **only**
in the knowledge graph used to generate the conversation/questions:

    none     no graph        -> dense-only retrieval        (≈ A baseline_ragdive)
    cooccur  co-occurrence   -> proximity edges             (≈ B graph_probe)
    typed    typed relations -> LLM-typed semantic edges    (≈ C relation_typing)

Held identical across arms (so the graph is the only variable):
    * the same corpus chunks (same documents, same count),
    * the same target VectorRAG (retrieves over the same chunks),
    * the same independent judge model (gpt-4o) for scoring,
    * the same number of conversations / turns / seed.

Reports per arm:
    graph stats         n_nodes, n_edges, % typed edges
    benchmark quality   groundedness (is the gold backed by its evidence?),
                        avg_difficulty
    RAG behaviour       correctness, context_recall   (same RAG, so comparable)

Run:
    python -m conv_rag_benchmark.compare_graphs --max-samples 20 --conversations 4 --turns 4
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from .config import Config, get_logger
from .datasets.loader import DatasetLoader, Sample
from .embeddings import Embedder
from .evaluation.failure_classifier import FailureClassifier
from .generation.conversation_generator import ConversationGenerator
from .graph.graph_builder import GraphBuilder
from .graph.retriever import GraphRetriever
from .interfaces.rag_interface import VectorRAG
from .llm import LLM

logger = get_logger("compare_graphs")

ARMS = ["none", "cooccur", "typed"]
_GENERIC_RELATIONS = {"co_occurs_with", "related_to"}


def _typed_edge_fraction(kg) -> float:
    edges = kg.edges()
    if not edges:
        return 0.0
    typed = sum(1 for e in edges if e.get("relation") not in _GENERIC_RELATIONS)
    return round(typed / len(edges), 3)


def run_arm(mode: str, samples: List[Sample], chunks: List[str], config: Config,
            gen_llm: LLM, judge: FailureClassifier, rag: VectorRAG,
            embedder: Embedder) -> Dict:
    """Build the arm's graph from the shared chunks, generate + evaluate."""
    logger.info("=== ARM '%s' ===", mode)
    builder = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                           graph_mode=mode)
    kg = builder.build(chunks)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    convgen = ConversationGenerator(kg, config=config, llm=gen_llm, retriever=retriever)

    conversations = convgen.generate_many(samples[: config.num_conversations])

    records: List[Dict] = []
    grounded_flags: List[float] = []
    for conv in conversations:
        rag.reset()
        for turn in conv.turns:
            history = conv.history_upto(turn["turn_id"])
            gold = turn["gold"]
            # groundedness: is the gold backed by the evidence it was generated from?
            supporting = turn["evidence"].get("chunks", [])
            grounded = judge._gold_in_context(gold["gold_answer"], supporting,
                                              turn.get("is_unanswerable", False))
            grounded_flags.append(1.0 if grounded else 0.0)

            resp = rag.answer(turn["question"], history=history)
            diag = judge.classify({
                "question": turn["question"],
                "query_type": turn["query_type"],
                "expected_failure": turn.get("expected_failure"),
                "gold_answer": gold["gold_answer"],
                "gold_evidence": " ".join(supporting)[:1200],
                "retrieved_evidence": resp.retrieved_context,
                "rag_answer": resp.answer,
                "is_unanswerable": turn.get("is_unanswerable", False),
                "history": history,
            })
            records.append({"difficulty": turn["difficulty"],
                            "diagnosis": diag.to_dict()})

    n = len(records) or 1
    corr = [r["diagnosis"]["correct"] for r in records
            if r["diagnosis"]["correct"] is not None]
    recall = [1.0 if r["diagnosis"]["gold_retrieved"] else 0.0 for r in records]
    return {
        "arm": mode,
        "n_nodes": kg.stats()["n_nodes"],
        "n_edges": kg.stats()["n_edges"],
        "pct_typed_edges": _typed_edge_fraction(kg),
        "turns": len(records),
        "groundedness": round(sum(grounded_flags) / (len(grounded_flags) or 1), 3),
        "correctness": round(sum(corr) / (len(corr) or 1), 3),
        "context_recall": round(sum(recall) / n, 3),
        "avg_difficulty": round(sum(r["difficulty"] for r in records) / n, 2),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Controlled graph ablation (none/cooccur/typed)")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--max-samples", type=int, default=20,
                    help="documents/questions used to build the shared corpus")
    ap.add_argument("--conversations", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args(argv)

    config = Config.load(dataset=args.dataset, max_samples=args.max_samples,
                         num_conversations=args.conversations,
                         min_turns=args.turns, max_turns=args.turns,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key (RAG-DIVE/.env) for a meaningful ablation")
        return

    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = FailureClassifier(config=config)  # uses the separate judge model

    # shared corpus (same documents -> same chunks for every arm)
    samples = DatasetLoader(config.dataset, max_samples=config.max_samples).load()
    chunks: List[str] = []
    for s in samples:
        chunks.extend(s.context)
    chunks = [c for c in chunks if c and c.strip()]
    print(f"# shared corpus: {len(samples)} docs -> {len(chunks)} chunks")
    print(f"# gen model: {config.gen_model} | judge model: {config.judge_model}")

    # one shared target RAG over the shared chunks (graph-independent)
    rag = VectorRAG(chunks, config=config, llm=gen_llm, embedder=embedder)

    results = [run_arm(mode, samples, chunks, config, gen_llm, judge, rag, embedder)
               for mode in ARMS]

    # ---- report ----
    print("\n============== CONTROLLED GRAPH ABLATION (same data, same RAG) ==============")
    cols = ["n_nodes", "n_edges", "pct_typed_edges", "turns",
            "groundedness", "correctness", "context_recall", "avg_difficulty"]
    header = f"{'metric':<18}" + "".join(f"{a:>14}" for a in ARMS)
    print(header)
    labels = {"none": "none (A)", "cooccur": "co-occur (B)", "typed": "typed (C)"}
    print(f"{'arm':<18}" + "".join(f"{labels[a]:>14}" for a in ARMS))
    by_arm = {r["arm"]: r for r in results}
    for c in cols:
        row = f"{c:<18}" + "".join(f"{str(by_arm[a][c]):>14}" for a in ARMS)
        print(row)

    print("\n--- how to read it ---")
    print("  groundedness   : gold answer backed by the evidence it was built from (higher=better generator)")
    print("  pct_typed_edges: share of edges with a real relation label (typed arm should win)")
    print("  correctness    : same VectorRAG's answer matches gold (comparable across arms)")
    print("  context_recall : gold info present in what the RAG retrieved")
    print(f"\ngen usage  ({config.gen_model}): {gen_llm.cost_report()}")
    print(f"judge usage({config.judge_model}): {judge.llm.cost_report()}")

    out = os.path.join(config.output_dir, "graph_ablation.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"config": {"dataset": config.dataset, "docs": len(samples),
                              "chunks": len(chunks), "conversations": config.num_conversations,
                              "turns": args.turns}, "arms": results}, fw, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
