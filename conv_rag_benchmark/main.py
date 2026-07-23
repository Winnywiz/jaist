"""
Command-line entry point for the conversational-RAG failure benchmark.

Examples
--------
Run fully offline on the built-in synthetic corpus (no API key needed)::

    python -m conv_rag_benchmark.main --dataset synthetic --offline \
        --conversations 3 --max-samples 5

Run on JudgeAgent's local MultiHopRAG with an OpenAI key in RAG-DIVE/.env::

    python -m conv_rag_benchmark.main --dataset multihoprag --conversations 5 \
        --target-rag mock

Outputs land in ``conv_rag_benchmark/output/`` (results.json, summary.csv,
evaluation_report.json) and ``output/figures/`` (four PNG charts).
"""
from __future__ import annotations

import argparse
import json

from .config import Config, get_logger
from .pipeline import BenchmarkPipeline

logger = get_logger("main")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conversational RAG failure benchmark")
    p.add_argument("--dataset", default=None,
                   help="multihoprag | hotpotqa | 2wikimultihopqa | musique | synthetic")
    p.add_argument("--dataset-path", default=None, help="override dataset dir/path")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--conversations", type=int, default=None,
                   help="number of conversations to generate")
    p.add_argument("--min-turns", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--target-rag", default=None, choices=["mock", "vector", "graph"])
    p.add_argument("--gen-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--offline", action="store_true", help="force no-network mode")
    p.add_argument("--no-local-embeddings", action="store_true",
                   help="skip sentence-transformers (use OpenAI or lexical)")
    p.add_argument("--config", default=None, help="path to config.yaml")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "max_samples": args.max_samples,
        "num_conversations": args.conversations,
        "min_turns": args.min_turns,
        "max_turns": args.max_turns,
        "target_rag": args.target_rag,
        "gen_model": args.gen_model,
        "judge_model": args.judge_model,
        "seed": args.seed,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.offline:
        overrides["offline"] = True
    if args.no_local_embeddings:
        overrides["prefer_local_embeddings"] = False
    return Config.load(yaml_path=args.config, **overrides)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    report = BenchmarkPipeline(config).run()

    print("\n================ BENCHMARK SUMMARY ================")
    print(json.dumps({
        "overall": report["overall"],
        "query_type_accuracy": report["query_type_accuracy"],
        "failure_distribution": report["failure_distribution"],
        "failure_category_distribution": report["failure_category_distribution"],
        "difficulty_accuracy": report["difficulty_accuracy"],
        "turn_depth_accuracy": report["turn_depth_accuracy"],
    }, indent=2))
    print("==================================================")


if __name__ == "__main__":
    main()
