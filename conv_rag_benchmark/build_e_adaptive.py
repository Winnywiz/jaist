"""
Run Method E (adaptive conversational probing) and compare it to Method D.

E reuses D's exact machinery — the same typed graph, retriever, query templates and
gold generator — and changes only the *loop*: each turn it asks a real target-RAG,
grades the answer, and lets that outcome choose the next question's type
(:mod:`conv_rag_benchmark.generation.adaptive_generator`).

It then reports TWO things:
  1. benchmark quality (well_formed / gold_supported / gold_correct) on the same
     scale as B / C / D, so E's questions are shown to be just as well-formed, and
  2. the RAG's failure profile (the diagnostic payoff): how often the live system
     was wrong / hallucinated / abstained, and which probe types caught it.

Run:
    python -m conv_rag_benchmark.build_e_adaptive --convos 5 --turns 6 --rag mock
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .generation.adaptive_generator import AdaptiveConversationGenerator, _has_placeholder
from .graph.graph_builder import GraphBuilder, KnowledgeGraph
from .graph.retriever import GraphRetriever
from .interfaces.rag_interface import build_rag
from .llm import LLM
from .quality_judge import _typed_edge_fraction
from .geval import geval_items, geval_breakdown_by_type

_LABELS = {"multihoprag": "MultiHopRAG", "medqa": "MedQA", "hotpotqa": "HotpotQA",
           "2wikimultihopqa": "2WikiMultiHopQA", "musique": "MuSiQue",
           "arxivcs": "ArXivCS"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Method E: adaptive probing vs D")
    ap.add_argument("--convos", type=int, default=5)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--d-chunks", type=int, default=600,
                    help="chunks to build the typed graph from (match D)")
    ap.add_argument("--rag", default="mock",
                    choices=["mock", "vector", "selfrag", "raptor", "graph", "longrag",
                             "crag"],
                    help="which target RAG to probe")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--graph-mode", default="typed", choices=["typed", "none", "textpairs"],
                    help="'typed' = typed-relation graph (default); 'none' = NO graph, "
                         "dense-only retrieval (ablation: does the graph help?); "
                         "'textpairs' = no graph, but entities/relations extracted from the "
                         "RETRIEVED CHUNKS at generation time. This is the FAIR control for "
                         "'typed': same entity extractor, relations only within a chunk, so "
                         "the graph's sole advantage is cross-chunk structure. Unlike 'none' "
                         "it still authors Comparative questions, so the arm is not "
                         "confounded with the question-type mix.")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--type-policy", default="controller", choices=["controller", "random"],
                    help="'controller' = outcome-driven type selection (Method E); "
                         "'random' = live loop that reads the RAG answer but picks the "
                         "next type at RANDOM (ignores outcome)")
    ap.add_argument("--strict-gold", action="store_true",
                    help="use the strict composer (gold from evidence ONLY) + verify each "
                         "gold is grounded AND correct before accepting (trustworthy key)")
    ap.add_argument("--quality-gate", action="store_true",
                    help="score each question's well-formedness and rewrite it if low "
                         "BEFORE sending to the RAG (pre-send self-refinement)")
    ap.add_argument("--tag", default="",
                    help="extra suffix for the output filename (e.g. 't10'), so runs that "
                         "differ only by turns/convos don't overwrite each other")
    args = ap.parse_args(argv)

    config = Config.load(dataset=args.dataset, max_samples=max(args.convos, 50),
                         num_conversations=args.convos,
                         min_turns=args.turns, max_turns=args.turns,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key (RAG-DIVE/.env or OPENAI_API_KEY)")
        return

    ds_label = _LABELS.get(args.dataset, args.dataset)
    out_dir = os.path.join(config.output_dir, ds_label)
    os.makedirs(out_dir, exist_ok=True)
    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model=config.judge_model, config=config)
    print(f"# dataset: {ds_label} | gen: {config.gen_model} | judge: {config.judge_model} | RAG: {args.rag}")

    # ---- 1. same typed graph D uses: load saved graph if present, else build ----
    #         (--graph-mode none = ABLATION: no graph, dense-only retrieval)
    seeds = DatasetLoader(config.dataset, max_samples=config.max_samples).load()
    graph_path = os.path.join(out_dir, "graph.json")
    local_extractor = None
    if args.graph_mode in ("none", "textpairs"):
        chunks = [c for s in seeds for c in s.context if c and c.strip()][:args.d_chunks]
        print(f"# {args.graph_mode.upper()} ablation: dense-only retrieval over "
              f"{len(chunks)} chunks")
        kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                          graph_mode="none").build(chunks)
        if args.graph_mode == "textpairs":
            # same extractor as the typed arm, but applied per-retrieval, not persisted
            local_extractor = GraphBuilder(config=config, llm=gen_llm,
                                           embedder=embedder, graph_mode="typed")
    elif os.path.exists(graph_path):
        print(f"# loading saved graph from {graph_path} (skipping rebuild)")
        kg = KnowledgeGraph.load(graph_path)
        try:
            kg.chunk_embeddings = np.asarray(embedder.encode(kg.chunks), dtype=float)
        except Exception as _e:
            print(f"  (chunk embedding failed: {str(_e)[:80]})")
    else:
        chunks = [c for s in seeds for c in s.context if c and c.strip()][:args.d_chunks]
        print(f"# building typed graph over {len(chunks)} chunks (same as D)")
        kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                          graph_mode="typed").build(chunks)
        kg.save(graph_path)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)

    # ---- 2. the system under test ----
    target_rag = build_rag(args.rag, kg.chunks, config=config, llm=gen_llm,
                           retriever=retriever, embedder=embedder)

    # ---- 3. run the adaptive loop ----
    gen = AdaptiveConversationGenerator(kg, target_rag, judge, config=config,
                                        gen_llm=gen_llm, retriever=retriever,
                                        type_policy=args.type_policy,
                                        quality_gate=args.quality_gate,
                                        strict_gold=args.strict_gold,
                                        local_extractor=local_extractor)
    # SEED filter: the seed question (turn 0) comes straight from the dataset, so it
    # bypasses the in-loop placeholder guard. Drop seeds whose question carries a raw
    # LaTeX label (Table TABREF1, ...); fall back to the raw list if too few remain.
    seed_pool = [s for s in seeds if not _has_placeholder(s.question)] or seeds
    convos = gen.generate_many(seed_pool[: config.num_conversations], args.turns)

    # ---- 4a. benchmark QUALITY (same metrics as D) ----
    items, turns = [], []
    for c in convos:
        for t in c.turns:
            items.append({"question": t.question, "gold": t.gold,
                          "evidence": t.evidence, "query_type": t.query_type})
            turns.append(t)
    # score E with G-Eval (same scorer D uses in benchmark_random.json)
    e_quality, scored = geval_items(items, model=config.judge_model)
    e_quality["n"] = len(items)
    e_by_type = geval_breakdown_by_type(scored)

    # ---- 4b. the RAG's FAILURE profile (the adaptive payoff) ----
    outcomes = Counter(t.outcome for t in turns)
    n = len(turns) or 1
    failure_rate = round((outcomes["wrong"] + outcomes["hallucinated"]) / n, 3)
    # which probe type surfaced each failure
    fails_by_type = Counter(t.query_type for t in turns
                            if t.outcome in ("wrong", "hallucinated"))

    graph_stats = {"graph_nodes": kg.stats()["n_nodes"],
                   "graph_edges": kg.stats()["n_edges"],
                   "graph_type": "typed-relation (LLM)",
                   "pct_typed_edges": _typed_edge_fraction(kg.edges())}

    # ---- 5. Method E question quality (G-Eval) ----
    print(f"\n===== question quality (G-Eval): Method E — {ds_label} =====")
    print(f"{'metric':<16}{'Method E':>12}")
    for k in ["well_formed", "gold_supported", "gold_correct"]:
        print(f"{k:<16}{str(e_quality.get(k,'-')):>12}")

    print(f"\n===== what E found about the '{args.rag}' RAG =====")
    print(f"turns probed         : {n}")
    print(f"outcome distribution : {dict(outcomes)}")
    print(f"RAG failure rate     : {failure_rate}  (wrong + hallucinated)")
    print(f"failures by probe    : {dict(fails_by_type)}")
    print(f"type sequence (c0)   : {convos[0].type_sequence if convos else []}")
    print(f"outcomes      (c0)   : {convos[0].outcome_sequence if convos else []}")

    print(f"\ngen usage  ({config.gen_model}): {gen_llm.cost_report()}")
    print(f"judge usage({config.judge_model}): {judge.cost_report()}")

    _suffix = ("_randomtype" if args.type_policy == "random" else "") + \
              ("_strictgold" if args.strict_gold else "") + \
              ("_qgate" if args.quality_gate else "") + \
              ("" if args.graph_mode == "typed" else f"_{args.graph_mode}graph") + \
              ("" if args.rag == "vector" else f"_{args.rag}") + \
              (f"_{args.tag}" if args.tag else "")   # keep RAGs / turn-settings from colliding
    out = os.path.join(out_dir, f"quality_e{_suffix}.json" if _suffix else "quality_e.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"rag": args.rag, "graph_stats": graph_stats,
                   "method": "E",
                   "quality": {"E": e_quality},
                   "e_by_query_type": e_by_type,
                   "rag_failure": {"n_turns": n, "outcomes": dict(outcomes),
                                   "failure_rate": failure_rate,
                                   "failures_by_probe": dict(fails_by_type)},
                   "conversations": [c.to_dict() for c in convos]},
                  fw, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")
    return e_quality


if __name__ == "__main__":
    main()
