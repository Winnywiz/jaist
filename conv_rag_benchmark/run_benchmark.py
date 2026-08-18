"""
Run the adaptive conversational benchmark (generate questions + score their quality).

The core is an adaptive loop: each turn it asks a real target-RAG, grades the answer,
and lets that outcome choose the next question's type
(:mod:`conv_rag_benchmark.generation.dynamic_generator`).

It then reports TWO things:
  1. benchmark quality (well_formed / gold_supported / gold_correct), so the
     generated questions are shown to be well-formed, and
  2. the RAG's failure profile (the diagnostic payoff): how often the live system
     was wrong / hallucinated / abstained, and which probe types caught it.

Where you plug in a RAG or a dataset: see :mod:`conv_rag_benchmark.connectors`
(RAGs via ``--rag``, datasets via ``--dataset``).

Run:
    python -m conv_rag_benchmark.run_benchmark --convos 5 --turns 6 --rag mock
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter

import numpy as np

from .config import Config
from .connectors import connect_rag, connect_dataset, available_rags
from .embeddings import Embedder
from .generation.dynamic_generator import AdaptiveConversationGenerator, has_latex_label
from .graph.graph_builder import GraphBuilder, KnowledgeGraph
from .graph.retriever import GraphRetriever
from .llm import LLM
from .quality_judge import _typed_edge_fraction
from .geval import geval_items, geval_breakdown_by_type

_LABELS = {"multihoprag": "MultiHopRAG", "medqa": "MedQA", "hotpotqa": "HotpotQA",
           "2wikimultihopqa": "2WikiMultiHopQA", "musique": "MuSiQue",
           "arxivcs": "ArXivCS"}


#: metrics summarised across repeats (quality first, then the RAG-side headline)
_SUMMARY_METRICS = ("well_formed", "gold_supported", "gold_correct", "failure_rate")


def _summarise_repeats(runs):
    """mean / std / min / max for each metric across repeated runs of one config.

    Uses the SAMPLE standard deviation (n-1), which is the honest estimator when the
    runs are a sample of all the runs the config could have produced. With a single
    run the spread is undefined and reported as 0.0.
    """
    out = {}
    for metric in _SUMMARY_METRICS:
        vals = []
        for r in runs:
            v = r["failure_rate"] if metric == "failure_rate" \
                else (r["quality"] or {}).get(metric)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            continue
        out[metric] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n_runs": len(vals),
            "values": [round(v, 4) for v in vals],
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Adaptive conversational RAG benchmark")
    ap.add_argument("--convos", type=int, default=5)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=1,
                    help="run this exact configuration N times and report mean +/- std. "
                         "Generation is stochastic, so a single run's score carries "
                         "run-to-run noise; use 3-5 when you need to claim that two "
                         "configurations really differ. Setup is built once and reused, "
                         "so N repeats cost roughly N x the generation+judging of one run.")
    ap.add_argument("--d-chunks", type=int, default=600,
                    help="chunks to build the typed graph from (match D)")
    ap.add_argument("--rag", default="mock", choices=available_rags(),
                    help="which target RAG to probe (see connectors.py to add your own)")
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
                    help="'controller' = outcome-driven type selection (adaptive); "
                         "'random' = live loop that reads the RAG answer but picks the "
                         "next type at RANDOM (ignores outcome)")
    ap.add_argument("--rag-k", type=int, default=None,
                    help="retrieval budget: number of docs the RAG retrieves per query "
                         "(overrides the backend default, e.g. vector's k=5). Difficulty "
                         "lever for the 2x2: standard k=5 vs constrained k=2. Only affects "
                         "how many chunks are retrieved, nothing else.")
    ap.add_argument("--content", default="static", choices=["static", "dynamic"],
                    help="follow-up question CONTENT conditioning (the thesis IV): "
                         "'static' = generated WITHOUT the RAG's previous answer "
                         "(generator sees gold/truth_history); 'dynamic' = generated AFTER "
                         "observing the RAG's actual previous answer (sees rag_history). "
                         "Gold answers always use truth_history regardless.")
    ap.add_argument("--types", default="",
                    help="comma-separated fixed follow-up type sequence, cycled over "
                         "turns 1.. (turn 0 is always the seed). Forces the SAME question "
                         "types in both static and dynamic runs — the paired-comparison "
                         "control. E.g. 'Correction,Multi-Hop,Unanswerable'. Empty = use "
                         "--type-policy instead.")
    ap.add_argument("--strict-gold", action="store_true",
                    help="use the strict composer (gold from evidence ONLY) + verify each "
                         "gold is grounded AND correct before accepting (trustworthy key)")
    ap.add_argument("--quality-gate", action="store_true",
                    help="score each question's well-formedness and rewrite it if low "
                         "BEFORE sending to the RAG (pre-send self-refinement)")
    ap.add_argument("--counterfactual", action="store_true",
                    help="after each answer, re-ask the SAME turn with the clean gold "
                         "history (one extra RAG call/turn) to split failures into "
                         "inherited (multi-turn) vs intrinsic; read by analyze_multiturn")
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
    seeds = connect_dataset(config.dataset, max_samples=config.max_samples)
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
    # REAL GraphRAG: it must be probed with an ACTUAL entity/relation graph, even when the
    # benchmark GENERATION runs graph-free (--graph-mode none). Build (or load) a typed
    # graph JUST for the RAG, decoupled from generation, so "graph" is a genuine GraphRAG
    # and not a dense variant on an empty store.
    rag_retriever = retriever
    rag_graph_stats = None
    if args.rag == "graph" and kg.stats()["n_edges"] == 0:
        rag_graph_path = os.path.join(out_dir, "graph_ragtest.json")
        if os.path.exists(rag_graph_path):
            print(f"# GraphRAG: loading real typed graph from {rag_graph_path}")
            rag_kg = KnowledgeGraph.load(rag_graph_path)
        else:
            rag_chunks = [c for s in seeds for c in s.context if c and c.strip()][:args.d_chunks]
            print(f"# GraphRAG: building REAL typed graph over {len(rag_chunks)} chunks")
            rag_kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                                  graph_mode="typed").build(rag_chunks)
            rag_kg.save(rag_graph_path)
        try:
            rag_kg.chunk_embeddings = np.asarray(embedder.encode(rag_kg.chunks), dtype=float)
        except Exception as _e:
            print(f"  (rag graph embedding failed: {str(_e)[:80]})")
        rag_retriever = GraphRetriever(rag_kg, config=config, llm=gen_llm, embedder=embedder)
        rag_graph_stats = rag_kg.stats()
        print(f"# GraphRAG graph ready: {rag_graph_stats}")
    target_rag = connect_rag(args.rag, kg.chunks, config=config, llm=gen_llm,
                             retriever=rag_retriever, embedder=embedder, cache_dir=out_dir,
                             **({} if args.rag_k is None else {"k": args.rag_k}))
    if args.rag_k is not None:
        print(f"# retrieval budget: k={args.rag_k} (constrained)")

    # ---- 3. run the adaptive loop ----
    gen = AdaptiveConversationGenerator(kg, target_rag, judge, config=config,
                                        gen_llm=gen_llm, retriever=retriever,
                                        type_policy=args.type_policy,
                                        content_policy=args.content,
                                        forced_types=([t.strip() for t in args.types.split(",")
                                                       if t.strip()] or None),
                                        quality_gate=args.quality_gate,
                                        strict_gold=args.strict_gold,
                                        counterfactual=args.counterfactual,
                                        local_extractor=local_extractor)
    # SEED filter: the seed question (turn 0) comes straight from the dataset, so it
    # bypasses the in-loop placeholder guard. Drop seeds whose question carries a raw
    # LaTeX label (Table TABREF1, ...); fall back to the raw list if too few remain.
    seed_pool = [s for s in seeds if not has_latex_label(s.question)] or seeds

    graph_stats = {"graph_nodes": kg.stats()["n_nodes"],
                   "graph_edges": kg.stats()["n_edges"],
                   # when GraphRAG builds its own graph (generation ran graph-free),
                   # record THAT graph too so the output proves the RAG used a real graph.
                   "rag_graph_stats": rag_graph_stats,
                   "graph_type": "typed-relation (LLM)",
                   "pct_typed_edges": _typed_edge_fraction(kg.edges())}

    # Output name: {rag}_{dataset}_t{turns}_c{convos}.json — self-describing, so a
    # run's configuration is readable straight off the filename.
    # The STANDARD configuration (strict gold + no generation graph) is the one the
    # saved results use and stays unmarked; only DEVIATIONS get an extra suffix, so
    # ablation runs can never silently overwrite a standard one.
    _flags = ("_dynamic" if args.content == "dynamic" else "_static") + \
             (f"_k{args.rag_k}" if args.rag_k is not None else "") + \
             ("_randomtype" if args.type_policy == "random" else "") + \
             ("_qgate" if args.quality_gate else "") + \
             ("_cf" if args.counterfactual else "") + \
             ("_nostrictgold" if not args.strict_gold else "") + \
             ("" if args.graph_mode == "none" else f"_{args.graph_mode}graph") + \
             ("" if not args.tag or args.tag == f"t{args.turns}" else f"_{args.tag}")
    # (a --tag of "t<turns>" is dropped: the turn count is already in the name, so the
    #  old habit of passing --tag t10 no longer yields "..._t10_c5_t10")

    # REPEATS: generation is stochastic (temperature > 0) and the loop is
    # path-dependent — one different RAG answer sends the rest of the conversation
    # down another branch — so a SINGLE run's score carries real run-to-run noise.
    # Repeating the same config N times and reporting mean +/- std tells you whether a
    # difference between two configs is real or just that noise. The expensive setup
    # (corpus, embeddings, graph, RAG) is built ONCE and reused across repeats.
    repeats = max(1, args.repeats)
    runs = []            # per-repeat quality dicts, for the summary
    e_quality = None
    for rep in range(repeats):
        if repeats > 1:
            print(f"\n########## repeat {rep + 1}/{repeats} ##########")
        convos = gen.generate_many(seed_pool[: config.num_conversations], args.turns)

        # ---- 4a. benchmark QUALITY ----
        items, turns = [], []
        for c in convos:
            for t in c.turns:
                items.append({"question": t.question, "gold": t.gold,
                              "evidence": t.evidence, "query_type": t.query_type})
                turns.append(t)
        # score the generated questions with G-Eval
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

        # ---- 5. question quality (G-Eval) ----
        print(f"\n===== question quality (G-Eval) — {ds_label} =====")
        print(f"{'metric':<16}{'benchmark':>12}")
        for k in ["well_formed", "gold_supported", "gold_correct"]:
            print(f"{k:<16}{str(e_quality.get(k, '-')):>12}")

        print(f"\n===== what the benchmark found about the '{args.rag}' RAG =====")
        print(f"turns probed         : {n}")
        print(f"outcome distribution : {dict(outcomes)}")
        print(f"RAG failure rate     : {failure_rate}  (wrong + hallucinated)")
        print(f"failures by probe    : {dict(fails_by_type)}")
        print(f"type sequence (c0)   : {convos[0].type_sequence if convos else []}")
        print(f"outcomes      (c0)   : {convos[0].outcome_sequence if convos else []}")

        # each repeat gets its own file (_r1, _r2, ...) so none overwrites another;
        # a single run keeps the plain, unsuffixed name.
        rsuffix = f"_r{rep + 1}" if repeats > 1 else ""
        out = os.path.join(
            out_dir,
            f"{args.rag}_{ds_label}_t{args.turns}_c{len(convos)}{_flags}{rsuffix}.json")
        # method label = the generation strategy actually run (self-describing).
        method_name = f"{args.content}_question_generation"   # static_/dynamic_
        with open(out, "w", encoding="utf-8") as fw:
            json.dump({"rag": args.rag, "graph_stats": graph_stats,
                       "method": method_name,
                       "repeat": rep + 1, "of_repeats": repeats,
                       "quality": {method_name: e_quality},
                       "e_by_query_type": e_by_type,
                       "rag_failure": {"n_turns": n, "outcomes": dict(outcomes),
                                       "failure_rate": failure_rate,
                                       "failures_by_probe": dict(fails_by_type)},
                       "conversations": [c.to_dict() for c in convos]},
                      fw, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {out}")
        runs.append({"file": os.path.basename(out), "quality": e_quality,
                     "failure_rate": failure_rate, "outcomes": dict(outcomes)})

    print(f"\ngen usage  ({config.gen_model}): {gen_llm.cost_report()}")
    print(f"judge usage({config.judge_model}): {judge.cost_report()}")

    # ---- 6. across-repeat summary: mean +/- std ----
    if repeats > 1:
        summary = _summarise_repeats(runs)
        print(f"\n===== across {repeats} repeats: mean +/- std =====")
        print(f"{'metric':<18}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
        for k, v in summary.items():
            print(f"{k:<18}{v['mean']:>9}{v['std']:>9}{v['min']:>9}{v['max']:>9}")
        print("\nInterpretation: two configurations differ only if their "
              "mean +/- std ranges do NOT overlap.")
        if repeats < 3:
            print("NOTE: with fewer than 3 repeats the spread is very rough.")
        spath = os.path.join(
            out_dir,
            f"{args.rag}_{ds_label}_t{args.turns}_c{config.num_conversations}"
            f"{_flags}_summary.json")
        with open(spath, "w", encoding="utf-8") as fw:
            json.dump({"rag": args.rag, "dataset": ds_label, "turns": args.turns,
                       "convos": config.num_conversations, "repeats": repeats,
                       "summary": summary, "runs": runs},
                      fw, ensure_ascii=False, indent=2)
        print(f"Saved summary -> {spath}")
    return e_quality


if __name__ == "__main__":
    main()
