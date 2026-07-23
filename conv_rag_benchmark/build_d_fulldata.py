"""
Build D's typed graph over the ENTIRE MultiHopRAG corpus (all 28,711 sentence
passages from corpus_chunks.jsonl — the same source the JudgeAgent B/C graph was
built from), so D becomes a genuine same-data comparison rather than a 600-chunk
subset.

To survive the scale (one LLM extraction call per passage), this does:
  * PARALLEL extraction with a thread pool, and
  * a RESUMABLE JSONL cache (output/d_fulldata_extract_cache.jsonl) keyed by
    passage index — re-running picks up where it left off, no re-spend.

Then it assembles the KnowledgeGraph, generates D's conversations/items exactly
like quality_judge.generate_d_items, re-judges D with the SAME judge model, reuses
the existing B/C scores from quality_bcd.json, and writes quality_bcd_fulldata.json.

Run:
    python -m conv_rag_benchmark.build_d_fulldata --convos 15 --turns 4 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from .config import Config, JUDGE_DATA_DIR, get_logger
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .generation.conversation_generator import ConversationGenerator
from .graph.graph_builder import GraphBuilder, KnowledgeGraph, slugify
from .graph.retriever import GraphRetriever
from .llm import LLM
from .quality_judge import (judge_items, breakdown_by_query_type,
                            _typed_edge_fraction)

logger = get_logger("build_d_fulldata")


def load_all_passages(max_passages=None):
    """Flatten corpus_chunks.jsonl into a flat list of sentence passages."""
    path = os.path.join(JUDGE_DATA_DIR, "MultiHopRAG", "corpus_chunks.jsonl")
    arts = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    passages = [c.strip() for a in arts for c in a.get("chunks", []) if c and c.strip()]
    if max_passages:
        passages = passages[:max_passages]
    return passages


def extract_all(builder: GraphBuilder, passages, cache_path, workers):
    """Parallel + resumable extraction. Returns list[(ents, rels)] aligned to passages."""
    results = [None] * len(passages)
    done = set()
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            i = rec["i"]
            if 0 <= i < len(passages):
                results[i] = (rec["e"], rec["r"])
                done.add(i)
        logger.info("resumed %d/%d passages from cache", len(done), len(passages))

    todo = [i for i in range(len(passages)) if i not in done]
    if not todo:
        return results

    lock = threading.Lock()
    cache_fh = open(cache_path, "a", encoding="utf-8")
    counter = {"n": len(done)}
    t0 = time.time()

    def work(i):
        ents, rels = builder._extract(passages[i])
        return i, ents, rels

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, i) for i in todo]
        for fut in as_completed(futures):
            i, ents, rels = fut.result()
            results[i] = (ents, rels)
            with lock:
                cache_fh.write(json.dumps({"i": i, "e": ents, "r": rels},
                                          ensure_ascii=False) + "\n")
                cache_fh.flush()
                counter["n"] += 1
                if counter["n"] % 250 == 0 or counter["n"] == len(passages):
                    rate = (counter["n"] - len(done)) / max(time.time() - t0, 1e-6)
                    eta = (len(passages) - counter["n"]) / max(rate, 1e-6)
                    logger.info("extracted %d/%d  (%.1f/s, ETA %.0f min)",
                                counter["n"], len(passages), rate, eta / 60)
    cache_fh.close()
    return results


def assemble_graph(extractions, passages, embedder) -> KnowledgeGraph:
    """Mirror GraphBuilder.build assembly, but from cached extractions."""
    graph = nx.MultiDiGraph()
    node_chunks = defaultdict(set)
    for cid, ex in enumerate(extractions):
        if ex is None:
            continue
        ents, rels = ex
        for ent in ents:
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            nid = slugify(ent["name"])
            node_chunks[nid].add(cid)
            if graph.has_node(nid):
                if graph.nodes[nid].get("type") in (None, "Other"):
                    graph.nodes[nid]["type"] = ent.get("type", "Other")
            else:
                graph.add_node(nid, entity=ent["name"],
                               type=ent.get("type", "Other"), chunks=set())
        for rel in rels:
            if not isinstance(rel, dict) or not rel.get("source") or not rel.get("target"):
                continue
            s, t = slugify(rel["source"]), slugify(rel["target"])
            if s == t:
                continue
            for nid, name in ((s, rel["source"]), (t, rel["target"])):
                if not graph.has_node(nid):
                    graph.add_node(nid, entity=name, type="Other", chunks=set())
                node_chunks[nid].add(cid)
            graph.add_edge(s, t, relation=rel.get("relation", "related_to"), chunk=cid)
    for nid, cids in node_chunks.items():
        if graph.has_node(nid):
            graph.nodes[nid]["chunks"] = set(cids)

    kg = KnowledgeGraph(graph=graph, chunks=passages)
    logger.info("graph assembled: %s ; attaching embeddings...", kg.stats())
    # reuse the builder's embedding routine
    if embedder.available:
        kg.chunk_embeddings = embedder.encode(kg.chunks)
        kg.node_ids = list(kg.graph.nodes)
        node_texts = [kg.graph.nodes[n].get("entity", n) for n in kg.node_ids]
        kg.node_embeddings = embedder.encode(node_texts) if node_texts else None
    return kg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build D on full MultiHopRAG + re-judge")
    ap.add_argument("--convos", type=int, default=15)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-passages", type=int, default=None, help="cap (default = all)")
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
        print("!! needs an OpenAI key (RAG-DIVE/.env or OPENAI_API_KEY)")
        return

    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model=config.judge_model, config=config)
    print(f"# gen: {config.gen_model} | judge: {config.judge_model} | workers: {args.workers}")

    # ---- 1. full-corpus typed graph ----
    passages = load_all_passages(args.max_passages)
    print(f"# building D graph over {len(passages):,} passages (ALL MultiHopRAG)")
    builder = GraphBuilder(config=config, llm=gen_llm, embedder=embedder, graph_mode="typed")
    cache_path = os.path.join(config.output_dir, "d_fulldata_extract_cache.jsonl")
    extractions = extract_all(builder, passages, cache_path, args.workers)
    kg = assemble_graph(extractions, passages, embedder)
    graph_path = os.path.join(config.output_dir, "d_fulldata_graph.json")
    kg.save(graph_path)
    d_graph_stats = {"graph_nodes": kg.stats()["n_nodes"],
                     "graph_edges": kg.stats()["n_edges"],
                     "graph_type": "typed-relation (LLM, full corpus)",
                     "pct_typed_edges": _typed_edge_fraction(kg.edges()),
                     "n_passages": len(passages)}
    print(f"# D full-data graph: {d_graph_stats}")

    # ---- 2. generate D items (same as quality_judge.generate_d_items) ----
    seeds = DatasetLoader(config.dataset, max_samples=config.max_samples).load()
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    convgen = ConversationGenerator(kg, config=config, llm=gen_llm, retriever=retriever)
    conversations = convgen.generate_many(seeds[:config.num_conversations])
    d_items = []
    for conv in conversations:
        for turn in conv.turns:
            ev = turn["evidence"].get("chunks", [])
            d_items.append({"question": turn["question"],
                            "gold": turn["gold"]["gold_answer"],
                            "evidence": " ".join(ev)[:1500],
                            "query_type": turn["query_type"]})
    print(f"# generated {len(d_items)} D items; judging...")

    # ---- 3. judge D; reuse B/C from existing quality_bcd.json ----
    d_agg, d_scored = judge_items(judge, d_items)
    d_by_type = breakdown_by_query_type(d_scored)

    prev_path = os.path.join(config.output_dir, "quality_bcd.json")
    prev = json.load(open(prev_path, encoding="utf-8")) if os.path.exists(prev_path) else {}
    scores = dict(prev.get("scores", {}))
    scores["D_full"] = d_agg
    graph_stats = dict(prev.get("graph_stats", {}))
    graph_stats["D_full"] = d_graph_stats

    # ---- 4. report ----
    print("\n===== D (FULL DATA) vs prior B / C / D(600) =====")
    methods = [m for m in ["B", "C", "D", "D_full"] if m in scores]
    names = {"B": "B co-occur", "C": "C typed", "D": "D(600)", "D_full": "D(FULL)"}
    print(f"{'metric':<18}" + "".join(f"{names[m]:>12}" for m in methods))
    for label in ["graph_nodes", "pct_typed_edges"]:
        print(f"{label:<18}" + "".join(
            f"{str(graph_stats.get(m, {}).get(label, '-'))[:11]:>12}" for m in methods))
    for label in ["n", "well_formed", "gold_supported", "gold_correct"]:
        print(f"{label:<18}" + "".join(f"{str(scores[m].get(label, '-')):>12}" for m in methods))

    print("\n--- D(FULL) breakdown by query type ---")
    for qt, s in d_by_type.items():
        print(f"  {qt:<22} n={s['n']:<3} wf={s['well_formed']:<6} "
              f"sup={s['gold_supported']:<6} cor={s['gold_correct']}")

    print(f"\ngen usage  ({config.gen_model}): {gen_llm.cost_report()}")
    print(f"judge usage({config.judge_model}): {judge.cost_report()}")

    out = os.path.join(config.output_dir, "quality_bcd_fulldata.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"graph_stats": graph_stats, "scores": scores,
                   "d_full_by_query_type": d_by_type,
                   "d_full_graph": d_graph_stats,
                   "counts": {m: scores[m].get("n") for m in methods}},
                  fw, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
