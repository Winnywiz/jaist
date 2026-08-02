"""
Neutral generator-quality judge for the B / C / D comparison.

Generates D's (question, gold, evidence) items, loads B and C's items from
``relation_graph/output/quality_bc.json`` (produced by ``quality_bc.py``), then
rates EVERY item with the SAME independent judge model on three axes:

    well_formed     the question is clear, self-contained and answerable
    gold_supported  the gold answer is grounded in / derivable from the evidence
                    (not invented)
    gold_correct    taking the evidence as truth, the gold correctly answers the
                    question

It also reports each method's graph statistics (nodes / edges / % typed edges),
so "which graph is better" and "are the question + gold correct" are answered
together, on identical footing, independent of any target RAG.

Run:
    python -m conv_rag_benchmark.quality_judge --convos 15 --turns 4 \
        --max-samples 200 --d-chunks 600
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from .config import Config, get_logger
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .generation.conversation_generator import ConversationGenerator
from .graph.graph_builder import ENTITY_TYPES, GraphBuilder
from .graph.retriever import GraphRetriever
from .llm import LLM

logger = get_logger("quality_judge")

_GENERIC_RELATIONS = {"co_occurs_with", "related_to"}

_JUDGE_SYS = (
    "You audit one item from an auto-generated QA benchmark. You are given the "
    "EVIDENCE it was built from, the generated QUESTION, and its GOLD answer. "
    "Judge three things strictly and independently:\n"
    "  well_formed   : the QUESTION is clear, self-contained, and actually "
    "answerable (not vague, broken, or under-specified).\n"
    "  gold_supported: the GOLD answer is grounded in / derivable from the "
    "EVIDENCE (not invented or contradicted).\n"
    "  gold_correct  : taking the EVIDENCE as ground truth, the GOLD is a correct "
    "answer to the QUESTION.\n"
    'Respond JSON: {"well_formed": true/false, "gold_supported": true/false, '
    '"gold_correct": true/false}'
)


def _typed_edge_fraction(edges: List[Dict]) -> float:
    if not edges:
        return 0.0
    typed = sum(1 for e in edges if e.get("relation") not in _GENERIC_RELATIONS)
    return round(typed / len(edges), 3)


def generate_d_items(config: Config, gen_llm: LLM, embedder: Embedder,
                     d_chunks: int) -> Dict:
    """Build D's graph + generate conversations; return items + graph stats."""
    samples = DatasetLoader(config.dataset, max_samples=config.max_samples).load()
    chunks: List[str] = []
    for s in samples:
        chunks.extend(s.context)
    chunks = [c for c in chunks if c and c.strip()][:d_chunks]
    logger.info("D: building typed graph over %d chunks", len(chunks))

    kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                      graph_mode="typed").build(chunks)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    convgen = ConversationGenerator(kg, config=config, llm=gen_llm, retriever=retriever)
    conversations = convgen.generate_many(samples[: config.num_conversations])

    items: List[Dict] = []
    for conv in conversations:
        for turn in conv.turns:
            ev = turn["evidence"].get("chunks", [])
            items.append({
                "seed": conv.seed_question,
                "question": turn["question"],
                "gold": turn["gold"]["gold_answer"],
                "evidence": " ".join(ev)[:1500],
                "query_type": turn["query_type"],
            })
    stats = {"graph_nodes": kg.stats()["n_nodes"], "graph_edges": kg.stats()["n_edges"],
             "graph_type": "typed-relation (LLM)",
             "pct_typed_edges": _typed_edge_fraction(kg.edges())}
    logger.info("D: %d items | graph %s", len(items), stats)
    return {"items": items, "graph_stats": stats}


def judge_items(judge: LLM, items: List[Dict]):
    """Score a list of items. Returns (aggregate_rates, items_with_scores)."""
    wf = sup = cor = n = 0
    scored: List[Dict] = []
    for it in items:
        q, gold, ev = it.get("question", ""), it.get("gold", ""), it.get("evidence", "")
        if not q or not gold:
            continue
        out = judge.chat_json(
            _JUDGE_SYS,
            f"EVIDENCE:\n{str(ev)[:1600]}\n\nQUESTION: {q}\nGOLD: {gold}")
        if out is None:
            continue
        n += 1
        rec = {"well_formed": bool(out.get("well_formed")),
               "gold_supported": bool(out.get("gold_supported")),
               "gold_correct": bool(out.get("gold_correct")),
               "query_type": it.get("query_type")}
        wf += rec["well_formed"]
        sup += rec["gold_supported"]
        cor += rec["gold_correct"]
        scored.append(rec)
    safe = lambda x: round(x / n, 3) if n else 0.0
    agg = {"n": n, "well_formed": safe(wf), "gold_supported": safe(sup),
           "gold_correct": safe(cor)}
    return agg, scored


def breakdown_by_query_type(scored: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-item judgements by query_type (for D's Comparative/Multi-Hop)."""
    buckets: Dict[str, List[Dict]] = {}
    for r in scored:
        buckets.setdefault(r.get("query_type") or "unknown", []).append(r)
    out = {}
    for qt, rows in sorted(buckets.items()):
        k = len(rows)
        out[qt] = {"n": k,
                   "well_formed": round(sum(r["well_formed"] for r in rows) / k, 3),
                   "gold_supported": round(sum(r["gold_supported"] for r in rows) / k, 3),
                   "gold_correct": round(sum(r["gold_correct"] for r in rows) / k, 3)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="B/C/D generator-quality judge")
    ap.add_argument("--dataset", default="multihoprag")
    ap.add_argument("--convos", type=int, default=15)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--d-chunks", type=int, default=600)
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--bc", default=None, help="path to quality_bc.json (B/C items)")
    args = ap.parse_args(argv)

    config = Config.load(dataset=args.dataset, max_samples=args.max_samples,
                         num_conversations=args.convos,
                         min_turns=args.turns, max_turns=args.turns,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key (RAG-DIVE/.env)")
        return

    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model=config.judge_model, config=config)
    print(f"# gen model: {config.gen_model} | judge model: {config.judge_model}")

    # --- B / C items (from quality_bc.py) ---
    bc_path = args.bc or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "relation_graph", "output", "quality_bc.json")
    if not os.path.exists(bc_path):
        print(f"!! B/C items not found at {bc_path}\n"
              f"   run relation_graph/quality_bc.py first.")
        return
    with open(bc_path, "r", encoding="utf-8") as fr:
        bc = json.load(fr)
    b_items, c_items = bc["items"]["B"], bc["items"]["C"]
    graph_stats = dict(bc["graph_stats"])

    # --- D items (generate now) ---
    d = generate_d_items(config, gen_llm, embedder, args.d_chunks)
    d_items = d["items"]
    graph_stats["D"] = d["graph_stats"]

    # --- judge all three with the same model ---
    print(f"# judging B={len(b_items)} C={len(c_items)} D={len(d_items)} items...")
    b_agg, _ = judge_items(judge, b_items)
    c_agg, _ = judge_items(judge, c_items)
    d_agg, d_scored = judge_items(judge, d_items)
    scores = {"B": b_agg, "C": c_agg, "D": d_agg}
    d_by_type = breakdown_by_query_type(d_scored)

    # --- report ---
    print("\n========== B / C / D GENERATOR-QUALITY (same judge, MultiHopRAG) ==========")
    methods = ["B", "C", "D"]
    names = {"B": "B co-occur", "C": "C typed", "D": "D conv_rag"}
    print(f"{'metric':<20}" + "".join(f"{names[m]:>14}" for m in methods))
    grows = [("graph_type", lambda m: graph_stats[m].get("graph_type", "-")),
             ("graph_nodes", lambda m: graph_stats[m].get("graph_nodes", "-")),
             ("pct_typed_edges", lambda m: graph_stats[m].get("pct_typed_edges", "-"))]
    for label, fn in grows:
        print(f"{label:<20}" + "".join(f"{str(fn(m))[:13]:>14}" for m in methods))
    for label in ["n", "well_formed", "gold_supported", "gold_correct"]:
        print(f"{label:<20}" + "".join(f"{str(scores[m][label]):>14}" for m in methods))

    print("\n--- how to read it ---")
    print("  pct_typed_edges: share of graph edges with a real relation label (higher=richer graph)")
    print("  well_formed    : generated question is clear & answerable")
    print("  gold_supported : gold answer is backed by the evidence (not invented)")
    print("  gold_correct   : gold actually answers the question, given the evidence")
    # D per-query-type (does the richer ontology help Comparative / Multi-Hop?)
    print("\n--- D breakdown by query type (well_formed / gold_supported / gold_correct) ---")
    for qt, s in d_by_type.items():
        print(f"  {qt:<22} n={s['n']:<3} "
              f"wf={s['well_formed']:<6} sup={s['gold_supported']:<6} cor={s['gold_correct']}")

    print(f"\ngen usage  ({config.gen_model}): {gen_llm.cost_report()}")
    print(f"judge usage({config.judge_model}): {judge.cost_report()}")

    out = os.path.join(config.output_dir, "quality_bcd.json")
    with open(out, "w", encoding="utf-8") as fw:
        json.dump({"graph_stats": graph_stats, "scores": scores,
                   "d_by_query_type": d_by_type,
                   "entity_types": ENTITY_TYPES,
                   "counts": {"B": len(b_items), "C": len(c_items), "D": len(d_items)}},
                  fw, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
