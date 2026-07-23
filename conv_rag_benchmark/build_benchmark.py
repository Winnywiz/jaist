"""
Random-type conversational RAG benchmark.

For each of N conversations (default 20), build a 3-turn dialogue:
  * turn 0 = the factual seed question;
  * turns 1-2 = a RANDOMLY chosen query type (from the 8 types).
Questions + gold are authored from the GRAPH/CORPUS truth (gold-based, so the
benchmark is fair + reproducible); the target RAG then answers each question and
we log everything needed to audit it:

  per turn -> {
     query_type,
     question,
     source_doc,                # which corpus doc/seed the question came from
     question_evidence_context, # the chunks (context in that doc) used to AUTHOR the question + gold
     gold_answer,
     rag_retrieved_context,     # the chunks the RAG retrieved to ANSWER
     rag_answer,                # the RAG's final answer
  }

Then it scores QUESTION QUALITY:
  well_formed, gold_supported, gold_correct        (LLM judge, reused from quality_judge)
  type_diversity                                    (algorithmic)
  doc_required  (anti-cheating)                     (LLM answers w/o the doc; if correct -> cheatable)
  followup_dependency (non-standalone)              (LLM judge: needs the prior turn?)

Run:
    python -m conv_rag_benchmark.build_benchmark --convos 20 --turns 3 --rag vector
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .generation.gold_answer_generator import GoldAnswerGenerator, ABSTENTION
from .generation.query_generator import QUERY_TYPES, QueryGenerator, QueryTurn
from .graph.graph_builder import GraphBuilder, KnowledgeGraph
from .graph.retriever import GraphRetriever, RetrievalResult
from .interfaces.rag_interface import build_rag
from .llm import LLM
from .quality_judge import judge_items, breakdown_by_query_type
from .geval import geval_items, geval_breakdown_by_type


# -- extra question-quality judges (not in quality_judge) -------------------- #
_NODOC_SYS = ('Answer the question using ONLY your own knowledge. Do not guess wildly; '
             'if you do not know, answer "UNKNOWN". Respond JSON: {"answer": "..."}')
_MATCH_SYS = ('Does the CANDIDATE answer match the GOLD answer (same fact)? '
             'Respond JSON: {"match": true/false}')
_STANDALONE_SYS = ('Can the QUESTION be fully understood on its own, WITHOUT the previous turn? '
                  'Respond JSON: {"standalone": true/false}')


def doc_required(gen_llm: LLM, judge: LLM, question: str, gold: str) -> bool:
    """True if the question genuinely needs the document (cannot be answered from
    parametric knowledge alone). i.e. NOT cheatable."""
    out = gen_llm.chat_json(_NODOC_SYS, f"QUESTION: {question}") or {}
    cand = str(out.get("answer", "")).strip()
    if not cand or cand.upper() == "UNKNOWN":
        return True
    m = judge.chat_json(_MATCH_SYS, f"GOLD: {gold}\nCANDIDATE: {cand}") or {}
    return not bool(m.get("match"))


def is_standalone(judge: LLM, prev_q: str, question: str) -> bool:
    out = judge.chat_json(_STANDALONE_SYS,
                          f"PREVIOUS TURN: {prev_q}\nQUESTION: {question}") or {}
    return bool(out.get("standalone", True))


def bridge_path_evidence(kg, rng, max_sources=200):
    """Semantic-Bridge style ENTITY BRIDGING: find a 2-hop path
    A --[r1]--> B --[r2]--> C across THREE DISTINCT entities, so a question over it
    requires real multi-hop reasoning (chain A->B->C), not a single-fact lookup.
    Returns a RetrievalResult (the path's nodes/edges/chunks) or None.
    Ref: Semantic Bridge (2025), Difficulty-Controllable Multi-hop QG from KGs (2019)."""
    from collections import defaultdict
    adj = defaultdict(list)
    for e in kg.edges():
        if e.get("relation"):
            adj[e["source"]].append((e["target"], e["relation"]))
    sources = [s for s in adj if adj[s]]
    rng.shuffle(sources)
    for a in sources[:max_sources]:
        for b, r1 in adj[a]:
            for c, r2 in adj.get(b, []):
                if len({a, b, c}) == 3:                      # distinct -> real bridge
                    chunks = []
                    for nid in (a, b, c):
                        for ch in kg.chunks_for(nid):
                            if ch not in chunks:
                                chunks.append(ch)
                    if len(chunks) < 2:                      # need evidence spread
                        continue
                    return RetrievalResult(
                        nodes=[kg.node(a), kg.node(b), kg.node(c)],
                        edges=[{"source": a, "target": b, "relation": r1},
                               {"source": b, "target": c, "relation": r2}],
                        chunks=chunks[:8])
    return None


def make_seed_question(gen_llm, evidence_text: str) -> str:
    """Generate ONE factual seed question from evidence (for corpus-only datasets
    like MedQA that ship no pre-made questions)."""
    out = gen_llm.chat_json(
        'Write ONE clear, specific factual question that is answerable from the TEXT. '
        'Do not mention "text" or "passage". Respond JSON: {"question": "..."}',
        f"TEXT: {evidence_text[:1500]}") or {}
    return str(out.get("question", "")).strip()


def algorithmic_grounding(items, embedder) -> dict:
    """MODEL-FREE grounding cross-check (no LLM judge): embedding cosine + lexical
    overlap between each gold answer and its evidence. Bias-free alternative to the
    LLM-judged gold_supported."""
    golds = [it["gold"] for it in items]
    evs = [it["evidence"] for it in items]
    gv = np.asarray(embedder.encode(golds), dtype=float)
    ev = np.asarray(embedder.encode(evs), dtype=float)

    def _norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n
    cos = (_norm(gv) * _norm(ev)).sum(axis=1)

    def _toks(s):
        return set(w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 3)
    overlaps = []
    for it in items:
        g = _toks(it["gold"])
        overlaps.append(len(g & _toks(it["evidence"])) / len(g) if g else 0.0)
    return {"embedding_cosine": round(float(cos.mean()), 3),
            "lexical_overlap": round(sum(overlaps) / len(overlaps), 3),
            "note": "model-free grounding (no LLM judge): gold vs evidence"}


def score_and_save(items, conversations, graph_stats, type_pool,
                   config, gen_llm, embedder, judge, out_dir, args):
    """Score generated items (G-Eval primary + plain + model-free cross-checks +
    diversity/anti-cheat/dependency) and write benchmark_random.json. Separated so
    a killed scoring phase can be resumed from the _progress.json checkpoint."""
    print("# scoring question quality ...")
    geval_agg, geval_scored = geval_items(items, model=config.judge_model)
    geval_by_type = geval_breakdown_by_type(geval_scored)
    agg, scored = judge_items(judge, items)
    by_type = breakdown_by_query_type(scored)
    algo_grounding = algorithmic_grounding(items, embedder)

    all_types = [it["query_type"] for it in items]
    distinct_per_conv = []
    for c in conversations:
        ts = [t["query_type"] for t in c["turns"]]
        distinct_per_conv.append(len(set(ts)) / len(ts) if ts else 0)
    type_diversity = {
        "distinct_types_used": len(set(all_types)),
        "of_total_types": len(type_pool),
        "avg_distinct_per_conversation": round(sum(distinct_per_conv) / len(distinct_per_conv), 3),
    }

    doc_req = nonstandalone = nonstandalone_n = 0
    for it in items:
        if doc_required(gen_llm, judge, it["question"], it["gold"]):
            doc_req += 1
        if it["turn_id"] > 0:
            nonstandalone_n += 1
            if not is_standalone(judge, it.get("prev_q") or "", it["question"]):
                nonstandalone += 1
    n = len(items)
    anti_cheating = {"doc_required_rate": round(doc_req / n, 3),
                     "note": "fraction that CANNOT be answered without the document (higher = less cheatable)"}
    followup_dependency = {"non_standalone_rate": round(nonstandalone / nonstandalone_n, 3) if nonstandalone_n else 0,
                           "note": "fraction of follow-up turns that genuinely depend on the prior turn"}

    report = {
        "config": {"convos": args.convos, "turns": args.turns, "rag": args.rag,
                   "seed": args.seed, "gen_model": config.gen_model,
                   "judge_model": config.judge_model},
        "graph_stats": graph_stats,
        "question_quality": {
            "scoring_method": "G-Eval (CoT + probability-weighted 1-5, normalized 0-1)",
            "well_formed": geval_agg["well_formed"],
            "gold_supported": geval_agg["gold_supported"],
            "gold_correct": geval_agg["gold_correct"],
            "by_query_type": geval_by_type,
            "cross_checks": {
                "plain_llm_judge": {"well_formed": agg["well_formed"],
                                    "gold_supported": agg["gold_supported"],
                                    "gold_correct": agg["gold_correct"],
                                    "by_query_type": by_type},
                "model_free_grounding": algo_grounding,
            },
            "type_diversity": type_diversity,
            "anti_cheating": anti_cheating,
            "followup_dependency": followup_dependency,
        },
        "llm_usage": {"gen": gen_llm.cost_report(), "judge": judge.cost_report()},
        "conversations": conversations,
    }
    out = os.path.join(out_dir, "benchmark_random.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n===== QUESTION QUALITY (n=%d questions) =====" % n)
    print("  [PRIMARY: G-Eval]      plain-judge (cross-check)")
    print(f"  well_formed    : {geval_agg['well_formed']:<8} {agg['well_formed']}")
    print(f"  gold_supported : {geval_agg['gold_supported']:<8} {agg['gold_supported']}")
    print(f"  gold_correct   : {geval_agg['gold_correct']:<8} {agg['gold_correct']}")
    print(f"  [model-free] grounding : embed_cos {algo_grounding['embedding_cosine']} | lexical {algo_grounding['lexical_overlap']}")
    print(f"  type diversity : {type_diversity['distinct_types_used']}/{type_diversity['of_total_types']}"
          f" | doc_required {anti_cheating['doc_required_rate']} | dependency {followup_dependency['non_standalone_rate']}")
    print(f"\nSaved -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Random-type conversational RAG benchmark")
    ap.add_argument("--dataset", default="multihoprag",
                    help="multihoprag | medqa | hotpotqa | 2wikimultihopqa | musique")
    ap.add_argument("--convos", type=int, default=20)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--d-chunks", type=int, default=600)
    ap.add_argument("--rag", default="vector", choices=["mock", "vector", "graph"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--score-only", action="store_true",
                    help="resume: skip graph+generation, score the saved _progress.json")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    config = Config.load(dataset=args.dataset, max_samples=max(args.convos, 50),
                         num_conversations=args.convos,
                         min_turns=args.turns, max_turns=args.turns,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key (RAG-DIVE/.env or OPENAI_API_KEY)")
        return

    # per-dataset output folder, e.g. output/MedQA/ , output/MultiHopRAG/
    LABELS = {"multihoprag": "MultiHopRAG", "medqa": "MedQA", "hotpotqa": "HotpotQA",
              "2wikimultihopqa": "2WikiMultiHopQA", "musique": "MuSiQue",
              "arxivcs": "ArXivCS"}
    ds_label = LABELS.get(args.dataset, args.dataset)
    out_dir = os.path.join(config.output_dir, ds_label)
    os.makedirs(out_dir, exist_ok=True)

    gen_llm = LLM(model=config.gen_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)
    judge = LLM(model=config.judge_model, config=config)
    print(f"# dataset: {ds_label} -> output {out_dir}")
    print(f"# gen: {config.gen_model} | judge: {config.judge_model} | RAG: {args.rag}")
    if config.gen_model == config.judge_model:
        print("!! WARNING: judge_model == gen_model -> self-preference bias. "
              "Use a DIFFERENT judge (e.g. --judge-model gpt-4o). Continuing anyway.")

    # RESUME: if scoring was killed, score the saved checkpoint and stop.
    if args.score_only:
        prog = json.load(open(os.path.join(out_dir, "_progress.json"), encoding="utf-8"))
        print(f"# score-only: scoring {len(prog['conversations'])} saved conversations")
        score_and_save(prog["items"], prog["conversations"], prog["graph_stats"],
                       prog["type_pool"], config, gen_llm, embedder, judge, out_dir, args)
        return

    # ---- 1. typed graph + retriever (same as D/E) ----
    seeds = DatasetLoader(config.dataset, max_samples=config.max_samples).load()
    chunks = [c for s in seeds for c in s.context if c and c.strip()][:args.d_chunks]
    graph_path = os.path.join(out_dir, "graph.json")
    if os.path.exists(graph_path):
        # reuse the saved graph -> skip the slow LLM relation-extraction build
        print(f"# loading saved graph from {graph_path} (skipping rebuild)")
        kg = KnowledgeGraph.load(graph_path)
        try:
            kg.chunk_embeddings = np.asarray(embedder.encode(kg.chunks), dtype=float)
        except Exception as _e:
            print(f"  (chunk embedding failed, using lexical retrieval: {str(_e)[:80]})")
    else:
        print(f"# building typed graph over {len(chunks)} chunks")
        kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                          graph_mode="typed").build(chunks)
        kg.save(graph_path)  # so the viewer / future runs can skip the rebuild
    graph_stats = {"nodes": len(kg.nodes()), "edges": len(kg.edges()), "chunks": len(kg.chunks)}
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    target_rag = build_rag(args.rag, kg.chunks, config=config, llm=gen_llm,
                           retriever=retriever, embedder=embedder)
    query_gen = QueryGenerator(config=config, llm=gen_llm)
    gold_gen = GoldAnswerGenerator(config=config, llm=gen_llm)

    # types available for the random follow-up turns (turn 0 is always the seed)
    type_pool = list(QUERY_TYPES)

    # RESUME generation from a per-conversation checkpoint if one exists
    progress_path = os.path.join(out_dir, "_progress.json")
    conversations, items, start_idx = [], [], 0
    if os.path.exists(progress_path):
        prog = json.load(open(progress_path, encoding="utf-8"))
        if 0 < len(prog.get("conversations", [])) < args.convos:
            conversations, items = prog["conversations"], prog["items"]
            start_idx = len(conversations)
            print(f"# resuming generation from conversation {start_idx}/{args.convos}")

    def _checkpoint():
        json.dump({"items": items, "conversations": conversations, "graph_stats": graph_stats,
                   "type_pool": type_pool, "convos": args.convos, "turns": args.turns},
                  open(progress_path, "w", encoding="utf-8"), ensure_ascii=False)

    for ci, seed in enumerate(seeds[: args.convos]):
        if ci < start_idx:
            continue  # already generated in a previous run
        truth_history, rag_history = [], []
        conv = {"conversation_id": f"conv-{ci:03d}", "source_doc": seed.id,
                "seed_question": seed.question, "turns": []}
        prev_q = None
        # corpus-only datasets (MedQA) have no seed question -> use first chunk as focus
        seed_focus = seed.question or (seed.context[0][:400] if seed.context else "")
        for turn_id in range(args.turns):
            # 1. choose the type: turn 0 = factual seed, else RANDOM
            qtype = "Follow-Up" if turn_id == 0 else rng.choice(type_pool)

            # 2. retrieve evidence + author question (gold-based, from truth history)
            focus = truth_history[-2]["content"] if len(truth_history) >= 2 else seed_focus
            evidence = retriever.retrieve(focus if turn_id else seed_focus, k=8,
                                          conversation_history=truth_history)
            if turn_id == 0:
                seed_q = seed.question or make_seed_question(gen_llm, evidence.evidence_text(1500))
                conv["seed_question"] = seed_q
                turn = QueryTurn(question=seed_q, query_type="Follow-Up",
                                 turn_id=0, difficulty=1,
                                 capability="Factual Retrieval (seed)",
                                 expected_failure="Missing Retrieval",
                                 conversation_history=list(truth_history),
                                 meta={"seed_id": seed.id})
            else:
                # NOTE: bridge walk (bridge_path_evidence) was tested for Multi-Hop but
                # LOWERED depth on this co-occurrence graph (2.53->1.73), so we keep the
                # similarity walk. See test_bridge.py for the negative result.
                turn = query_gen.generate(qtype, turn_id, evidence, truth_history)
                # re-retrieve for the authored (possibly drifted) question
                evidence = retriever.retrieve(turn.question, k=8,
                                              conversation_history=truth_history)
            gold = gold_gen.generate(turn, evidence, truth_history)
            gold_text = gold.gold_answer

            # 3. the RAG answers (it sees its own prior answers)
            rag_resp = target_rag.answer(turn.question, rag_history)

            # 4. log the full turn
            conv["turns"].append({
                "turn_id": turn_id,
                "query_type": turn.query_type,
                "question": turn.question,
                "source_doc": seed.id,
                "question_evidence_context": evidence.chunks[:5],
                "gold_answer": gold_text,
                "rag_retrieved_context": rag_resp.retrieved_context[:5],
                "rag_answer": rag_resp.answer,
            })
            items.append({"question": turn.question, "gold": gold_text,
                          "evidence": " ".join(evidence.chunks[:5])[:1800],
                          "query_type": turn.query_type,
                          "turn_id": turn_id, "prev_q": prev_q})

            truth_history += [{"role": "user", "content": turn.question},
                              {"role": "assistant", "content": gold_text}]
            rag_history += [{"role": "user", "content": turn.question},
                            {"role": "assistant", "content": rag_resp.answer}]
            prev_q = turn.question
        conversations.append(conv)
        _checkpoint()  # save after EVERY conversation so a kill never loses generation
        print(f"  [{ci+1}/{args.convos}] {conv['conversation_id']} done")

    # ---- 2. score (generation already checkpointed per-conversation above) ----
    score_and_save(items, conversations, graph_stats, type_pool,
                   config, gen_llm, embedder, judge, out_dir, args)


if __name__ == "__main__":
    main()
