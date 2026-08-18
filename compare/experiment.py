"""
compare/experiment.py — run the FOUR conversation-generation methods across the SAME
RAGs and save one directory per conversation, in a schema the failure classifier can
consume unchanged.

    proposed        dynamic follow-up generation  (content_policy="dynamic")
    xie             static  follow-up generation  (content_policy="static")
    single_turn_qa  seed turn only, no follow-up   (n_turns=1)
    mtrag           REPLAY human mtRAG conversations  (deferred: needs a corpus decision)

The method is the conversation SOURCE; the RAG is the system under test. Everything the
generation and RAG machinery needs is reused from ``conv_rag_benchmark`` — this file only
orchestrates and writes results. It changes no RAG, no generator semantics, and nothing
in ``failure/``.

Output (one dir per conversation, room for failuretypes.json / attribution.json beside it):

    compare/result/<method>/<rag>/conversation_NNN/conversation.json

The conversation.json wraps the log in a ``"conversations": [...]`` list, so
``python -m failure.run --file <that path>`` reads it with no changes.

Run from the repo root:

    # PILOT (this is what to run first): 1 conversation x 2 turns, proposed x vector x qasper
    python -m compare.experiment --method proposed --rag vector --dataset qasper \
        --convos 1 --turns 2 --seed 42

    # later, at scale:
    python -m compare.experiment --method all --rag all --dataset qasper \
        --convos 30 --turns 9 --seed 42
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Dict, List, Optional

import numpy as np

from conv_rag_benchmark.config import Config
from conv_rag_benchmark.connectors import (available_rags, connect_dataset,
                                           connect_rag)
from conv_rag_benchmark.embeddings import Embedder
from conv_rag_benchmark.embeddings_cache import load_or_build_embeddings
from conv_rag_benchmark.generation.dynamic_generator import (
    AdaptiveConversationGenerator, has_latex_label, LATEX_LABEL)
from conv_rag_benchmark.generation.mtrag_generator import MtragReplayGenerator
from conv_rag_benchmark.generation.single_turn_generator import SingleTurnQAGenerator
from conv_rag_benchmark.generation.xie_generator import XieSubQuestionGenerator
from conv_rag_benchmark.generation.gold_answer_generator import GoldAnswerGenerator, ABSTENTION
from conv_rag_benchmark.generation.query_generator import QueryTurn
from conv_rag_benchmark.datasets.mtrag_conversations import load_mtrag_conversations
from conv_rag_benchmark.graph.graph_builder import GraphBuilder
from conv_rag_benchmark.graph.retriever import GraphRetriever
from conv_rag_benchmark.llm import LLM

# compare/experiment.py -> compare/result
RESULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

#: The four experiment methods and how each maps onto the existing generator knobs.
#: ``content`` is the thesis IV (static vs dynamic); ``turns`` overrides the CLI turn
#: count where a method fixes it. ``replay`` marks the human-conversation method, which
#: is deferred until the mtRAG corpus decision is made.
METHODS: Dict[str, Dict] = {
    "proposed":       {"content": "dynamic", "turns": None, "replay": False, "generator": "adaptive"},
    "xie":            {"content": "static",  "turns": None, "replay": False, "generator": "adaptive"},
    "single_turn_qa": {"content": "static",  "turns": 1,    "replay": False, "generator": "single_turn"},
    "mtrag":          {"content": None,       "turns": None, "replay": True,  "generator": "replay"},
    # xie_subq = a THESIS ADAPTATION of Xie et al. (NAACL 2025): reuse the paper's
    # sub-question DECOMPOSITION (core/background/follow-up, computed once, statically) and
    # replay the sub-questions as conversation turns. Replaying-as-turns is NOT the paper's
    # method — the FAITHFUL coverage/attribution method is compare/xie_coverage.py. Distinct
    # from 'xie' (the older static content arm).
    "xie_subq":       {"content": "static",  "turns": None, "replay": False, "generator": "subquestion"},
}


def _build_setup(dataset: str, rag_name: str, d_chunks: int, config: Config,
                 gen_llm: LLM, judge: LLM, embedder: Embedder, out_dir: str,
                 rag_k: Optional[int]):
    """Build the corpus + retriever + target RAG once for a (dataset, rag) pair.

    Mirrors the minimal setup ``run_benchmark`` uses, but keeps generation graph-free
    (``graph_mode="none"`` = dense-only) unless the RAG under test is GraphRAG, which
    needs a real typed graph. The question-generation retriever and the corpus RAG are
    built over the SAME ``kg.chunks``, so the two document sets share a doc_id space and
    are directly comparable.
    """
    seeds = connect_dataset(dataset, max_samples=max(d_chunks, 50))
    chunks = [c for s in seeds for c in s.context if c and c.strip()][:d_chunks]
    if not chunks:
        raise RuntimeError(f"dataset '{dataset}' yielded no corpus chunks")

    # graph AND hippo both need the typed OpenIE graph (hippo runs PPR over it)
    graph_mode = "typed" if rag_name in ("graph", "hippo") else "none"
    kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                      graph_mode=graph_mode).build(chunks)
    try:
        kg.chunk_embeddings = np.asarray(embedder.encode(kg.chunks), dtype=float)
    except Exception as exc:                                   # pragma: no cover
        print(f"  (chunk embedding failed: {str(exc)[:80]})")
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    target_rag = connect_rag(rag_name, kg.chunks, config=config, llm=gen_llm,
                             retriever=retriever, embedder=embedder, cache_dir=out_dir,
                             **({} if rag_k is None else {"k": rag_k}))
    # SEED filter (same as run_benchmark): drop seeds whose question carries a raw LaTeX
    # label so turn 0 is well-formed; fall back to all seeds if too few remain.
    seed_pool = [s for s in seeds if not has_latex_label(s.question or "")] or seeds
    return kg, retriever, target_rag, seed_pool


def _conversation_metadata(method: str, rag_name: str, dataset: str, seed: int,
                           n_turns: int, content: str, config: Config,
                           rag_k: Optional[int], n_chunks: int) -> Dict:
    """Reproducibility block stored at the top of every conversation.json."""
    return {
        "method": method,
        "rag": rag_name,
        "dataset": dataset,
        "seed": seed,
        "n_turns": n_turns,
        "content_policy": content,
        "generation_source": ("rag_history" if content == "dynamic"
                              else "truth_history" if content == "static" else None),
        "gen_model": config.gen_model,
        "judge_model": config.judge_model,
        "retrieval": {
            "corpus_chunks": n_chunks,
            "rag_k": rag_k,
            "note": "score meaning differs per RAG; see rag_interface docstring. rank is "
                    "comparable across RAGs, raw score is not.",
        },
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
    }


def _save_conversation(path_dir: str, conv, meta: Dict, overwrite: bool) -> Optional[str]:
    """Write one conversation.json (wrapped as a run_benchmark-style log)."""
    os.makedirs(path_dir, exist_ok=True)
    out_path = os.path.join(path_dir, "conversation.json")
    if os.path.exists(out_path) and not overwrite:
        print(f"  ! exists, skipping (use --overwrite): {out_path}")
        return None
    payload = dict(meta)
    payload["conversation_id"] = conv.conversation_id
    # run_benchmark-compatible: the failure classifier reads data["conversations"][*].turns
    payload["conversations"] = [conv.to_dict()]
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)
    return out_path


#: mtRAG domains usable as a --dataset for the generated methods (same-corpus arm).
MTRAG_DOMAINS = {"clapnq", "govt", "fiqa", "cloud"}
#: RAGs feasible over the 366K-passage official corpus (embed-only, no per-passage LLM).
_MTRAG_FEASIBLE_RAGS = {"vector", "selfrag", "crag", "longrag", "mock"}


def _mtrag_domain_setup(domain: str, rag_name: str, corpus_path: str, convos: int,
                        config: Config, gen_llm: LLM, embedder: Embedder,
                        rag_k: Optional[int]):
    """Build a generated-method setup OVER the official mtRAG domain corpus, seeded by the
    domain's human seed questions — so Dynamic/Xie run on the SAME corpus + SAME starting
    questions as the mtRAG replay (the controlled same-corpus arm)."""
    from conv_rag_benchmark.datasets.loader import Sample
    chunks, _native = _load_official_domain(corpus_path, domain)
    emb = load_or_build_embeddings(embedder, chunks,
                                   os.path.join(corpus_path, f"{domain}.embindex"))
    kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                      graph_mode="none").build(chunks)          # embed-only (no OpenIE)
    if emb is not None:
        kg.chunk_embeddings = np.asarray(emb, dtype=float)
    retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
    target_rag = connect_rag(rag_name, chunks, config=config, llm=gen_llm,
                             retriever=retriever, embedder=embedder, precomputed_emb=emb,
                             **({} if rag_k is None else {"k": rag_k}))
    # seeds = each human conversation's FIRST question — taken from the SAME first-N mtRAG
    # conversations the mtRAG replay uses, filtered to this domain, so the per-domain counts
    # match mtRAG exactly (clapnq 3 / govt 4 / fiqa 6 / cloud 2 for the first 15).
    seeds = []
    for c in [c for c in load_mtrag_conversations()[:convos] if _domain_of(c) == domain]:
        if c.turns:
            t0 = c.turns[0]
            seeds.append(Sample(id=c.conversation_id, question=t0.question,
                                answer=t0.answer, context=[x.text for x in t0.contexts]))
    return kg, retriever, target_rag, seeds


def _ground_seed_text(seed_q: str, retriever, gold_gen: GoldAnswerGenerator,
                      gen_llm: LLM) -> str:
    """Ground ONE seed question against the corpus (rewrite it from evidence if the raw
    dataset question is unanswerable). Identical logic to the generators' own seed guard,
    lifted here so the SAME grounded seed can be shared across every method."""
    if not gen_llm.available:
        return seed_q
    for attempt in range(3):
        ev = retriever.retrieve(seed_q, k=8 if attempt == 0 else 16)
        qt = QueryTurn(question=seed_q, query_type="Seed", turn_id=0, difficulty=1,
                       capability="", expected_failure="", is_unanswerable=False)
        gold = gold_gen.generate(qt, ev, [])
        if not gold.gold_answer.startswith(ABSTENTION[:20]):
            return seed_q
        ev_text = LATEX_LABEL.sub(" ", ev.evidence_text(1500))
        out = gen_llm.chat_json(
            'Write ONE clear, specific factual question answerable from the TEXT. Do not '
            'mention "text" or "passage"; do not reference tables/sections/figures by label. '
            'Respond JSON: {"question": "..."}', f"TEXT: {ev_text}") or {}
        newq = str(out.get("question") or "").strip()
        if newq:
            seed_q = LATEX_LABEL.sub("", newq).strip()
    return seed_q


def _load_or_ground_seeds(seeds, cache_path: str, retriever,
                          gold_gen: GoldAnswerGenerator, gen_llm: LLM) -> Dict[str, str]:
    """Return {seed_id: grounded_question}, computing (and caching to disk) any missing.

    The cache is keyed by dataset/rag/corpus/run-seed (the file name), NOT by method, so
    the first method to run grounds the seeds and every later method reuses the EXACT same
    grounded question — that is what makes Dynamic and Xie start identically."""
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}
    out, changed = {}, False
    for s in seeds:
        sid = str(s.id)
        if sid not in cache:
            cache[sid] = _ground_seed_text((s.question or "").strip(), retriever,
                                           gold_gen, gen_llm)
            changed = True
        out[sid] = cache[sid]
    if changed:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fw:
            json.dump(cache, fw, ensure_ascii=False, indent=2)
    return out


def _with_question(seed, question: str):
    """Shallow copy of a Sample with its question replaced (leaves the original intact)."""
    import copy
    s2 = copy.copy(seed)
    s2.question = question
    return s2


def run_generated_method(method: str, rag_name: str, dataset: str, convos: int,
                         turns: int, seed: int, d_chunks: int, config: Config,
                         gen_llm: LLM, judge: LLM, embedder: Embedder,
                         rag_k: Optional[int], overwrite: bool,
                         forced_types: Optional[List[str]] = None,
                         label: Optional[str] = None,
                         shared_seed: bool = False,
                         inject_unanswerable_at: Optional[int] = None,
                         corpus_path: Optional[str] = None) -> List[str]:
    """Run one generation method (proposed / xie / single_turn_qa) against one RAG.

    ``forced_types`` (when given) fixes the per-turn follow-up type sequence via the
    generator's existing ``forced_types`` knob — used to reach ``Unanswerable`` reliably
    without changing the normal adaptive controller. ``label`` routes output to a separate
    tree ``compare/result/<label>/<method>/<rag>/`` so this never touches the main results.
    ``shared_seed`` grounds each seed ONCE (cached, keyed by dataset/rag/corpus/run-seed)
    and feeds the SAME grounded question to every method, so the only difference between
    Dynamic and Xie is the follow-up strategy — not the starting question.
    None of these change the static/dynamic content policy (the IV)."""
    spec = METHODS[method]
    content = spec["content"]
    n_turns = spec["turns"] or turns

    root = os.path.join(RESULT_ROOT, label) if label else RESULT_ROOT
    # layout: [<label>/]<method>/<rag>/<dataset>/conversation_NNN/  — dataset in the path
    out_root = os.path.join(root, method, rag_name, dataset)
    os.makedirs(out_root, exist_ok=True)
    print(f"\n=== method={method} | rag={rag_name} | dataset={dataset} | "
          f"convos={convos} | turns={n_turns} | seed={seed}"
          f"{' | forced_types=' + ','.join(forced_types) if forced_types else ''} ===")

    # SAME-CORPUS mtRAG arm: when the dataset is an mtRAG domain, build over the OFFICIAL
    # domain corpus with the human seed questions (instead of the qasper-style setup).
    if dataset in MTRAG_DOMAINS:
        if not corpus_path:
            raise RuntimeError(f"--dataset {dataset} (mtRAG domain) needs --mtrag-corpus-path")
        if rag_name not in _MTRAG_FEASIBLE_RAGS:
            print(f"  ! rag={rag_name} is infeasible over the 366K mtRAG corpus "
                  f"(needs per-passage LLM indexing) — skipping.")
            return []
        kg, retriever, target_rag, seed_pool = _mtrag_domain_setup(
            dataset, rag_name, corpus_path, convos, config, gen_llm, embedder, rag_k)
    else:
        kg, retriever, target_rag, seed_pool = _build_setup(
            dataset, rag_name, d_chunks, config, gen_llm, judge, embedder, out_root, rag_k)

    # deterministic seed selection: shuffle the pool by the run seed, take the first N.
    import random as _random
    pool = list(seed_pool)
    _random.Random(seed).shuffle(pool)
    config.num_conversations = convos
    used_seeds = pool[:convos]

    # SHARED SEED: ground each seed once (cached across methods) and feed the SAME
    # grounded question to every method. The cache key excludes method, so Dynamic and
    # Xie reuse the identical grounded seed.
    if shared_seed:
        gold_gen = GoldAnswerGenerator(config=config, llm=gen_llm)
        cache_path = os.path.join(RESULT_ROOT, "_shared_seeds",
                                  f"{dataset}_{rag_name}_d{d_chunks}_s{seed}.json")
        grounded = _load_or_ground_seeds(used_seeds, cache_path, retriever, gold_gen, gen_llm)
        used_seeds = [_with_question(s, grounded.get(str(s.id), s.question)) for s in used_seeds]

    # On the mtRAG-domain arm the seed IS a real human question — use it VERBATIM for both
    # methods (no rewrite), so Dynamic and Xie start from the IDENTICAL human seed.
    verbatim_seed = dataset in MTRAG_DOMAINS
    if spec.get("generator") == "subquestion":
        # the faithful Xie sub-question-coverage baseline (static by construction)
        gen = XieSubQuestionGenerator(
            kg, target_rag, judge, config=config, gen_llm=gen_llm, retriever=retriever,
            strict_gold=False, ground_seed=not (shared_seed or verbatim_seed))
    elif spec.get("generator") == "single_turn":
        # the single-turn control: the adaptive generator forced to the seed turn only
        # (no follow-ups). Same construction as the adaptive generator; n_turns is fixed
        # to 1 inside SingleTurnQAGenerator regardless of the CLI turn count.
        gen = SingleTurnQAGenerator(
            kg, target_rag, judge, config=config, gen_llm=gen_llm, retriever=retriever,
            type_policy="controller", content_policy=content, strict_gold=False,
            skip_seed_grounding=(shared_seed or verbatim_seed))
    else:
        gen = AdaptiveConversationGenerator(
            kg, target_rag, judge, config=config, gen_llm=gen_llm, retriever=retriever,
            type_policy="controller", content_policy=content, strict_gold=False,
            forced_types=forced_types,
            skip_seed_grounding=(shared_seed or verbatim_seed),
            inject_unanswerable_at=inject_unanswerable_at)

    # Generate AND save one conversation at a time, so a long run is resumable per
    # conversation (a crash loses at most the current one; a re-run without --overwrite
    # skips the finished ones) and progress is visible as files appear.
    written: List[str] = []
    for i, seed_i in enumerate(used_seeds, start=1):
        cdir = os.path.join(out_root, f"conversation_{i:03d}")
        if os.path.exists(os.path.join(cdir, "conversation.json")) and not overwrite:
            print(f"  skip (exists): conversation_{i:03d}")
            continue
        conv = gen.generate(seed_i, f"conv-{i - 1:03d}", n_turns)
        meta = _conversation_metadata(method, rag_name, dataset, seed, n_turns,
                                      content, config, rag_k, len(kg.chunks))
        if forced_types:
            meta["forced_types"] = forced_types
        if label:
            meta["experiment_label"] = label
        saved = _save_conversation(cdir, conv, meta, overwrite)
        if saved:
            written.append(saved)
            print(f"  saved -> conversation_{i:03d}  (types={conv.type_sequence}, "
                  f"outcomes={conv.outcome_sequence})")
    return written


# mtRAG conversation.domain (e.g. "mt-rag-clapnq-elser-...") -> corpus file stem.
_DOMAIN_KEYS = [("clapnq", "clapnq"), ("govt", "govt"), ("fiqa", "fiqa"),
                ("ibmcloud", "cloud"), ("cloud", "cloud")]


def _domain_of(conv) -> str:
    d = (conv.domain or "").lower()
    for key, val in _DOMAIN_KEYS:
        if key in d:
            return val
    return "unknown"


def _load_official_domain(path: str, domain: str) -> tuple:
    """Read one official mtRAG passage corpus (``<domain>.jsonl``).

    Returns ``(chunks, native_ids)`` row-aligned: ``native_ids[i]`` is the corpus passage
    ``_id`` (e.g. ``837799097_6931-7548-0-617``), so every retrieved doc_id is traceable
    back to the real source passage. Field names confirmed from the downloaded corpora:
    ``_id`` (== ``id``) and ``text``."""
    fp = os.path.join(path, f"{domain}.jsonl")
    if not os.path.isfile(fp):
        raise RuntimeError(f"official corpus for domain '{domain}' not found at {fp}")
    chunks: List[str] = []
    native_ids: List[str] = []
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if not text:
                continue
            chunks.append(text)
            native_ids.append(str(row.get("_id") or row.get("id") or ""))
    if not chunks:
        raise RuntimeError(f"no passages parsed from {fp}")
    return chunks, native_ids


def _union_domain(convs: List) -> tuple:
    """Union of the human turns' native contexts (the confound option). Returns
    ``(chunks, native_ids)`` where native_ids are the mtRAG context document_ids."""
    seen, chunks, native_ids = set(), [], []
    for c in convs:
        for t in c.turns:
            for ctx in t.contexts:
                if ctx.text and ctx.text not in seen:
                    seen.add(ctx.text); chunks.append(ctx.text)
                    native_ids.append(ctx.doc_id)
    return chunks, native_ids


def _remap_doc_ids(conv, native_ids: List[str]) -> None:
    """Replace each rag_retrieved_documents doc_id (array index into the corpus) with the
    real corpus passage id, so the stored id is traceable to the source passage."""
    n = len(native_ids)
    for turn in conv.turns:
        for d in turn.rag_retrieved_documents:
            idx = d.get("doc_id")
            if isinstance(idx, int) and 0 <= idx < n:
                d["corpus_doc_id"] = native_ids[idx]
                d["doc_id"] = native_ids[idx]


def run_mtrag_method(rag_name: str, convos: int, turns: int, seed: int, config: Config,
                     gen_llm: LLM, judge: LLM, embedder: Embedder, rag_k: Optional[int],
                     overwrite: bool, corpus_choice: str, corpus_path: Optional[str],
                     mtrag_domain: Optional[str], label: Optional[str] = None) -> List[str]:
    """Replay human mtRAG conversations through a target RAG. Requires an EXPLICIT
    --mtrag-corpus; otherwise reports the blocker and runs nothing. Each conversation is
    replayed against a RAG indexed on ITS OWN domain corpus (never a mix)."""
    if corpus_choice == "none":
        print("\n=== method=mtrag (REPLAY) — BLOCKED on corpus decision ===\n"
              "  Choose a corpus EXPLICITLY (never chosen silently):\n"
              "    --mtrag-corpus official --mtrag-corpus-path DIR  (real corpora; recommended)\n"
              "    --mtrag-corpus union                             (retrieval-biased confound)\n"
              "  Replay machinery is READY (generation/mtrag_generator.py). Skipping mtrag.")
        return []
    if corpus_choice == "official" and (not corpus_path or not os.path.isdir(corpus_path)):
        raise RuntimeError("--mtrag-corpus official needs --mtrag-corpus-path DIR with the "
                           "uncompressed <domain>.jsonl corpora")
    # graph / raptor build per-passage LLM structures (typed-graph extraction / RAPTOR
    # tree) over the WHOLE corpus. On the 366K-passage official corpora that is tens of
    # thousands of extra LLM calls (infeasible cost/time) and builds an index, not an
    # attribution test. Skip with a clear message rather than launch a runaway job.
    if rag_name in ("graph", "raptor", "hippo"):
        print(f"\n=== method=mtrag | rag={rag_name} — SKIPPED (infeasible over official "
              f"corpus) ===\n  '{rag_name}' would LLM-process all 366K passages to "
              "build its index (graph/OpenIE/tree). mtRAG runs on the corpus-embedding RAGs "
              "(vector/longrag/selfrag/crag/mock) instead.")
        return []

    all_convos = load_mtrag_conversations()
    if mtrag_domain:
        all_convos = [c for c in all_convos if _domain_of(c) == mtrag_domain]
    all_convos = all_convos[:convos]
    if not all_convos:
        print(f"  ! no mtRAG conversations matched domain={mtrag_domain!r}")
        return []

    root = os.path.join(RESULT_ROOT, label) if label else RESULT_ROOT
    # layout: [<label>/]mtrag/<rag>/<domain>/conversation_NNN/  — mtRAG's dataset IS the
    # domain (clapnq / govt / fiqa / cloud), each its own corpus. rag_base holds the shared
    # rag-level bits; conversations are written under a per-domain subfolder below.
    rag_base = os.path.join(root, "mtrag", rag_name)
    os.makedirs(rag_base, exist_ok=True)

    # group by domain so each conversation runs against its own corpus
    by_domain: Dict[str, List] = {}
    for c in all_convos:
        by_domain.setdefault(_domain_of(c), []).append(c)

    written: List[str] = []
    for domain, convs in by_domain.items():
        # each mtRAG domain is its OWN dataset/corpus -> its own folder + numbering
        dom_out = os.path.join(rag_base, domain)
        idx = 0
        if corpus_choice == "official":
            chunks, native_ids = _load_official_domain(corpus_path, domain)
            corpus_id = f"mtrag-official:{domain}"
            cache_path = os.path.join(corpus_path, f"{domain}.embindex")
        else:                                   # union
            chunks, native_ids = _union_domain(convs)
            corpus_id = f"mtrag-union:{domain}"
            cache_path = os.path.join(rag_base, f"{domain}_union.embindex")
        print(f"\n=== method=mtrag | rag={rag_name} | corpus={corpus_id} "
              f"({len(chunks)} passages) | convos={len(convs)} ===")

        emb = load_or_build_embeddings(embedder, chunks, cache_path)
        kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                          graph_mode=("typed" if rag_name == "graph" else "none")).build(chunks)
        if emb is not None:
            kg.chunk_embeddings = np.asarray(emb, dtype=float)
        retriever = GraphRetriever(kg, config=config, llm=gen_llm, embedder=embedder)
        target_rag = connect_rag(rag_name, chunks, config=config, llm=gen_llm,
                                 retriever=retriever, embedder=embedder, cache_dir=out_root,
                                 precomputed_emb=emb,
                                 **({} if rag_k is None else {"k": rag_k}))
        replay = MtragReplayGenerator(target_rag, judge)

        for mc in convs:
            idx += 1
            conv = replay.replay(mc, conversation_id=f"conversation_{idx:03d}",
                                 max_turns=turns if turns > 0 else None)
            _remap_doc_ids(conv, native_ids)
            # dataset = the domain (clapnq/govt/fiqa/cloud), so the path is accurate
            meta = _conversation_metadata("mtrag", rag_name, domain, seed,
                                          len(conv.turns), None, config, rag_k, len(chunks))
            meta["mtrag_corpus"] = corpus_id
            meta["mtrag_source_conversation"] = mc.conversation_id
            meta["mtrag_domain"] = mc.domain
            meta["note_mtrag"] = (
                "questions are human-authored (question_generation_documents is empty); "
                "rag_retrieved_documents are OUR RAG's cosine retrieval over the official "
                "corpus (doc_id = real corpus passage _id); mtRAG's native ELSER contexts "
                "are under each turn's provenance.native_contexts (score_type=elser).")
            cdir = os.path.join(dom_out, f"conversation_{idx:03d}")
            saved = _save_conversation(cdir, conv, meta, overwrite)
            if saved:
                written.append(saved)
                print(f"  saved -> {saved}  (outcomes={conv.outcome_sequence})")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the 4 conversation-generation methods across RAGs.")
    ap.add_argument("--method", default="proposed",
                    help="proposed | xie | single_turn_qa | mtrag | all")
    ap.add_argument("--rag", default="vector",
                    help="a RAG name (see connectors.available_rags) or 'all'")
    ap.add_argument("--dataset", default="qasper")
    ap.add_argument("--convos", type=int, default=1)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-chunks", type=int, default=300,
                    help="corpus size (chunks) to build retrieval over")
    ap.add_argument("--rag-k", type=int, default=None,
                    help="override the RAG's retrieval budget (docs per query)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite an existing conversation.json (default: skip it)")
    ap.add_argument("--mtrag-corpus", default="none",
                    choices=["none", "union", "official"],
                    help="corpus the mtRAG target RAG retrieves over. 'none' (default) "
                         "reports the blocker and runs nothing — no corpus is chosen "
                         "silently. 'union' = union of native contexts (confound); "
                         "'official' = real corpora (needs --mtrag-corpus-path).")
    ap.add_argument("--mtrag-corpus-path", default=None,
                    help="dir with the uncompressed official mtRAG *.jsonl corpora")
    ap.add_argument("--mtrag-domain", default=None,
                    choices=["clapnq", "govt", "fiqa", "cloud"],
                    help="restrict the mtRAG method to one domain (e.g. clapnq for the pilot)")
    ap.add_argument("--force-types", default=None,
                    help="comma-separated fixed follow-up type sequence (cycled over "
                         "turns 1..n-1; turn 0 is always the Seed). Overrides the adaptive "
                         "controller for THIS run only. e.g. "
                         "'Multi-Hop,Multi-Hop,Multi-Hop,Unanswerable,Multi-Hop,Multi-Hop,Multi-Hop' "
                         "puts one Unanswerable follow-up at turn 4. Generated methods only.")
    ap.add_argument("--label", default=None,
                    help="write results to compare/result/<label>/<method>/<rag>/ instead of "
                         "the main tree, so a stress-test dataset stays separate (e.g. 'unanswerable').")
    ap.add_argument("--shared-seed", action="store_true",
                    help="ground each seed ONCE (cached per dataset/rag/corpus/run-seed) and "
                         "feed the SAME grounded question to every method, so Dynamic and Xie "
                         "start from an IDENTICAL question. Generated methods only.")
    ap.add_argument("--inject-unanswerable-at", type=int, default=None, metavar="N",
                    help="force turn N of the Dynamic/adaptive method to be an Unanswerable "
                         "probe (knowledge-boundary test), leaving other turns adaptive. "
                         "e.g. --inject-unanswerable-at 4. Applies to proposed only.")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args(argv)

    forced_types = ([t.strip() for t in args.force_types.split(",") if t.strip()]
                    if args.force_types else None)

    methods = list(METHODS) if args.method == "all" else [args.method]
    for m in methods:
        if m not in METHODS:
            ap.error(f"unknown --method {m!r}; choose from {list(METHODS)} or 'all'")
    rags = available_rags() if args.rag == "all" else [args.rag]

    config = Config.load(dataset=args.dataset, max_samples=max(args.d_chunks, 50),
                         num_conversations=args.convos,
                         min_turns=args.turns, max_turns=args.turns,
                         gen_model=args.gen_model, judge_model=args.judge_model,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    if not config.has_openai:
        print("!! needs an OpenAI key (.env OPENAI_API_KEY) — generation calls the API.")
        return

    gen_llm = LLM(model=config.gen_model, config=config)
    judge = LLM(model=config.judge_model, config=config)
    embedder = Embedder(config=config, llm=gen_llm)

    if forced_types:
        print(f"# forced follow-up types: {forced_types}")
    all_written: List[str] = []
    for method in methods:
        for rag_name in rags:
            if METHODS[method]["replay"]:
                if forced_types:
                    print("  ! --force-types does not apply to mtrag (human replay) — skipping.")
                    continue
                all_written += run_mtrag_method(
                    rag_name, args.convos, args.turns, args.seed, config, gen_llm,
                    judge, embedder, args.rag_k, args.overwrite,
                    args.mtrag_corpus, args.mtrag_corpus_path, args.mtrag_domain,
                    label=args.label)
            else:
                all_written += run_generated_method(
                    method, rag_name, args.dataset, args.convos, args.turns, args.seed,
                    args.d_chunks, config, gen_llm, judge, embedder, args.rag_k,
                    args.overwrite, forced_types=forced_types, label=args.label,
                    shared_seed=args.shared_seed,
                    inject_unanswerable_at=args.inject_unanswerable_at,
                    corpus_path=args.mtrag_corpus_path)

    print(f"\n# gen usage  ({config.gen_model}): {gen_llm.cost_report()}")
    print(f"# judge usage({config.judge_model}): {judge.cost_report()}")
    print(f"\nwrote {len(all_written)} conversation file(s).")
    return all_written


if __name__ == "__main__":
    main()
