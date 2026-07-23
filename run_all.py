"""
run_all.py — one command, all three stages. Everything lands in result/.

STAGE 1  GENERATE   Method E builds an adaptive conversational benchmark from a corpus:
                    it asks the RAG a question, reads its REAL answer, and lets that
                    outcome choose the next question type. Also scores question quality
                    (well_formed / gold_supported / gold_correct) via G-Eval.
                    -> result/conversations/, result/benchmark_quality/

STAGE 2  EVALUATE   Independent checks that the BENCHMARK itself is trustworthy:
                      atomic faithfulness  decompose each gold into claims, check each is
                                           entailed by the evidence (RAGAS-style)
                      triple audit         audit (question, support, answer) triples for
                                           faithfulness + answer relevancy
                    -> result/benchmark_quality/

STAGE 3  CLASSIFY   The research question: given a FAILED RAG turn, can DYNAMIC follow-up
                    probing attribute the failure's CAUSE (Retrieval / Generation /
                    Conversation) more accurately than STATIC sub-question decomposition?
                    Failures are INJECTED so the true cause is known.
                    -> result/attribution/

Run from THIS folder:

    python run_all.py                                  # all 3 stages, default dataset
    python run_all.py --dataset qasper --convos 10     # smaller/cheaper
    python run_all.py --stages 3                       # only the attribution experiment
    python run_all.py --stages 1 2                     # only build + evaluate a benchmark

Needs an OpenAI key: put it in .env at this folder (see .env.example) or export
OPENAI_API_KEY. Datasets download from HuggingFace on first use.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # so `conv_rag_benchmark` / `DYNAMICQA` import

RESULT = os.path.join(HERE, "result")
CONV_DIR = os.path.join(RESULT, "conversations")
QUAL_DIR = os.path.join(RESULT, "benchmark_quality")
ATTR_DIR = os.path.join(RESULT, "attribution")

#: dataset name (loader) -> folder label used under result/benchmark_quality/
LABELS = {"multihoprag": "MultiHopRAG", "qasper": "qasper", "medqa": "MedQA"}


def _banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(f" {msg}")
    print("=" * 78, flush=True)


def stage1_generate(args) -> str:
    """Build the adaptive conversational benchmark + score question quality."""
    _banner("STAGE 1 — GENERATE the conversational benchmark (Method E)")
    from conv_rag_benchmark import build_e_adaptive
    build_e_adaptive.main([
        "--dataset", args.dataset, "--convos", str(args.convos),
        "--turns", str(args.turns), "--rag", args.rag,
        "--graph-mode", "none",        # graph ablated: measured as not helping (see README)
    ])
    label = LABELS.get(args.dataset, args.dataset)
    src = os.path.join(QUAL_DIR, label, "quality_e_nonegraph.json")
    # copy the conversations out to their own folder so they are easy to read/send
    if os.path.exists(src):
        os.makedirs(CONV_DIR, exist_ok=True)
        dst = os.path.join(CONV_DIR, f"conversations_{label}.json")
        shutil.copyfile(src, dst)
        print(f"\n  conversations -> {os.path.relpath(dst, HERE)}")
    else:
        print(f"\n  !! expected generated file not found: {src}")
    return src


def stage2_evaluate(args, quality_file: str) -> None:
    """Independent benchmark-quality checks on the generated triples."""
    _banner("STAGE 2 — EVALUATE the benchmark's own quality")
    if not os.path.exists(quality_file):
        print(f"  !! no generated benchmark at {quality_file} — run stage 1 first.")
        return

    print("\n-- atomic faithfulness (decompose gold into claims, check each is entailed) --")
    from conv_rag_benchmark import atomic_faithfulness
    atomic_faithfulness.main(["--file", quality_file, "--n", str(args.eval_n)])

    print("\n-- triple audit (question / support / answer) --")
    from conv_rag_benchmark import triple_audit
    triple_audit.main(["--file", quality_file, "--n", str(args.eval_n)])


def stage3_classify(args) -> None:
    """The RQ: dynamic vs static failure attribution on INJECTED (known-cause) failures."""
    _banner("STAGE 3 — CLASSIFY failures: DYNAMIC vs STATIC attribution")
    os.makedirs(ATTR_DIR, exist_ok=True)
    from DYNAMICQA import multiseed
    multiseed.main(["--dataset", args.dataset, "--n", str(args.attr_n),
                    "--seeds", *[str(s) for s in args.seeds]])

    print("\n-- fair re-analysis (macro over the categories EVERY method can emit) --")
    from DYNAMICQA import fair_macro
    fair_macro.main()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full pipeline: generate -> evaluate -> classify")
    ap.add_argument("--dataset", default="multihoprag",
                    help="multihoprag | qasper | medqa (downloads from HuggingFace)")
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3],
                    help="which stages to run (default: all)")
    # stage 1
    ap.add_argument("--convos", type=int, default=10, help="conversations to generate")
    ap.add_argument("--turns", type=int, default=6, help="turns per conversation")
    ap.add_argument("--rag", default="vector",
                    choices=["mock", "vector", "selfrag", "raptor", "graph", "longrag"],
                    help="which RAG system is being tested")
    # stage 2
    ap.add_argument("--eval-n", type=int, default=25, help="triples to audit (cost control)")
    # stage 3
    ap.add_argument("--attr-n", type=int, default=20, help="seeds per attribution run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="independent seeds (each shuffles WHICH docs get injected)")
    args = ap.parse_args()

    for d in (CONV_DIR, QUAL_DIR, ATTR_DIR):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    print(f"# dataset={args.dataset} | stages={args.stages} | results -> result/")

    label = LABELS.get(args.dataset, args.dataset)
    quality_file = os.path.join(QUAL_DIR, label, "quality_e_nonegraph.json")

    if 1 in args.stages:
        quality_file = stage1_generate(args)
    if 2 in args.stages:
        stage2_evaluate(args, quality_file)
    if 3 in args.stages:
        stage3_classify(args)

    _banner(f"DONE in {(time.time()-t0)/60:.1f} min — everything is in result/")
    for root, _, files in os.walk(RESULT):
        for f in sorted(files):
            if f.endswith(".json"):
                print("   " + os.path.relpath(os.path.join(root, f), HERE))


if __name__ == "__main__":
    main()
