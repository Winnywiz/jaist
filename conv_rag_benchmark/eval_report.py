"""
Consolidated evaluation report for the dynamic-vs-static × with-gold-vs-docs-only
comparison (Method E adaptive vs Method D all-types, on QASPER vs arxivcs).

For each build file it reports the metrics that matter for this project:
  * question quality  — well_formed / gold_supported / gold_correct, judged on a
                        sample of turns by the same LLM judge (quality_judge)
  * answer-key trust  — grounding-guard give-up rate (gate_giveup)
  * failures found    — RAG failure rate + which probe types found them
  * coverage          — how many distinct failure-probing types were exercised

Run:  python -m conv_rag_benchmark.eval_report [build.json ...]
      (default: the four eval_{E,D}_{qasper,arxivcs}.json files)
"""
import argparse
import json
import random

from .config import Config
from .llm import LLM
from .quality_judge import judge_items
from .gate_giveup import analyse, _gave_up

DEFAULT_FILES = [
    "conv_rag_benchmark/output/eval_E_qasper.json",
    "conv_rag_benchmark/output/eval_D_qasper.json",
    "conv_rag_benchmark/output/eval_E_arxivcs.json",
    "conv_rag_benchmark/output/eval_D_arxivcs.json",
]
OUT = "conv_rag_benchmark/output/eval_report.json"


def turns_of(build):
    for c in build.get("conversations", []):
        for t in c.get("turns", []):
            yield t


def failure_by_depth(turns):
    """MTRAG-style later-turn analysis: RAG failure rate by turn-depth bucket.

    Katsis et al. (TACL 2025) report that RAG performance degrades on later
    conversation turns; our external validation reproduced that curve on their
    human data (mtrag_validation/). This computes the same curve on OUR
    self-generated runs. ``turn_id`` is 0-indexed, so buckets 0-1 / 2-4 / 5+
    mirror MTRAG's 1-indexed 1-2 / 3-5 / 6+. Turns whose answer key the
    grounding guard gave up on are excluded (untrustworthy gold), matching the
    per-type grading rule.
    """
    agg = {}
    for t in turns:
        if _gave_up(t) or not t.get("outcome"):
            continue
        tid = int(t.get("turn_id") or 0)
        b = "1-2" if tid <= 1 else ("3-5" if tid <= 4 else "6+")
        bad, n = agg.get(b, (0, 0))
        agg[b] = (bad + (t["outcome"] in ("wrong", "hallucinated")), n + 1)
    return {k: {"failure_rate": round(v[0] / v[1], 3), "n": v[1]}
            for k, v in sorted(agg.items())}


def report_one(path, judge, sample_n, seed=0):
    build = json.load(open(path, encoding="utf-8"))
    turns = list(turns_of(build))
    judgeable = [t for t in turns
                 if t.get("query_type") != "Unanswerable"
                 and not t.get("is_unanswerable") and not _gave_up(t)]
    random.Random(seed).shuffle(judgeable)
    quality, _ = judge_items(judge, judgeable[:sample_n])

    ga = analyse(path)
    n_probe = sum(v["total"] for v in ga["by_type"].values())
    n_gaveup = sum(v["gave_up"] for v in ga["by_type"].values())

    return {
        "n_conversations": len(build.get("conversations", [])),
        "n_turns": len(turns),
        "question_quality": quality,
        "giveup_rate_overall": round(n_gaveup / n_probe, 3) if n_probe else None,
        "giveup_by_type": {k: v["rate"] for k, v in ga["by_type"].items()},
        "rag_failure": build.get("rag_failure"),
        "failure_by_turn_depth": failure_by_depth(turns),
        "type_coverage": sorted({t.get("query_type") for t in turns if t.get("query_type")}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=30,
                    help="turns per build to send to the quality judge")
    args = ap.parse_args(argv)
    files = args.files or DEFAULT_FILES

    cfg = Config.load()
    judge = LLM(model=getattr(cfg, "judge_model", None) or "gpt-4o", config=cfg)

    results = {}
    for f in files:
        try:
            results[f] = report_one(f, judge, args.sample)
        except Exception as e:
            print(f"skip {f}: {e}")

    print(f"\n{'build':<28}{'n_turns':>8}{'well_fmd':>9}{'gold_sup':>9}"
          f"{'gold_cor':>9}{'give-up':>8}{'fail_rate':>10}{'types':>6}")
    for f, r in results.items():
        name = f.split("/")[-1].replace("eval_", "").replace(".json", "")
        q = r["question_quality"]
        fr = (r["rag_failure"] or {}).get("failure_rate")
        print(f"{name:<28}{r['n_turns']:>8}{q['well_formed']:>9}{q['gold_supported']:>9}"
              f"{q['gold_correct']:>9}"
              f"{('' if r['giveup_rate_overall'] is None else r['giveup_rate_overall']):>8}"
              f"{('' if fr is None else fr):>10}{len(r['type_coverage']):>6}")
    print("\nfailures found, by probe type:")
    for f, r in results.items():
        name = f.split("/")[-1].replace("eval_", "").replace(".json", "")
        print(f"  {name:<26}{(r['rag_failure'] or {}).get('failures_by_probe')}")

    print("\nfailure rate by turn depth (MTRAG later-turn check):")
    for f, r in results.items():
        name = f.split("/")[-1].replace("eval_", "").replace(".json", "")
        depth = r.get("failure_by_turn_depth") or {}
        row = "  ".join(f"{k}: {v['failure_rate']} (n={v['n']})"
                        for k, v in depth.items())
        print(f"  {name:<26}{row}")

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
