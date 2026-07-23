"""
Answer the with-gold-vs-without research question on QASPER: score the HUMAN
question/answer pairs (written by NLP practitioners) and the GENERATED probe
turns from the qasper Method-E build with the SAME quality judge, so the two
get directly comparable well_formed / gold_supported / gold_correct rates.

Human items:     dataset loader `qasper` (question + gold + its evidence paras).
Generated items: output/qasper/quality_e.json turns (Unanswerable turns skipped —
                 their abstention gold is correct by design, not comparable).

Run:  python -m conv_rag_benchmark.compare_qasper_human --human 30 --generated 60
"""
import argparse
import json
import random

from .config import Config
from .llm import LLM
from .datasets.loader import DatasetLoader
from .quality_judge import judge_items

BUILD = "conv_rag_benchmark/output/qasper/quality_e.json"
OUT = "conv_rag_benchmark/output/qasper/human_vs_generated.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=int, default=30)
    ap.add_argument("--generated", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = Config.load()
    judge = LLM(model=getattr(cfg, "judge_model", None) or "gpt-4o", config=cfg)
    if not judge.available:
        raise SystemExit("judge LLM unavailable — set the OpenAI key first")

    human = [{"question": s.question, "gold": s.answer,
              "evidence": " ".join(s.context[:3])[:1600], "query_type": "Human"}
             for s in DatasetLoader("qasper", max_samples=args.human).load()]

    build = json.load(open(BUILD, encoding="utf-8"))
    gen_turns = [{"question": t["question"], "gold": t["gold"],
                  "evidence": t["evidence"], "query_type": t["query_type"]}
                 for c in build["conversations"] for t in c["turns"]
                 if not t.get("is_unanswerable")]
    random.Random(args.seed).shuffle(gen_turns)
    gen_turns = gen_turns[: args.generated]

    print(f"judging {len(human)} human + {len(gen_turns)} generated items "
          f"with {judge.model} ...")
    agg_h, _ = judge_items(judge, human)
    agg_g, scored_g = judge_items(judge, gen_turns)

    print(f"\n{'metric':<16}{'human':>8}{'generated':>11}")
    for m in ("well_formed", "gold_supported", "gold_correct"):
        print(f"{m:<16}{agg_h[m]:>8}{agg_g[m]:>11}")
    print(f"{'n':<16}{agg_h['n']:>8}{agg_g['n']:>11}")

    json.dump({"human": agg_h, "generated": agg_g,
               "generated_items": scored_g},
              open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
