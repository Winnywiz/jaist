"""
TOPIC COVERAGE: do E's generated questions cover the whole corpus, or cluster on
a few topics and miss the rest?

Method: cluster the corpus chunks into K topics (k-means on embeddings). Assign each
generated question's evidence to its nearest topic. Report:
  * coverage   = fraction of the K topics touched by at least one question
  * entropy    = how EVENLY the questions spread over topics (1.0 = perfectly even)
  * gini-ish   = concentration (lower = more even)
Compares graph vs no-graph (and is a benchmark-quality metric on its own).

Run:  python -m conv_rag_benchmark.coverage_check
"""
import json
import math
import os
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .llm import LLM

_LABELS = {"multihoprag": "MultiHopRAG", "medqa": "MedQA", "arxivcs": "ArXivCS"}


def _entropy(counts, K):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    return round(h / math.log(K), 3)        # normalised 0..1 (1 = perfectly even)


def coverage(dataset, K=20, chunk_limit=600):
    config = Config.load(dataset=dataset, max_samples=50, prefer_local_embeddings=False)
    llm = LLM(model=config.gen_model, config=config)
    emb = Embedder(config=config, llm=llm)
    label = _LABELS[dataset]

    seeds = DatasetLoader(dataset, max_samples=50).load()
    chunks = [c for s in seeds for c in s.context if c and c.strip()][:chunk_limit]
    X = np.asarray(emb.encode(chunks), dtype=float)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)

    base = os.path.join("conv_rag_benchmark", "output", label)
    rows = {}
    # D (static benchmark) uses 'question_evidence_context' (list); E uses 'evidence' (str)
    for tag, fn in [("D-static", "benchmark_random.json"),
                    ("E-dynamic", "quality_e_50conv.json"),
                    ("E-no-graph", "quality_e_nonegraph.json")]:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        evs = []
        for c in d["conversations"]:
            for t in c["turns"]:
                ev = t.get("evidence")
                if ev is None and t.get("question_evidence_context"):
                    ev = " ".join(t["question_evidence_context"])
                if (ev or "").strip():
                    evs.append(ev)
        if not evs:
            continue
        E = np.asarray(emb.encode(evs), dtype=float)
        topics = km.predict(E)
        cnt = Counter(int(t) for t in topics)
        rows[tag] = {"coverage": round(len(cnt) / K, 3), "topics_hit": len(cnt),
                     "K": K, "n_questions": len(evs), "evenness": _entropy(cnt, K),
                     "biggest_topic_share": round(max(cnt.values()) / len(evs), 3)}
    return label, rows


def main():
    print(f"# TOPIC COVERAGE (corpus clustered into K topics; do questions span them?)\n")
    allrows = {}
    for ds in ("multihoprag", "medqa", "arxivcs"):
        print(f"# clustering & scoring {ds} ...")
        label, rows = coverage(ds)
        allrows[label] = rows

    print(f"\n{'dataset':<14}{'arm':<10}{'coverage':>10}{'topics':>9}{'evenness':>10}{'top-share':>11}")
    for label, rows in allrows.items():
        for tag, r in rows.items():
            print(f"{label:<14}{tag:<10}{r['coverage']:>10}{str(r['topics_hit'])+'/'+str(r['K']):>9}"
                  f"{r['evenness']:>10}{r['biggest_topic_share']:>11}")
    json.dump(allrows, open("conv_rag_benchmark/output/coverage.json", "w",
                            encoding="utf-8"), indent=2)
    print("\nsaved -> conv_rag_benchmark/output/coverage.json")
    print("\nlegend: coverage = topics touched / K | evenness 1.0 = perfectly even spread |"
          " top-share = fraction of questions on the single biggest topic (lower = less clustered)")


if __name__ == "__main__":
    main()
