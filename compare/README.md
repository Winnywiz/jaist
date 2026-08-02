# compare/ — does *dynamic* really beat *static*?

This folder is the experiment that answers the thesis's core question:

> **Can a DYNAMIC follow-up probe (one that reacts to the RAG's actual answer)
> attribute RAG failures more accurately than STATIC probing (all sub-questions
> fixed up front)?**

Short answer, from the numbers below: **yes, on every dataset.**

---

## What is being compared

Four methods, all fed the *same* failed cases and all ending in the same
Retrieval-vs-Generation attribution signal. The **only** thing that differs is
*how they probe*:

| method | file | reacts to the RAG's answer? | kind |
|---|---|:--:|---|
| `1.single_turn` | [STATIC/method.py](STATIC/method.py) | no | static control (no probing) |
| `2.xie_static_core` | [STATIC/method.py](STATIC/method.py) | no | static — Xie et al. 2025, core sub-questions |
| `3.xie_followup_only` | [STATIC/method.py](STATIC/method.py) | no | static — Xie et al. 2025, follow-up sub-questions |
| **`4.dynamic_followup`** | [DYNAMIC/method.py](DYNAMIC/method.py) | **yes** | **the proposed method** |

`single_turn` and `xie` are the "single turn with Xie" static baselines; `dynamic_followup`
is ours.

---

## The result (this is the thing to look at)

**`macro_shared` accuracy** — the *fair* metric: averaged over only the failure
categories every method can emit (Retrieval + Generation). Higher = attributes the
true cause more often. Mean over seeds (± sample sd).

| dataset | single-turn | Xie core | Xie follow-up | **DYNAMIC (ours)** |
|---|:--:|:--:|:--:|:--:|
| hfdocqa      | 0.485 | 0.479 | 0.789 | **0.941** |
| mtrag        | 0.500 | 0.394 | 1.000 | **1.000** |
| multihoprag  | 0.491 | 0.815 | 0.920 | **0.970** |
| qasper       | 0.493 | 0.784 | 0.793 | **0.966** |

Dynamic is best (or tied-best) on all four. Pooled, the gap is largest on
**Generation** failures (dyn ≈ 0.99 vs best static ≈ 0.68) and small but positive on
**Retrieval** (dyn ≈ 0.94 vs 0.91).

**Plus a capability static structurally cannot have:** only the dynamic method can ever
emit the **Conversation** (coreference) category — a static method fixes its questions
before seeing the answer, so it can never detect a coreference failure. Static scores
**0.000** on it by construction; dynamic reaches **~0.21**. This is reported as a
*capability* result, not folded into the accuracy table above (that would be an unfair
fight).

Numbers regenerate from [result/attribution/fair_macro.json](result/attribution/fair_macro.json).

---

## Three honesty caveats (baked into the analysis)

1. **`macro_shared` is the comparison to quote**, not `macro_3class`. The 3-class
   average includes the Conversation column that static *cannot* score on, so part of
   that larger gap is design, not measurement.
2. **Conversation is a capability claim, reported separately** (see above).
3. With only 3 seeds the ± is a **rough spread, not a confidence interval.**

---

## How the ground truth is made (why "accuracy" means something)

Real failures have an unknown cause, so [shared/setup.py](shared/setup.py) **injects**
failures whose cause is *certain*, then asks each method to recover it:

* **Retrieval** — withhold the gold passage (answer verifiably absent from context).
* **Generation** — force an answer to a detail absent from all context → fabrication.
* **Conversation** — a real coreference follow-up with the history dropped.

Accuracy = fraction of injected cases whose category the method recovers.

---

## Reproduce

Run from the **clean-package root** (the folder that also contains `conv_rag_benchmark/`),
because this experiment reuses that package's LLM client, embeddings and dataset loaders.

```bash
# no API key needed: recompute the head-to-head table from the saved per-seed results
python -m compare.fair_macro
```

```bash
# a single fresh seed (needs OPENAI_API_KEY); writes result/attribution/attribution_<dataset>_seed0.json
python -m compare.run --dataset multihoprag --n 20
```

```bash
# plumbing smoke test, no API calls (numbers are NOT evidence)
python -m compare.run --dataset multihoprag --n 8 --offline
```

```bash
# many seeds → macro-average + spread, then the fair split
python -m compare.multiseed --dataset multihoprag --n 20 --seeds 0 1 2
python -m compare.fair_macro
```

## Layout

```
compare/
├── DYNAMIC/method.py    ← the proposed method (dynamic follow-up)
├── STATIC/method.py     ← the baselines (single-turn + Xie static decomposition)
├── shared/
│   ├── setup.py           inject controlled failures, wrap the RAG, judge correctness
│   └── harness.py         run every method on the same cases + score + save
├── run.py               single-seed entry point
├── multiseed.py         run across seeds → macro-average + spread
├── fair_macro.py        the defensible split (macro_shared) — no API calls
└── result/attribution/  the numbers (per-seed + multiseed_*.json + fair_macro.json)
```
