# Conversational RAG: benchmark generation, quality evaluation, and failure attribution

A self-contained package with three parts:

1. **Generate** an adaptive conversational benchmark (questions + gold answers) from a corpus.
2. **Evaluate** whether that benchmark is trustworthy (are the gold answers grounded?).
3. **Classify** *why* a RAG system failed — the research question:

> **Can dynamic follow-up question generation attribute RAG failures more accurately
> than static sub-question generation?**

Everything written by any stage lands in **`result/`**.

---

## Quick start

```bash
pip install -r requirements.txt

# put your OpenAI key in .env  (copy .env.example)
cp .env.example .env        # then edit it
# ...or just:  export OPENAI_API_KEY=sk-...

python run_all.py                                # all 3 stages (~30-50 min)
python run_all.py --dataset qasper --convos 5    # smaller / cheaper
python run_all.py --stages 3                     # only the attribution experiment
```

Datasets download from HuggingFace on first use (`multihoprag`, `qasper`, `medqa`).

---

## What each stage does

### Stage 1 — GENERATE  (`conv_rag_benchmark/`)
Method E builds a multi-turn conversational benchmark **adaptively**: it asks the RAG a
question, reads its **real** answer, grades it, and lets that outcome pick the *next*
question type (Follow-Up, Multi-Hop, Comparative, Correction, Ambiguous Reference, ...).
Questions and gold answers are authored from the corpus, so the answer key stays grounded.
It also scores question quality with **G-Eval** (logprob-weighted 1-5 rubrics):

* `well_formed` — is the question clear and answerable?
* `gold_supported` — is the gold answer grounded in the evidence?
* `gold_correct` — does the gold actually answer the question?

→ `result/conversations/`, `result/benchmark_quality/`

### Stage 2 — EVALUATE  (`conv_rag_benchmark/`)
Independent checks that the benchmark itself is sound (a benchmark with wrong golds cannot
diagnose anything):

* **atomic faithfulness** (`atomic_faithfulness.py`) — decompose each gold into atomic
  claims, check each is entailed by the evidence (RAGAS/ARES-style). Catches *partial*
  grounding that a single holistic score misses.
* **triple audit** (`triple_audit.py`) — audits each `(question, support, answer)` triple
  for faithfulness and answer relevancy, with per-item feedback.

→ `result/benchmark_quality/`

### Stage 3 — CLASSIFY  (`DYNAMICQA/`)
The research question. Failures are **injected**, so the true cause is *known* and
attribution accuracy is objectively measurable:

| injected cause | how it is created |
|---|---|
| **Retrieval** | withhold the gold passage (answer verifiably absent from context) |
| **Generation** | ask for a plausible detail absent from all context, force an answer → fabrication |
| **Conversation** | a real coreference follow-up with the dialogue history dropped |

Four methods must recover the cause:

| folder | method | reacts to the RAG's answer? | can output "Conversation"? |
|---|---|:--:|:--:|
| `STATIC/` | single-turn control | no | no |
| `STATIC/` | Xie static decomposition (core) | no | no |
| `STATIC/` | Xie static decomposition (follow-up) | no | no |
| **`DYNAMIC/`** | **dynamic follow-up (proposed)** | **yes** | **yes** |

→ `result/attribution/`

---

## Reading the attribution results

`fair_macro.py` (run automatically) reports three numbers, deliberately kept separate:

* **`macro_shared`** — accuracy over the categories **every** method can emit
  (Retrieval + Generation). **This is the fair accuracy comparison.**
* **`Conversation`** — reported **separately** as a *capability* result: static methods
  score exactly 0.000 because they never see the RAG's answer, so coreference failures are
  structurally invisible to them. This is reachability, not superiority on equal footing.
* **`macro_3class`** — kept for continuity only. **Do not lead with it**: it averages in a
  category static cannot score on, so part of that gap is design, not measurement.

---

## Folder map

```
THESIS_PACKAGE/
├── run_all.py              ← single entry point
├── .env.example            ← put your OPENAI_API_KEY in .env
├── conv_rag_benchmark/     ← engine + Stage 1 (generate) + Stage 2 (evaluate)
│   ├── build_e_adaptive.py     Method E runner
│   ├── generation/             adaptive generator, query + gold generators
│   ├── geval.py                G-Eval quality scorer
│   ├── atomic_faithfulness.py  decompose-then-entail grounding
│   └── triple_audit.py         (question, support, answer) auditor
├── DYNAMICQA/              ← Stage 3 (classify)
│   ├── DYNAMIC/method.py       the proposed method
│   ├── STATIC/method.py        the baselines
│   ├── shared/setup.py         failure injection + controlled RAG + judge
│   ├── shared/harness.py       run + score one experiment
│   ├── multiseed.py            repeat over independent seeds (mean ± spread)
│   └── fair_macro.py           the honest re-analysis (read this one)
└── result/                 ← ALL outputs
    ├── conversations/          generated conversations
    ├── benchmark_quality/      G-Eval, atomic faithfulness, triple audit
    └── attribution/            dynamic-vs-static results
```

---

## Notes / known limitations

* **The knowledge graph is disabled by default** (`--graph-mode none`). It was tested on
  two domains, including the most favourable case, and consistently traded question
  sophistication for gold groundedness (two independent measures agreed), so it is not used.
* **`support_focus`** in the triple audit returned ≈0.5 on every dataset tested, i.e. it
  reflects the auditor's prior rather than the data. Do not report it without redesign.
* **Judge caveat**: an LLM grades LLM output. Relative comparisons (dynamic vs static) are
  far more trustworthy than absolute values, because the same judge scores every arm.
* With only a few seeds, `±` is a rough spread, **not** a confidence interval.
