# Thesis package (clean)

A clean, self-contained copy of the thesis work — **only the proposed method and the
one comparison that matters**, nothing archived. Two parts:

```
THESIS_CLEAN/
├── conv_rag_benchmark/     ← PART 1: the proposed method (the benchmark generator)
├── compare/                ← PART 2: does dynamic beat static?  (the experiment + answer)
├── result/benchmark_quality/
│   ├── HotpotQA/  MuSiQue/   the current proposed-method results
├── CODE_WALKTHROUGH.md     deep, file-by-file tour of conv_rag_benchmark
├── requirements.txt        pip install -r requirements.txt
└── .env.example            put your OPENAI_API_KEY in .env
```

---

## Part 1 — the proposed method (`conv_rag_benchmark/`)

An **adaptive conversational RAG benchmark**. It generates a multi-turn conversation and,
crucially, closes the loop with the system under test:

```
ask a question → read the RAG's REAL answer → grade it → let that outcome pick the
NEXT question's type → repeat
```

Only the question *type* is chosen adaptively; the question text and gold answer are
always authored from the corpus, so the answer key stays grounded even when the RAG is
wrong. Each run reports **question quality** (G-Eval: well_formed / gold_supported /
gold_correct) and the **RAG's failure profile** (wrong / hallucinated / abstained, and
which probe type caught it).

The flow, end to end:

| stage | where |
|---|---|
| load dataset + build/load knowledge graph | [run_benchmark.py](conv_rag_benchmark/run_benchmark.py) |
| instantiate the RAG under test | [connectors.py](conv_rag_benchmark/connectors.py), [interfaces/rag_interface.py](conv_rag_benchmark/interfaces/rag_interface.py) |
| the adaptive loop (retrieve → author Q+gold → ask RAG → grade → pick next type) | [generation/adaptive_generator.py](conv_rag_benchmark/generation/adaptive_generator.py) |
| score question quality (G-Eval) | [geval.py](conv_rag_benchmark/geval.py) |
| write self-describing JSON to `result/` | [run_benchmark.py](conv_rag_benchmark/run_benchmark.py) |

Run it:

```bash
python -m conv_rag_benchmark.run_benchmark --dataset hotpotqa --rag vector --convos 5 --turns 10
```

Optional, after a run — score how similar each retrieved doc is to the question / gold /
RAG answer (cosine of embeddings). Writes a `*_docsim.json` companion beside the result;
does not touch the run itself:

```bash
python -m conv_rag_benchmark.question_doc_similarity --dataset hotpotqa \
    --file result/benchmark_quality/HotpotQA/vector_HotpotQA_t10_c5.json
```

Full tour: **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** and
[conv_rag_benchmark/README.md](conv_rag_benchmark/README.md).

---

## Part 2 — does dynamic beat static? (`compare/`)

The experiment behind the thesis claim: a **dynamic** follow-up probe (reacts to the
RAG's answer) attributes RAG failures more accurately than **static** probing
(single-turn control + Xie et al. static decomposition). The head-to-head table and the
answer live in **[compare/README.md](compare/README.md)** — dynamic wins on every dataset.

```bash
python -m compare.fair_macro        # recompute the head-to-head table (no API key needed)
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

Datasets download from HuggingFace on first use. Both parts share the
`conv_rag_benchmark/` engine, so **run every command from this folder** (the one holding
this README).
