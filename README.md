# Conversational RAG Failure Attribution

**Research question:** *Can **dynamic** follow-up question generation attribute RAG failures
more accurately than **static** sub-question generation?*

This repo generates controlled **multi-turn conversations** against RAG systems, records
rich per-turn evidence, and hands the logs to a **failure classifier** — so we can study
*where* and *why* a RAG fails, and whether the way we generate the conversation changes
what failures become visible.

```
   conversation generation  ─►  RAG under test  ─►  conversation log  ─►  failure classifier
   (Dynamic / Xie / Single /                        (per-turn evidence)     (6-type taxonomy)
    mtRAG human)
```

---

## The four conversation-generation methods

| method (code name) | what generates the follow-ups | sees the RAG's answer? |
|---|---|:--:|
| **Dynamic** (`proposed`) | LLM writes each follow-up *after observing the RAG's real previous answer* | **yes** (`rag_history`) |
| **Xie** (`xie_subq`) | **adapted from Xie et al. (NAACL 2025)**: reuses the paper's sub-question decomposition (**core / background / follow-up**), replayed as turns. The *faithful* paper method (coverage 2×2) is `compare/xie_coverage.py` | **no** (static) |
| **Single-turn** (`single_turn_qa`) | seed question only, no follow-ups (control) — `single_turn_generator.py` | — |
| **mtRAG** (`mtrag`) | real **human** multi-turn conversations (IBM mt-RAG benchmark) replayed | — (human) |

The dynamic-vs-static distinction is the independent variable and is logged explicitly per
turn (`generation_source`, `provenance.generator_saw_rag_answer`). Gold answers are always
authored from corpus truth, never from the RAG's (possibly wrong) reply.

Optional: `--inject-unanswerable-at N` forces one **Unanswerable** probe into the Dynamic
method (knowledge-boundary / hallucination test) while other turns stay adaptive.

## RAG systems under test

Embedding-based (feasible on any corpus): `vector`, `selfrag`, `crag`, `longrag`, `mock` —
plus graph-based (need an LLM-built index, small corpora only): `graph` (typed GraphRAG),
`raptor` (RAPTOR tree), **`hippo`** (HippoRAG: OpenIE knowledge graph + Personalized
PageRank). See `conv_rag_benchmark/interfaces/rag_interface.py`.

## Failure taxonomy (the classifier — a separate component)

`failure/` classifies each failed turn into six types across two layers:
- **retrieval layer:** `knowledge_boundary`, `chunking`, `retrieval`, `context_selection`
- **generation layer:** `grounding`, `response_coverage`

Every turn records **two separate document sets** — `question_generation_documents` (what
the generator used) and `rag_retrieved_documents` (what the RAG retrieved to answer) — each
with real `doc_id / rank / score / score_type`, so the classifier can tell a retrieval miss
from a grounding failure.

---

## Repository layout

```
conv_rag_benchmark/        # the generation engine + RAGs
├── generation/
│   ├── dynamic_generator.py     Dynamic method (proposed) + static content policy
│   ├── xie_generator.py         Xie method: faithful sub-question decomposition (xie_subq)
│   ├── single_turn_generator.py Single-turn control: adaptive generator, seed turn only
│   ├── mtrag_generator.py       mtRAG method: human-conversation replay
│   ├── query_generator.py       the 8 typed question generators
│   └── gold_answer_generator.py grounded gold answers
├── interfaces/rag_interface.py  all RAG systems (vector … hippo)
├── graph/                       typed knowledge graph + retriever (used by graph/hippo)
├── datasets/                    dataset loaders + conversation-preserving mtRAG loader
├── embeddings.py, embeddings_cache.py   embeddings + on-disk corpus index
└── connectors.py                the --rag / --dataset registry

compare/                   # the experiment
├── experiment.py               MAIN orchestrator (run methods × RAGs × datasets)
├── classify_all.py             run the failure classifier over all logs
├── xie_coverage.py             faithful Xie coverage-attribution method (answered×retrieved 2×2)
└── result/main/                the results (285 conversations, method/rag/dataset/…)

failure/                   # the failure classifier (6-type taxonomy) — consumes the logs
mtrag_validation/data/     # mtRAG human conversations (conversations.json)
```

Results are stored one directory per conversation:
`compare/result/main/<method>/<rag>/<dataset>/conversation_NNN/{conversation.json, conversation_failuretypes.json}`.

---

## Setup

```bash
pip install -r requirements.txt          # openai, numpy, datasets, networkx, scikit-learn
cp .env.example .env                      # then put your OPENAI_API_KEY in .env
```

qasper downloads from HuggingFace on first use. **Run all commands from this folder.**

The official mtRAG corpora are **not committed** (hundreds of MB, git-ignored). To run the
mtRAG method or the same-corpus arm, download them from
<https://github.com/IBM/mt-rag-benchmark> into `mtrag_validation/corpora/passage_level/`
(`clapnq/govt/fiqa/cloud.jsonl`) — see `compare/MTRAG_OFFICIAL_CORPUS.md`.

## Running the experiment

```bash
# generated methods on qasper (shared identical seed per RAG)
python -m compare.experiment --method proposed --rag vector --dataset qasper \
    --convos 15 --turns 8 --seed 42 --label main --shared-seed --inject-unanswerable-at 4
python -m compare.experiment --method xie_subq --rag vector --dataset qasper \
    --convos 15 --turns 8 --seed 42 --label main --shared-seed

# mtRAG human replay over the official corpus (per domain)
python -m compare.experiment --method mtrag --rag vector \
    --convos 15 --turns 8 --label main --mtrag-corpus official \
    --mtrag-corpus-path mtrag_validation/corpora/passage_level

# same-corpus arm: run Dynamic/Xie ON the mtRAG corpus (dataset = a domain)
python -m compare.experiment --method proposed --rag vector --dataset clapnq \
    --convos 15 --turns 8 --label main \
    --mtrag-corpus-path mtrag_validation/corpora/passage_level

# classify every log with the failure taxonomy
python -m compare.classify_all --glob "compare/result/main/**/conversation_*/conversation.json"

# classify a single log with the original entry point
python -m failure.run --file compare/result/main/proposed/vector/qasper/conversation_001/conversation.json
```

`--rag all` and `--method all` run the full matrix. `hippo`/`graph`/`raptor` are
auto-skipped for the mtRAG method (they can't build an index over the 366K-passage corpus).

## Results

`compare/result/main/` — **285 conversations**, all classified:

| method | conversations |
|---|--:|
| `proposed` (Dynamic + Unanswerable) | 90 |
| `xie_subq` (Xie sub-question) | 90 |
| `single_turn_qa` | 45 |
| `mtrag` (human) | 60 |

A results summary with all tables lives at `compare/result/main/result.md`.

---

## Notes

- **What's the dependent variable?** The failure *attribution* (the classifier's output), not
  the raw failure count — more failures ≠ better attribution.
- `compare/xie_coverage.py` implements the Xie paper's own coverage-attribution
  (`answered × retrieved` 2×2) as an alternative to the friend's classifier.
- `.env`, `__pycache__/`, the mtRAG corpora, and on-disk `*.embindex.*` files are
  git-ignored.
- Some files under `compare/` (`DYNAMIC/`, `STATIC/`, `shared/`, `fair_macro.py`,
  `multiseed.py`) and `conv_rag_benchmark/run_benchmark.py` are the earlier
  injected-failure experiment; the current pipeline is `compare/experiment.py`.
