# conv_rag_benchmark

A benchmark framework that **diagnoses *why* a conversational RAG system fails** —
not just whether its final answer is right. Inspired by **RAG-DIVE** (dynamic
multi-turn evaluation), **JudgeAgent** (agent-based dynamic evaluation) and
**GraphRAG** (graph-grounded retrieval).

The pipeline auto-generates multi-turn conversations with *typed* query turns,
synthesises **grounded** gold answers from a knowledge graph (so the answer key
isn't hallucinated), asks a pluggable target RAG, and classifies each failure into
a structured taxonomy.

> **Runs anywhere.** Every module degrades gracefully: with no OpenAI key and no
> downloaded datasets, the whole pipeline still runs end-to-end on a built-in
> synthetic corpus using deterministic heuristics. Add an `OPENAI_API_KEY` (in the
> repo's `RAG-DIVE/.env` or the environment) to get LLM-quality generation and
> LLM-as-a-judge classification.

## Pipeline

```
Dataset ─▶ Graph construction ─▶ GraphRAG retrieval ─▶ Conversation generation
       ─▶ Typed question generation ─▶ Grounded gold-answer generation
       ─▶ Target-RAG evaluation ─▶ Failure classification ─▶ Report + charts
```

## Quick start

```bash
pip install -r conv_rag_benchmark/requirements.txt   # only networkx/numpy/matplotlib are required

# fully offline, no API key, built-in corpus
python -m conv_rag_benchmark.main --dataset synthetic --offline --conversations 3 --max-samples 5

# JudgeAgent's local MultiHopRAG with an OpenAI key in RAG-DIVE/.env
python -m conv_rag_benchmark.main --dataset multihoprag --conversations 5 --target-rag mock
```

Outputs land in `conv_rag_benchmark/output/`:

| file | contents |
|---|---|
| `results.json` | every conversation + per-turn record (question, query_type, gold, rag_answer, failure_types, evidence) |
| `summary.csv` | one row per turn for spreadsheet analysis |
| `evaluation_report.json` | aggregate metrics |
| `figures/*.png` | failure distribution, query-type, difficulty and turn-depth charts |

## Modules

| module | class | role |
|---|---|---|
| `datasets/loader.py` | `DatasetLoader` | load + normalise MultiHopRAG / HotpotQA / 2WikiMultihopQA / MuSiQue / synthetic |
| `graph/graph_builder.py` | `GraphBuilder` | extract entities + typed relations → `networkx` graph linked to source chunks |
| `graph/retriever.py` | `GraphRetriever` | hybrid dense + graph-neighborhood retrieval |
| `generation/conversation_generator.py` | `ConversationGenerator` | coherent 3–8 turn conversations with mixed query types |
| `generation/query_generator.py` | `QueryGenerator` | the 8 typed query generators (+ difficulty, capability, expected failure) |
| `generation/gold_answer_generator.py` | `GoldAnswerGenerator` | **grounded** gold answers from graph evidence (not the raw dataset answer) |
| `interfaces/rag_interface.py` | `RAGInterface`, `MockRAG`, `VectorRAG`, `GraphRAGAdapter` | the target system under test |
| `evaluation/failure_taxonomy.py` | — | 4-category, 15-type failure taxonomy |
| `evaluation/failure_classifier.py` | `FailureClassifier` | multi-label LLM-as-a-judge + deterministic retrieval attribution |
| `evaluation/metrics.py` | `MetricsComputer` | query-type / failure / difficulty / turn-depth aggregates |
| `reports/visualization.py` | `Visualizer` | the four matplotlib charts |
| `pipeline.py` | `BenchmarkPipeline` | wires it all together |

## Query types ↔ capabilities ↔ failures

| query type | capability tested | expected failure |
|---|---|---|
| Follow-Up | coreference / history use | Coreference Failure |
| Clarification | context refinement | Incomplete Answer |
| Comparative | multi-entity retrieval / fusion | Comparative Failure |
| Correction | context replacement | Correction Failure |
| Topic Shift | memory flushing | Topic Shift Failure |
| Unanswerable | abstention / hallucination control | Overconfident Unknown |
| Multi-Hop | long-range reasoning | Multi-Hop Failure |
| Ambiguous Reference | ambiguity handling | Coreference Failure |

## Failure taxonomy

* **Retrieval** — Missing Retrieval, Wrong Retrieval, Partial Retrieval
* **Conversation** — Coreference Failure, History Ignoring, Context Drift, Correction Failure, Topic Shift Failure
* **Reasoning** — Multi-Hop Failure, Comparative Failure, Temporal Failure
* **Generation** — Hallucination, Overconfident Unknown, Contradiction, Incomplete Answer

## Configuration

Precedence: **CLI flags > environment > `config.yaml` > defaults** (see
`config.py` / `config.yaml`). The OpenAI key is read from the process env or the
repo's `RAG-DIVE/.env`. Generation (`gpt-4o-mini`) and judging (`gpt-4o`) use
*different* models on purpose, so the system that produces an answer never grades
itself.

## Plugging in your own target RAG

Subclass `RAGInterface` and implement `answer(question, history) -> RAGResponse`,
then pass an instance into `BenchmarkPipeline` (or extend `build_rag`). Returning
`retrieved_context` lets the classifier split retrieval failures from generation
failures.

## Notes on offline mode

Without an LLM the generators use templates and the classifier uses rule-based
attribution, so generated conversations are rougher (e.g. heuristic entity
extraction may pick odd spans). This mode exists to keep the framework testable
and CI-friendly — for research-quality benchmarks, run with an API key.
