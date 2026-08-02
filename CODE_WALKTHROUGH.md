# Adaptive Conversational RAG-Probing Benchmark — Complete Code & Method Walkthrough

*A self-contained explanation for someone who has never seen this codebase. It covers what
the project does, what every file is for, how documents are retrieved, how questions and
answers are generated, how quality is controlled, and how every metric is computed.*

---

## 0. What this project is (in one paragraph)

This is a system that **automatically generates multi-turn conversations to test a
Retrieval-Augmented Generation (RAG) system**, grades the RAG's real answers, and reports
two things: (1) how **good the generated questions** are, and (2) **where and why the RAG
fails**. Unlike a static test set written in advance, it is **adaptive** — it reads the
RAG's actual answer to each question and uses that outcome to choose the next question.

Everything runs on OpenAI models: `gpt-4o-mini` generates and judges cheaply,
`gpt-4o` is the stricter grader, and `text-embedding-3-small` powers retrieval.

> **Scope note.** The package ships the *proposed method*: generation + question quality.
> Earlier baselines, the RAG-answer grading suite, the failure-attribution experiment and
> assorted one-off diagnostics were moved to **`_archive/`**. Sections below that describe
> archived tools are marked 🗄️ — the *results* they produced are real and still in
> `result/`, but the scripts are no longer part of the core package.

---

## 1. The pipeline at a glance

```
 dataset (seed questions + passages)          ← plug in via connectors.py
        │
        ▼
 build corpus  ──►  dense retriever (embeddings + cosine)
        │
        ▼
 FOR EACH CONVERSATION, FOR EACH TURN (the adaptive loop):
     1. pick the question TYPE   ← based on the RAG's previous outcome
     2. write the QUESTION       (query generator)
     3. RETRIEVE evidence        (dense retrieval, top-k)
     4. write the GOLD answer    (from that evidence only)
     5. run 7 QUALITY GUARDS     (reject + retry if the turn is bad)
     6. ask the REAL RAG         (the system under test answers)  ← plug in via connectors.py
     7. GRADE the answer         → correct / wrong / hallucinated / abstained
     8. that outcome picks the NEXT type → back to step 1
        │
        ▼
 OUTPUTS  →  result/benchmark_quality/<dataset>/{rag}_{dataset}_t{turns}_c{convos}.json
   • question quality    (G-Eval: well_formed, gold_supported, gold_correct)
   • RAG failure profile (how often it failed, on which question types)
   • the full conversations (question, gold, RAG answer, outcome per turn)
```

---

## 2. File map — what each piece of code is for

### Core method

| File | Responsibility |
|---|---|
| `run_benchmark.py` | **Entry point.** Parses arguments, wires everything together, runs the loop, saves results. |
| `connectors.py` | **The plug-in point** — register your own RAG or dataset here. |
| `config.py` | Central settings: which models, the API key, file paths. |
| `datasets/loader.py` | Loads a dataset into `Sample` objects (`question`, `answer`, `context`). |
| `embeddings.py` | Wraps the OpenAI embedding model (text → vector). |
| `graph/retriever.py` | The retrieval engine — dense cosine search over the corpus. |
| `graph/graph_builder.py` | Builds the knowledge store from corpus chunks (dense-only when `--graph-mode none`). |
| `interfaces/rag_interface.py` | The RAG systems **under test**: vector, graph, raptor, selfrag, longrag, crag. |
| `generation/query_generator.py` | Writes the actual question for each question type. |
| `generation/gold_answer_generator.py` | Writes the gold answer strictly from retrieved evidence. |
| `generation/adaptive_generator.py` | **The core** — the adaptive loop + the 7 quality guards. |
| `llm.py` | Thin wrapper that actually calls the OpenAI API and returns JSON. |

### Question-quality measurement

| File | Responsibility |
|---|---|
| `geval.py` | Scores question quality (well_formed / gold_supported / gold_correct). |
| `quality_judge.py` | Judging helpers used by the quality scorers. |
| `atomic_faithfulness.py` | Decompose-then-entail grounding check (RAGAS/ARES style). |
| `grade_questions.py` | Type diversity, doc-required (anti-cheat), follow-up dependency, distinct-1/2. |
| `grade_gricean.py` | Follow-up quality on the 5 Gricean maxims. |
| `geval_compare.py` | Plain LLM judge vs real G-Eval, on well-formedness. |
| `make_quality_viewer.py` | Builds `result/quality_viewer.html` to browse everything. |
| `run_ablation.py` | Controller vs random-type selection (does adaptivity itself matter?). |

### 🗄️ Archived (in `_archive/`, produced results that remain in `result/`)

| File | Responsibility |
|---|---|
| `eval_llm_metrics.py` | Faithfulness, context recall, context precision, doc-required. |
| `question_doc_similarity.py` | Attribution: benchmark's evidence vs the RAG's retrieved evidence. |
| `make_validation_page.py` / `score_validation.py` | Human-validation tool for the faithfulness judge. |
| `build_benchmark.py`, `build_d_fulldata.py`, `build_alltypes.py` | Earlier baseline generators. |
| `grade_e.py`, `grade_def.py`, … | The RAG-answer grading suite. |
| `DYNAMICQA/` | The dynamic-vs-static failure-attribution experiment. |

---

## 3. Setup phase (before any conversation is generated)

`run_benchmark.py::main()` does the following, in order:

1. **Load config** — models, key, paths (`config.py`).
2. **Load the dataset** — `connect_dataset(name)` returns *seeds*, each a `Sample` with
   a `question`, a gold `answer`, and a list of `context` passages.
3. **Build the corpus** — flatten every passage from every seed into one big pool of text
   `chunks`. This pool is the "knowledge" the whole benchmark and the RAG draw from.
4. **Embed the corpus** — each chunk is turned into a 1536-dimension vector once, up front.
5. **Build the retriever** — `GraphRetriever` over those chunks. With `--graph-mode none`
   it does pure dense (embedding) retrieval; no knowledge graph is used. (The graph was
   tested and dropped — it did not help.)
6. **Build the RAG under test** — `connect_rag(name, chunks, ...)`. This is the *system
   being probed*.
7. **Seed filter** — drop any dataset seed question that still contains leftover LaTeX
   labels, so the first turn starts clean.

---

## 4. How documents are retrieved (the retrieval method)

Retrieval happens whenever the generator needs evidence to write a question and its gold.
The method is **dense passage retrieval** — three sub-steps:

**Step 4.1 — make the query self-contained.** A follow-up question like *"What is **its**
size?"* cannot retrieve anything, because "its" is meaningless alone. So it is first
rewritten using the conversation into a standalone query: *"What is the size of **CA**?"*
(function `_self_contained`). This is *conversational query rewriting*.

**Step 4.2 — dense retrieval** (`retriever.py::_dense_seed`):
```python
qv   = embedder.encode([query])     # 1. embed the query  -> a 1536-dim vector
sims = chunk_embeddings @ qv[0]     # 2. cosine similarity vs EVERY chunk (dot product of unit vectors)
top  = np.argsort(-sims)[:k]        # 3. keep the k most similar chunks
```
- The query is embedded with `text-embedding-3-small`.
- **Cosine similarity** is computed against every corpus chunk. Cosine = the dot product of
  two direction-normalized vectors; it measures *meaning* overlap, so "car" matches
  "automobile" even with no shared words.
- The **top-k** chunks are returned. `k = 8` normally; `k = 16` for Multi-Hop and
  Comparative questions, which need more evidence to span several facts.
- If embeddings are unavailable, it falls back to **IDF token-overlap** lexical scoring.

**Step 4.3 — entity merge.** For comparison questions, a second retrieval on the target
entity's name is merged in, so both compared entities appear in the evidence.

The result is a set of evidence chunks. The gold answer is then written **only** from these
chunks, so the benchmark stays grounded in the corpus, never in the model's memory.

> Important: the RAG under test does its **own, separate** retrieval when it answers. So each
> turn has two evidence sets — the benchmark's (used to write the gold) and the RAG's (used
> to answer). Comparing them is how failures are attributed to retrieval vs generation.

---

## 5. Generating one conversation (the adaptive loop)

`adaptive_generator.py::generate()` builds one conversation, keeping **two histories**:
- `truth_history` — question + **gold** answer. Drives the next question, so generation
  stays grounded in truth (not in the RAG's possibly-wrong replies).
- `rag_history` — question + the **RAG's real** answer. This is what the RAG sees on later
  turns, and what the adaptive controller reacts to.

**Turn 0 (the seed):** the question comes from the dataset. If its gold cannot be grounded,
a fallback rewrites the seed question from corpus text (with LaTeX labels stripped first).
Corpus-only datasets ship no seed question, so one is generated from the seed's evidence.

**Turns 1…N (the loop):** each turn runs steps 1–8 from the pipeline. Two steps deserve
detail:

- **Step 1, type selection (`_next_type`)** — this is what makes it *adaptive*:

  | RAG's last outcome | next question type | what it probes |
  |---|---|---|
  | `wrong` | Correction | can it recover from its own mistake? |
  | `hallucinated` | Unanswerable | will it ever admit it doesn't know? |
  | `abstained` | Clarification / Follow-Up | can it be re-engaged? |
  | `correct` | Multi-Hop → Comparative → Ambiguous Reference | escalate difficulty |
  | 4 correct in a row | Topic Shift | did sustained success leave it clinging to stale context? |

  A static benchmark fixes the whole type list in advance; here the RAG's real behaviour
  steers it.

- **Step 7, grading (`_grade`)** — the judge model (`gpt-4o`) reads the question, the gold,
  and the RAG's answer, and returns exactly one label: `correct`, `wrong`, `hallucinated`,
  or `abstained`.

The 8 question types each map to a conversational capability and an expected failure mode
(see `query_generator.py::QUERY_TYPES`), so a report can say *why* a RAG failed, not just
*that* it did.

---

## 6. The 7 quality guards (why the questions are trustworthy)

After a gold is written, the turn passes through guards. **If any guard fires, the turn is
rejected and regenerated** — up to 3 attempts, after which it is accepted but flagged
`guard_gave_up`. Each guard is a small, pure, testable function.

| # | Guard | Rejects a turn when… |
|---|---|---|
| 1 | Structural type check | a pronoun-type question has no pronoun, or names its target entity |
| 2 | Placeholder guard | question or gold contains a leftover LaTeX label (`TABREF1`, `SECREF27`) |
| 3 | Seed filter + rewrite-strip | the seed question carries a LaTeX label |
| 4 | Strict-gold | the gold is not *supported by AND correct against* the evidence |
| 5 | Citation-only guard | the gold is just a reference marker, e.g. "…in [7]" |
| 6 | No-repeat guard | the gold restates an answer already given earlier |
| 7 | Adds-new-info guard | a long gold restates the current question and adds nothing new |

**How the guards decide (examples):**
- *No-repeat* compares the new gold to earlier ones by exact match, containment, or ≥ 0.8
  word-overlap (Jaccard ratio).
- *Adds-new-info* rejects a gold of ≥ 5 meaningful words that adds fewer than 2 new words
  not already in the question — but **exempts short answers** (a number, name, or date), so
  "10000" or "Fubo" always pass.
- *Citation-only* fires only when a `[n]` marker exists AND removing it leaves < 3
  meaningful words.

When a guard fires, the generator is told *in words* what was wrong (e.g. "the answer just
restates the question; ask about a fact whose answer introduces new content") and tries
again — this is *self-refinement*.

---

## 7. How each metric is computed

### 7A. Question quality — G-Eval (`geval.py`)
An LLM judge reads each question against a 1–5 rubric; the score is weighted by the model's
output probabilities and normalized to 0–1. Three scores:
- **well_formed** — is the question clear, specific, answerable?
- **gold_supported** — is the gold backed by the evidence?
- **gold_correct** — is the gold a correct answer to the question?

### 7B. RAG failure profile (computed by counting)
From the per-turn outcomes: `failure_rate = (wrong + hallucinated) / total_turns`, plus a
breakdown of which question types produced the failures. No LLM needed — pure counting.

### 7C. Generation-quality extras (`grade_questions.py`, `grade_gricean.py`)
- **Type diversity** — how many distinct types appear, per conversation and overall.
- **Doc-required** — anti-cheat: answer the question from memory only, then check whether it
  matches the gold. If it does, the question didn't need the corpus.
- **Follow-up dependency** — is a follow-up genuinely un-understandable without the prior turn?
- **Distinct-1 / Distinct-2** — lexical diversity of the generated questions (non-LLM).
- **Gricean maxims** — relevance, informativeness, truthfulness, clarity, coherence, each
  scored 1–5 by a logprob-weighted judge.

### 7D. 🗄️ The 4 LLM-judged RAG metrics (`_archive/.../eval_llm_metrics.py`)
Still present in saved results under the `llm_metrics` key.

| Metric | Compares | Score = |
|---|---|---|
| **Faithfulness** | RAG answer ↔ retrieved chunks | supported claims / total claims |
| **Context recall** | gold ↔ retrieved chunks | gold facts present / total |
| **Context precision** | question ↔ retrieved chunks | relevant chunks / total |
| **Doc-required** | question (no docs) → gold | 1 if it needs docs, 0 if answerable from memory |

Faithfulness measures the **generation** step; context recall measures the **retrieval**
step; precision measures signal vs noise; doc-required is an **anti-cheat** check.

> **Known limitation:** context precision counts *chunks*, so a RAG that retrieves few but
> very large chunks (LongRAG) is penalized unfairly — a metric-vs-architecture confound, not
> a real weakness. Interpret precision qualitatively across architectures.

### 7E. 🗄️ Attribution — evidence overlap (`_archive/.../question_doc_similarity.py`)
Saved as `*_docsim.json` next to each run. Records `gen_evidence` (what the benchmark used)
and `rag_evidence` (what the RAG retrieved), plus a `gen_rag_overlap` number:
- **high overlap + wrong answer** → the RAG had the right evidence but answered wrong →
  **generation failure**.
- **low overlap + wrong answer** → the RAG never retrieved the answer passage →
  **retrieval failure**.

---

## 8. 🗄️ Validating the judge against a human

Because the LLM metrics are judged by an LLM, they need checking. The archived tool samples
turns, lets a human score faithfulness blind (without seeing the judge's score), then
computes agreement: correlation, mean error, and **Cohen's kappa**. Result: **κ = 0.69
(strong)** on 35 turns — so faithfulness is treated as validated; the other three metrics
remain directional.

---

## 9. The datasets (what is tested on)

Three kinds:
- **QA with human gold:** QASPER (CS papers), MultiHopRAG (news), MTRAG (human dialogues).
- **Corpus-only:** ArXivCS, mlarxiv — text only, no existing Q&A, so the method generates
  everything.
- **Multi-hop Wikipedia:** HotpotQA, MuSiQue — clean encyclopedia text, questions needing
  multiple facts.

Cleaner source text ⇒ higher `well_formed` (Wikipedia ≈ 0.72, dense CS papers ≈ 0.47),
which shows well_formed reflects the **corpus**, not the generator.

Add your own with `register_dataset(name, loader)` in `connectors.py`.

---

## 10. The RAG systems tested (`interfaces/rag_interface.py`)

Six architectures, all on the **same backend** (`text-embedding-3-small` + `gpt-4o-mini`)
so differences reflect the *architecture*, not a stronger model:
- **VectorRAG** — plain dense retrieve-then-generate.
- **GraphRAG** — retrieves via entity/relation graph expansion.
- **RaptorRAG** — collapsed-tree retrieval over leaves + LLM cluster summaries.
- **SelfRAG** — generate → self-critique the answer → re-retrieve if unsupported.
- **LongRAG** — retrieves few, very large chunks.
- **CRAG** — grades the *evidence* first, then refines/re-retrieves before answering.

Plus **MockRAG**, a deliberately weak RAG (drops history, hallucinates) used to make the
failure taxonomy light up during testing.

These are faithful re-implementations of each architecture's core idea, not the authors'
original code — a deliberate choice so the comparison is controlled and fair.

Add your own with `register_rag(name, factory)` in `connectors.py`.

---

## 11. Output files

```
result/benchmark_quality/<dataset>/{rag}_{dataset}_t{turns}_c{convos}.json
```

e.g. `graph_HotpotQA_t10_c5.json` = the `graph` RAG on HotpotQA, 10 turns × 5
conversations. The standard configuration (strict gold, no generation graph) is unmarked;
deviations add a suffix (`_randomtype`, `_qgate`, `_nostrictgold`, `_typedgraph`) so an
ablation can never overwrite a standard run. `*_docsim.json` files are the archived
evidence-overlap analysis, not benchmark runs.

---

## 11b. Run-to-run variance and `--repeats`

Two runs of the **identical command** do not give the same score. Two compounding causes:

1. **Generation is stochastic** — `llm.py::chat_json` defaults to `temperature=0.3`, and no
   API `seed` is passed, so each question is written slightly differently each time.
2. **The loop is path-dependent** — the controller picks the next question *type* from the
   RAG's outcome. So one differently-worded question that flips the RAG from `wrong` to
   `correct` sends the whole remainder of the conversation down a different branch, and the
   two runs end up asking **different questions**.

Because the quality score is an average over the questions actually asked, comparing two
single runs compares two different question sets — not a controlled comparison.

`--repeats N` runs the same config N times and reports **mean ± std** (setup is built once
and reused). Two configurations count as different only when their mean ± std ranges do not
overlap. Observed on MultiHopRAG/graph: `well_formed` moved 0.578 → 0.614 across two runs
with *no* configuration change at all, while `failure_rate` stayed identical at 0.12.

---

## 12. Honest limitations (stated plainly)

- **Sample sizes are modest** (50–120 turns per cell); treat differences under ~0.05 as noise.
- **The LLM metrics use a single, mostly-unvalidated judge** — only faithfulness has been
  human-validated (κ = 0.69).
- **Some graded "failures" are gold-precision artifacts** (e.g. gold "48" vs a more precise
  "48.4"), so failure rates are a mild upper bound.
- **Context precision is confounded by chunk size** across architectures.
- **The RAGs are simplified re-implementations**, so absolute numbers may differ from the
  original papers; the *relative* ranking is fair because the backend is shared.
- **An LLM grades LLM output.** Relative comparisons are far more trustworthy than absolute
  values, because the same judge scores every arm.

---

## 13. One-sentence summary

*A dataset provides seed passages; a dense retriever fetches evidence; an adaptive loop
writes typed questions and grounded gold answers, passes them through seven quality guards,
asks a real RAG, grades each answer, and lets the outcome steer the next question — then
reports how good the generated questions are and where the RAG failed.*
