# RAG Failure Attribution — Results

**Research question:** Can *dynamic* follow-up question generation attribute RAG failures
more accurately than *static* sub-question generation?

*Numbers below are failure **counts/rates** — supporting evidence, not the final
attribution-accuracy metric. "Gen / Ret" = the layer the classifier attributed failures to
(generation vs retrieval). 15 conversations per cell.*

---

## What we built

Four **conversation-generation methods**, each run against the same RAG systems, then
classified by the same six-type failure classifier (a separate component):

| Method | Follow-up questions come from | Sees RAG answer? |
|---|---|:--:|
| **Dynamic** (+Unanswerable) | LLM, *after* observing the RAG's real previous answer | **yes** |
| **Xie** | **adapted from Xie et al. (NAACL 2025)**: the paper's sub-question decomposition (core/background/follow-up) replayed as turns (the faithful coverage 2×2 is `compare/xie_coverage.py`) | **no** |
| **Single-turn** | seed only, no follow-ups (control) | — |
| **mtRAG** | real **human** multi-turn conversations (IBM mt-RAG) replayed | — |

**Controls:** identical shared seed per RAG (qasper runs); the same-corpus arm puts
Dynamic/Xie/mtRAG on the *same* corpora with the *same* human seeds; every turn logs two
separate document sets (question-generation vs RAG-retrieved) with real doc_id/rank/score.

---

## 1. RAG comparison on qasper (with HippoRAG)

| Method | RAG | Fail% | Halluc | Gen-layer | Ret-layer | Top failure types |
|---|---|--:|--:|--:|--:|---|
| **Dynamic** | vector | 19% | 0 | 20 | 3 | grounding 17, response_cov 3, retrieval 3 |
| **Dynamic** | graph | 13% | 0 | 15 | 1 | grounding 13, response_cov 2 |
| **Dynamic** | selfrag | 32% | 0 | 34 | 4 | grounding 32, retrieval 4 |
| **Dynamic** | **hippo** | 29% | 3 | 21 | **14** | grounding 16, **retrieval 14**, response_cov 5 |
| **Xie** | vector | 23% | 4 | 17 | 11 | grounding 14, retrieval 11 |
| **Xie** | graph | 8% | 1 | 3 | 6 | retrieval 5, grounding 2 |
| **Xie** | selfrag | 29% | 3 | 19 | 16 | retrieval 16, grounding 15 |
| **Xie** | **hippo** | 22% | 0 | 8 | **18** | **retrieval 18**, grounding 8 |
| **Single-turn** | vector | 27% | 0 | 3 | 1 | grounding 2, retrieval 1 |
| **Single-turn** | graph | 7% | 0 | 0 | 1 | retrieval 1 |
| **Single-turn** | selfrag | 20% | 0 | 2 | 1 | grounding 2, retrieval 1 |

**HippoRAG behaves distinctly** — it surfaces far more **retrieval-layer** failures (14–18)
than dense `vector` (3–11) or `graph` (1–6), because its Personalized-PageRank-over-entity-
graph retrieval is more brittle than dense retrieval on these questions.

---

## 2. Same-corpus comparison (the strongest result)

All three methods on the **identical mtRAG official corpora** (clapnq/govt/fiqa/cloud) with
the **identical 15 human seeds** — so nothing differs except the follow-up strategy.

| Method | RAG | Fail% | Halluc | Gen-layer | Ret-layer | Top failure types |
|---|---|--:|--:|--:|--:|---|
| **Dynamic (+unans)** | vector | 30% | 5 | 32 | 4 | grounding 18, response_cov 14, retrieval 3 |
| **Dynamic (+unans)** | selfrag | 33% | 5 | 36 | 4 | grounding 23, response_cov 13 |
| **Xie** | vector | 28% | 10 | 30 | 4 | grounding 23, response_cov 7, **knowledge_boundary 2** |
| **Xie** | selfrag | 28% | 11 | 28 | 5 | grounding 19, response_cov 9, **knowledge_boundary 3** |
| **mtRAG (human)** | vector | 38% | 5 | 20 | **23** | **retrieval 22**, grounding 13 |
| **mtRAG (human)** | selfrag | 39% | 4 | 19 | **25** | **retrieval 23**, grounding 13 |

**Key finding:** even on the *identical* corpus with the *same* seeds, the generated
methods stay **generation-dominated** (~4 retrieval failures) while human questions are
**retrieval-dominated** (~23–25). So the difference is the **question source, not the
corpus**: generated follow-ups are *grounded* (the generator ensures the answer is
retrievable, so retrieval rarely fails), while human follow-ups expose retrieval failures.
The same-corpus control isolates this cleanly.

**Unanswerable injection works:** Dynamic now produces hallucinations (0 → 5) and Xie 10–11,
plus `knowledge_boundary` failures — the knowledge-boundary axis is exercised.

---

## Caveats

1. These are failure **counts/rates**, i.e. supporting evidence — **not** the dependent
   variable. More failures ≠ better attribution. The attribution-*accuracy* analysis is the
   next step on these same logs (`compare/xie_coverage.py` gives Xie's own coverage-based
   attribution as one option).
2. Small n (15 conversations/cell) — trends, not significance tests.
3. mtRAG has no `graph`/`hippo` column — building an LLM graph over the 366K-passage corpus
   is infeasible.
4. **Xie is an *adaptation*, not the paper verbatim.** The `xie_subq` method reuses Xie et
   al.'s sub-question *decomposition* but replays the sub-questions as conversation turns;
   replaying-as-turns and the ungroundable-sub-question→`Unanswerable` handling are ours,
   not the paper's. (The faithful Xie coverage 2×2 is `compare/xie_coverage.py`.) Because of
   this handling, Xie's hallucination/knowledge_boundary counts include ungroundable
   sub-questions treated as unanswerable, whereas Dynamic retries and *excludes* such turns
   — so cross-method **hallucination counts are not apples-to-apples**.

*Data: `compare/result/main/<method>/<rag>/<dataset>/conversation_NNN/`, each with its
`conversation_failuretypes.json`.*
