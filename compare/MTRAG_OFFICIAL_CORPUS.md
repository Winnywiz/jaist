# mtRAG official-corpus setup (for the `mtrag` experiment method)

Decision: the `mtrag` method replays the **human** mtRAG conversations through the **same
target RAG** as the generated methods, retrieving over the **official mtRAG corpora** —
**not** the union of already-retrieved contexts (that would give mtRAG an unfair
retrieval advantage and confound the comparison).

This file documents exactly how to **download, store, index, and map** the four corpora.
Nothing here runs automatically; the orchestrator only touches these files when you pass
`--mtrag-corpus official --mtrag-corpus-path <DIR>`.

---

## 1. Download

Source repo: <https://github.com/IBM/mt-rag-benchmark> — corpora are committed **in the
repo** under `corpora/passage_level/` (no HF, no external script). Passage-level is the
retrieval unit; use it (not `document_level/`).

| domain (our label) | file | documents | passages |
|---|---|---:|---:|
| clapnq (Wikipedia) | `corpora/passage_level/clapnq.jsonl.zip` | 4,293 | 183,408 |
| fiqa (finance)     | `corpora/passage_level/fiqa.jsonl.zip`  | 7,661 | 49,607 |
| govt (government)  | `corpora/passage_level/govt.jsonl.zip`  | 8,578 | 72,422 |
| cloud (technical)  | `corpora/passage_level/cloud.json.zip` ⚠️ | 57,638 | 61,022 |

⚠️ The repo listing shows the cloud archive as `cloud.json.zip` (not `.jsonl.zip`) —
**verify the exact name on download.** Total ≈ **366K passages**.

Two ways to fetch (either is fine; this is a **large binary download**, so it is left for
you to run / approve):

```bash
# option A — sparse checkout of just the corpora (no full history)
git clone --filter=blob:none --sparse https://github.com/IBM/mt-rag-benchmark.git
cd mt-rag-benchmark && git sparse-checkout set corpora/passage_level

# option B — direct file download (repeat per file)
curl -L -o clapnq.jsonl.zip \
  https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/corpora/passage_level/clapnq.jsonl.zip
```

## 2. Store

Unzip into a stable directory the orchestrator will read:

```
mtrag_validation/corpora/passage_level/
    clapnq.jsonl
    fiqa.jsonl
    govt.jsonl
    cloud.jsonl        # (or cloud.json — see ⚠️ above)
```

Then: `--mtrag-corpus official --mtrag-corpus-path mtrag_validation/corpora/passage_level`.
(The dir is git-ignored; corpora are ~hundreds of MB and must not be committed.)

## 3. JSONL schema  ⚠️ CONFIRM ON DOWNLOAD

The corpora README does **not** publish the field names, so they must be read off one
file after download. BEIR-style corpora (mtRAG uses the BEIR retrieval codebase) are
typically:

```json
{"_id": "822086267_7384-8758", "title": "...", "text": "the passage text ..."}
```

`compare/experiment.py::_mtrag_corpus_chunks` currently reads `text` / `contents` /
`passage` for the body. **After download, confirm the id field (`_id` vs `id` vs
`document_id`) and the text field**, and I will pin the parser to the real names so every
stored `doc_id` is a real corpus id (not just an array index).

## 4. Map (conversation context → corpus passage)

A conversation context's `document_id` is a corpus passage id **with two extra ingestion
offsets appended**. Per the retrieval README: for chunk id

```
822086267_7384-8758-0-1374
```

the corpus passage id is `822086267_7384-8758` — **drop the last two `-` values**
(`-0-1374`). So the mapping is:

```
corpus_passage_id = "-".join(document_id.split("-")[:-2])
```

This lets us (a) store the RAG's retrieved `doc_id` as the real corpus id and (b) check
whether a passage the RAG retrieved is the same one the human turn cited.

## 5. Domain routing (per conversation)

Each conversation carries its domain (`clapnq` / `govt` / `fiqa` / `cloud`, parsed from
`MtragConversation.domain`). A conversation must be replayed against a RAG indexed on
**its own** domain corpus — never a mix. The orchestrator will build/reuse one index per
domain and route each conversation accordingly.

## 6. Retrieval scores across methods (interpretation)

- Our RAGs re-index the official passages and retrieve with **cosine** (embeddings) —
  this is a *different* retriever from mtRAG's native **ELSER**, so raw scores are on
  different scales.
- Report policy: **`rank` is the cross-method/cross-retriever comparison; raw `score` is
  interpreted only within the same `score_type`.** Each doc record carries `score_type`
  so this stays explicit. mtRAG's native ELSER contexts (+scores) are preserved under
  each turn's `provenance.native_contexts` for reference.

---

## Known limitation to resolve before the mtRAG pilot

**Persistent index needed.** The current pipeline embeds the corpus **fresh every run**
(`GraphBuilder(...).build(chunks)` then `embedder.encode(kg.chunks)`). That is fine for
the qasper pilot (~300 chunks) but re-embedding **183K clapnq passages on every run** is
prohibitive in cost and time. Before running even the "tiny" mtRAG pilot we need a
**one-time, on-disk embedding index per domain** that runs reuse. This is new (small)
infra I have not built yet — see the open question below.
