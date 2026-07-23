# DYNAMICQA

Clean, readable layout of the experiment that answers the thesis research question:

> **Can dynamic follow-up question generation attribute RAG failures more accurately
> than static sub-question generation?**

Each method lives in its own folder; results are written to `result/`.

```
DYNAMICQA/
├── run.py                 ← single entry point
├── DYNAMIC/               ← THE PROPOSED METHOD
│   └── method.py            dynamic follow-up attribution (reacts to the RAG's answer)
├── STATIC/                ← THE COMPARISON / BASELINE METHODS
│   └── method.py            Xie static decomposition + single-turn control
├── shared/                ← scaffolding both methods use (not "a method")
│   ├── setup.py             inject controlled failures, wrap the RAG, judge correctness
│   └── harness.py           run every method on the same cases + score + save
└── result/                ← output JSON lands here
```

## What is being compared

| folder | method | reacts to RAG answer? | can output "Conversation"? |
|--------|--------|:--:|:--:|
| `STATIC/` | 1. single-turn control | no | no |
| `STATIC/` | 2. Xie static decomposition (core) | no | no |
| `STATIC/` | 3. Xie static decomposition (follow-up) | no | no |
| `DYNAMIC/` | 4. **dynamic follow-up (proposed)** | **yes** | **yes** |

The **only** difference under test is *how each method probes*. Static methods fix all
sub-questions up front and never see the RAG's answer, so they are structurally blind to
coreference (Conversation) failures. The dynamic method synthesises probes from the RAG's
actual answer, so it can reach that category.

## How ground truth works (why "accuracy" is meaningful)

Real RAG failures have an unknown cause, so `shared/setup.py` **injects** failures whose
cause is *certain*:

* **Retrieval** — withhold the gold passage (answer verifiably absent from context).
* **Generation** — ask for a plausible detail absent from all context and force an answer,
  so the model fabricates.
* **Conversation** — a real coreference follow-up with the conversation history dropped.

Each method must recover the injected category. Accuracy = fraction recovered.

## Run it

From the **repo root** (`D:\intershippu`, where `conv_rag_benchmark/` also lives):

```bash
# real run (needs an OpenAI key in RAG-DIVE/.env or OPENAI_API_KEY)
python -m DYNAMICQA.run --dataset multihoprag --n 20

# plumbing smoke test, no API calls (numbers are NOT evidence)
python -m DYNAMICQA.run --dataset multihoprag --n 8 --offline
```

Output: `result/attribution_<dataset>.json` — per-method accuracy, per-category accuracy,
confusion pairs, and every injected case (question, RAG answer, and each method's verdict)
so a human can audit any decision.

## Dependency note

DYNAMICQA is the **readable experiment layer**. The heavy engine (RAG generator,
embeddings, LLM client, dataset loaders) is reused from the `conv_rag_benchmark/` package
in the same repo — that is why you run it from the repo root. This keeps DYNAMICQA small
and focused on the method comparison itself. (For a fully self-contained, sendable copy,
see the separate `DYQAGEN/` package.)
