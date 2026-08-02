"""
connectors.py — the ONE place to plug a RAG system or a dataset into the benchmark.

The benchmark needs two things from you, and nothing else:

  1. a TARGET RAG  — the system under test. Any object with
                     ``answer(question, history) -> RAGResponse`` (see
                     :class:`conv_rag_benchmark.interfaces.rag_interface.RAGInterface`).
  2. a DATASET     — a corpus (+ optional seed questions), returned as a list of
                     :class:`conv_rag_benchmark.datasets.loader.Sample` rows.

Several of each already ship built-in (see ``available_rags`` / ``available_datasets``).
Add your own HERE, without touching the generation or grading code.

--------------------------------------------------------------------------------
ADD YOUR OWN RAG
--------------------------------------------------------------------------------
    from conv_rag_benchmark.interfaces.rag_interface import RAGInterface, RAGResponse

    class MyRAG(RAGInterface):
        name = "myrag"
        def answer(self, question, history=None):
            ctx = my_retriever(question)              # your retrieval
            ans = my_generator(question, ctx, history)  # your generation
            return RAGResponse(answer=ans, retrieved_context=ctx)

    register_rag("myrag", lambda chunks, config, **kw: MyRAG(...))
    #  -> now: run_benchmark --rag myrag

--------------------------------------------------------------------------------
ADD YOUR OWN DATASET
--------------------------------------------------------------------------------
    from conv_rag_benchmark.datasets.loader import Sample

    def load_mydata(path, limit):
        rows = my_read(path)                          # your source
        return [Sample(id=str(i), question=r["q"], answer=r["a"],
                       context=r["passages"]) for i, r in enumerate(rows)][:limit]

    register_dataset("mydata", load_mydata)
    #  -> now: run_benchmark --dataset mydata
    # (question/answer may be empty — the benchmark then generates a seed from context)
"""
from __future__ import annotations

from typing import Callable, List, Optional

# ============================================================================ #
# 1. RAG SYSTEMS (the system under test)
# ============================================================================ #
from .interfaces.rag_interface import RAGInterface, RAGResponse, build_rag

#: RAGs that ship with the benchmark (implemented in interfaces/rag_interface.py).
BUILTIN_RAGS = ("mock", "vector", "selfrag", "crag", "raptor", "longrag", "graph")

#: RAGs you register at runtime: name -> factory(chunks, config, **kw) -> RAGInterface
CUSTOM_RAGS: dict = {}


def register_rag(name: str, factory: Callable[..., RAGInterface]) -> None:
    """Register a custom RAG under ``name`` so ``--rag name`` can select it."""
    CUSTOM_RAGS[name.lower().strip()] = factory


def connect_rag(name: str, chunks, config, **kw) -> RAGInterface:
    """Build the target RAG named ``name`` over ``chunks`` (built-in or registered)."""
    name = (name or "mock").lower().strip()
    if name in CUSTOM_RAGS:
        return CUSTOM_RAGS[name](chunks, config=config, **kw)
    return build_rag(name, chunks, config=config, **kw)


def available_rags() -> List[str]:
    """Every RAG name that ``connect_rag`` can build right now."""
    return sorted(set(BUILTIN_RAGS) | set(CUSTOM_RAGS))


# ============================================================================ #
# 2. DATASETS (the corpus the benchmark is built from)
# ============================================================================ #
from .datasets.loader import DatasetLoader, Sample, _ADAPTERS as DATASET_ADAPTERS


def register_dataset(name: str,
                     loader: Callable[[Optional[str], Optional[int]], List[Sample]]) -> None:
    """Register a custom dataset loader so ``--dataset name`` can select it.

    ``loader(path, limit) -> list[Sample]`` — return canonical rows (see Sample).
    """
    DATASET_ADAPTERS[name.lower().strip()] = loader


def connect_dataset(name: str, max_samples: Optional[int] = None,
                    path: Optional[str] = None) -> List[Sample]:
    """Load the dataset named ``name`` as a list of canonical Sample rows."""
    return DatasetLoader(name, path=path, max_samples=max_samples).load()


def available_datasets() -> List[str]:
    """Every dataset name that ``connect_dataset`` can load right now."""
    return sorted(DATASET_ADAPTERS)


__all__ = [
    "RAGInterface", "RAGResponse", "Sample",
    "register_rag", "connect_rag", "available_rags", "BUILTIN_RAGS", "CUSTOM_RAGS",
    "register_dataset", "connect_dataset", "available_datasets", "DATASET_ADAPTERS",
]
