"""
conv_rag_benchmark
==================

A benchmark framework for *diagnosing* conversational RAG (retrieval-augmented
generation) systems. Inspired by RAG-DIVE (dynamic multi-turn evaluation),
JudgeAgent (agent-based dynamic evaluation) and GraphRAG (graph-grounded
retrieval).

The framework does **not** build a production RAG system. Its job is to:

    1. build a GraphRAG knowledge graph from source documents,
    2. generate coherent multi-turn conversations with typed query turns,
    3. generate *grounded* gold answers from graph evidence (not the raw dataset
       answer),
    4. send each turn to a pluggable target RAG,
    5. classify *why* the RAG failed (structured failure taxonomy),
    6. and report per-query-type / per-difficulty / per-turn statistics.

Every component degrades gracefully: with no OpenAI key and no local datasets the
whole pipeline still runs end-to-end on a built-in synthetic corpus using
deterministic heuristics, so the code is always testable.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
