"""
Embedding backend with graceful degradation.

Preference order (configurable):
    1. ``sentence-transformers`` (local, free, no network) — ``all-MiniLM-L6-v2``,
    2. OpenAI embeddings via :class:`~conv_rag_benchmark.llm.LLM`,
    3. ``None`` — callers then fall back to lexical token-overlap retrieval.

The encoder always returns L2-normalised vectors so cosine similarity is a plain
dot product.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import Config, get_logger
from .llm import LLM

logger = get_logger("embeddings")


class Embedder:
    """Unified text -> vector encoder over whichever backend is available."""

    def __init__(self, config: Optional[Config] = None, llm: Optional[LLM] = None):
        self.config = config or Config.load()
        self.llm = llm
        self.backend: Optional[str] = None
        self._st_model = None

        if self.config.prefer_local_embeddings and self._try_load_st():
            self.backend = "sentence-transformers"
        elif (llm or LLM(config=self.config)).available:
            self.llm = llm or LLM(config=self.config)
            self.backend = "openai"
        else:
            self.backend = None
        logger.info("embedding backend: %s", self.backend or "lexical-fallback")

    @property
    def available(self) -> bool:
        return self.backend is not None

    def _try_load_st(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except Exception as exc:  # pragma: no cover - optional dep
            logger.info("sentence-transformers unavailable (%s)", exc)
            return False

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode a list of strings to a normalised (n, d) float32 array.

        Returns ``None`` if no embedding backend is available, signalling callers
        to use lexical retrieval instead.
        """
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "sentence-transformers":
            arr = self._st_model.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False)
            return np.asarray(arr, dtype=np.float32)
        if self.backend == "openai":
            # batch to stay within request limits
            mats = []
            for i in range(0, len(texts), 128):
                chunk = [t[:2000] for t in texts[i:i + 128]]
                vecs = self.llm.embed(chunk, model=self.config.embed_model)
                if vecs is None:
                    return None
                mats.append(vecs)
            return np.vstack(mats) if mats else None
        return None
