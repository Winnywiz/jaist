"""
On-disk embedding index — build once per corpus, reuse across runs.

The RAG/retriever machinery embeds its whole corpus at construction time
(``embedder.encode(chunks)``). That is cheap for a small generated corpus (~300 chunks)
but prohibitive for the official mtRAG corpora (clapnq alone = 183K passages) if repeated
every run. This module caches the corpus embedding matrix to disk, keyed by the embedding
model + the exact chunk set, so the ~1.4K-call encode happens ONCE per domain.

It is purely a cache: the vectors are exactly what ``embedder.encode`` would return (same
model, same normalisation) — nothing is approximated. Query embeddings are NOT cached
(each query is unique); only the corpus matrix is.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional

import numpy as np


def _fingerprint(chunks: List[str]) -> str:
    """Stable hash of the chunk set (order-sensitive), so a cache hit means the exact
    same corpus in the exact same order — otherwise row i would not line up with chunk i."""
    h = hashlib.sha1()
    h.update(str(len(chunks)).encode())
    for c in chunks:
        h.update(b"\x00")
        h.update((c or "").encode("utf-8", "ignore"))
    return h.hexdigest()


def load_or_build_embeddings(embedder, chunks: List[str], cache_path: str,
                             model: Optional[str] = None) -> Optional[np.ndarray]:
    """Return the (n, d) embedding matrix for ``chunks``, from ``cache_path`` if valid.

    Writes ``<cache_path>.npy`` (the matrix) and ``<cache_path>.json`` (metadata: model,
    count, fingerprint). A cached matrix is reused only when the model, count, and
    fingerprint all match — so a changed corpus never silently loads stale vectors.
    Returns None if the embedder has no backend (caller falls back to lexical).
    """
    if embedder is None or not getattr(embedder, "available", False):
        return None
    model = model or getattr(getattr(embedder, "config", None), "embed_model", "unknown")
    npy_path = cache_path + ".npy"
    meta_path = cache_path + ".json"
    fp = _fingerprint(chunks)

    if os.path.exists(npy_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if (meta.get("model") == model and meta.get("count") == len(chunks)
                    and meta.get("fingerprint") == fp):
                arr = np.load(npy_path)
                if arr.shape[0] == len(chunks):
                    return arr
        except Exception:
            pass  # fall through and rebuild

    arr = embedder.encode(chunks)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(npy_path, arr)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"model": model, "count": len(chunks), "fingerprint": fp,
                   "dim": int(arr.shape[1]) if arr.ndim == 2 else None}, fh, indent=2)
    return arr
