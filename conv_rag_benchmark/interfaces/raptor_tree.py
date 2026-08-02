"""
raptor_tree.py — a faithful implementation of RAPTOR's tree, on this project's OpenAI
backend (so it is comparable to the other RAGs), without RAPTOR's conflicting package.

RAPTOR (Sarthi et al., ICLR 2024) works like this, and so does this module:
  1. Start with the corpus chunks as LEAF nodes; embed them.
  2. CLUSTER the current level's nodes (dimensionality reduction + Gaussian mixture).
  3. SUMMARISE each cluster with an LLM -> a new, higher-level parent node.
  4. Embed the summaries and repeat on them, building a tree bottom-up.
  5. "Collapsed tree" retrieval: pool ALL nodes (leaf chunks AND every summary, from every
     level) into one set and retrieve the top-k by cosine similarity — so a broad question
     can match a high-level summary and a specific one can match a raw chunk.

The original uses UMAP for reduction; we use PCA (scikit-learn) since UMAP is not installed.
The clustering, recursive summarisation and collapsed-tree retrieval are otherwise faithful.
"""
from __future__ import annotations
import json, os
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

_SUMMARY_SYS = (
    "You are building a RAPTOR summary tree. Write a concise, information-dense summary of "
    "the passages below that preserves the key facts, names and numbers, so it can answer "
    "questions on its own. Respond as JSON: {\"summary\": \"...\"}"
)


def _reduce(embs: np.ndarray, target: int = 24) -> np.ndarray:
    """Reduce embedding dimensionality so the mixture model is stable on few samples."""
    n, d = embs.shape
    k = max(2, min(target, d, n - 1))
    if k >= d:
        return embs
    return PCA(n_components=k, random_state=0).fit_transform(embs)


def _cluster_labels(embs: np.ndarray, per_cluster: int = 8) -> np.ndarray:
    """Soft-ish Gaussian-mixture clustering; returns a hard label per node."""
    n = len(embs)
    if n <= 2:
        return np.zeros(n, dtype=int)
    n_clusters = max(2, min(n // per_cluster, 40))
    red = _reduce(embs)
    gm = GaussianMixture(n_components=n_clusters, covariance_type="diag",
                         random_state=0, reg_covar=1e-4).fit(red)
    return gm.predict(red)


def _summarise(llm, texts: List[str]) -> str:
    joined = "\n\n---\n\n".join(t[:700] for t in texts[:12])
    out = llm.chat_json(_SUMMARY_SYS, f"PASSAGES:\n{joined}") or {}
    return str(out.get("summary") or "").strip() or texts[0][:400]


def build_tree(chunks: List[str], llm, embedder, max_levels: int = 3,
               cache_path: str = None) -> List[str]:
    """Build the RAPTOR tree and return the FLAT list of all node texts (leaves + every
    summary). Embeddings are recomputed by the caller (cheap); only texts are cached, since
    the LLM summaries are the expensive part to rebuild."""
    if cache_path and os.path.exists(cache_path):
        return json.load(open(cache_path, encoding="utf-8")).get("nodes", chunks)

    all_nodes: List[str] = list(chunks)          # leaves are part of the collapsed tree
    current = list(chunks)
    for _level in range(max_levels):
        if len(current) <= 4:
            break
        embs = np.asarray(embedder.encode(current), dtype=float)
        labels = _cluster_labels(embs)
        summaries: List[str] = []
        for c in sorted(set(labels)):
            members = [current[i] for i in range(len(current)) if labels[i] == c]
            if members:
                summaries.append(_summarise(llm, members))
        all_nodes.extend(summaries)
        current = summaries                       # recurse on the new level
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump({"nodes": all_nodes, "n_leaves": len(chunks)},
                  open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return all_nodes
