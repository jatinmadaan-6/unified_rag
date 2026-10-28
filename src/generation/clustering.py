"""
Embedding-based clustering — the substitute for graph construction.

- Embed every chunk once (local model, no external API call needed).
- Cluster with KMeans (fast, O(n)) to get topical groups -> single-hop
  question candidates share a cluster.
- For multi-hop candidates, instead of traversing graph edges, find the
  nearest chunk in a *different* cluster to each cluster's centroid. This
  gives topically-adjacent-but-distinct pairs, which is what multi-hop
  questions need, without any path search.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .chunker import Chunk
except ImportError:  # pragma: no cover - direct script compatibility
    from chunker import Chunk


def embed_chunks(chunks: List[Chunk], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embeddings)


def cluster_chunks(chunks: List[Chunk], embeddings: np.ndarray,
                    target_chunks_per_cluster: int = 6) -> List[int]:
    n = len(chunks)
    if n <= 2:
        labels = [0] * n
    else:
        k = max(2, min(n // max(1, target_chunks_per_cluster), n - 1))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(embeddings).tolist()

    for chunk, label in zip(chunks, labels):
        chunk.cluster_id = int(label)
    return labels


@dataclass
class CrossClusterPair:
    chunk_a: Chunk
    chunk_b: Chunk
    similarity: float


def find_cross_cluster_pairs(chunks: List[Chunk], embeddings: np.ndarray,
                              top_k: int = 1,
                              min_similarity: float = 0.35,
                              max_similarity: float = 0.85) -> List[CrossClusterPair]:
    """
    For each cluster, find the top_k nearest chunks belonging to a different
    cluster. A similarity band (min/max) filters out pairs that are either
    unrelated (too low) or basically duplicates (too high) — both are poor
    multi-hop candidates.
    """
    sims = cosine_similarity(embeddings)
    labels = np.array([c.cluster_id for c in chunks])
    pairs: List[CrossClusterPair] = []

    for cluster_id in sorted(set(labels.tolist())):
        idx_in_cluster = np.where(labels == cluster_id)[0]
        idx_outside = np.where(labels != cluster_id)[0]
        if len(idx_in_cluster) == 0 or len(idx_outside) == 0:
            continue

        # use the cluster's centroid similarity to outside chunks
        centroid = embeddings[idx_in_cluster].mean(axis=0, keepdims=True)
        centroid_sims = cosine_similarity(centroid, embeddings[idx_outside])[0]

        candidates = [
            (idx_outside[i], centroid_sims[i])
            for i in range(len(idx_outside))
            if min_similarity <= centroid_sims[i] <= max_similarity
        ]
        candidates.sort(key=lambda x: -x[1])

        anchor_idx = idx_in_cluster[0]  # representative chunk from this cluster
        for outside_idx, sim in candidates[:top_k]:
            pairs.append(CrossClusterPair(
                chunk_a=chunks[anchor_idx],
                chunk_b=chunks[outside_idx],
                similarity=float(sim),
            ))

    return pairs
