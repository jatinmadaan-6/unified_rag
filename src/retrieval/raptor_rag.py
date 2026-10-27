"""RAPTOR-style recursive summary-tree baseline.

This is an in-repository, fully local implementation of RAPTOR's essential
bottom-up indexing idea: cluster a level, summarize each cluster, re-embed the
summaries, and repeat. Retrieval uses collapsed-tree search followed by leaf
evidence ranking.  It is deliberately labeled "RAPTOR-style" in reports: it
uses K-Means rather than the original paper's UMAP + GMM implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans

from src.retrieval.base import BaseRetriever
from src.retrieval.cluster_utils.summarizer import ClusterSummarizer


@dataclass
class _Node:
    node_id: str
    text: str
    embedding: np.ndarray
    leaf_ids: List[str]
    level: int


class RaptorRAG(BaseRetriever):
    """Local RAPTOR-style baseline with recursive summary embeddings."""

    def __init__(self, llm, embedder, config: Dict[str, Any]):
        self.llm = llm
        self.embedder = embedder
        self.config = config
        self.nodes: List[_Node] = []
        self.chunks: Dict[str, Dict[str, str]] = {}
        self.leaf_embeddings: Dict[str, np.ndarray] = {}

    def index(self, chunks: List[Dict], embeddings: Any = None):
        if not chunks:
            self.nodes, self.chunks, self.leaf_embeddings = [], {}, {}
            return
        chunk_embeddings = embeddings if embeddings is not None else self.embedder.encode([c["text"] for c in chunks])
        self.chunks = {
            c["unit_id"]: {"doc_id": c["doc_id"], "text": c["text"]}
            for c in chunks
        }
        self.leaf_embeddings = {
            c["unit_id"]: np.asarray(embedding)
            for c, embedding in zip(chunks, chunk_embeddings)
        }
        current = [
            _Node(c["unit_id"], c["text"], np.asarray(emb), [c["unit_id"]], 0)
            for c, emb in zip(chunks, chunk_embeddings)
        ]
        self.nodes = list(current)
        branch_factor = max(2, int(self.config.get("branch_factor", 5)))
        max_levels = max(1, int(self.config.get("max_levels", 3)))
        min_nodes_to_cluster = max(branch_factor + 1, int(self.config.get("min_nodes_to_cluster", 8)))
        summarizer = ClusterSummarizer(self.llm)

        for level in range(1, max_levels + 1):
            if len(current) < min_nodes_to_cluster:
                break
            k = min(branch_factor, len(current) - 1)
            matrix = np.vstack([node.embedding for node in current])
            labels = KMeans(n_clusters=k, n_init=10, random_state=self.config.get("seed", 0)).fit_predict(matrix)
            next_level = []
            for cluster_id in range(k):
                members = [node for node, label in zip(current, labels) if label == cluster_id]
                if not members:
                    continue
                summary, _ = summarizer.summarize_cluster([node.text for node in members], is_leaf=False)
                next_level.append(_Node(
                    node_id=f"raptor_L{level}_C{cluster_id}",
                    text=summary,
                    embedding=self.embedder.encode([summary])[0],
                    leaf_ids=[leaf_id for node in members for leaf_id in node.leaf_ids],
                    level=level,
                ))
            if len(next_level) >= len(current):
                break
            self.nodes.extend(next_level)
            current = next_level

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.nodes:
            return []
        query_embedding = self.embedder.encode([query])[0]
        node_matrix = np.vstack([node.embedding for node in self.nodes])
        node_scores = node_matrix @ query_embedding
        node_k = min(len(self.nodes), max(top_k, int(self.config.get("retrieval_node_k", 10))))
        selected = [self.nodes[i] for i in np.argsort(-node_scores)[:node_k]]

        candidate_ids = {leaf_id for node in selected for leaf_id in node.leaf_ids}
        scored = []
        for unit_id in candidate_ids:
            chunk = self.chunks[unit_id]
            score = float(self.leaf_embeddings[unit_id] @ query_embedding)
            scored.append((score, unit_id, chunk))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [
            {"unit_id": unit_id, "doc_id": chunk["doc_id"], "text": chunk["text"], "score": score}
            for score, unit_id, chunk in scored[:top_k]
        ]
