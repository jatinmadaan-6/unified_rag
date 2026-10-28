"""Independent reproduction of RAPTOR's UMAP + GMM summary-tree baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.mixture import GaussianMixture

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
    """Bottom-up RAPTOR reproduction with collapsed-tree retrieval."""

    def __init__(self, llm, embedder, config: Dict[str, Any]):
        self.llm, self.embedder, self.config = llm, embedder, config
        self.nodes: List[_Node] = []
        self.chunks: Dict[str, Dict[str, str]] = {}
        self.leaf_embeddings: Dict[str, np.ndarray] = {}

    def _reduce(self, embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
        """UMAP projection used before a GMM fit; small levels need no projection."""
        count = len(embeddings)
        if count < 3:
            return embeddings
        try:
            import umap
        except ImportError as error:
            raise ImportError("RAPTOR needs umap-learn; run `pip install -r requirements.txt`.") from error
        components = min(int(self.config.get("umap_n_components", 10)), embeddings.shape[1], count - 1)
        return umap.UMAP(
            n_neighbors=min(max(2, n_neighbors), count - 1),
            n_components=max(2, components), metric="cosine",
            # Random initialization avoids UMAP's spectral-eigensolver failure
            # on tiny recursive clusters while retaining a seeded projection.
            init="random", random_state=int(self.config.get("seed", 0)),
        ).fit_transform(embeddings)

    def _best_gmm(self, reduced: np.ndarray) -> GaussianMixture:
        """Select number of Gaussian components by Bayesian information criterion."""
        maximum = min(int(self.config.get("max_clusters", 50)), len(reduced) - 1)
        models = [GaussianMixture(
            n_components=components, covariance_type="full",
            random_state=int(self.config.get("seed", 0)),
            reg_covar=float(self.config.get("gmm_reg_covar", 1e-6)),
        ).fit(reduced) for components in range(1, max(1, maximum) + 1)]
        return min(models, key=lambda model: model.bic(reduced))

    def _soft_clusters(self, embeddings: np.ndarray) -> List[List[int]]:
        """RAPTOR global/local UMAP+GMM clustering with soft memberships."""
        if len(embeddings) < int(self.config.get("min_nodes_to_cluster", 8)):
            return []
        threshold = float(self.config.get("membership_threshold", 0.1))
        global_reduced = self._reduce(embeddings, int(self.config.get("global_n_neighbors", 50)))
        global_model = self._best_gmm(global_reduced)
        global_probs = global_model.predict_proba(global_reduced)
        memberships = [[] for _ in range(len(embeddings))]
        next_cluster = 0
        for global_id in range(global_model.n_components):
            global_indices = np.flatnonzero(global_probs[:, global_id] >= threshold)
            if not len(global_indices):
                continue
            if len(global_indices) <= 2:
                for index in global_indices:
                    memberships[index].append(next_cluster)
                next_cluster += 1
                continue
            local_reduced = self._reduce(embeddings[global_indices], int(self.config.get("local_n_neighbors", 10)))
            local_model = self._best_gmm(local_reduced)
            for local_id, probabilities in enumerate(local_model.predict_proba(local_reduced).T):
                for local_position, probability in enumerate(probabilities):
                    if probability >= threshold:
                        memberships[global_indices[local_position]].append(next_cluster)
                next_cluster += 1
        # Preserve every node if it fell below the threshold in every mixture.
        for membership in memberships:
            if not membership:
                membership.append(next_cluster)
                next_cluster += 1
        return [[index for index, member_of in enumerate(memberships) if cluster_id in member_of]
                for cluster_id in range(next_cluster)]

    def index(self, chunks: List[Dict], embeddings: Any = None):
        if not chunks:
            self.nodes, self.chunks, self.leaf_embeddings = [], {}, {}
            return
        chunk_embeddings = np.asarray(embeddings) if embeddings is not None else self.embedder.encode([c["text"] for c in chunks])
        self.chunks = {c["unit_id"]: {"doc_id": c["doc_id"], "text": c["text"]} for c in chunks}
        self.leaf_embeddings = {c["unit_id"]: np.asarray(embedding) for c, embedding in zip(chunks, chunk_embeddings)}
        current = [_Node(c["unit_id"], c["text"], np.asarray(embedding), [c["unit_id"]], 0)
                   for c, embedding in zip(chunks, chunk_embeddings)]
        self.nodes = list(current)
        summarizer = ClusterSummarizer(self.llm)
        for level in range(1, int(self.config.get("max_levels", 3)) + 1):
            clusters = self._soft_clusters(np.vstack([node.embedding for node in current]))
            if not clusters:
                break
            next_level = []
            for cluster_id, positions in enumerate(clusters):
                if not positions:
                    continue
                members = [current[position] for position in positions]
                summary, _ = summarizer.summarize_cluster([node.text for node in members], is_leaf=False)
                next_level.append(_Node(
                    f"raptor_L{level}_C{cluster_id}", summary, self.embedder.encode([summary])[0],
                    sorted({leaf for node in members for leaf in node.leaf_ids}), level,
                ))
            if not next_level or len(next_level) >= len(current):
                break
            self.nodes.extend(next_level)
            current = next_level

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.nodes:
            return []
        query_embedding = self.embedder.encode([query])[0]
        scores = np.vstack([node.embedding for node in self.nodes]) @ query_embedding
        node_k = min(len(self.nodes), max(top_k, int(self.config.get("retrieval_node_k", 10))))
        selected = [self.nodes[index] for index in np.argsort(-scores)[:node_k]]
        candidate_ids = {leaf_id for node in selected for leaf_id in node.leaf_ids}
        ranked = sorted(((float(self.leaf_embeddings[unit_id] @ query_embedding), unit_id) for unit_id in candidate_ids), reverse=True)
        return [{"unit_id": unit_id, "doc_id": self.chunks[unit_id]["doc_id"],
                 "text": self.chunks[unit_id]["text"], "score": score}
                for score, unit_id in ranked[:top_k]]
