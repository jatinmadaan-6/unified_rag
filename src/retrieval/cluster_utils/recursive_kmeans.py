"""
Recursive (hierarchical) K-Means clustering — the core novel component of Cluster-RAG
(Part 1.1 step 3, Part 2.2, Part 2.3 of the methodology).

Builds one ClusterTree per document. K is chosen per level via silhouette-guided
bisection rather than a fixed global K, because source documents vary by orders of
magnitude in size (Table 1 of HSG-RAG: 9k-289k tokens).
"""
from __future__ import annotations
import uuid
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.retrieval.cluster_utils.tree import ClusterNode, ClusterTree


def select_k(embeddings: np.ndarray, k_min: int = 2, k_max: int = 12,
             delta_threshold: float = 0.02, seed: int = 0) -> int:
    """Silhouette-guided bisection (Part 2.2): grow K while marginal silhouette
    improvement exceeds delta_threshold; cap at k_max so no level flattens out."""
    n = len(embeddings)
    if n <= k_min:
        return max(1, n)
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return k_min

    best_k, best_score = k_min, -1.0
    prev_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embeddings)
        try:
            score = silhouette_score(embeddings, km.labels_)
        except ValueError:
            break
        if score > best_score:
            best_score, best_k = score, k
        if k > k_min and (score - prev_score) < delta_threshold:
            break
        prev_score = score
    return best_k


class RecursiveKMeansIndexer:
    """
    Builds a per-document cluster tree (Part 1.1 step 3-4):
      Level 0: leaf chunk embeddings
      Level 1..L: recursively cluster the summary embeddings of the level below
    Stops when a level's cluster count falls below `min_clusters_to_stop`
    or when silhouette floor is not met.
    """

    def __init__(self, k_min: int = 2, k_max: int = 12, delta_threshold: float = 0.02,
                 min_clusters_to_stop: int = 8, max_levels: int = 6, seed: int = 0,
                 summarizer=None):
        self.k_min = k_min
        self.k_max = k_max
        self.delta_threshold = delta_threshold
        self.min_clusters_to_stop = min_clusters_to_stop
        self.max_levels = max_levels
        self.seed = seed
        self.summarizer = summarizer  # optional; injects LLM summary + YAKE keywords (Part 2.1 / 3)

    def build_tree(self, doc_id: str, chunks: List[dict], chunk_embeddings: np.ndarray,
                    hierarchical: bool = True) -> ClusterTree:
        tree = ClusterTree(doc_id)
        for c, emb in zip(chunks, chunk_embeddings):
            tree.add_chunk(c["unit_id"], c["text"], emb)

        # Keep tiny documents retrievable: the normal loop deliberately avoids
        # K-Means when there are too few samples, so materialize one leaf node.
        if len(chunks) <= self.min_clusters_to_stop:
            root_id = f"{doc_id}_L1_C00_{uuid.uuid4().hex[:6]}"
            root = ClusterNode(
                cluster_id=root_id, level=1, doc_id=doc_id,
                centroid_embedding=np.mean(chunk_embeddings, axis=0).tolist(),
                member_chunk_ids=[c["unit_id"] for c in chunks],
            )
            if self.summarizer is not None:
                root.summary, root.keywords = self.summarizer.summarize_cluster(
                    [c["text"] for c in chunks], is_leaf=True
                )
            tree.add_node(root, is_root=True)
            tree.level_k_report = [{"level": 1, "k": 1, "n_items": len(chunks)}]
            return tree

        # Level 0 nodes: one cluster per chunk conceptually, but we start clustering
        # directly on leaf embeddings to form level-1 leaf clusters.
        level = 1
        current_ids = [c["unit_id"] for c in chunks]
        current_embs = chunk_embeddings
        current_is_chunk_level = True
        level_k_report = []

        while True:
            n = len(current_embs)
            if n <= self.min_clusters_to_stop or level > self.max_levels:
                break

            k = select_k(current_embs, self.k_min, self.k_max, self.delta_threshold, self.seed) \
                if hierarchical else min(self.k_max, max(self.k_min, n // 20 or 1))
            level_k_report.append({"level": level, "k": k, "n_items": n})

            km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(current_embs)
            labels = km.labels_
            new_ids, new_embs = [], []

            for cluster_idx in range(k):
                member_mask = labels == cluster_idx
                member_ids = [current_ids[i] for i in range(n) if member_mask[i]]
                if not member_ids:
                    continue
                centroid = km.cluster_centers_[cluster_idx]
                node_id = f"{doc_id}_L{level}_C{cluster_idx:02d}_{uuid.uuid4().hex[:6]}"

                node = ClusterNode(
                    cluster_id=node_id,
                    level=level,
                    doc_id=doc_id,
                    centroid_embedding=centroid.tolist(),
                )
                if current_is_chunk_level:
                    node.member_chunk_ids = member_ids
                else:
                    node.child_clusters = member_ids
                    for child_id in member_ids:
                        tree.nodes[child_id].parent_cluster = node_id

                tree.add_node(node)
                new_ids.append(node_id)
                new_embs.append(centroid)

            if self.summarizer is not None:
                for node_id in new_ids:
                    node = tree.get(node_id)
                    if current_is_chunk_level:
                        member_texts = [tree.chunks[cid]["text"] for cid in node.member_chunk_ids]
                    else:
                        member_texts = [tree.get(c).summary for c in node.child_clusters]
                    node.summary, node.keywords = self.summarizer.summarize_cluster(
                        member_texts, is_leaf=current_is_chunk_level
                    )

            current_ids, current_embs = new_ids, np.vstack(new_embs)
            current_is_chunk_level = False

            if not hierarchical or k <= self.min_clusters_to_stop:
                level += 1
                break
            level += 1

        # Build a synthetic root if more than one node remains at the top level
        if len(current_ids) > 1:
            centroid = np.mean(current_embs, axis=0)
            root_id = f"{doc_id}_ROOT_{uuid.uuid4().hex[:6]}"
            root = ClusterNode(
                cluster_id=root_id, level=level, doc_id=doc_id,
                centroid_embedding=centroid.tolist(), child_clusters=current_ids,
                summary="root",
            )
            tree.add_node(root, is_root=True)
            for cid in current_ids:
                tree.get(cid).parent_cluster = root_id
        else:
            tree.root_id = current_ids[0]

        tree.level_k_report = level_k_report  # for Part 2.2 reporting table
        return tree

    def build_corpus(self, chunked_docs: dict, embedder, hierarchical: bool = True) -> dict:
        """chunked_docs: {doc_id: List[Chunk]} -> {doc_id: ClusterTree}
        Runs per-document (Part 1.1: 'partition by document', mirroring HSG-RAG's
        within-document constraint on long-distance relationship search)."""
        trees = {}
        for doc_id, chunks in chunked_docs.items():
            texts = [c["text"] for c in chunks]
            embs = embedder.encode(texts)
            trees[doc_id] = self.build_tree(doc_id, chunks, embs, hierarchical=hierarchical)
        return trees
