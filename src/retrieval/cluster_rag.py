import json
import numpy as np
from typing import List, Dict, Any

from src.retrieval.base import BaseRetriever
from src.retrieval.cluster_utils.recursive_kmeans import RecursiveKMeansIndexer
from src.retrieval.cluster_utils.tree import ClusterTree
from src.utils.prompts import QUERY_DECOMPOSITION_PROMPT

class ClusterRAG(BaseRetriever):
    def __init__(self, llm, embedder, config: Dict):
        self.llm = llm
        self.embedder = embedder
        self.config = config
        self.trees: Dict[str, ClusterTree] = {}
        
    def index(self, chunks: List[Dict], embeddings: Any = None):
        """Indexes chunks by document using Recursive KMeans."""
        docs = {}
        for c in chunks:
            docs.setdefault(c["doc_id"], []).append(c)
            
        indexer = RecursiveKMeansIndexer(
            k_min=self.config.get("k_min", 2),
            k_max=self.config.get("k_max", 12),
            delta_threshold=self.config.get("delta_threshold", 0.02),
            min_clusters_to_stop=self.config.get("min_clusters_to_stop", 8),
            max_levels=self.config.get("max_levels", 6),
            seed=self.config.get("seed", 0)
        )
        
        hierarchical = self.config.get("hierarchical", True)
        self.trees = indexer.build_corpus(docs, self.embedder, hierarchical=hierarchical)
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        document_list = list(self.trees.keys())
        prompt = QUERY_DECOMPOSITION_PROMPT.format(document_list=json.dumps(document_list), user_query=query)
        raw = self.llm.generate(prompt)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except Exception:
            data = {"sub_queries": [{"document": d, "sub_query": query} for d in document_list]}

        N = self.config.get("N", 7)
        D = self.config.get("D", 3)
        k_leaf = self.config.get("k_leaf", 5)
        alpha = self.config.get("alpha", 0.7)
        use_sibling_expansion = self.config.get("use_sibling_expansion", False)

        all_scored = []
        for sq in data.get("sub_queries", []):
            doc_id, subquery = sq["document"], sq["sub_query"]
            tree = self.trees.get(doc_id)
            if not tree:
                continue
                
            q_emb = self.embedder.encode([subquery])[0]
            
            leaf_clusters = self._top_down_select(tree, q_emb, N, D)
            if use_sibling_expansion:
                leaf_clusters = self._sibling_expansion(tree, leaf_clusters)
                
            scored = self._retrieve_leaf_chunks(tree, leaf_clusters, q_emb, k_leaf, alpha)
            all_scored.extend(scored)
            
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        
        results = []
        seen = set()
        for c in all_scored:
            if c["unit_id"] in seen:
                continue
            seen.add(c["unit_id"])
            results.append({
                "unit_id": c["unit_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "score": c["score"]
            })
            if len(results) >= top_k:
                break
        return results

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _top_down_select(self, tree: ClusterTree, query_emb: np.ndarray, N: int, D: int):
        frontier = [tree.root]
        depth = 0
        while depth < D and frontier:
            scored_children = []
            for node in frontier:
                for child in tree.children(node.cluster_id):
                    sim = self._cosine(query_emb, np.array(child.centroid_embedding))
                    scored_children.append((sim, child))
            if not scored_children:
                break
            scored_children.sort(key=lambda x: x[0], reverse=True)
            frontier = [c for _, c in scored_children[:N]]
            depth += 1
        return frontier

    def _sibling_expansion(self, tree: ClusterTree, leaf_clusters: list):
        seen = {c.cluster_id for c in leaf_clusters}
        expanded = list(leaf_clusters)
        for c in leaf_clusters:
            if c.parent_cluster is None:
                continue
            parent = tree.get(c.parent_cluster)
            for sib_id in parent.child_clusters:
                if sib_id not in seen:
                    seen.add(sib_id)
                    expanded.append(tree.get(sib_id))
        return expanded

    def _retrieve_leaf_chunks(self, tree: ClusterTree, leaf_clusters: list, query_emb: np.ndarray, k_leaf: int, alpha: float):
        results = []
        for cluster in leaf_clusters:
            ancestor_path = tree.ancestor_path(cluster.cluster_id)
            path_sims = [self._cosine(query_emb, np.array(a.centroid_embedding)) for a in ancestor_path]
            path_sim = float(np.mean(path_sims)) if path_sims else 0.0

            member_scores = []
            for cid in cluster.member_chunk_ids:
                emb = np.array(tree.chunks[cid]["embedding"])
                leaf_sim = self._cosine(query_emb, emb)
                member_scores.append((leaf_sim, cid))
            member_scores.sort(key=lambda x: x[0], reverse=True)

            for leaf_sim, cid in member_scores[:k_leaf]:
                score = alpha * leaf_sim + (1 - alpha) * path_sim
                results.append({
                    "unit_id": cid,
                    "doc_id": tree.doc_id,
                    "text": tree.chunks[cid]["text"],
                    "score": score
                })
        return results
