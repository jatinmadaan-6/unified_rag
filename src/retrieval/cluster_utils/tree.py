"""
ClusterNode data structure — the direct analog of HSG-RAG's knowledge graph.
Persists parent/child pointers explicitly and supports JSON serialization.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


@dataclass
class ClusterNode:
    cluster_id: str
    level: int
    doc_id: str
    centroid_embedding: List[float]
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    member_chunk_ids: List[str] = field(default_factory=list)   # populated at leaf level
    child_clusters: List[str] = field(default_factory=list)
    parent_cluster: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.child_clusters) == 0


class ClusterTree:
    """Per-document cluster tree. One instance per document (Section 1.1, step 4)."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.nodes: dict[str, ClusterNode] = {}
        self.root_id: Optional[str] = None
        # leaf chunk store: chunk_id -> {"text":..., "embedding": [...]}
        self.chunks: dict[str, dict] = {}

    def add_node(self, node: ClusterNode, is_root: bool = False):
        self.nodes[node.cluster_id] = node
        if is_root:
            self.root_id = node.cluster_id

    def add_chunk(self, chunk_id: str, text: str, embedding: np.ndarray):
        if not isinstance(embedding, list):
            embedding = np.asarray(embedding).tolist()
        self.chunks[chunk_id] = {"text": text, "embedding": embedding}

    def get(self, cluster_id: str) -> ClusterNode:
        return self.nodes[cluster_id]

    def children(self, cluster_id: str) -> List[ClusterNode]:
        return [self.nodes[c] for c in self.nodes[cluster_id].child_clusters]

    def ancestor_path(self, cluster_id: str) -> List[ClusterNode]:
        path = []
        cur = self.nodes.get(cluster_id)
        while cur is not None:
            path.append(cur)
            cur = self.nodes.get(cur.parent_cluster) if cur.parent_cluster else None
        return path

    @property
    def root(self) -> ClusterNode:
        return self.nodes[self.root_id]

    # ---- serialization ----
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "root_id": self.root_id,
            "nodes": {cid: asdict(n) for cid, n in self.nodes.items()},
            "chunks": self.chunks,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ClusterTree":
        with open(path) as f:
            data = json.load(f)
        tree = cls(data["doc_id"])
        tree.root_id = data["root_id"]
        tree.chunks = data["chunks"]
        for cid, nd in data["nodes"].items():
            tree.nodes[cid] = ClusterNode(**nd)
        return tree
