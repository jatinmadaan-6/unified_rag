from typing import List, Dict, Any
import numpy as np
from src.retrieval.base import BaseRetriever

class VanillaRAG(BaseRetriever):
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.embeddings = None

    def index(self, chunks: List[Dict], embeddings: Any = None):
        self.chunks = chunks
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            texts = [c["text"] for c in chunks]
            self.embeddings = self.embedder.encode(texts)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vec = self.embedder.encode([query])[0]
        
        # Cosine similarity
        # Assuming embeddings are normalized
        sims = self.embeddings @ query_vec
        
        top_idx = np.argsort(-sims)[:top_k]
        
        results = []
        for i in top_idx:
            c = self.chunks[i].copy()
            c["score"] = float(sims[i])
            results.append(c)
            
        return results
