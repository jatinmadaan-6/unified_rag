import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # BGE-M3 is highly recommended for RAG.
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

# Global instance for pipeline caching if needed
_embedder_instance = None

def get_embedder(model_name: str = "BAAI/bge-m3") -> LocalEmbedder:
    global _embedder_instance
    if _embedder_instance is None or _embedder_instance.model.tokenizer.name_or_path != model_name:
        _embedder_instance = LocalEmbedder(model_name)
    return _embedder_instance
