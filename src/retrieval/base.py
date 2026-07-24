from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def index(self, chunks: List[Dict], embeddings: Any = None):
        """
        Build the index for the given chunks.
        chunks: List of {'unit_id': str, 'doc_id': str, 'text': str}
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant chunks for the given query.
        Returns a list of dicts, typically containing at least 'unit_id', 'doc_id', 'text', 'score'.
        """
        pass
