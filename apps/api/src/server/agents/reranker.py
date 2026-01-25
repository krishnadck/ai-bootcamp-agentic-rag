import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

# Optional imports to prevent crashes if libraries aren't installed
try:
    import cohere
except ImportError:
    cohere = None

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None

class BaseReranker(ABC):
    """Abstract interface that all rerankers must follow."""
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        pass

class CohereReranker(BaseReranker):
    def __init__(self, model: str = "rerank-v4.0-fast"):
        # Automatically load key from environment
        self.client = cohere.ClientV2()
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        # Standardize output to match our generic format
        return [
            {
                "index": result.index,
                "text": documents[result.index],
                "score": result.relevance_score,
                "provider": "cohere"
            }
            for result in response.results
        ]

class FlashRankReranker(BaseReranker):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        if not Ranker:
            raise ImportError("FlashRank library not found. Run: pip install flashrank")
        
        # Loads model into CPU memory (takes ~1 sec once)
        self.ranker = Ranker(model_name=model_name, cache_dir="/opt")

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []

        # FlashRank requires a specific input format: [{"id": 1, "text": "..."}]
        passages = [
            {"id": i, "text": doc} 
            for i, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Standardize output
        return [
            {
                "index": result["id"],
                "text": result["text"],
                "score": result["score"],
                "provider": "flashrank"
            }
            for result in results[:top_n]
        ]

def get_reranker(provider: str = "cohere") -> BaseReranker:
    """Factory function to get the desired reranker."""
    if provider.lower() == "cohere":
        return CohereReranker()
    elif provider.lower() == "flashrank":
        return FlashRankReranker()
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")