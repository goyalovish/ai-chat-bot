import os
import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vector_store import VectorStore

class HybridRetriever:
    def __init__(self, chunks_path: str = os.path.join("data", "processed", "chunks.json")):
        self.vector_store = VectorStore()
        self.chunks = []
        self.bm25 = None

        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.corpus = [c["text"] for c in self.chunks]
            self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search_bm25(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "id": self.chunks[i]["id"],
                "text": self.chunks[i]["text"],
                "metadata": {"source": self.chunks[i]["source"], "page": self.chunks[i]["page"]},
                "score": float(scores[i])
            }
            for i in top_indices if scores[i] > 0
        ]

    def hybrid_search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict]:
        """
        Merges Dense Vector and BM25 results using Reciprocal Rank Fusion (RRF).
        """
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        rrf_scores = {}
        doc_lookup = {}

        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        return [
            {
                "id": doc_id,
                "text": doc_lookup[doc_id]["text"],
                "metadata": doc_lookup[doc_id]["metadata"],
                "fusion_score": score
            }
            for doc_id, score in sorted_docs
        ]

if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "What is the minimum stipend required for off campus internship?"
    print(f"\n--- Testing Hybrid Search for: '{query}' ---")
    results = retriever.hybrid_search(query, top_k=2)
    for idx, r in enumerate(results, 1):
        print(f"\n[{idx}] Section: {r['metadata']['page']} | RRF Score: {r['fusion_score']:.4f}")
        print(f"Snippet: {r['text'][:150]}...")