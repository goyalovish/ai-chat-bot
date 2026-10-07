import os
import json
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, collection_name: str = "nitj_placement_policy", persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_chunks(self, chunks: List[Dict]):
        if not chunks:
            print("No chunks provided to index.")
            return

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

        print(f"Embedding {len(texts)} chunks using all-MiniLM-L6-v2...")
        embeddings = self.model.encode(texts).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"Successfully indexed {len(ids)} chunks into ChromaDB.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        formatted = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                })
        return formatted

if __name__ == "__main__":
    processed_path = os.path.join("data", "processed", "chunks.json")
    if os.path.exists(processed_path):
        with open(processed_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        store = VectorStore()
        store.index_chunks(chunks)
        print("\n--- Test Dense Search: 'PPO rules' ---")
        for match in store.search("PPO rules", top_k=2):
            print(f"- [Section {match['metadata']['page']}] {match['text'][:120]}...")
    else:
        print(f"Chunks file missing: {processed_path}")