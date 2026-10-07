from typing import List, Dict

def chunk_documents(
    documents: List[Dict], 
    chunk_size: int = 250, 
    chunk_overlap: int = 40
) -> List[Dict]:
    chunks = []
    chunk_counter = 0

    for doc in documents:
        words = doc["text"].split()
        if not words:
            continue

        if len(words) <= chunk_size:
            chunk_counter += 1
            chunks.append({
                "id": f"{doc['source']}_p{doc['page']}_c{chunk_counter}",
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"]
            })
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk_counter += 1
            chunks.append({
                "id": f"{doc['source']}_p{doc['page']}_c{chunk_counter}",
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"]
            })

            if end >= len(words):
                break
            start += chunk_size - chunk_overlap

    return chunks