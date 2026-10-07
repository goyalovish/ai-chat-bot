import os
import glob
import json
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parsers import extract_text_from_file
from chunker import chunk_documents

def run_pipeline():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Match pdf, txt, and md
    supported_files = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        supported_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not supported_files:
        print(f"⚠️ No documents found in '{raw_dir}'.")
        return []

    print(f"🔍 Found {len(supported_files)} file(s) in {raw_dir}...\n")
    all_chunks = []

    for file_path in supported_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        pages = extract_text_from_file(file_path)
        if not pages:
            print(f"  -> Skipped (no text extracted).")
            continue
        chunks = chunk_documents(pages, chunk_size=150, chunk_overlap=30)
        all_chunks.extend(chunks)
        print(f"  -> Extracted {len(pages)} sections/pages | Generated {len(chunks)} chunks\n")

    output_path = os.path.join(processed_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Ingestion complete! Total chunks created: {len(all_chunks)}")
    print(f"📁 Chunks saved to: {output_path}")

    if all_chunks:
        print("\n--- Preview of First Chunk ---")
        print(f"ID:     {all_chunks[0]['id']}")
        print(f"Source: {all_chunks[0]['source']} (Section {all_chunks[0]['page']})")
        print(f"Text:   {all_chunks[0]['text'][:180]}...")

    return all_chunks

if __name__ == "__main__":
    run_pipeline()