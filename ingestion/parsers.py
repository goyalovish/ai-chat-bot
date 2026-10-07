import os
from typing import List, Dict
import pdfplumber

def extract_text_from_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    extracted_pages = []

    # Handle .txt or .md files
    if file_path.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split sections by heading markers
            sections = content.split("### ")
            for idx, sec in enumerate(sections):
                if sec.strip():
                    extracted_pages.append({
                        "source": filename,
                        "page": idx + 1,
                        "text": ("### " + sec).strip() if idx > 0 else sec.strip()
                    })
        return extracted_pages

    # Handle digital .pdf files
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    cleaned_text = " ".join(text.split())
                    extracted_pages.append({
                        "source": filename,
                        "page": page_idx + 1,
                        "text": cleaned_text
                    })
    except Exception as e:
        print(f"Error reading {filename}: {e}")

    return extracted_pages