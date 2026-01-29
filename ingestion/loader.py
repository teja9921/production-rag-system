import hashlib
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads and extracts text from a PDF file.

    Guarantees:
    - doc_id: A stable SHA-256 hash of the file content.
    - page_number: 1-indexed integer.
    - content: Whitespace-normalized string (no double spaces/newlines).
    - schema: Returns List[Dict] where each dict contains 'content' and 'metadata'.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No PDF found at {file_path}")

    # Generate stable doc_id
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    doc_id = sha256_hash.hexdigest()

    reader = PdfReader(path)
    documents = []
    print(f"Total pages detected by PdfReader: {len(reader.pages)}")
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        #print(f"Page {i+1}: {raw_text[:100]}...")  # Print first 100 chars
        # Normalize: Collapse all whitespace into single spaces
        normalized = " ".join(raw_text.split()).strip()

        if normalized:
            documents.append({
                "content": normalized,
                "metadata": {
                    "doc_id": doc_id,
                    "page_number": i + 1,
                    "source_file": path.name
                }
            })
    return documents
