"""
Extract and filter chunks from GALE PDF.
Output: data/chunks.jsonl
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm

from ingestion.loader import load_pdf
from ingestion.semantic_splitter import SemanticChunker
from api.config import settings
from evaluation.gale.scripts.utils import is_valid_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_chunks")

# Config
PDFS = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]
OUT_DIR = Path("evaluation/gale/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chunks.jsonl"

def main():
    logger.info("Loading PDFs...")
    pages = []
    for pdf_path in PDFS:
        pages.extend(load_pdf(pdf_path))
    
    logger.info(f"Loaded {len(pages)} pages")
    
    # Chunk
    logger.info("Chunking with semantic splitter...")
    chunker = SemanticChunker(
        max_chars=settings.MAX_CHARS,
        min_chars=settings.MIN_CHARS
    )
    chunks = chunker.split_pages(pages)
    
    logger.info(f"Total chunks: {len(chunks)}")
    
    # Filter
    valid_chunks = [c for c in chunks if is_valid_chunk(c["content"])]
    logger.info(f"Valid chunks after filtering: {len(valid_chunks)}")
    
    # Save as JSONL
    with OUT_FILE.open("w") as f:
        for chunk in valid_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    logger.info(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()