from pathlib import Path

from ingestion.loader import load_pdf
from ingestion.semantic_splitter import SemanticChunker


def run_semantic_chunker_smoke(
    pdf_path: str = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf",
) -> None:
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        return

    pages = load_pdf(pdf_path)
    chunker = SemanticChunker(max_chars=1000, min_chars=100)
    chunks = chunker.split_pages(pages)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    for i, c in enumerate(chunks[:3]):
        print(f"\nChunk {i}")
        print(len(c["content"]))
        print(c["content"][:300])


if __name__ == "__main__":
    run_semantic_chunker_smoke()
