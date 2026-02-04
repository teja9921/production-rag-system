import hashlib
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader

from core.logger import get_logger
from core.exceptions import CustomException


logger = get_logger("ingestion.loader")


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads and extracts text from a PDF file.

    Guarantees:
    - doc_id: Stable SHA-256 hash of file content
    - page_number: 1-indexed
    - content: Whitespace-normalized text
    - schema: List[{content, metadata}]
    """
    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"No PDF found at {file_path}")

        logger.info(
            "event=PDF_LOAD_START | file=%s",
            path.name
        )

        # Generate stable doc_id
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        doc_id = sha256_hash.hexdigest()

        reader = PdfReader(path)
        total_pages = len(reader.pages)

        logger.info(
            "event=PDF_PAGES_DETECTED | file=%s | pages=%d",
            path.name,
            total_pages
        )

        documents: List[Dict[str, Any]] = []

        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            normalized = " ".join(raw_text.split()).strip()

            if normalized:
                documents.append(
                    {
                        "content": normalized,
                        "metadata": {
                            "doc_id": doc_id,
                            "page_number": i + 1,
                            "source_file": path.name,
                        },
                    }
                )

        logger.info(
            "event=PDF_LOAD_COMPLETE | file=%s | pages_with_text=%d",
            path.name,
            len(documents),
        )

        return documents

    except Exception as e:
        logger.exception(
            "event=PDF_LOAD_FAILED | file=%s",
            file_path
        )
        raise CustomException(
            "Failed to load and parse PDF",
            error=e,
            context={"file_path": file_path},
        ) from e
