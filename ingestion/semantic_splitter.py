import re
from typing import List, Dict, Any

from core.logger import get_logger
from core.exceptions import CustomException


logger = get_logger("ingestion.semantic_chunker")


class SemanticChunker:
    """
    Semantic-aware document chunker.

    Strategy:
    - Split text into paragraph blocks
    - Merge adjacent blocks up to max_chars
    - Enforce minimum semantic size
    - Avoid sentence and paragraph fragmentation
    """

    def __init__(self, max_chars: int = 1800, min_chars: int = 400):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            logger.info(
                "event=SEMANTIC_CHUNKING_START | pages=%d | max_chars=%d | min_chars=%d",
                len(pages),
                self.max_chars,
                self.min_chars,
            )

            chunks: List[Dict[str, Any]] = []
            chunk_index = 0

            for page in pages:
                text = page["content"]
                metadata = page["metadata"]

                paragraphs = self._split_into_paragraphs(text)
                buffer = ""

                for para in paragraphs:
                    if len(buffer) + len(para) <= self.max_chars:
                        buffer = f"{buffer}\n\n{para}" if buffer else para
                    else:
                        if len(buffer) >= self.min_chars:
                            chunks.append(
                                self._make_chunk(buffer, metadata, chunk_index)
                            )
                            chunk_index += 1
                            buffer = para
                        else:
                            buffer = f"{buffer}\n\n{para}"

                if buffer:
                    chunks.append(
                        self._make_chunk(buffer, metadata, chunk_index)
                    )
                    chunk_index += 1

            logger.info(
                "event=SEMANTIC_CHUNKING_COMPLETE | total_chunks=%d",
                len(chunks),
            )

            return chunks

        except Exception as e:
            logger.exception("event=SEMANTIC_CHUNKING_FAILED")
            raise CustomException(
                "Semantic chunking failed",
                error=e,
                context={
                    "max_chars": self.max_chars,
                    "min_chars": self.min_chars,
                },
            ) from e

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Paragraph-level segmentation.

        Works across:
        - encyclopedias
        - research papers
        - clinical notes
        - books
        """
        paras = re.split("\\n{2,}", text)
        return [p.strip() for p in paras if len(p.strip()) > 50]

    def _make_chunk(
        self,
        text: str,
        meta: Dict[str, Any],
        idx: int,
    ) -> Dict[str, Any]:
        doc_id = meta["doc_id"]
        page_number = meta["page_number"]

        return {
            "chunk_id": f"{doc_id}_p{page_number}_s{idx}",
            "content": text.strip(),
            "metadata": {
                **meta,
                "semantic_index": idx,
            },
        }
