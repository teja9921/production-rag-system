import re
from typing import List, Dict, Any

class SemanticChunker:
    """
    Semantic-aware documetn chunker
    Strategy:
    - Split text into logical paragraph blocks
    - Merge adjacent blocks until size threshold
    - Preserve semantic coherence
    - Avoid sentence and paragraph fragmentation
    """

    def __init__(self, max_chars: int = 1800, min_chars: int = 400):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def _split_pages(self, pages: List[Dict[str, Any]])-> List[Dict[str, Any]]:
        chunks = []
        chunk_index =0

        for page in pages:
            text = page["content"]
            paragraphs = self._split_into_paragraphs(text)

            buffer = ""

            for para in paragraphs:
                if len(buffer) + len(para) <= self.max_chars:
                    buffer += ("\n\n" + para) if buffer else para

                else:
                    if len(buffer) >= self.min_chars:
                        chunks.append(self._make_chunks(buffer, page['metadata'], chunks_index))
                        chunk_index += 1
                        buffer = para
                    else:
                        buffer+= "\n\n" + para

            if buffer:
                chunks.append(self._make_chunks(buffer, page['metadata'], chunk_index))
                chunk_index += 1
        
        return chunks
                        
                    

    def _split_into_paragraphs(self, text:str)->List[str]:
        """
        Paragraph level segmentation.

        Works across:
        -encyclopedias
        -research papers
        -clinical notes
        -books
        """
        paras: List[str] = re.split(r"\n{2,}", text)
        return [p.strip() for p in paras if len(p.strip()) > 50]
    
    def _make_chunks(self, text: str, meta: Dict[str,Any], idx: int):

        doc_id = meta['doc_id']
        page_number = meta['page_number']
        return {
            "chunk_id": f"{doc_id}_p{page_number}_s{idx}",
            "content": text.strip(),
            "metadata": {
                **meta,
                "semantic_index": idx
            }
        } # type: ignore