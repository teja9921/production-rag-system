
import re
from typing import List, Dict, Any


class ParagraphChunker:

    def __init__(self, max_chars=1200, min_chars=200):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split_pages(self, pages: List[Dict[str, Any]]):
        chunks = []
        idx = 0

        for page in pages:
            paras = re.split(r"\n{2,}", page["content"])
            paras = [p.strip() for p in paras if len(p.strip()) > 30]

            buffer = ""

            for para in paras:
                if len(buffer) + len(para) <= self.max_chars:
                    buffer += ("\n\n" + para) if buffer else para
                else:
                    if len(buffer) >= self.min_chars:
                        chunks.append(self._make_chunk(buffer, page["metadata"], idx))
                        idx += 1
                        buffer = para
                    else:
                        buffer += "\n\n" + para

            if buffer:
                chunks.append(self._make_chunk(buffer, page["metadata"], idx))
                idx += 1

        return chunks

    def _make_chunk(self, text, meta, idx):
        return {
            "chunk_id": f"{meta['doc_id']}_p{meta['page_number']}_s{idx}",
            "content": text,
            "metadata": meta,
        }