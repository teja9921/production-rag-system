
import re
from typing import List, Dict, Any


class SentenceChunker:

    def __init__(self, target_chars=900):
        self.target_chars = target_chars

    def split_pages(self, pages: List[Dict[str, Any]]):
        chunks = []
        idx = 0

        for page in pages:
            sentences = re.split(r'(?<=[.!?]) +', page["content"])

            buffer = ""

            for sent in sentences:
                if len(buffer) + len(sent) <= self.target_chars:
                    buffer += " " + sent
                else:
                    chunks.append(self._make_chunk(buffer.strip(), page["metadata"], idx))
                    idx += 1
                    buffer = sent

            if buffer:
                chunks.append(self._make_chunk(buffer.strip(), page["metadata"], idx))
                idx += 1

        return chunks

    def _make_chunk(self, text, meta, idx):
        return {
            "chunk_id": f"{meta['doc_id']}_p{meta['page_number']}_s{idx}",
            "content": text,
            "metadata": meta,
        }