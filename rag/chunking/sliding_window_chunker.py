
from typing import List, Dict, Any


class SlidingWindowChunker:

    def __init__(self, window_chars=1000, overlap_chars=200):
        self.window = window_chars
        self.overlap = overlap_chars

    def split_pages(self, pages: List[Dict[str, Any]]):
        chunks = []
        idx = 0

        for page in pages:
            text = page["content"]
            start = 0

            while start < len(text):
                end = start + self.window
                segment = text[start:end]

                chunks.append(self._make_chunk(segment, page["metadata"], idx))
                idx += 1

                start += self.window - self.overlap

        return chunks

    def _make_chunk(self, text, meta, idx):
        return {
            "chunk_id": f"{meta['doc_id']}_p{meta['page_number']}_s{idx}",
            "content": text,
            "metadata": meta,
        }