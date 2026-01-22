from typing import List, Dict, Any

class SimpleChunker:
    """
    Splits documents into fixed-size chunks with overlap.
    
    Guarantees:
    - Stable IDs: Generated as {doc_id}_p{page}_c{index}.
    - Determinism: Same input text and config always produce identical chunks.
    - Overlap: Fixed character-based overlap applied between sequential chunks.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 1 token ≈ 4 characters
        self.char_size = chunk_size * 4
        self.char_overlap = chunk_overlap * 4

    def split_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []

        for page in pages:
            text = page["content"]
            metadata = page["metadata"]
            start = 0
            chunk_index = 0
            
            while start < len(text):
                end = start + self.char_size
                
                # Boundary refinement to avoid splitting words mid-character
                if end < len(text):
                    last_space = text[start:end].rfind(" ")
                    if last_space != -1:
                        end = start + last_space

                chunk_text = text[start:end].strip()
                
                # ID Contract: {doc_id}_p{page_number}_c{chunk_index}
                chunk_id = f"{metadata['doc_id']}_p{metadata['page_number']}_c{chunk_index}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index
                    }
                })

                # Progress window
                start = max(0, end - self.char_overlap)
                chunk_index += 1
                
                # Prevent infinite loops on tiny remainders
                if (len(text) - start) < self.char_overlap:
                    break


        return all_chunks
