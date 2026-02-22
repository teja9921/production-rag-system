"""
Filter evaluation_gale.json to keep only selected QA pairs.
Output: evaluation_gale_final.json (113 curated pairs)
"""

import json
from pathlib import Path

# Files
EVAL_FILE = Path("evaluation/gale/evaluation_gale.json")
CHUNKS_FILE = Path("evaluation/gale/data/chunks.jsonl")
OUTPUT_FILE = Path("evaluation/gale/evaluation_gale_final.json")

# Selected QA pair numbers (from manual review)
SELECTED_NUMS = [
    4, 5, 6, 11, 13, 17, 18, 19, 21, 24, 25,
    26, 27, 31, 33, 35, 37, 38, 42, 43, 44, 46,
    52, 59, 61, 63, 64, 65, 68, 73,
    76, 78, 79, 82, 87, 88, 90, 92, 94,
    102, 106, 109, 110, 116, 117, 119, 122, 125,
    127, 133, 134, 136, 137, 138, 140, 141, 142, 143, 146, 148,
    152, 153, 156, 157, 158, 159, 163, 164, 166, 168, 170, 171,
    177, 181, 183, 187, 188, 189, 192, 194, 195, 198, 199,
    201, 202, 204, 206, 207, 208, 211, 214, 217, 218, 220, 221, 222, 224, 225,
    226, 227, 228, 229, 230, 231, 234, 235, 240, 244, 246, 247, 250
]

def load_chunks():
    """Load chunks to extract answer_span text"""
    chunks = {}
    with CHUNKS_FILE.open("r") as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk["content"]
    return chunks

def main():
    # Load full evaluation data
    with EVAL_FILE.open("r") as f:
        all_qa_pairs = json.load(f)
    
    print(f"Total QA pairs: {len(all_qa_pairs)}")
    
    # Load chunks for answer_span extraction
    print("Loading chunks...")
    chunks = load_chunks()
    
    # Filter selected pairs
    selected_pairs = []
    for i, qa in enumerate(all_qa_pairs, 1):
        if i not in SELECTED_NUMS:
            continue
        
        # Extract answer_span from chunk
        chunk_id = qa["chunk_id"]
        if chunk_id in chunks:
            chunk_content = chunks[chunk_id]
            start = qa["answer_span_start"]
            end = qa["answer_span_end"]
            answer_span = chunk_content[start:end]
        else:
            # Fallback if chunk not found
            answer_span = qa["answer"]
        
        # Create cleaned QA pair
        cleaned_qa = {
            "id": qa["id"],
            "question": qa["question"],
            "answer": qa["answer"],
            "answer_span": answer_span,
            "doc_id": qa["doc_id"],
            "page_number": qa["page_number"],
            "chunk_id": qa["chunk_id"],
            "difficulty": qa["difficulty"]
        }
        selected_pairs.append(cleaned_qa)
    
    print(f"Selected QA pairs: {len(selected_pairs)}")
    
    # Difficulty distribution
    from collections import Counter
    difficulties = Counter(qa["difficulty"] for qa in selected_pairs)
    print(f"\nDifficulty distribution:")
    for diff, count in difficulties.items():
        print(f"  {diff}: {count}")
    
    # Save final dataset
    with OUTPUT_FILE.open("w") as f:
        json.dump(selected_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved final dataset to {OUTPUT_FILE}")
    
    # Show first 3 examples
    print(f"\nFirst 3 examples:")
    for qa in selected_pairs[:3]:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Answer span: {qa['answer_span']}")
        print(f"Page: {qa['page_number']} | Chunk: {qa['chunk_id']}")

if __name__ == "__main__":
    main()
