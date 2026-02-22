"""
Extract answers using extractive QA model.
Input: data/questions.jsonl
Output: data/qa_pairs.jsonl
"""

import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict

from evaluation.gale.scripts.utils import make_id, estimate_difficulty, deduplicate_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_answers")

# Config
DATA_DIR = Path("evaluation/gale/data")
IN_FILE = DATA_DIR / "questions.jsonl"
OUT_FILE = DATA_DIR / "qa_pairs.jsonl"

QA_MODEL = "deepset/roberta-base-squad2"
QA_CONF_THRESHOLD = 0.25
MAX_Q_PER_CHUNK = 2

def main():
    logger.info(f"Loading questions from {IN_FILE}")
    questions = []
    with IN_FILE.open("r") as f:
        for line in f:
            questions.append(json.loads(line))
    
    logger.info(f"Loaded {len(questions)} questions")
    
    # Group by chunk_id
    questions_by_chunk = defaultdict(list)
    for item in questions:
        questions_by_chunk[item["chunk_id"]].append(item)
    
    logger.info(f"Processing {len(questions_by_chunk)} chunks")
    
    # OPTIMIZATION: Dedupe questions BEFORE QA extraction
    logger.info("Deduplicating questions per chunk...")
    deduped_questions_by_chunk = {}
    total_before = 0
    total_after = 0
    
    for chunk_id, chunk_questions in questions_by_chunk.items():
        total_before += len(chunk_questions)
        
        # Convert to format expected by deduplicate_questions
        mock_qa = [
            {
                **q,
                "answer": "",  # Dummy answer
                "confidence": 1.0  # Dummy confidence
            }
            for q in chunk_questions
        ]
        
        # Dedupe within this chunk
        deduped_mock = deduplicate_questions(mock_qa, threshold=0.82)
        
        # Convert back to question format
        deduped_questions_by_chunk[chunk_id] = [
            {k: v for k, v in item.items() if k not in ["answer", "confidence"]}
            for item in deduped_mock
        ]
        
        total_after += len(deduped_questions_by_chunk[chunk_id])
    
    logger.info(f"Questions before chunk-level dedupe: {total_before}")
    logger.info(f"Questions after chunk-level dedupe: {total_after}")
    logger.info(f"Saved {total_before - total_after} redundant QA extractions!")
    
    # Load QA model
    logger.info(f"Loading QA model: {QA_MODEL}")
    qa_pipe = pipeline(
        "question-answering",
        model=QA_MODEL,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # NOW extract answers only for unique questions
    all_qa_pairs = []
    
    for chunk_id, chunk_questions in tqdm(deduped_questions_by_chunk.items(), desc="Extracting answers"):
        pairs_for_chunk = []
        
        for item in chunk_questions:
            try:
                result = qa_pipe(
                    question=item["question"],
                    context=item["context"]
                )
            except Exception as e:
                logger.debug(f"QA failed: {e}")
                continue
            
            answer = result.get("answer", "").strip()
            score = float(result.get("score", 0.0))
            
            # Validation
            if not answer:
                continue
            if score < QA_CONF_THRESHOLD:
                continue
            if answer.lower() not in item["context"].lower():
                continue
            if answer.lower() in item["question"].lower():
                continue
            if len(answer.split()) < 2 or len(answer.split()) > 100:
                continue
            
            qa_pair = {
                "id": make_id("gale", item["question"] + answer),
                "question": item["question"],
                "answer": answer,
                "answer_span_start": result.get("start", -1),
                "answer_span_end": result.get("end", -1),
                "doc_id": item["doc_id"],
                "page_number": item["page_number"],
                "chunk_id": item["chunk_id"],
                "confidence": score,
                "difficulty": estimate_difficulty(item["question"], answer)
            }
            pairs_for_chunk.append(qa_pair)
        
        if not pairs_for_chunk:
            continue
        
        # Sort by confidence and keep top-K per chunk
        pairs_for_chunk = sorted(pairs_for_chunk, key=lambda x: x["confidence"], reverse=True)
        pairs_for_chunk = pairs_for_chunk[:MAX_Q_PER_CHUNK]
        
        all_qa_pairs.extend(pairs_for_chunk)
    
    logger.info(f"Generated {len(all_qa_pairs)} QA pairs")
    
    # Save
    with OUT_FILE.open("w") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair) + "\n")
    
    logger.info(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
