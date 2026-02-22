"""
Generate questions from chunks using T5-QG.
Input: data/chunks.jsonl
Output: data/questions.jsonl
"""

import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from typing import List

from evaluation.gale.scripts.utils import is_valid_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_questions")

# Config
DATA_DIR = Path("evaluation/gale/data")
IN_FILE = DATA_DIR / "chunks.jsonl"
OUT_FILE = DATA_DIR / "questions.jsonl"

QG_MODEL = "iarfmoose/t5-base-question-generator"
GEN_RETURN_SEQS = 4
MAX_Q_PER_CHUNK = 3

def load_chunks(path: Path) -> List[dict]:
    """Load chunks from JSONL"""
    chunks = []
    with path.open("r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def generate_questions(model, tokenizer, context: str, max_q: int = 4) -> List[str]:
    """Generate questions using iarfmoose model"""
    prompt = f"generate questions: {context}"  # Note: plural!
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=max_q,              
            num_return_sequences=max_q,   # Return all beams
            do_sample=True,               # Enable sampling 
            temperature=0.8,              # Control randomness
            no_repeat_ngram_size=3,       # Reduce repetition
            early_stopping=True           # Stop when done
        )
    
    questions = [
        tokenizer.decode(o, skip_special_tokens=True).strip()
        for o in outputs
    ]
    return questions

def main():
    logger.info(f"Loading chunks from {IN_FILE}")
    chunks = load_chunks(IN_FILE)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Load model
    logger.info(f"Loading QG model: {QG_MODEL}")
    tokenizer = T5Tokenizer.from_pretrained(QG_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Using CUDA")
    
    # Generate questions
    results = []
    for chunk in tqdm(chunks, desc="Generating questions"):
        try:
            raw_questions = generate_questions(
                model, tokenizer, 
                chunk["content"], 
                max_q=GEN_RETURN_SEQS
            )
        except Exception as e:
            logger.warning(f"Failed on chunk {chunk.get('chunk_id')}: {e}")
            continue
        
        # Filter and dedupe
        raw_questions = list(dict.fromkeys(raw_questions))  # preserve order
        valid_qs = [q for q in raw_questions if is_valid_question(q)]
        valid_qs = valid_qs[:MAX_Q_PER_CHUNK * 2]  # Keep extras for QA filtering
        
        for q in valid_qs:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "question": q,
                "context": chunk["content"],
                "doc_id": chunk["metadata"]["doc_id"],
                "page_number": chunk["metadata"]["page_number"]
            })
    
    logger.info(f"Generated {len(results)} questions")
    
    # Save
    with OUT_FILE.open("w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()