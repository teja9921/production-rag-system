"""
Deduplicate, sample, and create final evaluation file.
Input: data/qa_pairs.jsonl
Output: evaluation_gale.json + sample_for_review.json
"""

import json
import logging
import random
from pathlib import Path

from evaluation.gale.scripts.utils import deduplicate_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finalize")

# Config
DATA_DIR = Path("evaluation/gale/data")
OUT_DIR = Path("evaluation/gale")
IN_FILE = DATA_DIR / "qa_pairs.jsonl"
OUT_FILE = OUT_DIR / "evaluation_gale.json"
SAMPLE_FILE = OUT_DIR / "sample_for_review.json"

TARGET_SIZE = 250
SAMPLE_SIZE = 20
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)
    
    logger.info(f"Loading QA pairs from {IN_FILE}")
    qa_pairs = []
    with IN_FILE.open("r") as f:
        for line in f:
            qa_pairs.append(json.loads(line))
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Deduplicate
    logger.info("Deduplicating...")
    qa_pairs = deduplicate_questions(qa_pairs, threshold=0.82)
    logger.info(f"After dedupe: {len(qa_pairs)}")
    
    # Sample if needed
    if len(qa_pairs) > TARGET_SIZE:
        logger.info(f"Sampling {TARGET_SIZE} pairs")
        random.shuffle(qa_pairs)
        qa_pairs = qa_pairs[:TARGET_SIZE]
    
    # Save final
    logger.info(f"Saving {len(qa_pairs)} pairs to {OUT_FILE}")
    OUT_FILE.write_text(json.dumps(qa_pairs, indent=2, ensure_ascii=False))
    
    # Save sample for review
    sample = random.sample(qa_pairs, min(SAMPLE_SIZE, len(qa_pairs)))
    SAMPLE_FILE.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
    logger.info(f"Sample saved to {SAMPLE_FILE}")
    
    # Stats
    difficulties = {}
    for pair in qa_pairs:
        diff = pair.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    logger.info("Difficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        logger.info(f"  {diff}: {count}")

if __name__ == "__main__":
    main()