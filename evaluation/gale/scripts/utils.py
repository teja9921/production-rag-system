"""Shared utilities for GALE evaluation generation"""

import hashlib
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KEYWORDS = [
    "symptom", "symptoms", "cause", "causes",
    "treatment", "diagnosis", "risk", "therapy",
    "management", "complication", "signs", "diagnose"
]

def is_valid_chunk(text: str, min_chars: int = 300) -> bool:
    """Check if chunk is suitable for QA generation"""
    if len(text) < min_chars:
        return False
    return any(k in text.lower() for k in KEYWORDS)

def is_valid_question(q: str, min_words: int = 5) -> bool:
    """Validate question quality"""
    if not q or len(q.split()) < min_words:
        return False
    q = q.strip()
    if not q.endswith("?"):
        return False
    
    ql = q.lower()
    bad_phrases = ["what is this", "what does this", "as mentioned", "what is mentioned"]
    if any(bad in ql for bad in bad_phrases):
        return False
    
    wh_words = ["what", "how", "why", "when", "which", "who", "where"]
    if not any(w in ql for w in wh_words):
        return False
    
    return True

def estimate_difficulty(question: str, answer: str) -> str:
    """Heuristic difficulty estimation"""
    if len(answer.split()) > 20:
        return "hard"
    if any(w in question.lower() for w in ["list", "describe", "explain"]):
        return "medium"
    return "easy"

def make_id(prefix: str, text: str) -> str:
    """Generate deterministic ID from text"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"

def deduplicate_questions(
    qa_pairs: List[Dict[str, Any]], 
    threshold: float = 0.82
) -> List[Dict[str, Any]]:
    """Remove similar questions using TF-IDF similarity"""
    if len(qa_pairs) < 2:
        return qa_pairs
    
    questions = [q["question"] for q in qa_pairs]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    
    keep = [True] * len(qa_pairs)
    for i in range(len(qa_pairs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(qa_pairs)):
            if not keep[j]:
                continue
            sim = cosine_similarity(vectors[i:i+1], vectors[j:j+1])[0, 0]
            if sim > threshold:
                # Keep the one with higher confidence
                if qa_pairs[i].get("confidence", 0) >= qa_pairs[j].get("confidence", 0):
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    return [qa for qa, k in zip(qa_pairs, keep) if k]