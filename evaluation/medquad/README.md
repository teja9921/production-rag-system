Source: MedQuAD (sampled subset)
Processing: field mapping → normalize/clean → dedupe (TF‑IDF + sentence‑transformers/all‑MiniLM‑L6‑v2) → filter heuristics → optional extractive validator (deepset/roberta‑base‑squad2) → stratified human review
Files: evaluation_medquad.json, sample_medquad_review.json, medquad_metadata.json
Use: NO_ANSWER, semantic similarity, ROUGE‑L (secondary), false‑positive rate, latency profiling — do not add MedQuAD docs to your index
Key metrics to report: NO_ANSWER rate, semantic_similarity (embedding cosine), ROUGE‑L, latency (p50/p95)