import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = Path("evaluation/chat/outputs/chat_eval_results.json")
OUTPUT_FILE = Path("evaluation/chat/outputs/metrics_chat_hybrid_rerank_v1.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedder = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Load results
# -----------------------------
with open(INPUT_FILE, encoding="utf-8") as f:
    rows = json.load(f)

total = len(rows)

# -----------------------------
# No-answer rate
# -----------------------------
no_answer_rate = sum(r["no_answer"] for r in rows) / total

# -----------------------------
# Latency metrics
# -----------------------------
latencies = [r["latency_ms"] for r in rows]

latency_stats = {
    "p50": float(np.percentile(latencies, 50)),
    "p95": float(np.percentile(latencies, 95)),
    "mean": float(np.mean(latencies)),
    "max": float(np.max(latencies)),
}

# -----------------------------
# Semantic similarity
# (query ↔ generated answer)
# -----------------------------
similarities = []

for r in rows:
    if r["no_answer"]:
        continue

    q_emb = embedder.encode(r["query"], convert_to_tensor=True)
    a_emb = embedder.encode(r["answer"], convert_to_tensor=True)

    sim = util.cos_sim(q_emb, a_emb).item()
    similarities.append(sim)

semantic_similarity = float(np.mean(similarities)) if similarities else 0.0

# -----------------------------
# Final metrics
# -----------------------------
metrics = {
    "total_samples": total,
    "no_answer_rate": no_answer_rate,
    "semantic_similarity_mean": semantic_similarity,
    "latency_ms": latency_stats,
    "config_id": rows[0]["config_id"] if rows else None,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"[DONE] Metrics written → {OUTPUT_FILE}")
