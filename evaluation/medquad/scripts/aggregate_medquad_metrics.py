# evaluation/aggregate_medquad_metrics.py

import csv
import json
import numpy as np
from pathlib import Path
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

CONFIG_ID = "hybrid_rerank_v1"
IN_FILE = Path(f"evaluation/medquad/outputs/medquad_answers_{CONFIG_ID}.csv")
OUT_FILE = f"metrics_medquad_{CONFIG_ID}.json"

model = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rows = []
with open(IN_FILE, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rouge_l, similarity_scores = [], []
latencies = []
no_answer_count = 0

for r in rows:
    gt = r["ground_truth"].strip().lower()
    pred = r["generated_answer"].strip().lower()

    if r["no_answer"] == "True":
        no_answer_count += 1
        continue

    rouge_l.append(scorer.score(gt, pred)["rougeL"].fmeasure)

    emb = model.encode([gt, pred], convert_to_tensor=True)
    similarity_scores.append(util.cos_sim(emb[0], emb[1]).item())

    latencies.append(int(r["latency_ms"]))

metrics = {
    "total_questions": len(rows),
    "rougeL": float(np.mean(rouge_l)) if rouge_l else 0.0,
    "semantic_similarity": float(np.mean(similarity_scores)) if similarity_scores else 0.0,
    "no_answer_rate": no_answer_count / len(rows),
    "latency_ms": {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "mean": float(np.mean(latencies)),
        "max": float(np.max(latencies)),
    },
    "config_id": CONFIG_ID,
}

with open(OUT_FILE, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
