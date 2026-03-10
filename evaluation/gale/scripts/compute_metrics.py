import csv
import json
import statistics
from pathlib import Path
from typing import List, Dict

INPUT_CSV = Path("evaluation/eval_outputs/bm25.csv")
OUTPUT_JSON = Path("evaluation/eval_outputs/metrics_bm25_v1.json")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def load_rows(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: List[Dict]) -> Dict:
    total = len(rows)

    reciprocal_ranks = []
    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    latencies = []

    for r in rows:
        rank = r["gold_rank"]
        latency = float(r["latency_ms"])

        latencies.append(latency)

        if rank:
            rank = int(rank)
            reciprocal_ranks.append(1.0 / rank)

            if rank <= 1:
                recall_1 += 1
            if rank <= 3:
                recall_3 += 1
            if rank <= 5:
                recall_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    return {
        "total_questions": total,
        "MRR": round(sum(reciprocal_ranks) / total, 4),
        "Recall@1": round(recall_1 / total, 4),
        "Recall@3": round(recall_3 / total, 4),
        "Recall@5": round(recall_5 / total, 4),
        "Latency_ms": {
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "max": round(max(latencies), 2),
            "mean": round(statistics.mean(latencies), 2),
        },
        "config_id": #bm25_v1, bm25_rerank_v1, dense_v1, dense_rerank_v1, hybrid_v1, hybrid_rerank_v1 
    }


def main():
    rows = load_rows(INPUT_CSV)
    metrics = compute_metrics(rows)

    Path(OUTPUT_JSON).write_text(
        json.dumps(metrics, indent=2)
    )

    print("Metrics written to:", OUTPUT_JSON)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
