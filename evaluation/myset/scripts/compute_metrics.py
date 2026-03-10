import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

from evaluation.monitoring.experiment_registry import register_experiment

DATASET_VERSION = "myset_v1"
CONFIG_ID = "myset_hybrid_rerank_v1"
INPUT_CSV = Path("evaluation/myset/outputs/eval_outputs/myset_hybrid_rerank.csv")
OUTPUT_JSON = Path(
    "evaluation/myset/outputs/eval_outputs/metrics_myset_hybrid_rerank_v1.json"
)


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


def load_rows(path: Path) -> List[Dict]:
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
        rank = str(r["gold_rank"]).strip()
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
        "MRR": round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        "Recall@1": round(recall_1 / total, 4) if total else 0.0,
        "Recall@3": round(recall_3 / total, 4) if total else 0.0,
        "Recall@5": round(recall_5 / total, 4) if total else 0.0,
        "Latency_ms": {
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "max": round(max(latencies), 2) if latencies else 0.0,
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        },
        "config_id": CONFIG_ID,
    }


def main():
    rows = load_rows(INPUT_CSV)
    metrics = compute_metrics(rows)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    register_experiment(
        dataset_version=DATASET_VERSION,
        config_id=CONFIG_ID,
        metrics=metrics,
    )

    print(f"Metrics written to: {OUTPUT_JSON}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
