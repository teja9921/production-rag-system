import json
import csv
from pathlib import Path

METRICS_DIR = Path("evaluation/eval_outputs")
OUTPUT_CSV = Path("evaluation/eval_outputs/ablation_table.csv")
OUTPUT_JSON = Path("evaluation/eval_outputs/ablation_table.json")

FIELDS = [
    "config_id",
    "total_questions",
    "MRR",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_mean_ms",
    "latency_max_ms",
]

rows = []

for metrics_file in sorted(METRICS_DIR.glob("metrics_*.json")):
    with open(metrics_file, "r") as f:
        data = json.load(f)

    row = {
        "config_id": data["config_id"],
        "total_questions": data["total_questions"],
        "MRR": round(data["MRR"], 4),
        "Recall@1": round(data["Recall@1"], 4),
        "Recall@3": round(data["Recall@3"], 4),
        "Recall@5": round(data["Recall@5"], 4),
        "latency_p50_ms": data["Latency_ms"]["p50"],
        "latency_p95_ms": data["Latency_ms"]["p95"],
        "latency_mean_ms": round(data["Latency_ms"]["mean"], 2),
        "latency_max_ms": data["Latency_ms"]["max"],
    }

    rows.append(row)

# Sort for readability:
# Primary: Recall@5 (descending)
# Secondary: latency_p50 (ascending)
rows.sort(
    key=lambda r: (-r["Recall@5"], r["latency_p50_ms"])
)

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

# Write JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(rows, f, indent=2)

print(f"Ablation CSV written → {OUTPUT_CSV}")
print(f"Ablation JSON written → {OUTPUT_JSON}")
