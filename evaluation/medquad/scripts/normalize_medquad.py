import csv
import json
import uuid
from pathlib import Path

CSV_PATH = "data/medquad.csv"
OUT_PATH = "evaluation/medquad/outputs/evaluation_medquad.json"

records = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row.get("question", "").strip()
        a = row.get("answer", "").strip()

        if len(q) < 5 or len(a) < 5:
            continue

        records.append({
            "id": f"medquad_{uuid.uuid4().hex[:8]}",
            "question": q,
            "answer": a,
            "difficulty": "medium"
        })

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Saved {len(records)} MedQuAD samples")
