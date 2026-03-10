import csv
import json
from pathlib import Path

INPUT_CSV = Path("evaluation/myset/outputs/eval_outputs/myset_hybrid_rerank.csv")
OUTPUT_JSON = Path("evaluation/myset/myset_failures.json")


def run():
    failures = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = str(row.get("gold_rank", "")).strip()
            is_failure = (not rank) or (rank.isdigit() and int(rank) > 5)
            if is_failure:
                question = row.get("question", "").strip()
                if question:
                    failures.append({"question": question})

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Saved {len(failures)} failures -> {OUTPUT_JSON}")


if __name__ == "__main__":
    run()
