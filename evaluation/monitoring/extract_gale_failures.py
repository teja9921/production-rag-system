import csv
import json
from pathlib import Path


INPUT_CANDIDATES = [
    Path("evaluation/gale/dense_rerank_v1.csv"),
    Path("evaluation/gale/outputs/eval_outputs/dense_rerank.csv"),
]
OUTPUT_JSON = Path("evaluation/gale/gale_failures.json")
GALE_DATASET = Path("evaluation/gale/evaluation_gale_final.json")


def _resolve_input() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No GALE evaluation CSV found. Checked: "
        + ", ".join(str(p) for p in INPUT_CANDIDATES)
    )


def run():
    failures = []
    input_csv = _resolve_input()
    id_to_question = {}
    if GALE_DATASET.exists():
        # GALE file contains extended characters; latin-1 keeps it loadable.
        dataset = json.loads(GALE_DATASET.read_text(encoding="latin-1"))
        id_to_question = {row["id"]: row.get("question", "") for row in dataset}

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hit_value = row.get("hit")
            if hit_value is not None:
                is_failure = str(hit_value).strip().lower() == "false"
            else:
                rank = row.get("gold_rank", "").strip()
                if not rank:
                    is_failure = True
                else:
                    try:
                        is_failure = int(rank) > 5
                    except ValueError:
                        is_failure = True

            if is_failure:
                question = row.get("question")
                if not question:
                    qid = row.get("question_id", "")
                    question = id_to_question.get(qid, "")
                if not question:
                    continue
                failures.append(
                    {
                        "question": question,
                    }
                )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Saved {len(failures)} failures -> {OUTPUT_JSON}")


if __name__ == "__main__":
    run()
