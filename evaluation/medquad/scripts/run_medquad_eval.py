import json
import time
import csv
from tqdm import tqdm
import random
from pathlib import Path
from api.agent_deps import REASONING_GRAPH
from orchestration.lc_llm import LLMRunnable

CONFIG_ID = "hybrid_rerank_v1"
IN_FILE = "evaluation/medquad/outputs/evaluation_medquad.json"
OUT_FILE = f"outputs/medquad_answers_{CONFIG_ID}.csv"

Path("outputs").mkdir(exist_ok=True)

with open(IN_FILE) as f:
    data = json.load(f)

# Sample 300 random items
sample_size = min(300, len(data))
data = random.sample(data, sample_size)

print(f"Loaded {sample_size} random samples from {IN_FILE}")

rows = []
llm = LLMRunnable()

for item in tqdm(data, desc = "starting the answer generation process"):
    start = time.perf_counter()

    try:
        result = REASONING_GRAPH.invoke({
            "query": item["question"],
            "conversation_id": "eval"
        })

        latency_ms = int((time.perf_counter() - start) * 1000)

        if result["status"] == "NO_ANSWER":
            rows.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["answer"],
                "generated_answer": "",
                "latency_ms": latency_ms,
                "no_answer": True,
            })
            continue

        generation = llm.invoke(
            item["question"],
            result["retrieved_chunks"]
        )

        rows.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "generated_answer": generation["answer"],
            "latency_ms": latency_ms,
            "no_answer": False,
        })

    except Exception as e:
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "generated_answer": "",
            "latency_ms": -1,
            "no_answer": True,
        })

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved MedQuAD answers → {OUT_FILE}")
