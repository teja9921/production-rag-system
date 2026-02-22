import json
import time
from pathlib import Path
from api.agent_deps import REASONING_GRAPH # uses current hybrid + rerank

EVAL_FILE = Path("evaluation/gale/evaluation_gale_final.json")
OUT_FILE = Path("evaluation/results/gale_hybrid_rerank.jsonl")
OUT_FILE.parent.mkdir(exist_ok=True)

def evaluate():
    data = json.loads(EVAL_FILE.read_text())

    with OUT_FILE.open("w") as f:
        for item in data:
            start = time.perf_counter()

            result = REASONING_GRAPH.invoke({
                "query": item["question"],
                "conversation_id": "__eval__"
            })

            latency = int((time.perf_counter() - start) * 1000)

            if result["status"] == "NO_ANSWER":
                row = {
                    "question_id": item["id"],
                    "difficulty": item["difficulty"],
                    "retrieved_chunk_ids": [],
                    "hit": False,
                    "no_answer": True,
                    "latency_ms": latency,
                }
            else:
                retrieved_ids = [
                    c["chunk_id"] for c in result["retrieved_chunks"]
                ]

                row = {
                    "question_id": item["id"],
                    "difficulty": item["difficulty"],
                    "retrieved_chunk_ids": retrieved_ids,
                    "hit": item["chunk_id"] in retrieved_ids,
                    "no_answer": False,
                    "latency_ms": latency,
                }

            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    evaluate()
