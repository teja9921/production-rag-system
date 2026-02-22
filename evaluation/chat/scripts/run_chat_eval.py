# run_chat_retrieval.py
import json
import time
from api.agent_deps import REASONING_GRAPH

INPUT = "evaluation/chat/outputs/evaluation_chat.json"
OUTPUT = "evaluation/chat/outputs/chat_retrieval_results.json"
CONFIG_ID = "hybrid_rerank_v1"

def run_conversation_retrieval(convo):
    """Only retrieve chunks, don't generate answers"""
    
    stats = {
        "conversation_id": convo["id"],
        "no_answer": True,
        "latency_ms": 0,
        "query": "",
        "retrieved_chunks": [],  # Store full chunk data
        "config_id": CONFIG_ID,
    }

    for turn in convo["dialogue"]:
        if turn["role"] != "user":
            continue

        start = time.perf_counter()
        stats["query"] = turn["content"]
        
        # Only retrieve, don't generate
        result = REASONING_GRAPH.invoke({
            "query": turn["content"]
        })

        latency = (time.perf_counter() - start) * 1000
        stats["latency_ms"] = latency

        if result["status"] == "NO_ANSWER":
            return stats

        # Store full chunks with metadata for generation later
        stats["retrieved_chunks"] = result["retrieved_chunks"]
        stats["no_answer"] = False

    return stats


with open(INPUT) as f:
    conversations = json.load(f)

retrieval_results = []

for i, convo in enumerate(conversations, 1):
    print(f"[{i}/{len(conversations)}] Processing conversation {convo['id']}...")
    stats = run_conversation_retrieval(convo)
    retrieval_results.append(stats)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, indent=2, ensure_ascii=False)

print(f"\n[DONE] Saved {len(retrieval_results)} retrieval results → {OUTPUT}")
print(f"[NEXT] Upload {OUTPUT} to Colab and run generation script")