import json
from pathlib import Path
from api.agent_deps import REASONING_GRAPH

EVAL_FILE = Path("evaluation/gale/evaluation_gale_final.json")

def check_retrieval_count():
    data = json.loads(EVAL_FILE.read_text())
    
    # Test on 10 random questions
    import random
    sample = random.sample(data, 10)
    
    chunk_counts = []
    
    for item in sample:
        result = REASONING_GRAPH.invoke({
            "query": item["question"],
            "conversation_id": "__eval__"
        })
        
        if result["status"] != "NO_ANSWER":
            num_chunks = len(result["retrieved_chunks"])
            chunk_counts.append(num_chunks)
            print(f"Q: {item['question'][:60]}...")
            print(f"   Retrieved: {num_chunks} chunks\n")
    
    if chunk_counts:
        avg = sum(chunk_counts) / len(chunk_counts)
        print(f"\n{'='*60}")
        print(f"Average chunks retrieved: {avg:.1f}")
        print(f"Min: {min(chunk_counts)}, Max: {max(chunk_counts)}")
        
        if avg > 15:
            print(f"\n⚠️  You're retrieving {avg:.0f} chunks on average!")
            print(f"   This is likely why you're getting 100% hit rate.")
            print(f"   Try reducing top_k in your retrieval config.")

if __name__ == "__main__":
    check_retrieval_count()