import json
from pathlib import Path
from api.agent_deps import REASONING_GRAPH

EVAL_FILE = Path("evaluation/gale/evaluation_gale_final.json")

def calculate_f1(predicted: str, gold: str) -> float:
    """Token-level F1"""
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    
    if not gold_tokens:
        return 0.0
    
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0
    
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gold_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

def check_answer_quality():
    data = json.loads(EVAL_FILE.read_text())
    
    # Test first 10
    print("Testing answer quality on first 10 questions...\n")
    
    results = []
    
    for i, item in enumerate(data[:10], 1):
        result = REASONING_GRAPH.invoke({
            "query": item["question"],
            "conversation_id": "__eval__"
        })
        
        if result["status"] == "NO_ANSWER":
            print(f"{i}. NO ANSWER")
            results.append({"f1": 0.0, "contains": False})
            continue
        
        predicted = result.get("answer", "")
        gold = item["answer"]
        
        f1 = calculate_f1(predicted, gold)
        contains = gold.lower() in predicted.lower()
        
        results.append({"f1": f1, "contains": contains})
        
        print(f"{i}. F1: {f1:.3f} | Contains: {contains}")
        print(f"   Q: {item['question'][:70]}...")
        print(f"   Gold:      {gold[:60]}...")
        print(f"   Predicted: {predicted[:60]}...")
        print()
    
    # Summary
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    contains_rate = sum(r["contains"] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print(f"Average F1 Score: {avg_f1:.3f}")
    print(f"Contains Answer Rate: {contains_rate:.1%}")
    print(f"{'='*60}")
    
    if avg_f1 < 0.3:
        print("\n⚠️  Low F1 despite 100% retrieval hit!")
        print("   Problem: LLM isn't using retrieved context well")
    elif avg_f1 > 0.6:
        print("\n✅ Good F1 + 100% retrieval = Excellent system!")

if __name__ == "__main__":
    check_answer_quality()

