import uuid
import json
from pathlib import Path
from datasets import load_dataset

MAX_CONVERSATIONS = 250
OUTPUT_PATH = Path("evaluation/chat/outputs/evaluation_chat.json")

ds = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

conversations = []
count = 0

for row in ds:
    patient = row.get("Patient", "").strip()
    doctor = row.get("Doctor", "").strip()

    if not patient or not doctor:
        continue

    dialogue = [
        {"role": "user", "content": patient},
        {"role": "assistant", "content": doctor},
    ]

    conversations.append({
        "id": str(uuid.uuid4()),
        "dialogue": dialogue,
    })

    count += 1
    if count >= MAX_CONVERSATIONS:
        break

with open(OUTPUT_PATH, "w") as f:
    json.dump(conversations, f, indent=2)

print(f"Saved {len(conversations)} conversations")
