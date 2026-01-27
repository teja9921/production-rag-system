from huggingface_hub import InferenceClient
from api.config import settings
from pathlib import Path
from typing import List, Dict, Any
import textwrap

class StreamingLLM:
    def __init__(self):
        self.client = InferenceClient(
            model = settings.LLM_MODEL_ID,
            token= settings.HF_TOKEN,
            timeout= settings.LLM_TIMEOUT_SECONDS
        )

        base = Path("prompts/v1")
        self.system_prompt = (base/"system.txt").read_text()
        self.answer_prompt = (base/"answer.txt").read_text()

    def stream(self, query: str, chunks: [List[Dict[str, Any]]]):
        sources_text = "\n\n".join(f"[Page {c['metadata']['page_number']}] {c['content']}"
                                   for c in chunks
                                   )
        
        user_prompt = self.answer_prompt.format(
            query=query,
            sources=textwrap.shorten(sources_text, 3000),
        )

        stream = self.client.chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=settings.MAX_PROMPT_TOKENS,
            stream = True
        )

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content