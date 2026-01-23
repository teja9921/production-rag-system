from langchain_core.runnables import Runnable
from huggingface_hub import InferenceClient
from api.config import settings
from pathlib import Path
from typing import Dict, Any
import textwrap

class LLMRunnable(Runnable):
    def __init__(self):
        self.client = InferenceClient(
            model=settings.LLM_MODEL_ID,
            token=settings.HF_TOKEN,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        base = Path("prompts/v1")
        self.system_prompt = (base / "system.txt").read_text()
        self.answer_prompt = (base / "answer.txt").read_text()

    def invoke(self, state: Dict[str, Any], config=None, **kwargs)-> Dict[str, Any]:
        if state["status"] != "ANSWER":
            return state

        sources_text = "\n\n".join(
            f"[Page {c['metadata']['page_number']}] {c['content']}"
            for c in state["retrieved_chunks"]
        )

        user_prompt = self.answer_prompt.format(
            query=state["rewritten_query"],
            sources=textwrap.shorten(sources_text, 3000),
        )

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=settings.MAX_PROMPT_TOKENS,
            temperature=0.2,
        )

        return {
            "answer": response.choices[0].message.content
        }
