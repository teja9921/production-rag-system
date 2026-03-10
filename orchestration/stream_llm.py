from huggingface_hub import InferenceClient
from api.config import settings
from pathlib import Path
from typing import List, Dict, Any, Iterable
import textwrap
from core.logger import get_logger


class StreamingLLM:
    def __init__(self):
        self.client = InferenceClient(
            model=settings.LLM_MODEL_ID,
            token=settings.HF_TOKEN,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        base = Path("prompts/v1")
        self.system_prompt = (base / "system.txt").read_text()
        self.answer_prompt = (base / "answer.txt").read_text()

        self.logger = get_logger("llm.streaming")

    def stream(self, query: str, chunks: List[Dict[str, Any]]) -> Iterable[str]:
        if not chunks:
            yield "I don’t have enough information to answer this question."
            return

        sources_text = "\n\n".join(
            f"[Page {c['metadata']['page_number']}] {c['content']}"
            for c in chunks
        )

        user_prompt = self.answer_prompt.format(
            query=query,
            sources=textwrap.shorten(sources_text, 3000),
        )

        try:
            stream = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=settings.MAX_PROMPT_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token

        except Exception as e:
            self.logger.error("event=LLM_STREAM_ERROR | error=%s", str(e))
            yield "⚠️ The model is currently unavailable."
