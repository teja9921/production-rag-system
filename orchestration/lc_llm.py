import random
import time
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.runnables import Runnable
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

from api.config import settings
from core.logger import get_logger


MAX_RETRIES = 5
BASE_DELAY = 0.8
MAX_DELAY = 10.0


class LLMRunnable(Runnable):
    """
    LangGraph adapter for LLM answer generation.
    """

    def __init__(self):
        self.client = InferenceClient(
            model=settings.LLM_MODEL_ID,
            token=settings.HF_TOKEN,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        base = Path("prompts/v1")
        self.system_prompt = (base / "system.txt").read_text()
        self.answer_prompt = (base / "answer.txt").read_text()

        self.logger = get_logger("orchestration.llm")

        self.logger.info(
            "event=LLM_INIT | model=%s",
            settings.LLM_MODEL_ID,
        )

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "model is currently unavailable" in msg
            or "503" in msg
            or "timeout" in msg
            or isinstance(exc, HfHubHTTPError)
        )

    @staticmethod
    def _backoff(attempt: int):
        delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
        delay += random.uniform(0, 0.3 * delay)
        time.sleep(delay)

    def invoke(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config=None,
        **kwargs,
    ) -> Dict[str, Any]:

        if not chunks:
            return {"answer": "I don’t have enough information to answer this question."}

        sources_text = "\n\n".join(
            f"[Page {c['metadata']['page_number']}] {c['content']}"
            for c in chunks
        )

        user_prompt = self.answer_prompt.format(
            query=query,
            sources=sources_text,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat_completion(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=settings.MAX_PROMPT_TOKENS,
                    temperature=0.2,
                )

                answer = response.choices[0].message.content

                self.logger.info(
                    "event=LLM_SUCCESS | attempt=%d | sources=%d | answer_len=%d",
                    attempt,
                    len(chunks),
                    len(answer),
                )

                return {"answer": answer}

            except Exception as e:
                if not self._should_retry(e):
                    self.logger.exception(
                        "event=LLM_FATAL | attempt=%d", attempt
                    )
                    break

                self.logger.warning(
                    "event=LLM_RETRY | attempt=%d | error=%s",
                    attempt,
                    str(e)[:200],
                )

                self._backoff(attempt)

        self.logger.error(
            "event=LLM_DEGRADED | retries_exhausted | model=%s",
            settings.LLM_MODEL_ID,
        )

        return {
            "answer": "Model unavailable during evaluation. Answer not generated."
        }
