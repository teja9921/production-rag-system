from pathlib import Path
from typing import Dict, Any

from langchain_core.runnables import Runnable
from huggingface_hub import InferenceClient

from api.config import settings
from core.logger import get_logger
from core.exceptions import CustomException

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

    def invoke(
        self,
        state: Dict[str, Any],
        config=None,
        **kwargs,
    ) -> Dict[str, Any]:

        if state.get("status") != "ANSWER":
            return state

        query = state.get("rewritten_query") or state["query"]
        chunks = state.get("retrieved_chunks", [])

        sources_text = "\n\n".join(
            f"[Page {c['metadata']['page_number']}] {c['content']}"
            for c in chunks
        )

        user_prompt = self.answer_prompt.format(
            query=query,
            sources=sources_text,
        )

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
                "event=LLM_SUCCESS | sources=%d | answer_len=%d",
                len(chunks),
                len(answer),
            )

            return {
                **state,
                "answer": answer,
            }

        except Exception as e:
            self.logger.exception("event=LLM_FAILURE")
            raise CustomException(
                "LLM generation failed",
                error=e,
                context={
                    "model": settings.LLM_MODEL_ID,
                    "source_count": len(chunks),
                },
            ) from e
