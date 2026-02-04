from orchestration.state import GraphState
from api.config import settings
from pathlib import Path
from huggingface_hub import InferenceClient
import re


class QueryWriter:
    """
    Optional query rewriter.
    Rewrites ONLY when the query is conversationally incomplete.
    """

    def __init__(self):
        self.client = InferenceClient(
            model=settings.LLM_MODEL_ID,
            token=settings.HF_TOKEN,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        base = Path("prompts/v2")
        self.system_prompt = (base / "rewrite_system.txt").read_text()
        self.user_prompt = (base / "rewrite_user.txt").read_text()

        # Cheap linguistic signals
        self.pronoun_pattern = re.compile(r"\b(it|they|that|this|those|these|he|she)\b", re.I)

    def _needs_rewrite(self, query: str, history: str | None) -> bool:
        """
        Heuristic gate to decide whether rewrite is necessary.
        """
        if not history:
            return False

        if len(query.strip()) < 12:
            return True

        if self.pronoun_pattern.search(query):
            return True

        if query.lower().startswith(("and ", "also ", "what about", "then ")):
            return True

        return False

    def __call__(self, state: GraphState) -> GraphState:
        query = state["query"]
        history = state.get("history")

        # Default: no rewrite
        state["rewritten_query"] = None

        if not self._needs_rewrite(query, history):
            return state

        try:
            prompt = self.user_prompt.format(
                context=history or "",
                query=query,
            )

            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=128,
            )

            rewritten = response.choices[0].message.content.strip()

            # Safety: empty or nonsense rewrite → ignore
            if rewritten and rewritten.lower() != query.lower():
                state["rewritten_query"] = rewritten

        except Exception:
            # Rewrite failure must NEVER break the graph
            state["rewritten_query"] = None

        return state
