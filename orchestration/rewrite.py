from orchestration.state import GraphState
from api.config import settings
from pathlib import Path
from huggingface_hub import InferenceClient
import textwrap

class QueryWriter:
    def __init__(self):
        self.client = InferenceClient(
            model = settings.LLM_MODEL_ID,
            token = settings.HF_TOKEN,
            timeout = settings.LLM_TIMEOUT_SECONDS
        )

        base = Path("prompts/v2")
        self.system_prompt = (base/"rewrite_system.txt").read_text()
        self.user_prompt = (base/"rewrite_user.txt").read_text()

    def __call__(self, state: GraphState) -> GraphState:
        prompt = self.user_prompt.format(
            context = state.get("history", ""),
            query = state["query"]
        )

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=128
        )

        rewritten = response.choices[0].message.content.strip()

        state["rewritten_query"]= rewritten
        print(rewritten)
        return state
        


