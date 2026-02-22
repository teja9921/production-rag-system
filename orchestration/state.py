from typing import TypedDict, List, Dict, Any, Optional, Literal

class GraphState(TypedDict):
    query: str
    history: Optional[str]
    rewritten_query: Optional[str]
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    status: Optional[Literal["ANSWER", "NO_ANSWER"]]
    answer: Optional[str]

    