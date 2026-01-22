from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)

class AnswerSource(BaseModel):
    page_number: int
    content : str

class QueryResponse(BaseModel):
    status: Literal["ANSWER", "NO_ANSWER"]
    answer: Optional[str] = None
    sources: Optional[list[AnswerSource]] = None



