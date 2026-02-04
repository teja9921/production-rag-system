from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class CreateUserRequest(BaseModel):
    user_id: Optional[str] = None
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)

class AnswerSource(BaseModel):
    page_number: int
    content : str

class QueryResponse(BaseModel):
    status: Literal["ANSWER", "NO_ANSWER"]
    answer: Optional[str] = None
    sources: Optional[list[AnswerSource]] = None

class UserResponse(BaseModel):
    user_id: str

class ConversationResponse(BaseModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime | None = None

class GetConversations(BaseModel):
    conversations: List[ConversationResponse]
class MessageResponse(BaseModel):
    role: str
    content: str

class UpdateTitleRequest(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
