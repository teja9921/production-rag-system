from dotenv import load_dotenv
load_dotenv()
from typing import List
from fastapi import FastAPI, HTTPException
from api.schemas import (
    QueryResponse, 
    QueryRequest,
    AnswerSource,
    UserResponse,
    ConversationResponse,
    MessageResponse
)
from api.deps import GRAPH
from db.session import SessionLocal
from db import crud

app = FastAPI(title="Medical RAG API", version="0.2")

# ---------- User ----------

@app.post("/users", response_model= UserResponse)
def create_user():
    db = SessionLocal()
    user = crud.create_user(db)
    db.close()
    return UserResponse(user_id= user.id)

# ---------- Conversations ----------

@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(user_id:str):
    db= SessionLocal()
    convo = crud.create_conversation(db, user_id)
    db.close()
    return ConversationResponse(conversation_id= convo.id)

# ---------- Query ----------

@app.post("/conversations/{conversation_id}/query", response_model=QueryResponse)
def query_conversation(conversation_id: str, payload: QueryRequest):

    db = SessionLocal()

    try:
        crud.add_message(db, conversation_id, "user", payload.query)
        result = GRAPH.invoke({"query": payload.query})
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("status") == "NO_ANSWER":
        crud.add_message(db, conversation_id, "assistant", "NO_ANSWER")
        db.close()
        return QueryResponse(status = "NO_ANSWER")
    
    sources = [
        AnswerSource(
            page_number = c["metadata"]["page_number"],
            content = c["content"]
        )
            for c in result.get("sources", [])      
    ]

    crud.add_message(db, conversation_id, "assistant", result.get("answer"))
    db.close()

    return QueryResponse(
        status = "ANSWER",
        answer = result.get("answer"),
        sources = sources
    )

# ---------- History ----------

@app.get(
    "/conversations/{conversation_id}/messages",
    response_model= List[MessageResponse]
)
def get_messages(conversation_id: str):
    db= SessionLocal()

    messages = crud.get_conversation_messages(db, conversation_id)
    db.close()

    return [
        MessageResponse(
            role = m.role, 
            content = m.content
        )
        for m in messages
    ]
    


