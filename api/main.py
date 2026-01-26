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
    MessageResponse,
    GetConversations
)
from api.agent_deps import GRAPH, RETRIEVAL_GRAPH
from db.session import SessionLocal
from db import crud
import textwrap
from fastapi.responses import StreamingResponse
from orchestration.stream_llm import StreamingLLM
import json



app = FastAPI(title="Medical RAG API", version="0.3")
stream_llm = StreamingLLM()

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

@app.get("/users/{user_id}/conversations", response_model= GetConversations)
def get_user_conversations(user_id: str):
    db= SessionLocal()
    convos = crud.get_user_conversations(db, user_id)
    db.close()
    return GetConversations(
        conversations=[
        ConversationResponse(conversation_id=c.id)
        for c in convos
        ]
    )

# ---------- Query ----------

@app.post("/conversations/{conversation_id}/query", response_model=QueryResponse)
def query_conversation(conversation_id: str, payload: QueryRequest):

    db = SessionLocal()

    try:
        crud.add_message(db, conversation_id, "user", payload.query)
        result = GRAPH.invoke({"query": payload.query, "conversation_id": conversation_id})
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))
    
    status = result.get("status")

    if status == "NO_ANSWER":
        db=SessionLocal()
        crud.add_message(db, conversation_id, "assistant", "NO_ANSWER")
        db.close()
        return QueryResponse(status = "NO_ANSWER")
    
    sources = [
        AnswerSource(
            page_number = c["metadata"]["page_number"],
            content = textwrap.shorten(c["content"], width=100, placeholder="...")
        )
            for c in result.get("retrieved_chunks", [])      
    ]

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
    
@app.post("/conversations/{conversation_id}/stream")
def stream_query(conversation_id: str, payload: QueryRequest):

    db = SessionLocal()
    convo = crud.get_conversation(db, conversation_id)
    if not convo: 
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    crud.add_message(db, conversation_id, "user", payload.query)
    db.close()

    result = RETRIEVAL_GRAPH.invoke({
        "conversation_id": conversation_id,
        "query": payload.query
    })

    if result.get("status") == "NO_ANSWER":
        return StreamingResponse(
            iter(["NO_ANSWER"]),
            media_type="text/event-stream"
        )
    
    def token_stream():
        answer_accum = ""
        for token in stream_llm.stream(payload.query, result["retrieved_chunks"]):
            answer_accum+= token
            yield token

        db = SessionLocal()
        crud.add_message(db, conversation_id, "assistant", answer_accum)
        db.close()

    return StreamingResponse(token_stream(), media_type="text/event-stream")    



