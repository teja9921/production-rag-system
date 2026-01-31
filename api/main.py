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
from utils.title_generator import generate_simple_title, generate_llm_title
import json

app = FastAPI(title="Medical RAG API", version="0.3")
stream_llm = StreamingLLM()

# ---------- User ----------

@app.post("/users", response_model= UserResponse)
def create_user(payload: dict | None =None):
    db = SessionLocal()

    if payload and "user_id" in payload:
        user = crud.get_user(db, payload["user_id"])
        if user:
            db.close()
            return UserResponse(user_id= user.id)
        
        user = crud.add_user(db, payload["user_id"])

    else:
        user = crud.create_user(db)
    
    db.close()
    return UserResponse(user_id= user.id)

# ---------- Conversations ----------

@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(user_id:str):
    db= SessionLocal()

    user = crud.get_user(db, user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail = "User not found")
    
    convo = crud.create_conversation(db, user_id)
    db.close()

    return ConversationResponse(conversation_id= convo.id)

@app.get("/users/{user_id}/conversations", response_model= GetConversations)
def get_user_conversations(user_id: str):
    db= SessionLocal()

    user = crud.get_user(db, user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    convos = crud.get_user_conversations(db, user_id)
    db.close()
    return GetConversations(
        conversations=[
        ConversationResponse(
            conversation_id=c.id,
            title = c.title,
            created_at = c.created_at)
        for c in convos
        ]
    )

# ---------- Query ----------

@app.post("/conversations/{conversation_id}/query", response_model=QueryResponse)
def query_conversation(user_id:str, conversation_id: str, payload: QueryRequest):

    db = SessionLocal()

    user = crud.get_user(db, user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

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
def get_messages(conversation_id: str, user_id: str):
    db= SessionLocal()

    user = crud.get_user(db, user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

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
def stream_query(conversation_id: str, user_id:str, payload: QueryRequest):

    db = SessionLocal()

    user = crud.get_user(db, user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")


    convo = crud.get_conversation(db, conversation_id)
    if not convo: 
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    needs_title = convo.title is None

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
        
        # ✅ Generate title only if needed
        if needs_title:
            try: 
                title =  generate_llm_title(payload.query, answer_accum)
                crud.update_conversation_title(db, conversation_id, title)
            except Exception as e:
                print(f"Title generation failed: {e}")
                #Fallback to simple title
                try:
                    title = generate_simple_title(payload.query, 30)
                    crud.update_conversation_title(db, conversation_id, title)
                except Exception as fallback_error:
                    print(f"Fallback title generation failed: {fallback_error}")

        db.close()

    return StreamingResponse(token_stream(), media_type="text/event-stream")    

#------------converstion Title---------------

@app.put("/conversations/{conversation_id}/title")
def update_custom_title(conversation_id: str, user_id: str, payload: dict):
    """
    Docstring for update_custom_title
    
    :param conversation_id: Description
    :type conversation_id: str
    :param user_id: Description
    :type user_id: str
    :param payload: Description
    :type payload: dict
    """
    db = SessionLocal()

    try: 
        user = crud.get_user(db, user_id)
        if not user: 
            raise HTTPException(status_code=404, detail="User not found")
        
        new_title = payload.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
   
        updated_convo = crud.update_conversation_title(db, conversation_id, new_title)

        return {"conversation_id": updated_convo.id, "title": updated_convo.title}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()

@app.post("/conversations/{conversation_id}/title/regenerate")
def regenerate_title(conversation_id: str, user_id:str):
    """
    Docstring for regenerate_title
    
    :param conversation_id: Description
    :type conversation_id: str
    :param user_id: Description
    :type user_id: str
    """
    db = SessionLocal()

    try:
        user = crud.get_user(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        messages = crud.get_conversation_messages(db, conversation_id)
        if len(messages)<2:
            raise HTTPException(status_code=400, detail="Not enough messages")
        
        first_query = messages[0].content
        first_answer = messages[1].content

        title = generate_llm_title(first_query, first_answer)

        updated_convo = crud.update_conversation_title(db, conversation_id, title)
        
        return {"conversation_id": updated_convo.id, "title": updated_convo.title}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user_id: str):
    """Delete a conversation and all its messages"""
    db = SessionLocal()
    
    try:
        user = crud.get_user(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        convo = crud.get_conversation(db, conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        db.delete(convo)
        db.commit()
        
        return {"message": "Conversation deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    