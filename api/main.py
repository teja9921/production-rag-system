import textwrap
import time
from typing import List
from db import crud
from db.session import get_db
from fastapi import FastAPI, HTTPException
from api.schemas import (
    QueryResponse, 
    QueryRequest,
    AnswerSource,
    UserResponse,
    CreateUserRequest,
    ConversationResponse,
    MessageResponse,
    GetConversations,
    UpdateTitleRequest
)
from api.agent_deps import REASONING_GRAPH
from fastapi import Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from orchestration.stream_llm import StreamingLLM
from utils.title_generator import generate_simple_title, generate_llm_title
from core.logger import get_logger

logger = get_logger("api.main")

app = FastAPI(title="Medical RAG API", version="0.3")
stream_llm = StreamingLLM()

# ---------- User ----------

@app.post("/users", response_model= UserResponse)
def create_user(payload: CreateUserRequest, db:Session = Depends(get_db)):

    if payload.user_id:
        user = crud.get_user(db, payload.user_id)
        if user:
            return UserResponse(user_id= user.id)
        
        user = crud.add_user(db, payload.user_id)

    else:
        user = crud.add_user(db)
    
    return UserResponse(user_id= user.id)

# ---------- Conversations ----------

@app.post("/users/{user_id}/conversations", response_model=ConversationResponse)
def create_conversation(user_id:str, db:Session = Depends(get_db)):

    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail = "User not found")
    
    convo = crud.create_conversation(db, user_id)

    return ConversationResponse(conversation_id= convo.id)

@app.get("/users/{user_id}/conversations", response_model= GetConversations)
def get_user_conversations(user_id: str, db: Session = Depends(get_db)):

    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    convos = crud.get_user_conversations(db, user_id)
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
def query_conversation(
    conversation_id: str,
    payload: QueryRequest,
    user_id: str,
    db: Session = Depends(get_db),
):

    start = time.perf_counter()

    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    convo = crud.get_conversation(db, conversation_id, user_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        crud.add_message(db, conversation_id, "user", payload.query)
        result = REASONING_GRAPH.invoke({
            "query": payload.query, 
             "conversation_id": conversation_id
            })
        
        if result["status"] == "NO_ANSWER":
            crud.add_message(db, conversation_id, "assistant", "NO_ANSWER")
            return QueryResponse(status = "NO_ANSWER")

        sources = [
        AnswerSource(
            page_number = c["metadata"]["page_number"],
            content = textwrap.shorten(c["content"], width=100, placeholder="...")
        )
            for c in result.get("retrieved_chunks", [])      
        ]
        
        answer = result["answer"]
        crud.add_message(db, conversation_id, "assistant", answer)

        if convo.title is None:
            crud.update_conversation_title(
                db,
                conversation_id,
                generate_simple_title(payload.query)
            )

        return QueryResponse(
        status = "ANSWER",
        answer = answer,
        sources = sources,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        latency = time.perf_counter() - start
        logger.info(
            "query_complete | latency_ms=%d",
            int(latency * 1000),
            )
    
# ---------- History ----------

@app.get(
    "/conversations/{conversation_id}/messages",
    response_model= List[MessageResponse]
)
def get_messages(conversation_id: str, user_id: str, db: Session = Depends(get_db)):

    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    convo = crud.get_conversation(db, conversation_id, user_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = crud.get_conversation_messages(db, conversation_id)

    return [
        MessageResponse(
            role = m.role, 
            content = m.content
        )
        for m in messages
    ]
    
@app.post("/conversations/{conversation_id}/stream")
def stream_query(
    conversation_id: str,
    payload: QueryRequest,
    user_id:str, 
    db: Session = Depends(get_db), 
):
    start = time.perf_counter()
    first_token_time = None

    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    convo = crud.get_conversation(db, conversation_id, user_id)
    if not convo: 
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    needs_title = convo.title is None
    crud.add_message(db, conversation_id, "user", payload.query)

    try:
        result = REASONING_GRAPH.invoke({
            "query": payload.query,
            "conversation_id": conversation_id,
        })
    except Exception: 
        logger.exception("retrieval_graph_failed")
        return StreamingResponse(
            iter(["ERROR"]),
            media_type="text/event-stream"
        )

    if result["status"] == "NO_ANSWER":
        crud.add_message(db, conversation_id, "assistant", "NO_ANSWER")
        return StreamingResponse(
            iter(["NO_ANSWER"]),
            media_type="text/event-stream"
        )

    def token_stream():
        nonlocal first_token_time
        answer_accum = ""

        try:
            for token in stream_llm.stream(payload.query, result["retrieved_chunks"]):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                answer_accum+= token
                yield token

            crud.add_message(db, conversation_id, "assistant", answer_accum)
            
            # ✅ Generate title only if needed
            if needs_title:
                try: 
                    title =  generate_llm_title(payload.query, answer_accum)
                    crud.update_conversation_title(db, conversation_id, title)
                except Exception as e:
                    logger.warning("llm_title_genration_failed | falling back to simple_title")
                    #Fallback to simple title
                    try:
                        title = generate_simple_title(payload.query, 30)
                        crud.update_conversation_title(db, conversation_id, title)
                    except Exception as fallback_error:
                        logger.warning("Fallback title generation failed: {fallback_error}")

        except Exception:
            logger.exception("streaming_failed")
            yield "\nERROR\n"

        finally:
            logger.info(
                "event=stream_complete | ttfb_ms=%d | total_ms=%d",
                int((first_token_time - start) * 1000) if first_token_time else -1,
                int((time.perf_counter() - start) * 1000),
            )
    return StreamingResponse(token_stream(), media_type="text/event-stream")    

#------------converstion Title---------------

@app.put("/conversations/{conversation_id}/title")
def update_custom_title(payload: UpdateTitleRequest, conversation_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Docstring for update_custom_title
    
    :param conversation_id: Description
    :type conversation_id: str
    :param user_id: Description
    :type user_id: str
    :param payload: Description
    :type payload: dict
    """

    try: 
        user = crud.get_user(db, user_id)
        if not user: 
            raise HTTPException(status_code=404, detail="User not found")
        
        new_title = payload.title.strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
   
        updated_convo = crud.update_conversation_title(db, conversation_id, new_title)

        return {"conversation_id": updated_convo.id, "title": updated_convo.title}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations/{conversation_id}/title/regenerate")
def regenerate_title(conversation_id: str, user_id:str, db: Session = Depends(get_db)):
    """
    Docstring for regenerate_title
    
    :param conversation_id: Description
    :type conversation_id: str
    :param user_id: Description
    :type user_id: str
    """
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

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user_id: str, db: Session = Depends(get_db)):
    """Delete a conversation and all its messages"""
    
    try:
        user = crud.get_user(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        convo = crud.get_conversation(db, conversation_id, user_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        db.delete(convo)
        db.commit()
        
        return {"message": "Conversation deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    