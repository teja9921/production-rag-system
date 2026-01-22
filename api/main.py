from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from api.schemas import (
    QueryResponse, 
    QueryRequest,
    AnswerSource,
)
from api.deps import GRAPH

app = FastAPI(title="Medical RAG API", version="0.1")

@app.post(path="/query", response_model= QueryResponse)
def query_rag(payload: QueryRequest):
    try:
        result = GRAPH.invoke({"query":payload.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if result.get("status") == "NO_ANSWER":
        return QueryResponse(status="NO_ANSWER")
    
    sources = [
        AnswerSource(
            page_number = c["metadata"]["page_number"],
            content = c["content"]
        )
        for c in result.get("sources",[])
    ]

    return QueryResponse(
        status = "ANSWER",
        answer = result.get("answer"),
        sources = sources
    )
    


