from db.session import SessionLocal
from db import crud
from orchestration.state import GraphState

MAX_HISTORY_MESSAGES = 6

def memory_reader(state: GraphState)-> GraphState:
    conversation_id = state["conversation_id"]
    db = SessionLocal()

    messages = crud.get_conversation_messages(db, conversation_id)
    db.close()

    if not messages:
        return state
    
    # only last N messages
    recent = messages[-MAX_HISTORY_MESSAGES:]

    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in recent
    )

    # Inject memory into query for rewrite stage
    state["history"] = history_text

    return state

def memory_writer(state: GraphState) ->GraphState:
    conversation_id = state["conversation_id"]
    answer = state.get("answer")

    if not answer:
        return state
    
    db = SessionLocal()
    crud.add_message(db, conversation_id, "assistant", answer)
    db.close()

    return state

