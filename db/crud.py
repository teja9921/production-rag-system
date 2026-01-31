from sqlalchemy.orm import Session
from typing import List
from db import models

# ----------Users------------

def create_user(db: Session)->models.User:
    user = models.User()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def add_user(db: Session, user_id: str)->models.User:
    user = models.User(id = user_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user(db: Session, user_id:str)->models.User:
    return (
        db.query(models.User)
        .filter(models.User.id == user_id)
        .first()
      )

# ---------- Conversations ----------

def create_conversation(db: Session, user_id: str)-> models.Conversation:
    convo = models.Conversation(user_id= user_id)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo

def get_user_conversations(db: Session, user_id: str):
    return (db.query(models.Conversation)
            .filter(models.Conversation.user_id == user_id)
            .order_by(models.Conversation.created_at.desc())
            .all()
    )

def get_conversation(db: Session, conversation_id: str)-> models.Conversation:
    return (db.query(models.Conversation)
            .filter(models.Conversation.id == conversation_id)
            .first()
    )

# ---------- Messages ----------

def add_message(
        db: Session, 
        conversation_id: str, 
        role:str, 
        content: str
    )->models.Message:

    msg = models.Message(
        conversation_id= conversation_id, 
        role = role, 
        content = content
    )

    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_conversation_messages(
        db: Session,
        conversation_id: str
)-> List[models.Message]:
    return (db.query(models.Message)
            .filter(models.Message.conversation_id == conversation_id)
            .order_by(models.Message.created_at.asc())
            .all()
        )

# ---------- Conversation Title ----------

def update_conversation_title(
    db: Session, 
    conversation_id: str, 
    title: str
) -> models.Conversation:
    """
    Update the title of a conversation.
    
    Args:
        db: Database session
        conversation_id: ID of conversation to update
        title: New title (will be trimmed and truncated to 200 chars)
    
    Returns:
        Updated Conversation object
    
    Raises:
        ValueError: If conversation not found or title is invalid
    """
    # Fetch conversation
    convo = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id
    ).first()
    
    if not convo:
        raise ValueError(f"Conversation {conversation_id} not found")
    
    # Validate title
    if not title or len(title.strip()) == 0:
        raise ValueError("Title cannot be empty")
    
    # Sanitize and truncate
    sanitized_title = title.strip()
    if len(sanitized_title) > 200:
        sanitized_title = sanitized_title[:197] + "..."
    
    # Update
    convo.title = sanitized_title
    
    try:
        db.commit()
        db.refresh(convo)
        return convo
    except Exception as e:
        db.rollback()
        raise Exception(f"Failed to update conversation title: {e}")
