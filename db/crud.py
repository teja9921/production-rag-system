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
            .order_by(models.Conversation.created_at.asc())
            .all()
    )

def get_conversation(db: Session, conversation_id: str)-> models.Conversation | None:
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