from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from db import models
import uuid


# ---------- Users ----------

def add_user(db: Session, user_id: Optional[str] = None) -> models.User:
    """
    Idempotent user creation.
    """
    if user_id is None:
        user_id = str(uuid.uuid4())
    user = models.User(id=user_id)
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        return get_user(db, user_id)


def get_user(db: Session, user_id: str) -> Optional[models.User]:
    return (
        db.query(models.User)
        .filter(models.User.id == user_id)
        .first()
    )


# ---------- Conversations ----------

def create_conversation(db: Session, user_id: str) -> models.Conversation:
    convo = models.Conversation(user_id=user_id)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


def get_user_conversations(db: Session, user_id: str) -> List[models.Conversation]:
    return (
        db.query(models.Conversation)
        .filter(models.Conversation.user_id == user_id)
        .order_by(models.Conversation.created_at.desc())
        .all()
    )


def get_conversation(
    db: Session,
    conversation_id: str,
    user_id: str,
) -> Optional[models.Conversation]:
    return (
        db.query(models.Conversation)
        .filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == user_id,
        )
        .first()
    )


# ---------- Messages ----------

def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    commit: bool = True,
) -> models.Message:

    if role not in ("user", "assistant"):
        raise ValueError("Invalid role")

    msg = models.Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )

    db.add(msg)

    if commit:
        db.commit()
        db.refresh(msg)

    return msg


def get_conversation_messages(
    db: Session,
    conversation_id: str,
) -> List[models.Message]:
    return (
        db.query(models.Message)
        .filter(models.Message.conversation_id == conversation_id)
        .order_by(models.Message.created_at.asc())
        .all()
    )


# ---------- Conversation Title ----------

class ConversationNotFoundError(Exception):
    pass

class InvalidTitleError(Exception):
    pass

class DatabaseWriteError(Exception):
    pass

def update_conversation_title(
    db: Session,
    conversation_id: str,
    title: str,
) -> models.Conversation:

    convo = (
        db.query(models.Conversation)
        .filter(models.Conversation.id == conversation_id)
        .first()
    )

    if not convo:
        raise ConversationNotFoundError(conversation_id)

    if not title or not title.strip():
        raise InvalidTitleError("Title cannot be empty")

    sanitized = title.strip()
    if len(sanitized) > 200:
        sanitized = sanitized[:197] + "..."

    convo.title = sanitized

    try:
        db.commit()
        db.refresh(convo)
        return convo
    except Exception as e:
        db.rollback()
        raise DatabaseWriteError(str(e))
