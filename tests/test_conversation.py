from db import models
from db.session import SessionLocal


def list_conversations() -> list[models.Conversation]:
    db = SessionLocal()
    try:
        return db.query(models.Conversation).all()
    finally:
        db.close()


if __name__ == "__main__":
    conversations = list_conversations()
    print(conversations)
