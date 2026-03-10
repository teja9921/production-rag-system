from db import crud
from db.session import SessionLocal


def run_db_smoke() -> None:
    db = SessionLocal()
    try:
        user = crud.add_user(db)
        print("User:", user.id)

        convo = crud.create_conversation(db, user.id)
        print("Conversation:", convo.id)

        crud.add_message(db, convo.id, "user", "What is a large language model")
        crud.add_message(
            db,
            convo.id,
            "assistant",
            "An LLM is a model trained on large corpora.",
        )

        messages = crud.get_conversation_messages(db, convo.id)
        for m in messages:
            print(m.role, ":", m.content)
    finally:
        db.close()


if __name__ == "__main__":
    run_db_smoke()
