from db.base import Base
from db.session import engine
from db import models  # noqa: F401


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


if __name__ == "__main__":
    create_tables()
