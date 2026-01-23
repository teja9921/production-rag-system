from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./rag_app.db"

engine = create_engine(
    url = DATABASE_URL,
    connect_args = {"check_same_thread":False}
)

SessionLocal = sessionmaker(
    bind = engine,
    autoflush=False,
    autocommit = False
)


