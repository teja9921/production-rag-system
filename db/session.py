from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    url = DATABASE_URL,
    connect_args = {"check_same_thread":False}
)

SessionLocal = sessionmaker(
    bind = engine,
    autoflush=False,
    autocommit = False
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
