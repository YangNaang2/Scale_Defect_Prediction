"""DB 엔진/세션 설정. DATABASE_URL 미지정 시 로컬 SQLite(scale_system.db) 사용."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///scale_system.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
