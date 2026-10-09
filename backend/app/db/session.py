"""Database engine/session management."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def make_engine(url: str | None = None, echo: bool | None = None):
    settings = get_settings()
    url = url or settings.database_url
    echo = settings.db_echo if echo is None else echo
    kwargs: dict = {"echo": echo, "pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kwargs.pop("pool_pre_ping")
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = None
SessionLocal: sessionmaker[Session] | None = None


def init_engine(url: str | None = None) -> None:
    """(Re)create the global engine -- called at startup and in tests."""
    global engine, SessionLocal
    engine = make_engine(url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    if SessionLocal is None:
        init_engine()
    assert SessionLocal is not None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
