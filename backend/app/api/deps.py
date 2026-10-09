"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session as DbSession

from app.db.session import get_db
from app.models.user import User
from app.services.engine import EngineHolder, get_engine_holder
from app.services.user_service import get_user


def get_current_db() -> DbSession:  # re-export for routers
    yield from get_db()


def dependency_db(db: DbSession = Depends(get_db)) -> DbSession:
    return db


def dependency_user(user_id: str, db: DbSession = Depends(get_db)) -> User:
    return get_user(db, user_id)


def dependency_engine() -> EngineHolder:
    return get_engine_holder()
