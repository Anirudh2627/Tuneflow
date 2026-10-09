"""User service: registration, retrieval, onboarding preferences."""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session as DbSession

from app.models.user import User
from app.services.event_service import EventServiceError


def create_user(db: DbSession, display_name: str, email: str | None = None,
                preferred_genres: list[str] | None = None) -> User:
    user = User(
        id=f"usr_{uuid.uuid4().hex[:16]}",
        display_name=display_name.strip(),
        email=email.strip().lower() if email else None,
        preferred_genres=[g.strip().lower() for g in (preferred_genres or [])],
    )
    db.add(user)
    db.flush()
    return user


def get_user(db: DbSession, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise EventServiceError(f"Unknown user {user_id!r}", 404)
    return user


def update_preferred_genres(db: DbSession, user_id: str, genres: list[str]) -> User:
    user = get_user(db, user_id)
    user.preferred_genres = [g.strip().lower() for g in genres]
    db.flush()
    return user
