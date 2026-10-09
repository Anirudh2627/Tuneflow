"""Session management service.

A session groups consecutive listening events for a user.  Sessions
auto-close after ``session_timeout_minutes`` of inactivity; the current
(or most recent) session's ordered song list feeds the SessionRecommender.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.models.events import Session as UserSession
from app.models.events import UserSongEvent
from app.models.user import User, utcnow
from app.schemas.events import EventOut, SessionOut

POSITIVE_EVENTS = ("play", "like", "save")


def _event_out(event: UserSongEvent) -> EventOut:
    return EventOut(
        id=event.id, user_id=event.user_id, song_id=event.song_id,
        event_type=event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
        timestamp=event.timestamp, session_id=event.session_id, weight=event.weight,
    )


def session_out(session: UserSession) -> SessionOut:
    song_ids = [
        e.song_id for e in sorted(session.events or [], key=lambda e: e.timestamp)
        if (e.event_type.value if hasattr(e.event_type, "value") else e.event_type) in POSITIVE_EVENTS
    ]
    return SessionOut(
        id=session.id, user_id=session.user_id, start_time=session.start_time,
        last_activity=session.last_activity, ended=session.ended,
        song_ids=song_ids, events=[_event_out(e) for e in session.events or []],
    )


def create_session(db: DbSession, user_id: str, close_previous: bool = True) -> UserSession:
    if close_previous:
        active = db.execute(
            select(UserSession).where(UserSession.user_id == user_id, UserSession.ended.is_(False))
        ).scalars().all()
        for session in active:
            session.ended = True
    session = UserSession(id=f"ses_{uuid.uuid4().hex[:16]}", user_id=user_id,
                          start_time=utcnow(), last_activity=utcnow(), ended=False)
    db.add(session)
    db.flush()
    return session


def get_or_create_active(db: DbSession, user_id: str, timeout_minutes: int = 30) -> UserSession:
    """Return the user's active session, creating one when missing/expired."""
    session = db.execute(
        select(UserSession)
        .where(UserSession.user_id == user_id, UserSession.ended.is_(False))
        .order_by(UserSession.last_activity.desc())
    ).scalars().first()

    now = utcnow()
    if session is not None:
        last = session.last_activity
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now - last <= timedelta(minutes=timeout_minutes):
            return session
        session.ended = True
    return create_session(db, user_id, close_previous=True)


def get_session(db: DbSession, session_id: str) -> UserSession | None:
    return db.get(UserSession, session_id)


def recent_session_items(db: DbSession, user_id: str, timeout_minutes: int = 30 * 24,
                         limit: int = 12) -> list[str]:
    """Ordered (oldest -> newest) song ids from the user's current/most recent session."""
    session = db.execute(
        select(UserSession)
        .where(UserSession.user_id == user_id)
        .order_by(UserSession.last_activity.desc())
    ).scalars().first()
    if session is None:
        return []
    events = db.execute(
        select(UserSongEvent)
        .where(UserSongEvent.session_id == session.id,
               UserSongEvent.event_type.in_(POSITIVE_EVENTS))
        .order_by(UserSongEvent.timestamp.asc())
    ).scalars().all()
    song_ids = [e.song_id for e in events]
    return song_ids[-limit:]


def touch_session(db: DbSession, session: UserSession, user: User | None = None) -> None:
    now = utcnow()
    session.last_activity = now
    if user is not None:
        user.last_active_at = now
