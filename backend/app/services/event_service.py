"""Event/feedback ingestion service.

Every PLAY / SKIP / LIKE / DISLIKE / SAVE lands here, gets its
configurable numeric weight, is attached to the user's active session
and updates activity timestamps.  This is the feedback loop feeding
future training data.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.core.logging import get_logger
from app.models.catalog import Song
from app.models.events import EventType, UserSongEvent
from app.models.user import User
from app.services import session_service
from app.services.engine import get_engine_holder

logger = get_logger(__name__)


class EventServiceError(Exception):
    """Domain error mapped to HTTP 4xx by the router."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _weight_for(event_type: str, weights: dict[str, float]) -> float:
    return float(weights.get(event_type, weights.get("play", 1.0)))


def record_event(db: DbSession, user_id: str, song_id: str, event_type: str,
                 session_id: str | None = None, timestamp: datetime | None = None) -> UserSongEvent:
    user = db.get(User, user_id)
    if user is None:
        raise EventServiceError(f"Unknown user {user_id!r}", 404)
    song = db.get(Song, song_id)
    if song is None:
        raise EventServiceError(f"Unknown song {song_id!r}", 404)

    EventType(event_type)  # validate
    weights = get_engine_holder().event_weights()
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    if session_id:
        session = session_service.get_session(db, session_id)
        if session is None or session.user_id != user_id:
            raise EventServiceError(f"Unknown session {session_id!r} for user {user_id!r}", 404)
    else:
        session = session_service.get_or_create_active(db, user_id)

    event = UserSongEvent(
        id=f"ev_{uuid.uuid4().hex[:20]}",
        user_id=user_id, song_id=song_id,
        event_type=event_type,
        timestamp=ts, session_id=session.id,
        weight=_weight_for(event_type, weights),
    )
    db.add(event)
    session_service.touch_session(db, session, user)
    db.flush()
    logger.debug("Recorded %s event user=%s song=%s session=%s", event_type, user_id, song_id, session.id)
    return event


def record_events_bulk(db: DbSession, events: list[dict]) -> list[UserSongEvent]:
    return [record_event(db, **event) for event in events]


def list_user_events(db: DbSession, user_id: str, limit: int = 50, offset: int = 0,
                     event_type: str | None = None) -> list[UserSongEvent]:
    query = (
        select(UserSongEvent)
        .where(UserSongEvent.user_id == user_id)
        .order_by(UserSongEvent.timestamp.desc())
        .limit(limit).offset(offset)
    )
    if event_type:
        query = query.where(UserSongEvent.event_type == EventType(event_type).value)
    return list(db.execute(query).scalars().all())


def user_interactions_df(db: DbSession, user_id: str) -> pd.DataFrame:
    """Interaction history in the canonical ML schema (engine input)."""
    events = db.execute(
        select(UserSongEvent).where(UserSongEvent.user_id == user_id)
        .order_by(UserSongEvent.timestamp.asc())
    ).scalars().all()
    if not events:
        return pd.DataFrame(columns=["user_id", "song_id", "event_type", "timestamp", "session_id", "weight"])
    return pd.DataFrame([{
        "user_id": e.user_id,
        "song_id": e.song_id,
        "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
        "timestamp": e.timestamp,
        "session_id": e.session_id,
        "weight": e.weight,
    } for e in events])
