"""Event, feedback and session routers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DbSession

from app.api.deps import dependency_db
from app.schemas.events import (
    EventBatchCreate,
    EventCreate,
    EventOut,
    FeedbackCreate,
    SessionCreate,
    SessionOut,
)
from app.services import event_service, session_service, user_service

events_router = APIRouter(tags=["events"])
sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])


def _event_out(e) -> EventOut:
    return EventOut(
        id=e.id, user_id=e.user_id, song_id=e.song_id,
        event_type=e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
        timestamp=e.timestamp, session_id=e.session_id, weight=e.weight,
    )


@events_router.post("/events", response_model=EventOut, status_code=201)
def post_event(payload: EventCreate, db: DbSession = Depends(dependency_db)) -> EventOut:
    try:
        event = event_service.record_event(
            db, payload.user_id, payload.song_id, payload.event_type,
            session_id=payload.session_id, timestamp=payload.timestamp,
        )
        db.commit()
    except event_service.EventServiceError as exc:
        db.rollback()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _event_out(event)


@events_router.post("/events/batch", response_model=list[EventOut], status_code=201)
def post_events_batch(payload: EventBatchCreate, db: DbSession = Depends(dependency_db)) -> list[EventOut]:
    try:
        events = [
            event_service.record_event(
                db, e.user_id, e.song_id, e.event_type,
                session_id=e.session_id, timestamp=e.timestamp,
            )
            for e in payload.events
        ]
        db.commit()
    except event_service.EventServiceError as exc:
        db.rollback()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return [_event_out(e) for e in events]


@events_router.post("/feedback", response_model=EventOut, status_code=201)
def post_feedback(payload: FeedbackCreate, db: DbSession = Depends(dependency_db)) -> EventOut:
    """Convenience endpoint: LIKE/DISLIKE/SAVE/SKIP/PLAY feedback."""
    try:
        event = event_service.record_event(
            db, payload.user_id, payload.song_id, payload.feedback, session_id=payload.session_id)
        db.commit()
    except event_service.EventServiceError as exc:
        db.rollback()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _event_out(event)


@sessions_router.post("", response_model=SessionOut, status_code=201)
def create_session(payload: SessionCreate, db: DbSession = Depends(dependency_db)) -> SessionOut:
    try:
        user_service.get_user(db, payload.user_id)
    except event_service.EventServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    session = session_service.create_session(db, payload.user_id)
    db.commit()
    db.refresh(session)
    return session_service.session_out(session)


@sessions_router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: DbSession = Depends(dependency_db)) -> SessionOut:
    session = session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id!r}")
    return session_service.session_out(session)


@sessions_router.post("/{session_id}/end", response_model=SessionOut)
def end_session(session_id: str, db: DbSession = Depends(dependency_db)) -> SessionOut:
    session = session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id!r}")
    session.ended = True
    db.commit()
    db.refresh(session)
    return session_service.session_out(session)
