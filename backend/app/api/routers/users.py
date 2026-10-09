"""User, profile, history and recommendation routers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.api.deps import dependency_db
from app.models.catalog import Song
from app.models.events import UserSongEvent
from app.schemas.catalog import SongOut
from app.schemas.events import EventOut
from app.schemas.recommendation import RecommendationsOut
from app.schemas.user import UserCreate, UserOut, UserProfileOut
from app.services import event_service, recommendation_service, user_service
from app.services.catalog_service import to_song_out
from app.services.profile_service import build_user_profile

router = APIRouter(prefix="/users", tags=["users"])


def _get_user_or_404(db: DbSession, user_id: str):
    try:
        return user_service.get_user(db, user_id)
    except event_service.EventServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _event_out(e: UserSongEvent) -> EventOut:
    return EventOut(
        id=e.id, user_id=e.user_id, song_id=e.song_id,
        event_type=e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
        timestamp=e.timestamp, session_id=e.session_id, weight=e.weight,
    )


@router.post("", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: DbSession = Depends(dependency_db)) -> UserOut:
    user = user_service.create_user(db, payload.display_name, payload.email, payload.preferred_genres)
    db.commit()
    db.refresh(user)
    return UserOut.model_validate(user)


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: str, db: DbSession = Depends(dependency_db)) -> UserOut:
    return UserOut.model_validate(_get_user_or_404(db, user_id))


@router.get("/{user_id}/profile", response_model=UserProfileOut)
def get_user_profile(user_id: str, db: DbSession = Depends(dependency_db)) -> UserProfileOut:
    return build_user_profile(db, _get_user_or_404(db, user_id))


@router.get("/{user_id}/recommendations", response_model=RecommendationsOut)
def get_recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    mode: str = Query(default="personalized", pattern="^(personalized|discover|recent|trending)$"),
    genre: str | None = Query(default=None),
    db: DbSession = Depends(dependency_db),
) -> RecommendationsOut:
    user = _get_user_or_404(db, user_id)
    try:
        result = recommendation_service.get_recommendations(db, user, mode=mode, limit=limit, genre=genre)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return result


@router.get("/{user_id}/history")
def get_history(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    event_type: str | None = Query(default=None, pattern="^(play|skip|like|dislike|save)$"),
    db: DbSession = Depends(dependency_db),
) -> list:
    """Listening history with embedded song payloads (newest first)."""
    from app.schemas.events import HistoryItemOut

    _get_user_or_404(db, user_id)
    events = event_service.list_user_events(db, user_id, limit=limit, offset=offset, event_type=event_type)
    song_ids = list(dict.fromkeys(e.song_id for e in events))
    songs = {
        s.id: s for s in db.execute(select(Song).where(Song.id.in_(song_ids))).scalars().all()
    } if song_ids else {}
    items = []
    for e in events:
        song = songs.get(e.song_id)
        if song is None:
            continue
        items.append(HistoryItemOut(
            id=e.id,
            event_type=e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
            timestamp=e.timestamp, session_id=e.session_id, weight=e.weight,
            song=to_song_out(song),
        ))
    return items
@router.get("/{user_id}/recent", response_model=list[SongOut])
def get_recently_played(
    user_id: str,
    limit: int = Query(default=12, ge=1, le=50),
    db: DbSession = Depends(dependency_db),
) -> list[SongOut]:
    _get_user_or_404(db, user_id)
    songs = recommendation_service.recently_played(db, user_id, limit=limit)
    return [to_song_out(s) for s in songs]


@router.get("/{user_id}/liked")
def get_liked(user_id: str, db: DbSession = Depends(dependency_db)) -> dict:
    """Liked / saved / disliked song lists (most recent first, deduped)."""
    _get_user_or_404(db, user_id)
    rows = db.execute(
        select(UserSongEvent)
        .where(UserSongEvent.user_id == user_id,
               UserSongEvent.event_type.in_(["like", "save", "dislike"]))
        .order_by(UserSongEvent.timestamp.desc())
    ).scalars().all()

    buckets: dict[str, list[SongOut]] = {"liked": [], "saved": [], "disliked": []}
    seen: dict[str, set[str]] = {"like": set(), "save": set(), "dislike": set()}
    bucket_for = {"like": "liked", "save": "saved", "dislike": "disliked"}
    song_ids = list(dict.fromkeys(e.song_id for e in rows))
    songs = {s.id: s for s in db.execute(select(Song).where(Song.id.in_(song_ids))).scalars().all()} if song_ids else {}

    for e in rows:
        et = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
        if e.song_id in seen[et]:
            continue
        seen[et].add(e.song_id)
        song = songs.get(e.song_id)
        if song is not None:
            buckets[bucket_for[et]].append(to_song_out(song))
    return {"user_id": user_id, **buckets}
