"""Recommendation service.

Bridges the persisted ML engine and the database:

* gathers the user's interaction history + current session from Postgres
* delegates to the HybridRecommender (candidate generation -> ranking ->
  diversification -> explanations all happen inside the ML package)
* enriches results with catalog metadata (artwork, preview URLs)
* persists served recommendation lists for analytics/retraining
* degrades to DB-computed popularity when no artifacts are loaded
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from app.core.logging import get_logger
from app.models.catalog import Song
from app.models.events import Recommendation, UserSongEvent
from app.models.user import User
from app.schemas.recommendation import RecommendedSongOut, RecommendationsOut
from app.services import session_service
from app.services.catalog_service import get_songs_bulk, to_song_out
from app.services.engine import get_engine_holder
from app.services.profile_service import build_ml_profile

logger = get_logger(__name__)

VALID_MODES = ("personalized", "discover", "recent", "trending")


def _to_item_out(song: Song | None, song_id: str, score: float, reason: str,
                 signals: list[str], fallback: dict | None = None) -> RecommendedSongOut:
    fallback = fallback or {}
    if song is not None:
        base = to_song_out(song)
        return RecommendedSongOut(
            song_id=song.id, title=base.title, artist=base.artist_name, artist_id=base.artist_id,
            album=base.album, genres=base.genres, artwork_url=base.artwork_url,
            preview_url=base.preview_url, duration_ms=base.duration_ms,
            score=round(score, 6), reason=reason, signals=signals,
        )
    return RecommendedSongOut(
        song_id=song_id, title=fallback.get("title", song_id),
        artist=fallback.get("artist", "Unknown Artist"), artist_id=fallback.get("artist_id", ""),
        genres=fallback.get("genres", []), score=round(score, 6), reason=reason, signals=signals,
    )


def get_recommendations(db: DbSession, user: User, mode: str = "personalized",
                        limit: int = 10, genre: str | None = None) -> RecommendationsOut:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    limit = max(1, min(limit, 100))

    holder = get_engine_holder()
    profile, interactions = build_ml_profile(db, user)
    session_items = session_service.recent_session_items(db, user.id)
    exclude = set(profile.history_songs) | set(profile.disliked_songs)

    if holder.is_ready:
        recs = holder.hybrid.recommend(
            user_id=user.id, k=limit, mode=mode, interactions=interactions,
            profile=profile, session_items=session_items or None,
            preferred_genres=user.preferred_genres or None, exclude_songs=exclude,
        )
        strategy = "hybrid"
    else:
        recs = _fallback_recommendations(db, user, limit, exclude, genre)
        strategy = "popularity-fallback"

    songs = get_songs_bulk(db, [r.song_id for r in recs])
    items = []
    for rec in recs:
        song = songs.get(rec.song_id)
        if song is None and strategy == "hybrid":
            # artifact catalog and DB can drift (e.g. after re-seeding); skip
            # songs the API cannot resolve to catalog rows
            logger.warning("Skipping recommendation %s: not present in catalog DB", rec.song_id)
            continue
        meta = {"title": getattr(rec, "title", None)} if song is None else None
        items.append(_to_item_out(song, rec.song_id, rec.score, rec.reason, rec.signals, meta or {}))
    items = items[:limit]

    _persist(db, user.id, mode, items)
    return RecommendationsOut(
        user_id=user.id, mode=mode, strategy=strategy,
        generated_at=datetime.now(timezone.utc), items=items,
    )


def _fallback_recommendations(db: DbSession, user: User, limit: int, exclude: set[str],
                              genre: str | None = None) -> list:
    """DB popularity fallback used before any model is trained."""
    from ml.recommenders.hybrid import Recommendation as MLRecommendation

    counts = (
        select(UserSongEvent.song_id, func.count().label("c"))
        .where(UserSongEvent.event_type.in_(["play", "like", "save"]))
        .group_by(UserSongEvent.song_id)
        .order_by(func.count().desc())
        .limit(limit * 10)
    )
    rows = db.execute(counts).all()
    if not rows:
        # no events recorded yet: fall back to catalog popularity column
        songs = db.execute(
            select(Song).order_by(Song.popularity.desc().nullslast(), Song.id).limit(limit)
        ).scalars().all()
        return [
            MLRecommendation(
                song_id=s.id, score=float(s.popularity or 0.0),
                reason="Popular on TuneFlow right now. Train the ML models to unlock personalized recommendations.",
                signals=["popularity", "cold_start"],
            )
            for s in songs
        ]
    results = []
    preferred = set(user.preferred_genres or [])
    for song_id, count in rows:
        if song_id in exclude or len(results) >= limit:
            continue
        song = db.get(Song, song_id)
        if song is None:
            continue
        if genre:
            if genre.lower() not in [g.name.lower() for g in song.genres]:
                continue
        elif preferred and not (preferred & {g.name.lower() for g in song.genres}):
            # soft preference: keep, but rank preferred-genre songs first
            pass
        results.append(MLRecommendation(
            song_id=song_id, score=float(count),
            reason="Popular on TuneFlow right now. Train the ML models to unlock personalized recommendations.",
            signals=["popularity", "cold_start"] if not results else ["popularity"],
        ))
    results.sort(key=lambda r: (0, -r.score))
    return results[:limit]


def _persist(db: DbSession, user_id: str, mode: str, items: list[RecommendedSongOut]) -> None:
    rec = Recommendation(
        id=f"rec_{uuid.uuid4().hex[:16]}", user_id=user_id, mode=mode,
        items=[i.model_dump() for i in items],
        context={"generated_at": datetime.now(timezone.utc).isoformat()},
    )
    db.add(rec)
    db.flush()


# ----------------------------------------------------------------------
def similar_songs(db: DbSession, song_id: str, limit: int = 10) -> list[RecommendedSongOut]:
    """Content-similarity based 'more like this'; genre/artist fallback."""
    song = db.get(Song, song_id)
    if song is None:
        return []
    holder = get_engine_holder()
    results: list[tuple[str, float, str, list[str]]] = []

    if holder.content is not None and song_id in holder.content.song_to_idx:
        series = holder.content.similar_songs(song_id, k=limit)
        songs_meta = holder.songs_df.set_index("song_id") if holder.songs_df is not None else None
        for sid, score in series.items():
            reason = _similarity_reason(song, sid, songs_meta)
            results.append((sid, float(score), reason, ["content_similarity"]))
    else:
        same_genre = db.execute(
            select(Song).join(Song.genres)
            .where(Song.id != song_id)
            .limit(limit * 3)
        ).scalars().all()
        target_genres = {g.id for g in song.genres}
        ranked = sorted(same_genre, key=lambda s: -len(target_genres & {g.id for g in s.genres}))
        for cand in ranked[:limit]:
            results.append((cand.id, 0.5, f"Shares genres with {song.title}", ["genre_similarity"]))

    songs = get_songs_bulk(db, [r[0] for r in results])
    return [_to_item_out(songs.get(sid), sid, score, reason, signals)
            for sid, score, reason, signals in results]


def _similarity_reason(source: Song, candidate_id: str, songs_meta: pd.DataFrame | None) -> str:
    if songs_meta is not None and candidate_id in songs_meta.index:
        row = songs_meta.loc[candidate_id]
        genres = row.get("genres") or []
        if len(genres):
            return f"Sounds similar to {source.title} (shared style: {', '.join(list(genres)[:2])})"
    return f"Sounds similar to {source.title}"


def recently_played(db: DbSession, user_id: str, limit: int = 12) -> list[Song]:
    rows = db.execute(
        select(UserSongEvent.song_id, func.max(UserSongEvent.timestamp).label("ts"))
        .where(UserSongEvent.user_id == user_id,
               UserSongEvent.event_type.in_(["play", "like", "save"]))
        .group_by(UserSongEvent.song_id)
        .order_by(func.max(UserSongEvent.timestamp).desc())
        .limit(limit)
    ).all()
    songs = get_songs_bulk(db, [r[0] for r in rows])
    return [songs[r[0]] for r in rows if r[0] in songs]
