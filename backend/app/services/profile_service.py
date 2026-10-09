"""User profile service: derives taste profile from stored events."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.models.catalog import Artist
from app.models.user import User
from app.schemas.user import UserProfileOut
from app.services.catalog_service import songs_dataframe
from app.services.engine import get_engine_holder
from app.services.event_service import user_interactions_df


def _songs_frame(db: DbSession):
    holder = get_engine_holder()
    if holder.songs_df is not None and not holder.songs_df.empty:
        return holder.songs_df
    return songs_dataframe(db)


def build_user_profile(db: DbSession, user: User) -> UserProfileOut:
    from ml.features.user_profile import UserProfileService

    interactions = user_interactions_df(db, user.id)
    songs = _songs_frame(db)
    service = UserProfileService(songs, affinity_events=get_engine_holder().event_weights() | {})
    profile = service.build(user.id, interactions)

    artist_ids = list(profile.artist_affinity)[:10]
    artists = {}
    if artist_ids:
        rows = db.execute(select(Artist).where(Artist.id.in_(artist_ids))).scalars().all()
        artists = {a.id: a.name for a in rows}
    top_artists = [
        {"artist_id": aid, "name": artists.get(aid, aid), "affinity": round(score, 4)}
        for aid, score in sorted(profile.artist_affinity.items(), key=lambda kv: -kv[1])[:10]
    ]

    return UserProfileOut(
        user_id=user.id,
        display_name=user.display_name,
        top_genres=profile.top_genres,
        genre_affinity={g: round(v, 4) for g, v in profile.genre_affinity.items()},
        top_artists=top_artists,
        tag_affinity={t: round(v, 4) for t, v in profile.tag_affinity.items()},
        audio_feature_means={k: round(v, 4) for k, v in profile.audio_feature_means.items()},
        liked_songs=profile.liked_songs[-50:],
        disliked_songs=profile.disliked_songs[-50:],
        saved_songs=profile.saved_songs[-50:],
        recent_songs=profile.recent_songs[:20],
        listening_stats=profile.listening_stats,
        is_cold_start=profile.is_cold_start,
    )


def build_ml_profile(db: DbSession, user: User):
    """ML-side UserProfile object used by the recommendation service."""
    from ml.features.user_profile import UserProfileService

    interactions = user_interactions_df(db, user.id)
    service = UserProfileService(_songs_frame(db))
    return service.build(user.id, interactions), interactions
