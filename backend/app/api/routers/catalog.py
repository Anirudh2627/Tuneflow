"""Song, artist and search routers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DbSession

from app.api.deps import dependency_db
from app.schemas.catalog import AlbumOut, ArtistDetailOut, ArtistOut, GenreOut, SearchResultsOut, SongOut
from app.schemas.recommendation import RecommendedSongOut, SimilarSongsOut
from app.services import catalog_service, recommendation_service

songs_router = APIRouter(prefix="/songs", tags=["songs"])
artists_router = APIRouter(prefix="/artists", tags=["artists"])
search_router = APIRouter(tags=["search"])


@songs_router.get("/{song_id}", response_model=SongOut)
def get_song(song_id: str, db: DbSession = Depends(dependency_db)) -> SongOut:
    song = catalog_service.get_song(db, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail=f"Unknown song {song_id!r}")
    return catalog_service.to_song_out(song)


@songs_router.get("/{song_id}/similar", response_model=SimilarSongsOut)
def get_similar_songs(
    song_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    db: DbSession = Depends(dependency_db),
) -> SimilarSongsOut:
    song = catalog_service.get_song(db, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail=f"Unknown song {song_id!r}")
    items = recommendation_service.similar_songs(db, song_id, limit=limit)
    return SimilarSongsOut(song_id=song_id, items=items)


@artists_router.get("/{artist_id}", response_model=ArtistDetailOut)
def get_artist(artist_id: str, db: DbSession = Depends(dependency_db)) -> ArtistDetailOut:
    artist = catalog_service.get_artist(db, artist_id)
    if artist is None:
        raise HTTPException(status_code=404, detail=f"Unknown artist {artist_id!r}")
    base = ArtistOut.model_validate(artist)
    return ArtistDetailOut(
        **base.model_dump(),
        top_songs=[catalog_service.to_song_out(s)
                   for s in catalog_service.artist_top_songs(db, artist_id)],
        albums=[AlbumOut.model_validate(a) for a in catalog_service.artist_albums(db, artist_id)],
    )


@search_router.get("/search", response_model=SearchResultsOut)
def search(
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=10, ge=1, le=50),
    db: DbSession = Depends(dependency_db),
) -> SearchResultsOut:
    results = catalog_service.search_catalog(db, q, limit=limit)
    return SearchResultsOut(query=q, **results)


genres_router = APIRouter(prefix="/genres", tags=["genres"])


@genres_router.get("", response_model=list[GenreOut])
def list_genres(db: DbSession = Depends(dependency_db)) -> list[GenreOut]:
    from app.models.catalog import Genre
    from sqlalchemy import select

    genres = db.execute(select(Genre).order_by(Genre.name)).scalars().all()
    return [GenreOut.model_validate(g) for g in genres]


@genres_router.get("/{genre_id}/songs", response_model=list[SongOut])
def genre_songs(
    genre_id: str,
    limit: int = Query(default=20, ge=1, le=50),
    db: DbSession = Depends(dependency_db),
) -> list[SongOut]:
    """Top songs for a genre: popularity-artifact ranked, DB-count fallback."""
    from app.services.engine import get_engine_holder
    from app.models.events import UserSongEvent
    from sqlalchemy import func, select as sa_select

    holder = get_engine_holder()
    song_ids: list[str] = []
    if holder.popularity is not None:
        top = holder.popularity.top(k=limit, genre=genre_id.replace("-", " "))
        song_ids = [str(s) for s in top.index]
        if len(song_ids) < limit:
            # popularity genre vocab may use names, retry with the raw slug
            top = holder.popularity.top(k=limit, genre=genre_id)
            song_ids = list(dict.fromkeys(song_ids + [str(s) for s in top.index]))
    if not song_ids:
        from app.models.catalog import song_genres

        rows = db.execute(
            sa_select(UserSongEvent.song_id, func.count().label("c"))
            .join(song_genres, song_genres.c.song_id == UserSongEvent.song_id)
            .where(song_genres.c.genre_id == genre_id,
                   UserSongEvent.event_type.in_(["play", "like", "save"]))
            .group_by(UserSongEvent.song_id)
            .order_by(func.count().desc())
            .limit(limit * 3)
        ).all()
        song_ids = [r[0] for r in rows]
    songs = catalog_service.get_songs_bulk(db, song_ids[:limit])
    ordered = [songs[sid] for sid in song_ids[:limit] if sid in songs]
    if not ordered:
        from app.models.catalog import Genre, song_genres

        genre = db.get(Genre, genre_id)
        if genre is None:
            raise HTTPException(status_code=404, detail=f"Unknown genre {genre_id!r}")
        ordered = list(db.execute(
            sa_select(Song).join(song_genres, song_genres.c.song_id == Song.id)
            .where(song_genres.c.genre_id == genre_id).limit(limit)
        ).scalars().all())
    return [catalog_service.to_song_out(s) for s in ordered]
