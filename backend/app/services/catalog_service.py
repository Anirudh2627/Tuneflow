"""Catalog service: ORM -> schema mapping and catalog queries."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.catalog import Album, Artist, Genre, Song
from app.schemas.catalog import AlbumOut, ArtistOut, GenreOut, SongOut


def to_song_out(song: Song) -> SongOut:
    return SongOut(
        id=song.id,
        title=song.title,
        artist_id=song.artist_id,
        artist_name=song.artist.name if song.artist else "Unknown Artist",
        album=song.album.title if song.album else None,
        album_id=song.album_id,
        release_year=song.release_year,
        duration_ms=song.duration_ms,
        genres=[g.name for g in (song.genres or [])],
        tags=list(song.tags or []),
        preview_url=song.preview_url,
        artwork_url=song.artwork_url,
        energy=song.energy,
        danceability=song.danceability,
        valence=song.valence,
        acousticness=song.acousticness,
        instrumentalness=song.instrumentalness,
        speechiness=song.speechiness,
        loudness=song.loudness,
        tempo=song.tempo,
        popularity=song.popularity,
    )


def get_song(db: Session, song_id: str) -> Song | None:
    return db.get(Song, song_id)


def get_songs_bulk(db: Session, song_ids: list[str]) -> dict[str, Song]:
    if not song_ids:
        return {}
    rows = db.execute(select(Song).where(Song.id.in_(song_ids))).scalars().all()
    return {s.id: s for s in rows}


def get_artist(db: Session, artist_id: str) -> Artist | None:
    return db.get(Artist, artist_id)


def artist_top_songs(db: Session, artist_id: str, limit: int = 20) -> list[Song]:
    """Top songs for an artist by play/like/save counts (popularity falls back to catalog order)."""
    from app.models.events import UserSongEvent

    counts = (
        select(UserSongEvent.song_id, func.count().label("c"))
        .join(Song, Song.id == UserSongEvent.song_id)
        .where(Song.artist_id == artist_id,
               UserSongEvent.event_type.in_(["play", "like", "save"]))
        .group_by(UserSongEvent.song_id)
        .order_by(func.count().desc())
        .limit(limit)
    )
    song_ids = [row[0] for row in db.execute(counts).all()]
    if song_ids:
        songs = get_songs_bulk(db, song_ids)
        ordered = [songs[sid] for sid in song_ids if sid in songs]
        if len(ordered) >= limit:
            return ordered
    else:
        ordered = []
    seen = {s.id for s in ordered}
    extra = db.execute(
        select(Song).where(Song.artist_id == artist_id, ~Song.id.in_(seen or {"-"})).limit(limit)
    ).scalars().all()
    return ordered + list(extra)


def artist_albums(db: Session, artist_id: str, limit: int = 20) -> list[Album]:
    return list(db.execute(
        select(Album).where(Album.artist_id == artist_id).limit(limit)).scalars().all())


def songs_dataframe(db: Session) -> pd.DataFrame:
    """Catalog as a DataFrame (for the ML user-profile service)."""
    rows = db.execute(select(Song)).scalars().all()
    records = []
    for song in rows:
        records.append({
            "song_id": song.id,
            "title": song.title,
            "artist_id": song.artist_id,
            "artist_name": song.artist.name if song.artist else "",
            "genres": [g.name for g in (song.genres or [])],
            "tags": list(song.tags or []),
            "energy": song.energy, "danceability": song.danceability, "valence": song.valence,
            "acousticness": song.acousticness, "instrumentalness": song.instrumentalness,
            "speechiness": song.speechiness, "loudness": song.loudness, "tempo": song.tempo,
        })
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["song_id", "title", "artist_id", "artist_name", "genres", "tags"])
    return df


def search_catalog(db: Session, query: str, limit: int = 10) -> dict:
    """ILIKE/LIKE search across songs, artists, albums, genres."""
    like = f"%{query}%"
    ilike = func.lower(Song.title).like(f"%{query.lower()}%")

    songs = db.execute(
        select(Song)
        .join(Artist, Artist.id == Song.artist_id)
        .where(or_(ilike, func.lower(Artist.name).like(f"%{query.lower()}%")))
        .limit(limit)
    ).scalars().unique().all()

    artists = db.execute(
        select(Artist).where(func.lower(Artist.name).like(f"%{query.lower()}%")).limit(limit)
    ).scalars().all()

    albums = db.execute(
        select(Album).where(func.lower(Album.title).like(f"%{query.lower()}%")).limit(limit)
    ).scalars().all()

    genres = db.execute(
        select(Genre).where(func.lower(Genre.name).like(like.lower())).limit(limit)
    ).scalars().all()

    return {
        "songs": [to_song_out(s) for s in songs],
        "artists": [ArtistOut.model_validate(a) for a in artists],
        "albums": [AlbumOut.model_validate(a) for a in albums],
        "genres": [GenreOut.model_validate(g) for g in genres],
    }
