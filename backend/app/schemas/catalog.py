"""Pydantic schemas for catalog objects (songs, artists, albums, genres)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GenreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str


class AlbumOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    artist_id: str
    release_year: int | None = None
    image_url: str | None = None


class ArtistOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    genres: list[str] = []
    followers: float = 0.0
    image_url: str | None = None


class SongOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    artist_id: str
    artist_name: str
    album: str | None = None
    album_id: str | None = None
    release_year: int | None = None
    duration_ms: float | None = None
    genres: list[str] = []
    tags: list[str] = []
    preview_url: str | None = None
    artwork_url: str | None = None
    energy: float | None = None
    danceability: float | None = None
    valence: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    speechiness: float | None = None
    loudness: float | None = None
    tempo: float | None = None
    popularity: float | None = None


class ArtistDetailOut(ArtistOut):
    top_songs: list[SongOut] = []
    albums: list[AlbumOut] = []


class SearchResultsOut(BaseModel):
    query: str
    songs: list[SongOut] = []
    artists: list[ArtistOut] = []
    albums: list[AlbumOut] = []
    genres: list[GenreOut] = []
