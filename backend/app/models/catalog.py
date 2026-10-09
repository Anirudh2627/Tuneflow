"""Catalog ORM models: artists, albums, genres, songs."""
from __future__ import annotations

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Table, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

song_genres = Table(
    "song_genres",
    Base.metadata,
    Column("song_id", String(64), ForeignKey("songs.id", ondelete="CASCADE"), primary_key=True),
    Column("genre_id", String(64), ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True),
)


class Artist(Base):
    __tablename__ = "artists"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    genres: Mapped[list] = mapped_column(JSON, default=list)
    followers: Mapped[float] = mapped_column(Float, default=0.0)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    songs: Mapped[list["Song"]] = relationship(back_populates="artist", lazy="noload")
    albums: Mapped[list["Album"]] = relationship(back_populates="artist", lazy="noload")


class Album(Base):
    __tablename__ = "albums"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    artist_id: Mapped[str] = mapped_column(ForeignKey("artists.id", ondelete="CASCADE"), index=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    artist: Mapped[Artist] = relationship(back_populates="albums")
    songs: Mapped[list["Song"]] = relationship(back_populates="album", lazy="noload")


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)   # slug, e.g. "hip-hop"
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)

    songs: Mapped[list["Song"]] = relationship(
        secondary=song_genres, back_populates="genres", lazy="noload")


class Song(Base):
    __tablename__ = "songs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    artist_id: Mapped[str] = mapped_column(ForeignKey("artists.id", ondelete="CASCADE"), index=True)
    album_id: Mapped[str | None] = mapped_column(ForeignKey("albums.id", ondelete="SET NULL"), nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    preview_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artwork_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # audio features (nullable: datasets without them degrade gracefully)
    energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    danceability: Mapped[float | None] = mapped_column(Float, nullable=True)
    valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    acousticness: Mapped[float | None] = mapped_column(Float, nullable=True)
    instrumentalness: Mapped[float | None] = mapped_column(Float, nullable=True)
    speechiness: Mapped[float | None] = mapped_column(Float, nullable=True)
    loudness: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo: Mapped[float | None] = mapped_column(Float, nullable=True)
    popularity: Mapped[float | None] = mapped_column(Float, nullable=True)

    artist: Mapped[Artist] = relationship(back_populates="songs", lazy="joined")
    album: Mapped[Album | None] = relationship(back_populates="songs", lazy="joined")
    genres: Mapped[list[Genre]] = relationship(
        secondary=song_genres, back_populates="songs", lazy="selectin")
