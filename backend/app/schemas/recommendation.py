"""Pydantic schemas for recommendations."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RecommendedSongOut(BaseModel):
    song_id: str
    title: str
    artist: str
    artist_id: str
    album: str | None = None
    genres: list[str] = []
    artwork_url: str | None = None
    preview_url: str | None = None
    duration_ms: float | None = None
    score: float
    reason: str
    signals: list[str] = []


class RecommendationsOut(BaseModel):
    user_id: str
    mode: str
    generated_at: datetime
    strategy: str = "hybrid"          # hybrid | popularity-fallback
    items: list[RecommendedSongOut] = []


class SimilarSongsOut(BaseModel):
    song_id: str
    items: list[RecommendedSongOut] = []
