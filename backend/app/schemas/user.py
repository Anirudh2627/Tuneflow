"""Pydantic schemas for users and profiles."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    email: str | None = Field(default=None, max_length=255)
    preferred_genres: list[str] = Field(default_factory=list, max_length=10)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    email: str | None = None
    preferred_genres: list[str] = []
    created_at: datetime
    last_active_at: datetime | None = None


class UserProfileOut(BaseModel):
    user_id: str
    display_name: str
    top_genres: list[str] = []
    genre_affinity: dict[str, float] = {}
    top_artists: list[dict] = []           # [{artist_id, name, affinity}]
    tag_affinity: dict[str, float] = {}
    audio_feature_means: dict[str, float] = {}
    liked_songs: list[str] = []
    disliked_songs: list[str] = []
    saved_songs: list[str] = []
    recent_songs: list[str] = []
    listening_stats: dict[str, float] = {}
    is_cold_start: bool = True
