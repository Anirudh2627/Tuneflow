"""Pydantic schemas for events, feedback and sessions."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.catalog import SongOut

EVENT_TYPES = ("play", "skip", "like", "dislike", "save")


class EventCreate(BaseModel):
    user_id: str
    song_id: str
    event_type: str = Field(pattern="^(play|skip|like|dislike|save)$")
    session_id: str | None = None
    timestamp: datetime | None = None


class EventBatchCreate(BaseModel):
    events: list[EventCreate] = Field(min_length=1, max_length=500)


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    song_id: str
    event_type: str
    timestamp: datetime
    session_id: str | None = None
    weight: float


class FeedbackCreate(BaseModel):
    user_id: str
    song_id: str
    feedback: str = Field(pattern="^(play|skip|like|dislike|save)$")
    session_id: str | None = None


class SessionCreate(BaseModel):
    user_id: str


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    ended: bool
    song_ids: list[str] = []
    events: list[EventOut] = []


class HistoryItemOut(BaseModel):
    """A listening-history entry with the full song payload for rendering."""

    id: str
    event_type: str
    timestamp: datetime
    session_id: str | None = None
    weight: float
    song: SongOut
