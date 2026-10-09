"""Interaction ORM models: sessions, events, served recommendations."""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.user import utcnow


class EventType(str, enum.Enum):
    PLAY = "play"
    SKIP = "skip"
    LIKE = "like"
    DISLIKE = "dislike"
    SAVE = "save"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended: Mapped[bool] = mapped_column(default=False)

    user: Mapped[User] = relationship(back_populates="sessions")  # noqa: F821
    events: Mapped[list["UserSongEvent"]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
        order_by="UserSongEvent.timestamp", lazy="selectin")


class UserSongEvent(Base):
    __tablename__ = "user_song_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    song_id: Mapped[str] = mapped_column(ForeignKey("songs.id", ondelete="CASCADE"), index=True)
    # stored as plain string (see EventType enum for valid values) so the
    # schema is identical across PostgreSQL and SQLite
    event_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("sessions.id", ondelete="SET NULL"),
                                                   nullable=True, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    user: Mapped[User] = relationship(back_populates="events")  # noqa: F821
    song: Mapped[Song] = relationship(lazy="joined")            # noqa: F821
    session: Mapped[Session | None] = relationship(back_populates="events")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    mode: Mapped[str] = mapped_column(String(32), default="personalized")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    # ordered payload: [{song_id, score, reason, signals}, ...]
    items: Mapped[list] = mapped_column(JSON, default=list)
    context: Mapped[dict] = mapped_column(JSON, default=dict)

    user: Mapped[User] = relationship(back_populates="recommendations")  # noqa: F821
