"""User ORM model."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    # onboarding preferences used for cold-start recommendations
    preferred_genres: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["UserSongEvent"]] = relationship(          # noqa: F821
        back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    sessions: Mapped[list["Session"]] = relationship(               # noqa: F821
        back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    recommendations: Mapped[list["Recommendation"]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan", lazy="noload")

    def __repr__(self) -> str:
        return f"<User id={self.id!r} name={self.display_name!r}>"
