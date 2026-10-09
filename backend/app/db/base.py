"""Declarative base shared by all ORM models (Alembic target)."""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
