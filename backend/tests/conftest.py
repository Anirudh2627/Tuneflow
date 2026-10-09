"""Backend API test fixtures: in-memory-style SQLite file + tiny seeded catalog."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
for p in (str(BACKEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh SQLite database + seeded tiny catalog per test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # hermetic tests: skip loading on-disk ML artifacts, use DB fallback
    monkeypatch.setenv("TUNEFLOW_DISABLE_ENGINE", "1")

    # reload settings-dependent singletons with the test database
    from app.core.config import get_settings
    from app.db import session as db_session
    from app.services.engine import reset_engine_holder

    get_settings.cache_clear()
    reset_engine_holder()
    db_session.init_engine(f"sqlite:///{db_path}")

    from app.db.base import Base
    from app.db.seed import DEMO_USER_ID
    from app.models.catalog import Artist, Genre, Song
    from app.models.user import User

    Base.metadata.create_all(bind=db_session.engine)
    db = db_session.SessionLocal()
    try:
        _seed_tiny_catalog(db)
        db.add(User(id=DEMO_USER_ID, display_name="Demo Listener",
                    preferred_genres=["electronic"], created_at=datetime.now(timezone.utc)))
        db.add(User(id="usr_alice", display_name="Alice", preferred_genres=[],
                    created_at=datetime.now(timezone.utc)))
        db.commit()
    finally:
        db.close()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
    reset_engine_holder()


def _seed_tiny_catalog(db) -> None:
    from app.models.catalog import Artist, Genre, Song

    artists = [
        Artist(id="ar1", name="Pulse Machine", genres=["electronic"], followers=5000),
        Artist(id="ar2", name="Blue Note Trio", genres=["jazz"], followers=1200),
    ]
    genres = [Genre(id="electronic", name="electronic"), Genre(id="jazz", name="jazz")]
    db.add_all(artists + genres)
    db.flush()

    songs = []
    for i in range(12):
        artist = artists[0] if i < 8 else artists[1]
        genre = genres[0] if i < 8 else genres[1]
        song = Song(
            id=f"s{i:02d}", title=f"Track {i}", artist_id=artist.id,
            release_year=2000 + i, duration_ms=200000,
            energy=0.9 - i * 0.05, valence=0.6, tempo=120, popularity=float(100 - i),
            preview_url=None, artwork_url=None,
        )
        song.genres = [genre]
        songs.append(song)
    db.add_all(songs)
    db.flush()


@pytest.fixture()
def demo_events(client) -> None:
    """Give the demo user a small listening history via the public API."""
    for i, (song, event) in enumerate([
        ("s00", "play"), ("s01", "like"), ("s02", "play"), ("s03", "save"),
        ("s09", "play"), ("s10", "dislike"), ("s04", "skip"),
    ]):
        response = client.post("/events", json={
            "user_id": "usr_demo", "song_id": song, "event_type": event})
        assert response.status_code == 201, response.text
