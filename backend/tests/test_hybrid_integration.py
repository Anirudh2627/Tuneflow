"""Integration test: full hybrid engine served through the API.

Skipped when trained artifacts / processed data are absent -- run the
sample pipeline first (see README quickstart).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.utils.artifacts import artifact_exists  # noqa: E402
from ml.utils.paths import PROCESSED_DATA_DIR, SAMPLE_DATA_DIR  # noqa: E402

READY = (
    artifact_exists("content") and artifact_exists("collaborative")
    and artifact_exists("session") and (PROCESSED_DATA_DIR / "songs.parquet").exists()
    and (SAMPLE_DATA_DIR / "songs.csv").exists()
)

pytestmark = pytest.mark.skipif(not READY, reason="trained sample artifacts not present")


@pytest.fixture()
def hybrid_client(tmp_path, monkeypatch):
    db_path = tmp_path / "hybrid.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.delenv("TUNEFLOW_DISABLE_ENGINE", raising=False)

    from app.core.config import get_settings
    from app.db import session as db_session
    from app.db.base import Base
    from app.db.seed import seed_database
    from app.services.engine import reset_engine_holder

    get_settings.cache_clear()
    reset_engine_holder()
    db_session.init_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=db_session.engine)

    db = db_session.SessionLocal()
    try:
        seed_database(db, history_limit=200)
    finally:
        db.close()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
    reset_engine_holder()


def test_health_reports_hybrid(hybrid_client):
    body = hybrid_client.get("/health").json()
    assert body["engine"] == "hybrid"
    assert body["components"]["content"] is True


def test_personalized_recommendations_with_engine(hybrid_client):
    response = hybrid_client.get("/users/usr_demo/recommendations?limit=6&mode=personalized")
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid"
    assert len(body["items"]) == 6
    for item in body["items"]:
        assert item["reason"]
        assert item["signals"]
        assert item["title"] and item["artist"]
        assert item["score"] > 0


def test_discover_mode_differs_from_personalized(hybrid_client):
    personalized = hybrid_client.get("/users/usr_demo/recommendations?limit=10").json()["items"]
    discover = hybrid_client.get("/users/usr_demo/recommendations?limit=10&mode=discover").json()["items"]
    assert {i["song_id"] for i in personalized} != {i["song_id"] for i in discover}


def test_feedback_then_recommendations(hybrid_client):
    user = hybrid_client.post("/users", json={"display_name": "Fresh"}).json()
    for song, fb in (("so_000010", "like"), ("so_000011", "like"), ("so_000012", "skip")):
        response = hybrid_client.post("/feedback", json={
            "user_id": user["id"], "song_id": song, "feedback": fb})
        assert response.status_code == 201
    recs = hybrid_client.get(f"/users/{user['id']}/recommendations?limit=5").json()["items"]
    assert len(recs) > 0
    assert all(r["song_id"] not in {"so_000010", "so_000011", "so_000012"} for r in recs)
