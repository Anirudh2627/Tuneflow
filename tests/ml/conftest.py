"""Shared fixtures for ML tests -- tiny synthetic data, no downloads."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tiny_songs() -> pd.DataFrame:
    """12 songs across 3 artists and 3 genres with audio features."""
    rows = []
    spec = [
        # (artist, genre, energy, valence, tempo)
        ("ar1", "electronic", 0.9, 0.7, 128),
        ("ar1", "electronic", 0.85, 0.6, 126),
        ("ar1", "electronic", 0.8, 0.65, 130),
        ("ar1", "electronic", 0.88, 0.72, 124),
        ("ar2", "jazz", 0.3, 0.5, 110),
        ("ar2", "jazz", 0.35, 0.45, 105),
        ("ar2", "jazz", 0.28, 0.55, 112),
        ("ar2", "jazz", 0.33, 0.48, 108),
        ("ar3", "classical", 0.15, 0.3, 90),
        ("ar3", "classical", 0.12, 0.35, 85),
        ("ar3", "classical", 0.18, 0.28, 95),
        ("ar3", "classical", 0.1, 0.4, 80),
    ]
    for i, (artist, genre, energy, valence, tempo) in enumerate(spec):
        rows.append({
            "song_id": f"s{i:02d}",
            "title": f"Track {i}",
            "artist_id": artist,
            "artist_name": {"ar1": "Pulse Machine", "ar2": "Blue Note Trio", "ar3": "Strings Quartet"}[artist],
            "album": f"Album {i % 3}",
            "release_year": 2000 + i,
            "duration_ms": 200000 + i * 5000,
            "genres": [genre],
            "tags": ["party"] if genre == "electronic" else ["chill"],
            "preview_url": None,
            "artwork_url": None,
            "energy": energy,
            "danceability": energy * 0.9,
            "valence": valence,
            "acousticness": 1 - energy,
            "instrumentalness": 0.5 if genre != "electronic" else 0.2,
            "speechiness": 0.05,
            "loudness": -6 - (1 - energy) * 12,
            "tempo": tempo,
            "popularity": float(100 - i * 5),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_artists() -> pd.DataFrame:
    return pd.DataFrame([
        {"artist_id": "ar1", "name": "Pulse Machine", "genres": ["electronic"], "followers": 50000.0},
        {"artist_id": "ar2", "name": "Blue Note Trio", "genres": ["jazz"], "followers": 12000.0},
        {"artist_id": "ar3", "name": "Strings Quartet", "genres": ["classical"], "followers": 8000.0},
    ])


@pytest.fixture(scope="session")
def tiny_interactions() -> pd.DataFrame:
    """6 users with chronological, sessionized listening histories."""
    rng = np.random.default_rng(7)
    base = datetime(2025, 1, 1)
    rows = []
    # user -> (preferred songs, sessions)
    plan = {
        "u1": (["s00", "s01", "s02", "s03"], 4),   # electronic fan
        "u2": (["s00", "s01", "s03"], 3),
        "u3": (["s04", "s05", "s06"], 4),          # jazz fan
        "u4": (["s04", "s06", "s07"], 3),
        "u5": (["s08", "s09", "s10"], 3),          # classical fan
        "u6": (["s00", "s04", "s08"], 4),          # eclectic
    }
    for user, (songs, n_sessions) in plan.items():
        t = base
        for s in range(n_sessions):
            session_id = f"{user}_ses{s}"
            for j in range(rng.integers(3, 6)):
                song = songs[int(rng.integers(0, len(songs)))]
                r = rng.random()
                event = "like" if r < 0.2 else ("skip" if r < 0.3 else "play")
                rows.append({
                    "user_id": user, "song_id": song, "event_type": event,
                    "timestamp": t, "session_id": session_id,
                })
                t += timedelta(minutes=4)
            t += timedelta(hours=8)
    weights = {"play": 1.0, "like": 3.0, "save": 2.0, "skip": -0.5, "dislike": -2.0}
    df = pd.DataFrame(rows)
    df["weight"] = df["event_type"].map(weights)
    return df


@pytest.fixture(scope="session")
def event_weights() -> dict[str, float]:
    return {"play": 1.0, "like": 3.0, "save": 2.0, "skip": -0.5, "dislike": -2.0}


@pytest.fixture(scope="session")
def sample_config(tmp_path_factory) -> "Config":  # noqa: F821
    from ml.utils.config import Config

    return Config.load("default.yaml")
