"""Cleaning: duplicates, missing values, normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.utils.logging import get_logger

logger = get_logger(__name__)


def dedupe_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (user, song, event_type, session) rows.

    Keeps the earliest timestamp per duplicate group. Exact duplicate
    rows are dropped unconditionally.
    """
    before = len(interactions)
    df = interactions.drop_duplicates()
    keys = ["user_id", "song_id", "event_type"]
    if "session_id" in df.columns:
        keys.append("session_id")
    sort_col = "timestamp" if "timestamp" in df.columns else keys[0]
    df = df.sort_values(sort_col, kind="mergesort")
    df = df.drop_duplicates(subset=keys, keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info("dedupe_interactions: removed %d duplicate rows", dropped)
    return df.reset_index(drop=True)


def normalize_song_metadata(songs: pd.DataFrame) -> pd.DataFrame:
    """Normalize song metadata columns and handle missing values."""
    df = songs.copy()
    df["title"] = df["title"].fillna("Unknown Title").astype(str).str.strip()
    df.loc[df["title"] == "", "title"] = "Unknown Title"
    df["artist_name"] = df["artist_name"].fillna("Unknown Artist").astype(str).str.strip()
    df["artist_id"] = df["artist_id"].fillna("").astype(str).str.strip()
    empty_artist = df["artist_id"] == ""
    if empty_artist.any():
        df.loc[empty_artist, "artist_id"] = [
            f"ar_unknown_{i:06d}" for i in range(int(empty_artist.sum()))
        ]
    for col in ("genres", "tags"):
        df[col] = df[col].apply(lambda v: [] if v is None or (isinstance(v, float) and np.isnan(v)) else list(v))
    df["genres"] = df["genres"].apply(lambda gs: [str(g).strip().lower() for g in gs if str(g).strip()])
    df["tags"] = df["tags"].apply(lambda ts: [str(t).strip().lower() for t in ts if str(t).strip()])
    if "duration_ms" in df.columns:
        dur = pd.to_numeric(df["duration_ms"], errors="coerce")
        df["duration_ms"] = dur.mask((dur <= 0) | (dur > 3.6e6))  # drop absurd values
    if "release_year" in df.columns:
        year = pd.to_numeric(df["release_year"], errors="coerce")
        df["release_year"] = year.mask((year < 1800) | (year > 2100))
    return df


def attach_event_weights(interactions: pd.DataFrame, event_weights: dict[str, float]) -> pd.DataFrame:
    """Add a numeric ``weight`` column from configurable event weights."""
    df = interactions.copy()
    unknown = set(df["event_type"].unique()) - set(event_weights)
    if unknown:
        logger.warning("Unknown event types %s defaulted to weight of 'play'", unknown)
    default = event_weights.get("play", 1.0)
    df["weight"] = df["event_type"].map(event_weights).fillna(default).astype(float)
    return df


def clean_dataset(songs: pd.DataFrame, artists: pd.DataFrame, interactions: pd.DataFrame,
                  event_weights: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full cleaning pass over a canonical dataset."""
    songs = normalize_song_metadata(songs)
    interactions = dedupe_interactions(interactions)
    interactions = attach_event_weights(interactions, event_weights)
    if "timestamp" in interactions.columns and interactions["timestamp"].notna().any():
        interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")
    artists = artists.copy()
    artists["name"] = artists["name"].fillna("Unknown Artist").astype(str)
    if "genres" in artists.columns:
        artists["genres"] = artists["genres"].apply(
            lambda v: [] if v is None or (isinstance(v, float) and np.isnan(v)) else list(v))
    return songs, artists, interactions
