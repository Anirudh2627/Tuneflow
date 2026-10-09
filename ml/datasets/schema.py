"""Canonical dataset schemas.

All dataset adapters must emit these three DataFrames so the rest of
the pipeline never depends on a specific source dataset.

songs
    song_id (str, required)
    title (str)
    artist_id (str)
    artist_name (str)
    album (str)
    release_year (int, nullable)
    duration_ms (float, nullable)
    genres (list[str], may be empty)
    tags (list[str], may be empty)
    preview_url (str, nullable)
    artwork_url (str, nullable)
    Audio features (all nullable): energy, danceability, valence,
    acousticness, instrumentalness, speechiness, loudness, tempo,
    popularity

artists
    artist_id (str), name (str), genres (list[str]), followers (float)

interactions
    user_id (str), song_id (str), event_type (str),
    timestamp (datetime, nullable), session_id (str, nullable),
    weight (float, derived from event_type by the pipeline)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SONG_COLUMNS = [
    "song_id", "title", "artist_id", "artist_name", "album", "release_year",
    "duration_ms", "genres", "tags", "preview_url", "artwork_url",
    "energy", "danceability", "valence", "acousticness", "instrumentalness",
    "speechiness", "loudness", "tempo", "popularity",
]

ARTIST_COLUMNS = ["artist_id", "name", "genres", "followers"]

INTERACTION_COLUMNS = ["user_id", "song_id", "event_type", "timestamp", "session_id"]

AUDIO_FEATURE_COLUMNS = [
    "energy", "danceability", "valence", "acousticness", "instrumentalness",
    "speechiness", "loudness", "tempo",
]

VALID_EVENT_TYPES = {"play", "like", "save", "skip", "dislike"}


@dataclass
class Dataset:
    """Validated raw dataset container."""

    songs: pd.DataFrame
    artists: pd.DataFrame
    interactions: pd.DataFrame
    name: str = "unknown"
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_songs": int(self.songs["song_id"].nunique()),
            "n_artists": int(self.artists["artist_id"].nunique()) if not self.artists.empty else 0,
            "n_users": int(self.interactions["user_id"].nunique()),
            "n_interactions": int(len(self.interactions)),
            **self.stats,
        }


class SchemaValidationError(ValueError):
    """Raised when a dataset does not match the canonical schema."""


def validate_dataset(dataset: Dataset) -> Dataset:
    """Validate canonical schemas and enforce key invariants."""
    songs, artists, interactions = dataset.songs, dataset.artists, dataset.interactions

    missing = [c for c in ("song_id", "title", "artist_id", "artist_name") if c not in songs.columns]
    if missing:
        raise SchemaValidationError(f"songs missing required columns: {missing}")
    missing = [c for c in ("user_id", "song_id", "event_type") if c not in interactions.columns]
    if missing:
        raise SchemaValidationError(f"interactions missing required columns: {missing}")
    if songs["song_id"].duplicated().any():
        raise SchemaValidationError("songs.song_id contains duplicates")
    if artists is not None and not artists.empty and artists["artist_id"].duplicated().any():
        raise SchemaValidationError("artists.artist_id contains duplicates")

    invalid_events = set(interactions["event_type"].unique()) - VALID_EVENT_TYPES
    if invalid_events:
        raise SchemaValidationError(f"invalid event_type values: {invalid_events}")

    known_songs = set(songs["song_id"])
    orphan = (~interactions["song_id"].isin(known_songs)).sum()
    if orphan:
        raise SchemaValidationError(f"{orphan} interactions reference unknown songs")

    for column in SONG_COLUMNS:
        if column not in songs.columns:
            songs[column] = np.nan if column not in ("genres", "tags", "preview_url", "artwork_url", "album") else None
    for column in ARTIST_COLUMNS:
        if column not in artists.columns:
            artists[column] = np.nan if column != "genres" else None
    for column in INTERACTION_COLUMNS:
        if column not in interactions.columns:
            interactions[column] = None

    songs["genres"] = songs["genres"].apply(lambda g: [] if g is None else list(g))
    songs["tags"] = songs["tags"].apply(lambda t: [] if t is None else list(t))
    return dataset
