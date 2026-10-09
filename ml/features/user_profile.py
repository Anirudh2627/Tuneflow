"""User profile derivation.

Turns raw interaction history + song metadata into the structured
profile consumed by the hybrid recommender, ranker, explanations and
the API ``/users/{id}/profile`` endpoint.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.features.song_features import AUDIO_COLUMNS


@dataclass
class UserProfile:
    user_id: str
    genre_affinity: dict[str, float] = field(default_factory=dict)
    artist_affinity: dict[str, float] = field(default_factory=dict)
    tag_affinity: dict[str, float] = field(default_factory=dict)
    liked_songs: list[str] = field(default_factory=list)
    disliked_songs: list[str] = field(default_factory=list)
    saved_songs: list[str] = field(default_factory=list)
    history_songs: list[str] = field(default_factory=list)          # chronological
    recent_songs: list[str] = field(default_factory=list)           # most recent first
    recent_sessions: dict[str, list[str]] = field(default_factory=dict)
    audio_feature_means: dict[str, float] = field(default_factory=dict)
    listening_stats: dict[str, float] = field(default_factory=dict)

    @property
    def is_cold_start(self) -> bool:
        return len(self.history_songs) == 0

    @property
    def top_genres(self) -> list[str]:
        return [g for g, _ in sorted(self.genre_affinity.items(), key=lambda kv: -kv[1])[:5]]

    @property
    def top_artists(self) -> list[str]:
        return [a for a, _ in sorted(self.artist_affinity.items(), key=lambda kv: -kv[1])[:5]]


def _normalize(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


class UserProfileService:
    """Derives :class:`UserProfile` objects from interaction history."""

    def __init__(self, songs: pd.DataFrame, affinity_events: dict[str, float] | None = None,
                 recent_limit: int = 50, n_recent_sessions: int = 3):
        self.songs = songs.set_index("song_id") if "song_id" in songs.columns else songs
        # extra boost applied when counting affinity (LIKE counts more than PLAY)
        self.affinity_events = affinity_events or {"play": 1.0, "like": 2.5, "save": 2.0, "skip": -0.5, "dislike": -1.5}
        self.recent_limit = recent_limit
        self.n_recent_sessions = n_recent_sessions

    def build(self, user_id: str, interactions: pd.DataFrame) -> UserProfile:
        profile = UserProfile(user_id=user_id)
        if interactions is None or len(interactions) == 0:
            return profile

        df = interactions.copy()
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            df = df.sort_values("timestamp", kind="mergesort")

        genre_counts: Counter = Counter()
        artist_counts: Counter = Counter()
        tag_counts: Counter = Counter()
        audio_sums: dict[str, list[float]] = {c: [] for c in AUDIO_COLUMNS}

        for row in df.itertuples(index=False):
            song = self.songs.loc[row.song_id] if row.song_id in self.songs.index else None
            boost = self.affinity_events.get(row.event_type, 1.0)
            if song is not None and boost > 0:
                for g in (song.get("genres") or []):
                    genre_counts[g] += boost
                for t in (song.get("tags") or []):
                    tag_counts[t] += boost
                artist_counts[str(song.get("artist_id", ""))] += boost
                for c in AUDIO_COLUMNS:
                    value = song.get(c)
                    if value is not None and not pd.isna(value):
                        audio_sums[c].append(float(value))

            event = row.event_type
            if event == "like":
                profile.liked_songs.append(row.song_id)
            elif event == "dislike":
                profile.disliked_songs.append(row.song_id)
            elif event == "save":
                profile.saved_songs.append(row.song_id)
            if event in ("play", "like", "save"):
                profile.history_songs.append(row.song_id)

        profile.genre_affinity = _normalize(Counter({k: v for k, v in genre_counts.items() if v > 0}))
        profile.artist_affinity = _normalize(Counter({k: v for k, v in artist_counts.items() if v > 0 and k}))
        profile.tag_affinity = _normalize(Counter({k: v for k, v in tag_counts.items() if v > 0}))
        profile.audio_feature_means = {c: float(np.mean(v)) for c, v in audio_sums.items() if v}

        profile.recent_songs = list(dict.fromkeys(reversed(profile.history_songs)))[: self.recent_limit]
        if "session_id" in df.columns:
            sessions: dict[str, list[str]] = {}
            for row in df.itertuples(index=False):
                if row.event_type in ("play", "like", "save") and getattr(row, "session_id", None):
                    sessions.setdefault(str(row.session_id), []).append(row.song_id)
            ordered = list(sessions.items())[-self.n_recent_sessions:]
            profile.recent_sessions = {k: v for k, v in ordered}

        n_pos = len(profile.history_songs)
        profile.listening_stats = {
            "total_events": float(len(df)),
            "positive_events": float(n_pos),
            "likes": float(len(profile.liked_songs)),
            "dislikes": float(len(profile.disliked_songs)),
            "saves": float(len(profile.saved_songs)),
            "unique_songs": float(len(set(profile.history_songs))),
            "unique_artists": float(len([a for a in profile.artist_affinity])),
        }
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            ts = pd.to_datetime(df["timestamp"])
            profile.listening_stats["days_active"] = float((ts.max() - ts.min()).days + 1)
            profile.listening_stats["last_active"] = float(pd.Timestamp(ts.max()).timestamp())
        return profile
