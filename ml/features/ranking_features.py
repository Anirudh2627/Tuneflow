"""Ranking feature extraction.

Computes the feature frame consumed by rankers.  Kept out of FastAPI
routes: services call :class:`RankingFeatureExtractor` with a user
profile, candidate ids and per-source scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features.song_features import AUDIO_COLUMNS
from ml.features.user_profile import UserProfile

#: Canonical ranking feature columns (rankers see a stable schema)
RANKING_FEATURES = [
    "content_score",
    "collaborative_score",
    "session_score",
    "popularity_score",
    "genre_affinity",
    "artist_affinity",
    "tag_affinity",
    "audio_match",
    "song_popularity",
    "artist_popularity",
    "user_artist_interactions",
    "user_genre_interactions",
    "novelty",
    "release_recency",
]


class RankingFeatureExtractor:
    """Builds per-candidate ranking features from profile + scores + metadata."""

    def __init__(self, songs: pd.DataFrame, popularity: pd.Series | None = None,
                 artist_popularity: pd.Series | None = None):
        self.songs = songs.set_index("song_id", drop=False) if "song_id" in songs.columns else songs
        self.popularity = popularity
        self.artist_popularity = artist_popularity
        self._max_pop = float(popularity.max()) if popularity is not None and len(popularity) else 1.0
        self._max_artist_pop = float(artist_popularity.max()) if artist_popularity is not None and len(artist_popularity) else 1.0

    # ------------------------------------------------------------------
    def extract(
        self,
        profile: UserProfile | None,
        candidate_ids: list[str],
        source_scores: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by song_id with RANKING_FEATURES columns.

        ``source_scores`` maps a score-source name (``content``,
        ``collaborative``, ``session``, ``popularity``) to a Series of
        raw scores for the candidates it produced.
        """
        songs = self.songs.reindex(candidate_ids)
        n = len(candidate_ids)
        features = pd.DataFrame(index=pd.Index(candidate_ids, name="song_id"))

        # --- per-source scores (min-max normalized across candidates) ---
        for source, column in (("content", "content_score"), ("collaborative", "collaborative_score"),
                               ("session", "session_score"), ("popularity", "popularity_score")):
            series = source_scores.get(source)
            if series is None:
                features[column] = 0.0
                continue
            values = series.reindex(candidate_ids).fillna(0.0).to_numpy(dtype=float)
            features[column] = self._normalize(values)

        # --- affinity features from the user profile ---
        if profile is not None and not profile.is_cold_start:
            features["genre_affinity"] = [
                self._max_affinity(profile.genre_affinity, row.get("genres")) for row in songs.to_dict("records")
            ]
            features["artist_affinity"] = [
                float(profile.artist_affinity.get(str(row.get("artist_id", "")), 0.0)) for row in songs.to_dict("records")
            ]
            features["tag_affinity"] = [
                self._max_affinity(profile.tag_affinity, row.get("tags")) for row in songs.to_dict("records")
            ]
            features["audio_match"] = [
                self._audio_match(profile, row) for row in songs.to_dict("records")
            ]
            features["user_artist_interactions"] = [
                self._normalize_scalar(
                    float(profile.artist_affinity.get(str(row.get("artist_id", "")), 0.0)), 0.25)
                for row in songs.to_dict("records")
            ]
            features["user_genre_interactions"] = [
                self._normalize_scalar(
                    self._max_affinity(profile.genre_affinity, row.get("genres")), 0.4)
                for row in songs.to_dict("records")
            ]
        else:
            for col in ("genre_affinity", "artist_affinity", "tag_affinity", "audio_match",
                        "user_artist_interactions", "user_genre_interactions"):
                features[col] = 0.0

        # --- catalog-side features ---
        if self.popularity is not None:
            pop = self.popularity.reindex(candidate_ids).fillna(0.0).to_numpy(dtype=float)
            features["song_popularity"] = self._normalize(np.log1p(pop))
            features["novelty"] = 1.0 - features["song_popularity"]
        else:
            features["song_popularity"] = 0.0
            features["novelty"] = 0.5

        if self.artist_popularity is not None:
            apop = (
                songs["artist_id"].astype(str)
                .map(self.artist_popularity).fillna(0.0).to_numpy(dtype=float)
            )
            features["artist_popularity"] = self._normalize(np.log1p(apop))
        else:
            features["artist_popularity"] = 0.0

        if "release_year" in songs.columns:
            years = pd.to_numeric(songs["release_year"], errors="coerce")
            max_year = 2025.0
            features["release_recency"] = ((years.fillna(max_year - 10) - 1950) / (max_year - 1950)).clip(0, 1)
        else:
            features["release_recency"] = 0.5

        for col in RANKING_FEATURES:
            if col not in features.columns:
                features[col] = 0.0
        return features[RANKING_FEATURES]

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
        lo, hi = values.min(), values.max()
        if hi <= lo:
            return np.zeros_like(values) if hi == 0 else np.full_like(values, 0.5)
        return (values - lo) / (hi - lo)

    @staticmethod
    def _normalize_scalar(value: float, cap: float) -> float:
        return float(min(max(value / cap, 0.0), 1.0))

    @staticmethod
    def _max_affinity(affinity: dict[str, float], values) -> float:
        if not isinstance(values, (list, tuple)) or not values:
            return 0.0
        return max((affinity.get(str(v), 0.0) for v in values), default=0.0)

    @staticmethod
    def _audio_match(profile: UserProfile, song_row: dict) -> float:
        diffs = []
        for col in AUDIO_COLUMNS:
            target = profile.audio_feature_means.get(col)
            value = song_row.get(col)
            if target is None or value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            scale = 60.0 if col == "loudness" else (200.0 if col in ("tempo", "duration_ms") else 1.0)
            diffs.append(min(abs(float(value) - target) / scale, 1.0))
        if not diffs:
            return 0.0
        return float(1.0 - np.mean(diffs))
