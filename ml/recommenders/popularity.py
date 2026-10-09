"""Popularity recommender.

Supports global popularity, recent popularity (trailing window) and
per-genre popularity.  Used for cold-start users, fallbacks and the
exploration candidate pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.recommenders.base import BaseRecommender
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class PopularityRecommender(BaseRecommender):
    name = "popularity"
    family = "collaborative"  # stored under models/collaborative/popularity

    def __init__(self, recent_window_days: int = 30):
        self.recent_window_days = recent_window_days
        self.global_popularity: pd.Series = pd.Series(dtype=float)
        self.recent_popularity: pd.Series = pd.Series(dtype=float)
        self.genre_popularity: dict[str, pd.Series] = {}
        self.catalog_song_ids: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, interactions: pd.DataFrame, songs: pd.DataFrame) -> "PopularityRecommender":
        positive = interactions[interactions["weight"] > 0] if "weight" in interactions.columns else interactions
        weights = positive.get("weight", pd.Series(1.0, index=positive.index)).astype(float)
        self.global_popularity = (
            positive.assign(w=weights).groupby("song_id")["w"].sum().sort_values(ascending=False)
        )
        self.catalog_song_ids = list(songs["song_id"].astype(str))

        self.recent_popularity = pd.Series(dtype=float)
        if "timestamp" in positive.columns and positive["timestamp"].notna().any():
            ts = pd.to_datetime(positive["timestamp"])
            cutoff = ts.max() - pd.Timedelta(days=self.recent_window_days)
            recent = positive[ts >= cutoff].assign(w=weights[ts >= cutoff])
            self.recent_popularity = recent.groupby("song_id")["w"].sum().sort_values(ascending=False)

        self.genre_popularity = {}
        if "genres" in songs.columns:
            song_genres = songs.set_index("song_id")["genres"].to_dict()
            exploded_rows = []
            for song_id, weight in self.global_popularity.items():
                for genre in song_genres.get(song_id, []) or []:
                    exploded_rows.append((genre, song_id, weight))
            if exploded_rows:
                exploded = pd.DataFrame(exploded_rows, columns=["genre", "song_id", "w"])
                self.genre_popularity = {
                    genre: grp.set_index("song_id")["w"].sort_values(ascending=False)
                    for genre, grp in exploded.groupby("genre")
                }

        self._fitted = True
        logger.info(
            "PopularityRecommender fitted: %d songs, %d genres, recent window=%dd",
            len(self.global_popularity), len(self.genre_popularity), self.recent_window_days,
        )
        return self

    # ------------------------------------------------------------------
    def top(self, k: int = 10, genre: str | None = None, recent: bool = False,
            exclude: set[str] | None = None) -> pd.Series:
        exclude = exclude or set()
        if genre:
            source = self.genre_popularity.get(genre.lower(), self.global_popularity)
        elif recent and len(self.recent_popularity):
            source = self.recent_popularity
        else:
            source = self.global_popularity
        source = source[~source.index.isin(exclude)]
        return source.head(k)

    def scores(self, song_ids: list[str] | pd.Index, genre: str | None = None,
               recent: bool = False) -> pd.Series:
        """Raw popularity score for arbitrary song ids (0 when unknown)."""
        if genre:
            source = self.genre_popularity.get(genre.lower(), self.global_popularity)
        elif recent and len(self.recent_popularity):
            source = self.recent_popularity
        else:
            source = self.global_popularity
        return source.reindex(song_ids).fillna(0.0)

    def recommend_for_user(self, user_id: str, k: int = 10, exclude: set[str] | None = None,
                           genre: str | None = None, recent: bool = False) -> pd.Series:
        return self.top(k=k, genre=genre, recent=recent, exclude=exclude)

    def popular_genres(self) -> list[str]:
        return sorted(self.genre_popularity)
