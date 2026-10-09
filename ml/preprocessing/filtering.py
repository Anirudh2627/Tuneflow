"""Interaction filtering (k-core) and user/item statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.utils.logging import get_logger

logger = get_logger(__name__)


def k_core_filter(interactions: pd.DataFrame, min_user: int = 1, min_item: int = 1,
                  max_iter: int = 10) -> pd.DataFrame:
    """Iteratively drop users/items with too few interactions."""
    df = interactions
    for i in range(max_iter):
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
        item_counts = df["song_id"].value_counts()
        df = df[df["song_id"].isin(item_counts[item_counts >= min_item].index)]
        if len(df) == len(interactions) or i == max_iter - 1:
            break
        interactions = df
    removed = len(interactions) - len(df) if len(interactions) > len(df) else 0
    if removed:
        logger.info("k_core_filter(min_user=%d, min_item=%d): removed %d interactions",
                    min_user, min_item, removed)
    return df.reset_index(drop=True)


def drop_negative_only_users(interactions: pd.DataFrame) -> pd.DataFrame:
    """Users with only skip/dislike carry no positive training signal."""
    pos = interactions[interactions["weight"] > 0]
    keep_users = set(pos["user_id"].unique())
    return interactions[interactions["user_id"].isin(keep_users)].reset_index(drop=True)


def compute_statistics(songs: pd.DataFrame, artists: pd.DataFrame, interactions: pd.DataFrame) -> dict:
    """User/item statistics used by EDA and reports."""
    n_users = int(interactions["user_id"].nunique())
    n_songs = int(interactions["song_id"].nunique())
    n_events = int(len(interactions))
    n_unique_pairs = int(interactions[["user_id", "song_id"]].drop_duplicates().shape[0])
    per_user = interactions.groupby("user_id").size()
    per_song = interactions.groupby("song_id").size()
    sparsity = 1.0 - (n_unique_pairs / max(n_users * int(songs["song_id"].nunique()), 1))
    stats = {
        "n_users": n_users,
        "n_songs_in_catalog": int(songs["song_id"].nunique()),
        "n_songs_with_interactions": n_songs,
        "n_artists": int(artists["artist_id"].nunique()) if not artists.empty else 0,
        "n_interactions": n_events,
        "sparsity": float(sparsity),
        "events_per_user_mean": float(per_user.mean()) if n_users else 0.0,
        "events_per_user_median": float(per_user.median()) if n_users else 0.0,
        "events_per_user_p95": float(per_user.quantile(0.95)) if n_users else 0.0,
        "events_per_song_mean": float(per_song.mean()) if n_songs else 0.0,
        "events_per_song_median": float(per_song.median()) if n_songs else 0.0,
        "event_type_distribution": interactions["event_type"].value_counts(normalize=True).round(4).to_dict(),
        "cold_users_le5": int((per_user <= 5).sum()),
        "cold_songs_le2": int((per_song <= 2).sum()),
    }
    if "genres" in songs.columns:
        all_genres = [g for gs in songs["genres"] for g in gs]
        genre_counts = pd.Series(all_genres).value_counts()
        stats["top_genres"] = genre_counts.head(20).to_dict()
        stats["n_genres"] = int(len(genre_counts))
    return stats
