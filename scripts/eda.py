#!/usr/bin/env python3
"""Exploratory data analysis.

Usage::

    python scripts/eda.py --config configs/sample.yaml

Produces dataset statistics (stdout + reports/eda_stats.json) and
figures under ``reports/figures`` (matplotlib).  The same functions are
reused by ``notebooks/01_eda.ipynb``.
"""
from __future__ import annotations

import argparse
import json
import sys

import bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ml.preprocessing.filtering import compute_statistics
from ml.preprocessing.pipeline import load_processed, load_processed_stats
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import FIGURES_DIR, REPORTS_DIR, ensure_dirs

logger = get_logger("eda")


def plot_all(songs: pd.DataFrame, artists: pd.DataFrame, interactions: pd.DataFrame) -> list[str]:
    """Generate every EDA figure; returns saved paths. Degrades without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping figures (stats still computed)")
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def _save(fig, name: str) -> None:
        path = FIGURES_DIR / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(str(path))
        logger.info("Saved %s", path)

    # 1. interactions per user
    per_user = interactions.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_user.clip(upper=per_user.quantile(0.99)), bins=50, color="#6366f1")
    ax.set_title("Interactions per user (99th pct clipped)")
    ax.set_xlabel("interactions"); ax.set_ylabel("users")
    _save(fig, "interactions_per_user.png")

    # 2. song popularity distribution
    per_song = interactions.groupby("song_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(per_song.clip(lower=1)), bins=40, color="#22c55e")
    ax.set_title("Song popularity (log10 interactions)")
    ax.set_xlabel("log10 interactions"); ax.set_ylabel("songs")
    _save(fig, "song_popularity.png")

    # 3. artist popularity (top 20)
    if "artist_id" in songs.columns and "artist_name" in songs.columns:
        song_artist = songs.set_index("song_id")["artist_id"]
        artist_names = songs.drop_duplicates("artist_id").set_index("artist_id")["artist_name"]
        artist_counts = interactions.groupby("song_id").size().groupby(song_artist).sum().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([artist_names.get(a, a)[:25] for a in artist_counts.index][::-1], artist_counts.to_numpy()[::-1], color="#f59e0b")
        ax.set_title("Top 20 artists by interactions")
        _save(fig, "artist_popularity.png")

    # 4. genre distribution
    if "genres" in songs.columns:
        exploded = songs.explode("genres")
        exploded = exploded[exploded["genres"].notna() & (exploded["genres"].astype(str) != "")]
        if not exploded.empty:
            genre_counts = exploded.groupby("genres").size().sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(genre_counts.index, genre_counts.to_numpy(), color="#0ea5e9")
            ax.set_title("Songs per genre (top 20)")
            ax.tick_params(axis="x", rotation=45)
            _save(fig, "genre_distribution.png")

    # 5. event type distribution
    if "event_type" in interactions.columns:
        counts = interactions["event_type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.to_numpy(), labels=counts.index, autopct="%1.1f%%",
               colors=["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#0ea5e9"][: len(counts)])
        ax.set_title("Event type distribution")
        _save(fig, "event_types.png")

    # 6. listening activity over time
    if "timestamp" in interactions.columns and interactions["timestamp"].notna().any():
        ts = pd.to_datetime(interactions["timestamp"]).dt.floor("D")
        daily = ts.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(daily.index, daily.to_numpy(), color="#6366f1", lw=1)
        ax.set_title("Daily listening activity")
        _save(fig, "listening_activity.png")

    # 7. audio feature distributions
    audio_cols = [c for c in ("energy", "danceability", "valence", "acousticness", "tempo")
                  if c in songs.columns and pd.to_numeric(songs[c], errors="coerce").notna().any()]
    if audio_cols:
        fig, axes = plt.subplots(1, len(audio_cols), figsize=(4 * len(audio_cols), 3.2))
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, audio_cols):
            ax.hist(pd.to_numeric(songs[col], errors="coerce").dropna(), bins=30, color="#a855f7")
            ax.set_title(col)
        _save(fig, "audio_features.png")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--from-raw", action="store_true",
                        help="recompute stats from raw adapter instead of processed tables")
    args = parser.parse_args()

    config = Config.load(args.config)
    setup_logging(config.get("project.log_level", "INFO"))
    ensure_dirs()

    try:
        tables = load_processed()
    except FileNotFoundError:
        logger.info("No processed data found -- running preprocessing first")
        from ml.preprocessing.pipeline import PreprocessingPipeline
        tables = PreprocessingPipeline(config).run()

    songs, artists, interactions = tables["songs"], tables["artists"], tables["interactions"]
    stats = compute_statistics(songs, artists, interactions)
    stats["splits"] = {k: int(len(tables[k])) for k in ("train", "val", "test")}

    print("\n=== TuneFlow EDA ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "eda_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    logger.info("Wrote %s", REPORTS_DIR / "eda_stats.json")

    figures = plot_all(songs, artists, interactions)
    logger.info("Generated %d figures under %s", len(figures), FIGURES_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
