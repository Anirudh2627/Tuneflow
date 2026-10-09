"""Million Song Dataset adapter (taste_profile triplets + unique tracks).

Expected raw layout under ``data/raw/msd``::

    train_triplets.txt(.gz)   user_id <TAB> song_id <TAB> play_count
    unique_tracks.txt(.gz)    track_id <SEP> song_id <SEP> artist <SEP> title <SEP> year

The files are downloaded by ``scripts/download_data.py`` (or can be
placed manually -- see README).  MSD has no audio features or genres;
the pipeline handles that gracefully (content model falls back to
artist/title/year text features).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets.adapters.base import BaseAdapter
from ml.datasets.schema import Dataset
from ml.utils.logging import get_logger
from ml.utils.paths import RAW_DATA_DIR

logger = get_logger(__name__)

SEP = "<SEP>"


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else open(path, "r", encoding="utf-8", errors="replace")


class MsdAdapter(BaseAdapter):
    name = "msd"

    def __init__(self, raw_dir: Path | None = None, max_interactions: int | None = None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR / "msd"
        self.max_interactions = max_interactions

    def _find(self, stem: str) -> Path:
        for candidate in (self.raw_dir / f"{stem}.txt.gz", self.raw_dir / f"{stem}.txt"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Missing {stem}.txt(.gz) in {self.raw_dir}. Run `python scripts/download_data.py --source msd` "
            f"or download manually (see README)."
        )

    def load(self) -> Dataset:
        tracks_path = self._find("unique_tracks")
        triplets_path = self._find("train_triplets")

        logger.info("Loading MSD tracks from %s", tracks_path)
        tracks = pd.read_csv(
            tracks_path, sep=SEP, engine="python", quoting=3,
            names=["track_id", "song_id", "artist_name", "title", "release_year"],
            on_bad_lines="skip",
        )
        tracks = tracks[tracks["song_id"].notna()].copy()
        tracks["song_id"] = tracks["song_id"].astype(str).str.strip()
        tracks = tracks.drop_duplicates(subset="song_id")
        # Keep unknown years as NaN instead of 0
        tracks["release_year"] = pd.to_numeric(tracks["release_year"], errors="coerce").replace(0, np.nan)

        # Stable artist ids derived from normalized artist names
        artist_names = tracks["artist_name"].fillna("Unknown Artist").astype(str).str.strip()
        artist_ids = (
            pd.factorize(artist_names.str.lower())[0]
        )
        tracks["artist_id"] = [f"ar_{i:07d}" for i in artist_ids]
        artists = (
            tracks[["artist_id", "artist_name"]]
            .drop_duplicates("artist_id")
            .rename(columns={"artist_name": "name"})
            .reset_index(drop=True)
        )
        artists["genres"] = [[] for _ in range(len(artists))]
        artists["followers"] = (
            tracks.groupby("artist_id").size().reindex(artists["artist_id"]).fillna(0).to_numpy() * 100.0
        )

        songs = tracks[["song_id", "title", "artist_id", "artist_name", "release_year"]].copy()
        songs["album"] = None
        songs["duration_ms"] = np.nan
        songs["genres"] = [[] for _ in range(len(songs))]
        songs["tags"] = [[] for _ in range(len(songs))]
        songs["preview_url"] = None
        songs["artwork_url"] = None
        for col in ["energy", "danceability", "valence", "acousticness", "instrumentalness",
                    "speechiness", "loudness", "tempo"]:
            songs[col] = np.nan
        songs["popularity"] = np.nan

        logger.info("Loading MSD triplets from %s", triplets_path)
        interactions = pd.read_csv(
            triplets_path, sep="\t", names=["user_id", "song_id", "play_count"],
        )
        if self.max_interactions:
            interactions = interactions.head(self.max_interactions)
        interactions["song_id"] = interactions["song_id"].astype(str).str.strip()

        known = set(songs["song_id"])
        before = len(interactions)
        interactions = interactions[interactions["song_id"].isin(known)]
        logger.info("Dropped %d/%d triplets referencing unknown tracks", before - len(interactions), before)

        # Expand play counts capped (a play_count of N -> N implicit plays is
        # wasteful; cap to keep the matrix manageable, confidence in CF uses it)
        interactions["play_count"] = interactions["play_count"].clip(upper=50)
        interactions["event_type"] = np.where(interactions["play_count"] >= 3, "like", "play")
        interactions["timestamp"] = pd.NaT
        interactions["session_id"] = None

        return Dataset(
            songs=songs.reset_index(drop=True),
            artists=artists,
            interactions=interactions.reset_index(drop=True),
            name="msd",
            stats={"dropped_unknown_track_triplets": int(before - len(interactions))},
        )
