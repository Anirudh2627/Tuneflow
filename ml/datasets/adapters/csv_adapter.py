"""Generic CSV adapter.

Lets you plug in ANY dataset (e.g. a Kaggle export) without touching
source code.  Point it at three CSVs matching the canonical schema
(see ml/datasets/schema.py).  List-valued columns (genres, tags) use
pipe-separated strings, e.g. ``rock|pop``.

Example::

    python scripts/prepare_data.py --source csv \
        --csv-songs data/raw/mydata/songs.csv \
        --csv-artists data/raw/mydata/artists.csv \
        --csv-interactions data/raw/mydata/interactions.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.datasets.adapters.base import BaseAdapter
from ml.datasets.schema import Dataset
from ml.utils.logging import get_logger

logger = get_logger(__name__)


def _split_list_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: [] if (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "")
                else [p.strip() for p in str(v).split("|") if p.strip()]
            )
        else:
            df[col] = [[] for _ in range(len(df))]
    return df


class CsvAdapter(BaseAdapter):
    name = "csv"

    def __init__(self, songs_path: str | Path, interactions_path: str | Path, artists_path: str | Path | None = None):
        self.songs_path = Path(songs_path)
        self.interactions_path = Path(interactions_path)
        self.artists_path = Path(artists_path) if artists_path else None

    def load(self) -> Dataset:
        logger.info("Loading songs from %s", self.songs_path)
        songs = _split_list_columns(pd.read_csv(self.songs_path), ("genres", "tags"))
        if "artist_name" not in songs.columns:
            songs["artist_name"] = songs.get("artist", "Unknown Artist")
        if "artist_id" not in songs.columns:
            songs["artist_id"] = [f"ar_{i:07d}" for i in pd.factorize(songs["artist_name"].str.lower())[0]]

        logger.info("Loading interactions from %s", self.interactions_path)
        interactions = pd.read_csv(self.interactions_path)
        if "timestamp" in interactions.columns:
            interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")
        if "event_type" not in interactions.columns:
            interactions["event_type"] = "play"
        interactions["event_type"] = interactions["event_type"].str.lower()

        if self.artists_path and self.artists_path.exists():
            artists = _split_list_columns(pd.read_csv(self.artists_path), ("genres",))
        else:
            artists = (
                songs[["artist_id", "artist_name"]]
                .drop_duplicates("artist_id")
                .rename(columns={"artist_name": "name"})
                .reset_index(drop=True)
            )
            artists["genres"] = [[] for _ in range(len(artists))]
            artists["followers"] = 0.0

        return Dataset(songs=songs, artists=artists, interactions=interactions, name="csv")
