"""End-to-end preprocessing pipeline.

raw Dataset -> clean -> filter -> split -> feature inputs -> processed
files under ``data/processed`` consumed by every training script.

Outputs (parquet when pyarrow is available, else csv):
    songs, artists, interactions (all weighted), train, val, test,
    stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets import get_adapter
from ml.datasets.schema import Dataset
from ml.preprocessing.cleaning import clean_dataset
from ml.preprocessing.filtering import compute_statistics, drop_negative_only_users, k_core_filter
from ml.preprocessing.splits import split_interactions
from ml.utils.config import Config
from ml.utils.logging import get_logger
from ml.utils.paths import PROCESSED_DATA_DIR

logger = get_logger(__name__)

TABLES = ("songs", "artists", "interactions", "train", "val", "test")


def write_table(df: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        path = directory / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError):
        path = directory / f"{name}.csv"
        out = df.copy()
        for col in out.columns:
            if out[col].apply(lambda v: isinstance(v, list)).any():
                out[col] = out[col].apply(lambda v: "|".join(map(str, v)) if isinstance(v, list) else v)
        out.to_csv(path, index=False)
        return path


def read_table(directory: Path, name: str) -> pd.DataFrame:
    parquet, csv = directory / f"{name}.parquet", directory / f"{name}.csv"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"No processed table {name!r} in {directory}. Run scripts/prepare_data.py first.")
    for col in ("genres", "tags"):
        if col in df.columns:
            def _to_list(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return []
                if isinstance(v, (list, tuple)):
                    return list(v)
                if isinstance(v, np.ndarray):
                    return v.tolist()
                return [p for p in str(v).split("|") if p]

            df[col] = df[col].apply(_to_list)
    return df


class PreprocessingPipeline:
    """Runs the full ingestion -> cleaning -> filtering -> split flow."""

    def __init__(self, config: Config, output_dir: Path | None = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR

    def load_dataset(self, source: str | None = None, adapter_kwargs: dict | None = None) -> Dataset:
        source = source or self.config.get("data.source", "sample")
        adapter_kwargs = {**(self.config.section("data.adapter_kwargs", {}) or {}), **(adapter_kwargs or {})}
        adapter = get_adapter(source, **adapter_kwargs)
        logger.info("Loading dataset via adapter %r", adapter.name)
        return adapter.load_validated()

    def run(self, dataset: Dataset | None = None, source: str | None = None,
            adapter_kwargs: dict | None = None) -> dict[str, pd.DataFrame]:
        if dataset is None:
            dataset = self.load_dataset(source, adapter_kwargs)
        logger.info("Raw dataset: %s", dataset.summary())

        event_weights = self.config.section("feedback.event_weights", {"play": 1.0, "like": 3.0, "save": 2.0, "skip": -0.5, "dislike": -2.0})
        songs, artists, interactions = clean_dataset(
            dataset.songs, dataset.artists, dataset.interactions, event_weights)

        filt = self.config.section("data.filtering", {})
        interactions = k_core_filter(
            interactions,
            min_user=int(filt.get("min_interactions_per_user", 2)),
            min_item=int(filt.get("min_interactions_per_song", 1)),
        )
        interactions = drop_negative_only_users(interactions)

        split_cfg = self.config.section("data.split", {})
        splits = split_interactions(
            interactions,
            train_frac=float(split_cfg.get("train_frac", 0.7)),
            val_frac=float(split_cfg.get("val_frac", 0.15)),
            seed=int(split_cfg.get("seed", 42)),
            min_events_for_split=int(split_cfg.get("min_events_for_split", 5)),
        )

        tables = {
            "songs": songs,
            "artists": artists,
            "interactions": interactions,
            **splits,
        }
        stats = compute_statistics(songs, artists, interactions)
        stats.update({
            "split_sizes": {k: int(len(v)) for k, v in splits.items()},
            "event_weights": event_weights,
        })

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            path = write_table(df, self.output_dir, name)
            logger.info("Wrote %-12s -> %s (%d rows)", name, path, len(df))
        with open(self.output_dir / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        logger.info("Dataset statistics: %s", {k: v for k, v in stats.items() if k != "top_genres"})
        return tables


def load_processed(output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all processed tables for training/inference scripts."""
    directory = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    return {name: read_table(directory, name) for name in TABLES}


def load_processed_stats(output_dir: Path | None = None) -> dict:
    directory = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    path = directory / "stats.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
