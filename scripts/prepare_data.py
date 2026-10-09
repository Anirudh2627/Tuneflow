#!/usr/bin/env python3
"""Run the full preprocessing pipeline.

Usage::

    python scripts/prepare_data.py                              # uses configs/default.yaml
    python scripts/prepare_data.py --config configs/sample.yaml
    python scripts/prepare_data.py --source msd --config configs/msd.yaml
    python scripts/prepare_data.py --source csv \
        --csv-songs data/raw/my/songs.csv \
        --csv-artists data/raw/my/artists.csv \
        --csv-interactions data/raw/my/interactions.csv

Outputs go to ``data/processed`` (songs, artists, interactions, train,
val, test, stats.json).
"""
from __future__ import annotations

import argparse
import sys

import bootstrap  # noqa: F401

from ml.datasets import get_adapter
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import ensure_dirs

logger = get_logger("prepare_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="default.yaml", help="YAML config (name in configs/ or path)")
    parser.add_argument("--override-config", default=None, help="second YAML merged on top of --config")
    parser.add_argument("--source", default=None, help="dataset source override (sample|msd|csv)")
    parser.add_argument("--csv-songs", default=None)
    parser.add_argument("--csv-artists", default=None)
    parser.add_argument("--csv-interactions", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = Config.load(args.config, args.override_config)
    setup_logging(config.get("project.log_level", "INFO"))
    ensure_dirs()

    adapter_kwargs = dict(config.section("data.adapter_kwargs", {}) or {})
    if args.source == "csv":
        if not (args.csv_songs and args.csv_interactions):
            parser.error("--source csv requires --csv-songs and --csv-interactions")
        adapter_kwargs = {
            "songs_path": args.csv_songs,
            "interactions_path": args.csv_interactions,
            "artists_path": args.csv_artists,
        }

    pipeline = PreprocessingPipeline(config, output_dir=args.output_dir)
    adapter = get_adapter(args.source or config.get("data.source", "sample"), **adapter_kwargs)
    dataset = adapter.load_validated()
    tables = pipeline.run(dataset=dataset)

    logger.info("Preprocessing complete. Tables: %s", {k: len(v) for k, v in tables.items()})
    logger.info("Next: python scripts/train_content.py --config %s", args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
