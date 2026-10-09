#!/usr/bin/env python3
"""Train the collaborative filtering model (+ popularity tables).

Usage::

    python scripts/train_collaborative.py --config configs/sample.yaml

Backend selection (configs): ``collaborative.backend`` = auto | als |
item_item.  ``als`` uses the `implicit` library when importable;
otherwise a pure-scipy item-item CF fallback with identical API.
Also fits and saves the PopularityRecommender used for cold start and
exploration candidates.
"""
from __future__ import annotations

import argparse
import sys
import time

import bootstrap  # noqa: F401

from ml.features.interaction_matrix import InteractionIndex
from ml.preprocessing.pipeline import load_processed
from ml.recommenders.collaborative import CollaborativeRecommender
from ml.recommenders.popularity import PopularityRecommender
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import ensure_dirs

logger = get_logger("train_collaborative")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--artifact-name", default="latest")
    parser.add_argument("--backend", default=None, help="override collaborative.backend")
    args = parser.parse_args()

    config = Config.load(args.config, args.override_config)
    setup_logging(config.get("project.log_level", "INFO"))
    ensure_dirs()

    tables = load_processed()
    songs, train = tables["songs"], tables["train"]

    cf_cfg = config.section("collaborative", {})
    recommender = CollaborativeRecommender(
        backend=args.backend or cf_cfg.get("backend", "auto"),
        factors=int(cf_cfg.get("factors", 64)),
        iterations=int(cf_cfg.get("iterations", 15)),
        regularization=float(cf_cfg.get("regularization", 0.05)),
        alpha=float(cf_cfg.get("alpha", 20.0)),
        n_neighbors=int(cf_cfg.get("n_neighbors", 100)),
    )
    start = time.time()
    recommender.fit(train)
    elapsed = time.time() - start

    sample_user = train["user_id"].iloc[0]
    sample = recommender.recommend_for_user(sample_user, k=5)
    logger.info("Sanity check user=%s top-5: %s", sample_user, list(sample.index))

    directory = recommender.save(
        args.artifact_name,
        metadata={
            "backend": recommender.backend,
            "n_users": recommender.index.n_users if recommender.index else 0,
            "n_songs": recommender.index.n_songs if recommender.index else 0,
            "train_seconds": round(elapsed, 2),
        },
        config=config.to_dict(),
    )
    logger.info("Saved collaborative artifact (%s backend) -> %s (%.1fs)",
                recommender.backend, directory, elapsed)

    # popularity tables share the collaborative artifact family
    pop_cfg = config.section("popularity", {})
    start = time.time()
    popularity = PopularityRecommender(
        recent_window_days=int(pop_cfg.get("recent_window_days", 30)),
    ).fit(train, songs)
    pop_dir = popularity.save(
        "popularity",
        metadata={"n_songs": int(len(popularity.global_popularity)),
                  "n_genres": len(popularity.genre_popularity),
                  "train_seconds": round(time.time() - start, 2)},
        config=config.to_dict(),
    )
    logger.info("Saved popularity artifact -> %s", pop_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
