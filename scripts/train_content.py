#!/usr/bin/env python3
"""Train the content-based recommender.

Usage::

    python scripts/train_content.py --config configs/sample.yaml

Reads ``data/processed``, fits SongFeatureBuilder + similarity space,
and saves the artifact to ``models/content/latest``.
"""
from __future__ import annotations

import argparse
import sys
import time

import bootstrap  # noqa: F401

from ml.preprocessing.pipeline import load_processed
from ml.recommenders.content import ContentRecommender
from ml.utils.artifacts import save_artifact
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import ensure_dirs

logger = get_logger("train_content")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--artifact-name", default="latest")
    args = parser.parse_args()

    config = Config.load(args.config, args.override_config)
    setup_logging(config.get("project.log_level", "INFO"))
    ensure_dirs()

    tables = load_processed()
    songs, train = tables["songs"], tables["train"]
    logger.info("Training content model: %d songs, %d train interactions", len(songs), len(train))

    content_cfg = config.section("content", {})
    start = time.time()
    recommender = ContentRecommender(
        embedding_config=content_cfg.get("text_embedding", {"provider": "none"}),
        embedding_weight=float(content_cfg.get("embedding_weight", 0.3)),
        like_weight=float(content_cfg.get("like_weight", 2.0)),
        play_weight=float(content_cfg.get("play_weight", 1.0)),
        dislike_penalty=float(content_cfg.get("dislike_penalty", 1.5)),
        profile_history_limit=int(content_cfg.get("profile_history_limit", 60)),
    ).fit(songs, interactions=train)
    elapsed = time.time() - start

    # quick sanity recommendations so the log shows the model works
    sample_user = train["user_id"].iloc[0]
    user_interactions = train[train["user_id"] == sample_user]
    sample = recommender.recommend_for_user(sample_user, k=5, interactions=user_interactions)
    logger.info("Sanity check user=%s top-5: %s", sample_user, list(sample.index))

    directory = recommender.save(
        args.artifact_name,
        metadata={
            "n_songs": len(songs),
            "n_features": int(recommender.X.shape[1]) if recommender.X is not None else 0,
            "embeddings": recommender.embeddings is not None,
            "train_seconds": round(elapsed, 2),
        },
        config=config.to_dict(),
    )
    logger.info("Saved content artifact -> %s (%.1fs)", directory, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
