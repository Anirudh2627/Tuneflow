#!/usr/bin/env python3
"""Train the session-based recommender.

Usage::

    python scripts/train_session.py --config configs/sample.yaml

Fits item-item transition similarity over sessionized training logs.
To replace this baseline with GRU4Rec/Transformer later, implement the
same public API (``fit``, ``recommend_for_session``, ``session_scores``)
in a new class and point ``--artifact-name``/loading at it.
"""
from __future__ import annotations

import argparse
import sys
import time

import bootstrap  # noqa: F401

from ml.preprocessing.pipeline import load_processed
from ml.recommenders.session import SessionRecommender
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import ensure_dirs

logger = get_logger("train_session")


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
    train = tables["train"]

    session_cfg = config.section("session", {})
    recommender = SessionRecommender(
        decay=float(session_cfg.get("decay", 0.6)),
        n_neighbors=int(session_cfg.get("n_neighbors", 100)),
        session_gap_minutes=int(session_cfg.get("session_gap_minutes", 30)),
        max_session_len=int(session_cfg.get("max_session_len", 20)),
    )
    start = time.time()
    recommender.fit(train)
    elapsed = time.time() - start

    # sanity check on a random user's cached session
    if recommender.user_last_sessions:
        user, session = next(iter(recommender.user_last_sessions.items()))
        sample = recommender.recommend_for_session(session, k=5)
        logger.info("Sanity check user=%s session=%s -> %s", user, session[-3:], list(sample.index))

    directory = recommender.save(
        args.artifact_name,
        metadata={
            "n_songs": len(recommender.song_ids),
            "n_cached_user_sessions": len(recommender.user_last_sessions),
            "train_seconds": round(elapsed, 2),
        },
        config=config.to_dict(),
    )
    logger.info("Saved session artifact -> %s (%.1fs)", directory, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
