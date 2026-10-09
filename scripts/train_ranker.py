#!/usr/bin/env python3
"""Train (or persist) the ranking model.

Usage::

    python scripts/train_ranker.py --ranker weighted     # persist config weights
    python scripts/train_ranker.py --ranker learned      # logistic regression on val labels
    python scripts/train_ranker.py --ranker lightgbm     # requires `pip install lightgbm`

Supervised modes build a training set from the VALIDATION split:
candidates come from the sub-recommenders, positives are the user's
val events, negatives are sampled from the catalog.  Features are the
canonical ranking features (ml/features/ranking_features.py), so any
future ranker using the same schema plugs straight in.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

import bootstrap  # noqa: F401

from ml.features.ranking_features import RankingFeatureExtractor
from ml.features.user_profile import UserProfileService
from ml.preprocessing.pipeline import load_processed
from ml.ranking.learned import LearnedRanker, LightGBMRanker
from ml.ranking.weighted import WeightedRanker
from ml.recommenders.factory import load_components
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import ensure_dirs

logger = get_logger("train_ranker")

POSITIVE_EVENTS = {"play", "like", "save"}


def build_supervised_dataset(config: Config, tables: dict, components: dict,
                             max_users: int = 2000, negatives_per_user: int = 20,
                             seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Label candidates from the validation split (leakage-safe)."""
    songs, train, val = tables["songs"], tables["train"], tables["val"]
    if val.empty:
        raise RuntimeError("Validation split is empty; rerun prepare_data with val_frac > 0")

    rng = np.random.default_rng(seed)
    catalog = songs["song_id"].astype(str).to_numpy()
    popularity = components.get("popularity")
    content = components.get("content")
    collaborative = components.get("collaborative")
    profile_service = UserProfileService(songs)
    extractor = RankingFeatureExtractor(
        songs=songs,
        popularity=popularity.global_popularity if popularity else None,
        artist_popularity=HybridArtistPopularity(songs, popularity) if popularity else None,
    )

    train_by_user = {u: g for u, g in train.groupby("user_id")}
    val_positive = val[val["event_type"].isin(POSITIVE_EVENTS)]
    users = [u for u in val_positive["user_id"].unique() if u in train_by_user]
    if len(users) > max_users:
        users = list(rng.choice(users, size=max_users, replace=False))
    logger.info("Building ranker training set from %d users", len(users))

    feature_rows, labels = [], []
    for n, user in enumerate(users):
        user_train = train_by_user[user]
        profile = profile_service.build(user, user_train)
        seen = set(user_train["song_id"].astype(str))
        positives = set(val_positive[val_positive["user_id"] == user]["song_id"].astype(str)) - seen
        if not positives:
            continue

        candidate_pool: set[str] = set(positives)
        source_scores: dict[str, pd.Series] = {}
        if content is not None:
            s = content.recommend_for_user(user, k=50, exclude=seen, interactions=user_train)
            source_scores["content"] = s
            candidate_pool |= set(s.index)
        if collaborative is not None:
            s = collaborative.recommend_for_user(user, k=50, exclude=seen)
            source_scores["collaborative"] = s
            candidate_pool |= set(s.index)
        if popularity is not None:
            s = popularity.top(k=30, exclude=seen)
            source_scores["popularity"] = s
            candidate_pool |= set(s.index)

        negatives = list(candidate_pool - positives)
        if len(negatives) < negatives_per_user:
            extra = rng.choice(catalog, size=negatives_per_user - len(negatives), replace=False)
            negatives.extend(str(e) for e in extra if str(e) not in positives and str(e) not in seen)
        candidates = list(positives) + negatives[: negatives_per_user * 2]
        candidates = [c for c in dict.fromkeys(candidates) if c not in seen]
        if not candidates:
            continue

        features = extractor.extract(profile, candidates, source_scores)
        feature_rows.append(features)
        labels.extend([1 if c in positives else 0 for c in candidates])
        if (n + 1) % 200 == 0:
            logger.info("  ...%d/%d users processed", n + 1, len(users))

    if not feature_rows:
        raise RuntimeError("No training rows generated -- check that sub-model artifacts exist")
    X = pd.concat(feature_rows)
    y = np.asarray(labels[: len(X)], dtype=int)
    logger.info("Ranker training set: %d rows, positive rate=%.3f", len(X), y.mean())
    return X, y


def HybridArtistPopularity(songs: pd.DataFrame, popularity) -> pd.Series:
    from ml.recommenders.hybrid import HybridRecommender

    return HybridRecommender._artist_popularity(songs, popularity.global_popularity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--ranker", default=None, choices=["weighted", "learned", "lightgbm"],
                        help="defaults to ranker.type in config")
    parser.add_argument("--artifact-name", default="latest")
    parser.add_argument("--max-users", type=int, default=2000)
    args = parser.parse_args()

    config = Config.load(args.config, args.override_config)
    setup_logging(config.get("project.log_level", "INFO"))
    ensure_dirs()

    ranker_type = args.ranker or config.get("ranker.type", "weighted")
    ranker_cfg = config.section("ranker", {})
    start = time.time()

    if ranker_type == "weighted":
        ranker = WeightedRanker(weights=ranker_cfg.get("weights"))
        logger.info("Persisting WeightedRanker with config weights: %s", ranker.weights)
    else:
        tables = load_processed()
        components = load_components(config, args.artifact_name)
        X, y = build_supervised_dataset(
            config, tables, components, max_users=args.max_users,
            seed=int(config.get("project.seed", 42)),
        )
        if ranker_type == "lightgbm":
            ranker = LightGBMRanker().fit(X, y)
        else:
            ranker = LearnedRanker().fit(X, y)

    elapsed = time.time() - start
    directory = ranker.save(
        args.artifact_name,
        metadata={"ranker_type": ranker_type, "train_seconds": round(elapsed, 2),
                  "weights": getattr(ranker, "weights", {})},
        config=config.to_dict(),
    )
    logger.info("Saved ranking artifact -> %s (%.1fs)", directory, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
