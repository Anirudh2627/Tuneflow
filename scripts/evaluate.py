#!/usr/bin/env python3
"""Evaluate all recommenders on the held-out test split.

Usage::

    python scripts/evaluate.py --config configs/sample.yaml
    python scripts/evaluate.py --recommenders popularity,content,hybrid --k 10

Metrics: Precision@K, Recall@K, MAP@K, NDCG@K, HitRate@K, MRR,
Coverage, Novelty, Intra-list Diversity, latency.  Results are written
to ``reports/evaluation/<run_name>.json`` and ``.md``.  Numbers are
computed live -- nothing is pre-baked.
"""
from __future__ import annotations

import argparse
import sys

import bootstrap  # noqa: F401

from ml.evaluation.evaluator import RecommenderEvaluator, save_results
from ml.preprocessing.pipeline import load_processed, load_processed_stats
from ml.recommenders.collaborative import CollaborativeRecommender
from ml.recommenders.content import ContentRecommender
from ml.recommenders.factory import build_hybrid_recommender, load_components
from ml.recommenders.popularity import PopularityRecommender
from ml.recommenders.session import SessionRecommender
from ml.utils.config import Config
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import ensure_dirs

logger = get_logger("evaluate")

ALL_RECOMMENDERS = ("popularity", "content", "collaborative", "session", "hybrid")


def get_or_fit(name: str, config: Config, tables: dict, components: dict):
    """Prefer persisted artifacts; fall back to fitting on train split."""
    if components.get(name) is not None:
        logger.info("%s: using persisted artifact", name)
        return components[name]
    logger.info("%s: no artifact found -- fitting on train split for evaluation", name)
    songs, train = tables["songs"], tables["train"]
    if name == "popularity":
        cfg = config.section("popularity", {})
        return PopularityRecommender(recent_window_days=int(cfg.get("recent_window_days", 30))).fit(train, songs)
    if name == "content":
        cfg = config.section("content", {})
        return ContentRecommender(embedding_config=cfg.get("text_embedding", {"provider": "none"})).fit(songs, train)
    if name == "collaborative":
        cfg = config.section("collaborative", {})
        return CollaborativeRecommender(
            backend=cfg.get("backend", "auto"), factors=int(cfg.get("factors", 64)),
            iterations=int(cfg.get("iterations", 15)),
            regularization=float(cfg.get("regularization", 0.05)),
            alpha=float(cfg.get("alpha", 20.0)), n_neighbors=int(cfg.get("n_neighbors", 100)),
        ).fit(train)
    if name == "session":
        cfg = config.section("session", {})
        return SessionRecommender(
            decay=float(cfg.get("decay", 0.6)), n_neighbors=int(cfg.get("n_neighbors", 100)),
            session_gap_minutes=int(cfg.get("session_gap_minutes", 30)),
            max_session_len=int(cfg.get("max_session_len", 20)),
        ).fit(train)
    raise ValueError(f"Unknown recommender {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--artifact-name", default="latest")
    parser.add_argument("--recommenders", default=",".join(ALL_RECOMMENDERS),
                        help="comma separated subset of: " + ",".join(ALL_RECOMMENDERS))
    parser.add_argument("--k", type=int, default=None, help="single K (default: evaluation.k_values)")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = Config.load(args.config, args.override_config)
    setup_logging(config.get("project.log_level", "INFO"))
    ensure_dirs()

    tables = load_processed()
    stats = load_processed_stats()
    songs, train, test = tables["songs"], tables["train"], tables["test"]
    if test.empty:
        logger.error("Test split is empty -- nothing to evaluate")
        return 1

    names = [n.strip() for n in args.recommenders.split(",") if n.strip()]
    unknown = set(names) - set(ALL_RECOMMENDERS)
    if unknown:
        parser.error(f"Unknown recommenders: {unknown}")

    components = load_components(config, args.artifact_name)
    popularity = get_or_fit("popularity", config, tables, components)
    components["popularity"] = popularity

    models = {}
    for name in names:
        if name == "hybrid":
            models["hybrid"] = build_hybrid_recommender(config, songs, args.artifact_name, components)
        else:
            models[name] = get_or_fit(name, config, tables, components)

    content = models.get("content") or components.get("content")
    sim_cache: dict[tuple[str, str], float] = {}

    def similarity_lookup(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        if key not in sim_cache:
            if content is None:
                sim_cache[key] = 0.0
            else:
                sim = content.similarity([key[0], key[1]])
                sim_cache[key] = float(sim[0, 1]) if sim is not None else 0.0
        return sim_cache[key]

    k_values = [args.k] if args.k else [int(k) for k in config.get("evaluation.k_values", [10])]
    max_users = args.max_users or config.get("evaluation.max_users")

    results = []
    for k in k_values:
        evaluator = RecommenderEvaluator(
            songs=songs, train=train, test=test,
            popularity=popularity.global_popularity,
            similarity_lookup=similarity_lookup if content is not None else None,
            max_users=max_users,
            seed=int(config.get("project.seed", 42)),
        )
        for name in names:
            session_aware = name in ("session",)
            result = evaluator.evaluate(models[name], name=f"{name}@{k}", k=k, session_aware=session_aware)
            results.append(result)
            logger.info(
                "%-16s P=%.4f R=%.4f NDCG=%.4f HR=%.4f MAP=%.4f Cov=%.3f Nov=%.3f Div=%.3f",
                result.recommender, result.precision, result.recall, result.ndcg,
                result.hit_rate, result.map, result.coverage, result.novelty, result.diversity,
            )

    run_name = args.run_name or config.get("evaluation.run_name", "latest")
    save_results(results, run_name=run_name, extra={"dataset_stats": stats,
                                                     "k_values": k_values,
                                                     "recommenders": names})
    return 0


if __name__ == "__main__":
    sys.exit(main())
