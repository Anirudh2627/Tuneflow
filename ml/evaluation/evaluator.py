"""Multi-recommender evaluation harness.

Evaluates Popularity / Content / Collaborative / Session / Hybrid on
the held-out test split and writes machine-readable JSON plus a
markdown summary under ``reports/evaluation``.  Nothing is faked: all
numbers are computed when you run ``scripts/evaluate.py``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation import metrics as M
from ml.utils.logging import get_logger
from ml.utils.paths import EVALUATION_DIR

logger = get_logger(__name__)

POSITIVE_EVENTS = {"play", "like", "save"}
EVENT_GAINS = {"play": 1.0, "like": 3.0, "save": 2.0}


@dataclass
class EvaluationResult:
    recommender: str
    k: int
    n_users_evaluated: int
    precision: float
    recall: float
    map: float
    ndcg: float
    hit_rate: float
    mrr: float
    coverage: float
    novelty: float
    diversity: float
    latency_ms_per_user: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RecommenderEvaluator:
    """Runs the same protocol for every recommender under test."""

    def __init__(self, songs: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                 popularity: pd.Series, similarity_lookup=None, max_users: int | None = None,
                 seed: int = 42):
        self.songs = songs
        self.train = train
        self.test = test
        self.popularity = popularity
        self.similarity_lookup = similarity_lookup
        self.catalog_size = int(songs["song_id"].nunique())
        self.catalog_ids = set(songs["song_id"].astype(str))
        self.max_users = max_users
        self.rng = np.random.default_rng(seed)

        # ground truth: positive test events per user with graded gains
        positive_test = test[test["event_type"].isin(POSITIVE_EVENTS)]
        self.relevant = {
            user: set(group["song_id"].astype(str)) for user, group in positive_test.groupby("user_id")
        }
        self.gains = {
            user: group.groupby(group["song_id"].astype(str))["event_type"]
            .apply(lambda s: float(sum(EVENT_GAINS.get(e, 1.0) for e in s)))
            .to_dict()
            for user, group in positive_test.groupby("user_id")
        }
        # training history per user (for exclusion + session context)
        self.history = {
            user: group.sort_values("timestamp")["song_id"].astype(str).tolist()
            if "timestamp" in group.columns and group["timestamp"].notna().any()
            else group["song_id"].astype(str).tolist()
            for user, group in train[train["event_type"].isin(POSITIVE_EVENTS)].groupby("user_id")
        }
        self.n_train_events = int(len(train))

    # ------------------------------------------------------------------
    def evaluate(self, recommender, name: str, k: int = 10, session_aware: bool = False) -> EvaluationResult:
        users = [u for u in self.relevant if u in self.history or not session_aware]
        if not users:
            users = list(self.relevant)
        if self.max_users and len(users) > self.max_users:
            users = list(self.rng.choice(users, size=self.max_users, replace=False))
        logger.info("Evaluating %-14s on %d users (k=%d)", name, len(users), k)

        all_recs: list[list[str]] = []
        precisions, recalls, ndcgs, hits, aps = [], [], [], [], []
        start = time.perf_counter()
        for user in users:
            exclude = set(self.history.get(user, []))
            recommended = self._recommend(recommender, name, user, k, exclude, session_aware)
            recommended = [str(r) for r in recommended]
            all_recs.append(recommended)
            relevant = self.relevant[user] - exclude
            if not relevant:
                continue
            precisions.append(M.precision_at_k(recommended, relevant, k))
            recalls.append(M.recall_at_k(recommended, relevant, k))
            ndcgs.append(M.ndcg_at_k(recommended, self.gains[user], k))
            hits.append(M.hit_rate_at_k(recommended, relevant, k))
            aps.append(M.average_precision_at_k(recommended, relevant, k))
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(len(users), 1)

        item_counts = np.zeros(self.catalog_size)
        song_index = {s: i for i, s in enumerate(sorted(self.catalog_ids))}
        for rec in all_recs:
            for item in rec:
                if item in song_index:
                    item_counts[song_index[item]] += 1

        diversities = []
        if self.similarity_lookup is not None:
            for rec in all_recs[: min(len(all_recs), 500)]:
                diversities.append(M.intra_list_diversity(rec, self.similarity_lookup))

        return EvaluationResult(
            recommender=name,
            k=k,
            n_users_evaluated=len(precisions),
            precision=float(np.mean(precisions)) if precisions else 0.0,
            recall=float(np.mean(recalls)) if recalls else 0.0,
            map=float(np.mean(aps)) if aps else 0.0,
            ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
            hit_rate=float(np.mean(hits)) if hits else 0.0,
            mrr=M.mrr(all_recs, [self.relevant[u] for u in users]),
            coverage=M.catalog_coverage(all_recs, self.catalog_ids),
            novelty=M.novelty([item for rec in all_recs for item in rec], self.popularity,
                              max(self.n_train_events, 1)),
            diversity=float(np.mean(diversities)) if diversities else float("nan"),
            latency_ms_per_user=elapsed_ms,
        )

    def _recommend(self, recommender, name: str, user: str, k: int, exclude: set[str],
                   session_aware: bool) -> list[str]:
        base_name = name.split("@")[0]
        if base_name == "hybrid" and hasattr(recommender, "recommend"):
            interactions = self._user_interactions(user)
            recs = recommender.recommend(user, k=k, exclude_songs=exclude, interactions=interactions)
            return [r.song_id for r in recs]
        if session_aware and hasattr(recommender, "recommend_for_session"):
            session = self.history.get(user, [])[-12:]
            series = recommender.recommend_for_session(session, k=k, exclude=exclude)
        elif hasattr(recommender, "recommend_for_user"):
            if base_name == "content":
                series = recommender.recommend_for_user(
                    user, k=k, exclude=exclude, interactions=self._user_interactions(user))
            else:
                series = recommender.recommend_for_user(user, k=k, exclude=exclude)
        else:  # pragma: no cover
            series = pd.Series(dtype=float)
        return list(series.index)

    def _user_interactions(self, user: str) -> pd.DataFrame:
        mask = (self.train["user_id"] == user)
        return self.train[mask]


def save_results(results: list[EvaluationResult], run_name: str = "latest",
                 output_dir: Path | None = None, extra: dict | None = None) -> Path:
    directory = Path(output_dir) if output_dir else EVALUATION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [r.to_dict() for r in results],
        **(extra or {}),
    }
    json_path = directory / f"{run_name}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    md_path = directory / f"{run_name}.md"
    df = pd.DataFrame([r.to_dict() for r in results])
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# TuneFlow evaluation - {run_name}\n\n")
        fh.write(f"Generated at {payload['generated_at']}\n\n")
        if not df.empty:
            fh.write(df.round(4).to_markdown(index=False))
        fh.write("\n")
    logger.info("Saved evaluation results: %s, %s", json_path, md_path)
    return json_path
