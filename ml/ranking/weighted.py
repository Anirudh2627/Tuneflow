"""Weighted linear ranker (configurable, no training required).

final_score = sum_f  w_f * minmax_normalized(feature_f)

Feature normalization happens per-request over the candidate set, so
weights are on a comparable scale.  Learned rankers (LightGBM etc.)
plug into the same BaseRanker interface and can be trained with
``scripts/train_ranker.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.ranking.base import BaseRanker


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return np.zeros_like(values) if not np.isfinite(lo) else np.full_like(values, 0.5)
    return (np.nan_to_num(values, nan=lo) - lo) / (hi - lo)


DEFAULT_WEIGHTS: dict[str, float] = {
    "content_score": 0.30,
    "collaborative_score": 0.35,
    "session_score": 0.20,
    "popularity_score": 0.15,
    "genre_affinity": 0.10,
    "artist_affinity": 0.08,
    "audio_match": 0.06,
    "novelty": 0.04,
}


class WeightedRanker(BaseRanker):
    name = "weighted"

    def __init__(self, weights: dict[str, float] | None = None, normalize: bool = True):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.normalize = normalize

    def with_weight_overrides(self, overrides: dict) -> "WeightedRanker":
        merged = {k: v for k, v in self.weights.items() if k not in overrides}
        merged.update({k: float(v) for k, v in overrides.items()})
        return WeightedRanker(weights=merged, normalize=self.normalize)

    def rank(self, candidates: pd.DataFrame) -> np.ndarray:
        if candidates.empty:
            return np.empty(0)
        total = np.zeros(len(candidates), dtype=np.float64)
        for feature, weight in self.weights.items():
            if weight == 0 or feature not in candidates.columns:
                continue
            values = pd.to_numeric(candidates[feature], errors="coerce").fillna(0.0).to_numpy()
            if self.normalize:
                values = minmax_normalize(values)
            total += weight * values
        return total
