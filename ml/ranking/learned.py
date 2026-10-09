"""Learned rankers implementing the BaseRanker interface.

``LearnedRanker``  - logistic regression over the canonical ranking
features (pure sklearn, always available).  ``LightGBMRanker`` - same
interface backed by LightGBM when installed.  Both are trained by
``scripts/train_ranker.py`` on labels derived from the validation split
and then persisted like every other artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.features.ranking_features import RANKING_FEATURES
from ml.ranking.base import BaseRanker
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class LearnedRanker(BaseRanker):
    """Supervised logistic-regression ranker over ranking features."""

    name = "learned"

    def __init__(self, class_weight: str = "balanced", max_iter: int = 1000, C: float = 1.0):
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.C = C
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = list(RANKING_FEATURES)

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "LearnedRanker":
        columns = [c for c in self.feature_names if c in features.columns]
        self.feature_names = columns
        X = features[columns].astype(float).fillna(0.0).to_numpy()
        y = (np.asarray(labels) > 0).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels contain a single class; check the validation split")
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(
            class_weight=self.class_weight, max_iter=self.max_iter, C=self.C)
        self.model.fit(self.scaler.transform(X), y)
        acc = self.model.score(self.scaler.transform(X), y)
        logger.info("LearnedRanker fitted on %d rows (%d features), train acc=%.3f",
                    len(y), len(columns), acc)
        return self

    def rank(self, candidates: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("LearnedRanker used before fit/load")
        if candidates.empty:
            return np.empty(0)
        X = candidates.reindex(columns=self.feature_names).astype(float).fillna(0.0).to_numpy()
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    @property
    def weights(self) -> dict[str, float]:
        """Effective per-feature weights (for explanations)."""
        if self.model is None or self.scaler is None:
            return {f: 0.0 for f in self.feature_names}
        coef = self.model.coef_[0] / np.maximum(self.scaler.scale_, 1e-9)
        return {f: float(c) for f, c in zip(self.feature_names, coef)}

    def with_weight_overrides(self, overrides: dict) -> "LearnedRanker":
        return self  # learned weights are not hand-tuned


class LightGBMRanker(LearnedRanker):
    """LambdaRank-style LightGBM ranker (optional dependency)."""

    name = "lightgbm"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 31):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self._lgbm = None

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "LightGBMRanker":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "lightgbm is not installed. `pip install lightgbm` or use --ranker learned/weighted."
            ) from exc
        columns = [c for c in RANKING_FEATURES if c in features.columns]
        self.feature_names = columns
        X = features[columns].astype(float).fillna(0.0).to_numpy()
        y = (np.asarray(labels) > 0).astype(int)
        self._lgbm = LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, verbose=-1)
        self._lgbm.fit(X, y)
        logger.info("LightGBMRanker fitted on %d rows (%d features)", len(y), len(columns))
        return self

    def rank(self, candidates: pd.DataFrame) -> np.ndarray:
        if self._lgbm is None:
            raise RuntimeError("LightGBMRanker used before fit/load")
        if candidates.empty:
            return np.empty(0)
        X = candidates.reindex(columns=self.feature_names).astype(float).fillna(0.0).to_numpy()
        return self._lgbm.predict_proba(X)[:, 1]

    @property
    def weights(self) -> dict[str, float]:
        if self._lgbm is None:
            return {}
        importances = self._lgbm.feature_importances_
        total = max(float(importances.sum()), 1.0)
        return {f: float(i / total) for f, i in zip(self.feature_names, importances)}
