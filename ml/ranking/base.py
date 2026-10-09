"""Ranker interfaces.

``BaseRanker`` is the stable API: candidate feature frames in, scores
out.  ``WeightedRanker`` is the production baseline; a learned ranker
(LightGBM/XGBoost/neural) can be added later by implementing the same
interface plus ``fit()``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """Ranks candidate rows described by ranking features."""

    name: str = "base"

    @abstractmethod
    def rank(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return a score per candidate row (higher is better)."""

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "BaseRanker":
        """Optional supervised fitting; stateless rankers ignore it."""
        return self

    def with_weight_overrides(self, overrides: dict) -> "BaseRanker":
        """Return a ranker variant with adjusted weights (mode-specific ranking).

        Rankers without configurable weights return themselves.
        """
        return self

    def save(self, name: str = "latest", metadata: dict | None = None,
             config: dict | None = None) -> str:
        from ml.utils.artifacts import save_artifact

        save_artifact(self, "ranking", name, metadata=metadata, config=config)
        return name

    @classmethod
    def load(cls, name: str = "latest") -> "BaseRanker":
        from ml.utils.artifacts import load_artifact

        obj = load_artifact("ranking", name)
        if not isinstance(obj, BaseRanker):
            raise TypeError(f"Artifact ranking/{name} is not a BaseRanker")
        return obj
