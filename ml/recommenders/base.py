"""Recommender interfaces.

All recommenders return scores as ``pd.Series`` indexed by ``song_id``
so the hybrid layer can merge sources uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Common contract for every recommendation strategy."""

    name: str = "base"
    family: str = "base"  # artifact folder under models/

    @abstractmethod
    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
        exclude: set[str] | None = None,
    ) -> pd.Series:
        """Return top-k scores (Series indexed by song_id, desc)."""

    # ------------------------------------------------------------------
    def save(self, name: str = "latest", metadata: dict | None = None,
             config: dict | None = None) -> str:
        from ml.utils.artifacts import save_artifact

        save_artifact(self, self.family, name, metadata=metadata, config=config)
        return name

    @classmethod
    def load(cls, name: str = "latest") -> "BaseRecommender":
        from ml.utils.artifacts import load_artifact

        obj = load_artifact(cls.family, name)
        if not isinstance(obj, BaseRecommender):
            raise TypeError(f"Artifact {cls.family}/{name} is not a BaseRecommender")
        return obj

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)
