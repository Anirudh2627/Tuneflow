"""Recommendation engine holder.

Loads trained ML artifacts ONCE at application startup (never per
request, never trains).  If artifacts or processed data are missing the
holder degrades to a database-popularity fallback so the API stays up
while you train your first models.
"""
from __future__ import annotations

import os
import threading

import pandas as pd

from app.core.bootstrap import ensure_ml_on_path
from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EngineHolder:
    """Thread-safe lazily-initialized ML engine container."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._initialized = False
        self.hybrid = None                 # HybridRecommender | None
        self.components: dict = {}         # content / collaborative / session / popularity / ranker
        self.songs_df: pd.DataFrame | None = None
        self.config = None                 # ml Config
        self.degraded_reason: str | None = None

    # ------------------------------------------------------------------
    def initialize(self) -> "EngineHolder":
        with self._lock:
            if self._initialized:
                return self
            self._initialized = True
            ensure_ml_on_path()
            if os.environ.get("TUNEFLOW_DISABLE_ENGINE") == "1":
                self.degraded_reason = "engine disabled via TUNEFLOW_DISABLE_ENGINE=1 (test mode)"
                logger.warning("ML engine disabled by environment; using DB popularity fallback")
                return self
            try:
                from ml.preprocessing.pipeline import load_processed
                from ml.recommenders.factory import build_hybrid_recommender, load_components
                from ml.utils.config import Config

                self.config = Config.load(*self.settings.config_file_list)
                tables = load_processed()
                self.songs_df = tables["songs"]
                self.components = load_components(self.config, self.settings.artifact_name)
                self.hybrid = build_hybrid_recommender(
                    self.config, self.songs_df, self.settings.artifact_name, self.components)
                logger.info("Hybrid recommendation engine loaded from artifacts (%s)",
                            self.settings.artifact_name)
            except FileNotFoundError as exc:
                self.degraded_reason = str(exc)
                logger.warning("Engine running in DEGRADED mode (DB popularity fallback): %s", exc)
            except Exception as exc:  # pragma: no cover
                self.degraded_reason = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to load recommendation engine; using fallback")
        return self

    @property
    def is_ready(self) -> bool:
        return self.hybrid is not None

    @property
    def content(self):
        return self.components.get("content")

    @property
    def popularity(self):
        return self.components.get("popularity")

    def event_weights(self) -> dict[str, float]:
        if self.config is not None:
            return dict(self.config.section("feedback.event_weights", {}))
        return {"play": 1.0, "like": 3.0, "save": 2.0, "skip": -0.5, "dislike": -2.0}


_holder: EngineHolder | None = None
_holder_lock = threading.Lock()


def get_engine_holder(settings: Settings | None = None) -> EngineHolder:
    """Process-wide singleton used by FastAPI dependencies."""
    global _holder
    with _holder_lock:
        if _holder is None:
            from app.core.config import get_settings

            _holder = EngineHolder(settings or get_settings())
        return _holder


def reset_engine_holder() -> None:
    """Used by tests."""
    global _holder
    with _holder_lock:
        _holder = None
