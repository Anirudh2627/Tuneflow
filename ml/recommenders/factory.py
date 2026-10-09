"""Factory: assemble the full hybrid engine from persisted artifacts.

Shared by ``scripts/evaluate.py`` and the backend service layer so both
load models exactly the same way.  The factory NEVER trains; missing
artifacts degrade the engine (fewer sources) with a warning.
"""
from __future__ import annotations

import pandas as pd

from ml.ranking.base import BaseRanker
from ml.ranking.weighted import WeightedRanker
from ml.recommenders.collaborative import CollaborativeRecommender
from ml.recommenders.content import ContentRecommender
from ml.recommenders.hybrid import HybridRecommender
from ml.recommenders.popularity import PopularityRecommender
from ml.recommenders.session import SessionRecommender
from ml.utils.artifacts import load_artifact, artifact_exists
from ml.utils.config import Config
from ml.utils.logging import get_logger

logger = get_logger(__name__)


def _try_load(cls, family: str, name: str):
    if not artifact_exists(family, name):
        logger.warning("Artifact %s/%s missing -- %s disabled. Run scripts/train_%s.py.",
                       family, name, cls.__name__, family)
        return None
    obj = load_artifact(family, name)
    if not isinstance(obj, cls):
        logger.warning("Artifact %s/%s is %s, expected %s -- disabled",
                       family, name, type(obj).__name__, cls.__name__)
        return None
    return obj


def load_components(config: Config, artifact_name: str = "latest") -> dict:
    """Load every trained component, tolerating missing pieces."""
    ranker_cfg = config.section("ranker", {})
    ranker: BaseRanker | None = _try_load(BaseRanker, "ranking", artifact_name)
    if ranker is None:
        logger.info("Using WeightedRanker from config (no persisted ranker artifact)")
        ranker = WeightedRanker(weights=ranker_cfg.get("weights"))
    return {
        "content": _try_load(ContentRecommender, "content", artifact_name),
        "collaborative": _try_load(CollaborativeRecommender, "collaborative", artifact_name),
        "popularity": _try_load(PopularityRecommender, "collaborative", "popularity"),
        "session": _try_load(SessionRecommender, "session", artifact_name),
        "ranker": ranker,
    }


def build_hybrid_recommender(config: Config, songs: pd.DataFrame,
                             artifact_name: str = "latest",
                             components: dict | None = None) -> HybridRecommender:
    """Assemble a HybridRecommender from persisted artifacts + config."""
    components = components or load_components(config, artifact_name)
    popularity = components.get("popularity")
    if popularity is None:
        raise FileNotFoundError(
            "Popularity artifact is required for cold start/fallback. "
            "Run: python scripts/train_collaborative.py (it saves popularity too)."
        )
    return HybridRecommender(
        songs=songs,
        content=components.get("content"),
        collaborative=components.get("collaborative"),
        session=components.get("session"),
        popularity=popularity,
        ranker=components["ranker"],
        hybrid_config=config.section("hybrid", {}),
    )
