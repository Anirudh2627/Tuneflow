"""Backend recommender surface.

The ML implementations live in the shared ``ml`` package (single source of
truth for training AND serving).  This module re-exports them plus the
serving-side engine holder so ``app.recommenders`` is a complete facade.
"""
from ml.recommenders.base import BaseRecommender
from ml.recommenders.collaborative import CollaborativeRecommender
from ml.recommenders.content import ContentRecommender
from ml.recommenders.hybrid import HybridRecommender, Recommendation
from ml.recommenders.popularity import PopularityRecommender
from ml.recommenders.session import SessionRecommender

from app.services.engine import EngineHolder, get_engine_holder

__all__ = [
    "BaseRecommender", "CollaborativeRecommender", "ContentRecommender",
    "HybridRecommender", "Recommendation", "PopularityRecommender",
    "SessionRecommender", "EngineHolder", "get_engine_holder",
]
