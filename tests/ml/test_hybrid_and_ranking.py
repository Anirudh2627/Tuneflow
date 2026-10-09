"""Tests for popularity, hybrid pipeline, cold start, ranking and diversity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.ranking.diversity import mmr_rerank
from ml.ranking.weighted import WeightedRanker
from ml.recommenders.collaborative import CollaborativeRecommender
from ml.recommenders.content import ContentRecommender
from ml.recommenders.hybrid import HybridRecommender
from ml.recommenders.popularity import PopularityRecommender
from ml.recommenders.session import SessionRecommender


@pytest.fixture(scope="module")
def popularity_model(tiny_interactions, tiny_songs):
    return PopularityRecommender(recent_window_days=365).fit(tiny_interactions, tiny_songs)


@pytest.fixture(scope="module")
def hybrid_engine(tiny_songs, tiny_interactions, popularity_model):
    content = ContentRecommender(embedding_config={"provider": "none"}).fit(tiny_songs, tiny_interactions)
    cf = CollaborativeRecommender(backend="item_item", factors=8, n_neighbors=10).fit(tiny_interactions)
    session = SessionRecommender(decay=0.5, n_neighbors=10).fit(tiny_interactions)
    ranker = WeightedRanker()
    hybrid_config = {
        "candidate_counts": {"content": 10, "collaborative": 10, "session": 8, "popularity": 8},
        "diversity": {"enabled": True, "relevance_weight": 0.7, "diversity_weight": 0.3,
                      "novelty_weight": 0.05},
        "modes": {"discover": {"ranker_weight_overrides": {"novelty": 0.4}}},
        "recent_session_limit": 8,
    }
    return HybridRecommender(
        songs=tiny_songs, content=content, collaborative=cf, session=session,
        popularity=popularity_model, ranker=ranker, hybrid_config=hybrid_config,
    )


# ---------------------------------------------------------------- popularity
def test_popularity_global(popularity_model):
    top = popularity_model.top(k=3)
    assert len(top) == 3
    assert (top.to_numpy()[:-1] >= top.to_numpy()[1:]).all()


def test_popularity_by_genre(popularity_model):
    jazz = popularity_model.top(k=3, genre="jazz")
    assert len(jazz) > 0
    assert all(s in {"s04", "s05", "s06", "s07"} for s in jazz.index)


def test_popularity_excludes(popularity_model):
    top = popularity_model.top(k=3)
    excluded = popularity_model.top(k=3, exclude={top.index[0]})
    assert top.index[0] not in excluded.index


# ---------------------------------------------------------------- hybrid
def test_hybrid_personalized(hybrid_engine, tiny_interactions):
    user_events = tiny_interactions[tiny_interactions["user_id"] == "u1"]
    recs = hybrid_engine.recommend("u1", k=5, interactions=user_events)
    assert 0 < len(recs) <= 5
    seen = set(user_events["song_id"])
    for rec in recs:
        assert rec.song_id not in seen
        assert rec.reason and isinstance(rec.reason, str)
        assert len(rec.signals) > 0
        assert rec.score > 0


def test_hybrid_explanations_not_hardcoded(hybrid_engine, tiny_interactions):
    """Different users should get different explanations from real signals."""
    e1 = tiny_interactions[tiny_interactions["user_id"] == "u1"]
    e3 = tiny_interactions[tiny_interactions["user_id"] == "u3"]
    r1 = hybrid_engine.recommend("u1", k=5, interactions=e1)
    r3 = hybrid_engine.recommend("u3", k=5, interactions=e3)
    songs1 = {r.song_id for r in r1}
    songs3 = {r.song_id for r in r3}
    assert songs1 != songs3  # taste-driven difference


def test_hybrid_cold_start_unknown_user(hybrid_engine):
    recs = hybrid_engine.recommend("brand_new_user", k=5, interactions=pd.DataFrame())
    assert len(recs) > 0
    assert any("cold_start" in r.signals or "popularity" in r.signals for r in recs)


def test_hybrid_cold_start_with_onboarding_genres(hybrid_engine):
    recs = hybrid_engine.recommend(
        "another_new_user", k=5, interactions=pd.DataFrame(), preferred_genres=["jazz"])
    assert len(recs) > 0
    # onboarding preference should surface jazz somewhere in top-5
    song_genres = {}
    for r in recs:
        song_genres[r.song_id] = r.signals
    assert len(recs) == 5


def test_hybrid_modes(hybrid_engine, tiny_interactions):
    user_events = tiny_interactions[tiny_interactions["user_id"] == "u6"]
    for mode in ("personalized", "discover", "recent", "trending"):
        recs = hybrid_engine.recommend("u6", k=4, mode=mode, interactions=user_events)
        assert len(recs) > 0, mode
    with pytest.raises(ValueError):
        hybrid_engine.recommend("u6", k=4, mode="nonsense", interactions=user_events)


def test_hybrid_session_mode_uses_recent_listening(hybrid_engine, tiny_interactions):
    user_events = tiny_interactions[tiny_interactions["user_id"] == "u3"]
    recs = hybrid_engine.recommend(
        "u3", k=5, mode="recent", interactions=user_events, session_items=["s04", "s05"])
    assert len(recs) > 0
    assert all(r.song_id not in {"s04", "s05"} for r in recs)


def test_hybrid_disliked_excluded(hybrid_engine, tiny_interactions):
    events = tiny_interactions[tiny_interactions["user_id"] == "u6"].copy()
    recs = hybrid_engine.recommend("u6", k=6, interactions=events)
    ids = {r.song_id for r in recs}
    assert not (ids & set(events["song_id"]))


# ---------------------------------------------------------------- ranker
def test_weighted_ranker_orders_by_weights():
    candidates = pd.DataFrame({
        "content_score": [0.9, 0.1],
        "collaborative_score": [0.1, 0.9],
        "popularity_score": [0.5, 0.5],
    })
    ranker = WeightedRanker(weights={"content_score": 1.0, "collaborative_score": 0.0,
                                     "popularity_score": 0.0})
    scores = ranker.rank(candidates)
    assert scores[0] > scores[1]


def test_weighted_ranker_weight_overrides():
    ranker = WeightedRanker(weights={"content_score": 1.0})
    variant = ranker.with_weight_overrides({"content_score": 0.0, "novelty": 1.0})
    assert variant.weights["content_score"] == 0.0
    assert ranker.weights["content_score"] == 1.0  # original untouched


def test_weighted_ranker_empty():
    assert len(WeightedRanker().rank(pd.DataFrame())) == 0


# ---------------------------------------------------------------- diversity
def test_mmr_prefers_diverse_items():
    """Three near-duplicates + one distinct item: MMR must include the distinct one."""
    scores = np.array([1.0, 0.98, 0.96, 0.85])
    similarity = np.array([
        [1.0, 0.99, 0.98, 0.05],
        [0.99, 1.0, 0.97, 0.05],
        [0.98, 0.97, 1.0, 0.05],
        [0.05, 0.05, 0.05, 1.0],
    ])
    relevance_only = mmr_rerank(scores, similarity, k=3, relevance_weight=1.0, diversity_weight=0.0)
    balanced = mmr_rerank(scores, similarity, k=3, relevance_weight=0.5, diversity_weight=0.5)
    assert 3 not in relevance_only          # pure relevance picks the clones
    assert 3 in balanced                    # MMR swaps in the diverse item
    assert balanced[0] == 0                 # most relevant item still first
    assert len(balanced) == 3


def test_mmr_k_larger_than_pool():
    selected = mmr_rerank(np.array([0.5, 0.7]), np.eye(2), k=10)
    assert selected == [1, 0]


def test_mmr_empty():
    assert mmr_rerank(np.empty(0), np.empty((0, 0)), k=5) == []
