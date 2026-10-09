"""Tests for collaborative filtering (scipy fallback backend)."""
from __future__ import annotations

import pytest

from ml.recommenders.collaborative import CollaborativeRecommender


@pytest.fixture(scope="module")
def cf_model(tiny_interactions):
    return CollaborativeRecommender(backend="item_item", factors=8, n_neighbors=10).fit(tiny_interactions)


def test_fit_builds_matrix(cf_model):
    assert cf_model.is_fitted
    assert cf_model.index.n_users == 6
    assert cf_model.index.n_songs == 11  # s11 never appears in the tiny log
    assert cf_model.matrix.nnz > 0


def test_recommend_excludes_history(cf_model, tiny_interactions):
    seen = set(tiny_interactions[tiny_interactions["user_id"] == "u1"]["song_id"])
    recs = cf_model.recommend_for_user("u1", k=5, exclude=seen)
    assert not (set(recs.index) & seen)


def test_recommendations_follow_collaborative_signal(cf_model, tiny_interactions):
    """u2 listens to the same electronic songs as u1 -> gets electronic recs."""
    recs = cf_model.recommend_for_user("u2", k=4)
    electronic = {"s00", "s01", "s02", "s03"}
    seen = set(tiny_interactions[tiny_interactions["user_id"] == "u2"]["song_id"])
    assert len(electronic & (set(recs.index) | seen)) >= 2


def test_unknown_user_returns_empty(cf_model):
    assert cf_model.recommend_for_user("nobody").empty


def test_similar_items(cf_model):
    similar = cf_model.similar_items("s00", k=3)
    assert len(similar) > 0
    assert "s00" not in similar.index


def test_similar_users(cf_model):
    similar = cf_model.similar_users("u1", k=3)
    assert "u1" not in similar.index
    assert "u2" in similar.index  # shares electronic taste


def test_scores_for_candidates(cf_model, tiny_interactions):
    seen = set(tiny_interactions[tiny_interactions["user_id"] == "u1"]["song_id"])
    candidates = ["s04", "s05", "s08"]
    scores = cf_model.scores_for_candidates("u1", candidates)
    assert list(scores.index) == candidates
    assert (scores >= 0).all()
