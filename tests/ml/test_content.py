"""Tests for the content-based recommender."""
from __future__ import annotations

import pytest

from ml.recommenders.content import ContentRecommender


@pytest.fixture(scope="module")
def content_model(tiny_songs, tiny_interactions):
    return ContentRecommender(embedding_config={"provider": "none"}).fit(tiny_songs, tiny_interactions)


def test_fit_builds_feature_space(content_model, tiny_songs):
    assert content_model.is_fitted
    assert content_model.X.shape[0] == len(tiny_songs)
    assert content_model.X.shape[1] > 0


def test_similar_songs_prefers_same_genre(content_model):
    similar = content_model.similar_songs("s00", k=3)
    assert len(similar) == 3
    assert "s00" not in similar.index
    # electronic track should have electronic neighbors
    assert all(s in {"s01", "s02", "s03"} for s in similar.index)


def test_similar_songs_unknown_id(content_model):
    assert content_model.similar_songs("nope").empty


def test_recommend_for_user_matches_taste(content_model, tiny_interactions):
    user_events = tiny_interactions[tiny_interactions["user_id"] == "u3"]  # jazz fan
    recs = content_model.recommend_for_user("u3", k=3, interactions=user_events)
    assert len(recs) > 0
    # recommendations should be jazz songs the user has not positively consumed only
    jazz = {"s04", "s05", "s06", "s07"}
    assert len(jazz & set(recs.index)) >= 1


def test_recommend_for_user_excludes_history(content_model, tiny_interactions):
    user_events = tiny_interactions[tiny_interactions["user_id"] == "u1"]
    seen = set(user_events["song_id"])
    recs = content_model.recommend_for_user("u1", k=10, exclude=seen, interactions=user_events)
    assert not (set(recs.index) & seen)


def test_dislikes_penalized(content_model, tiny_interactions):
    """A user disliking electronic should not get electronic tracks on top."""
    events = tiny_interactions[tiny_interactions["user_id"] == "u6"].copy()
    # turn electronic plays into dislikes
    events.loc[events["song_id"].isin(["s00", "s01", "s03"]), "event_type"] = "dislike"
    recs = content_model.recommend_for_user("u6", k=3, interactions=events)
    assert not ({"s00", "s01", "s03"} & set(recs.index))


def test_add_songs_cold_start_item(content_model, tiny_songs):
    new_song = tiny_songs.iloc[[0]].copy()
    new_song["song_id"] = "s99"
    new_song["title"] = "Brand New Electronic Track"
    content_model.add_songs(new_song)
    assert "s99" in content_model.song_to_idx
    similar = content_model.similar_songs("s99", k=3)
    assert len(similar) == 3 and "s99" not in similar.index


def test_similarity_matrix_symmetric(content_model):
    sim = content_model.similarity(["s00", "s04", "s08"])
    assert sim.shape == (3, 3)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(sim[1, 0], abs=1e-6)
    # electronic vs jazz should be less similar than electronic vs electronic
    sim2 = content_model.similarity(["s00", "s01"])
    assert sim2[0, 1] > sim[0, 1]
