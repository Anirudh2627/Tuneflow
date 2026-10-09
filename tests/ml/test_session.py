"""Tests for the session-based recommender."""
from __future__ import annotations

import pandas as pd
import pytest

from ml.recommenders.session import SessionRecommender


@pytest.fixture(scope="module")
def session_model(tiny_interactions):
    return SessionRecommender(decay=0.5, n_neighbors=10).fit(tiny_interactions)


def test_fit(session_model):
    assert session_model.is_fitted
    assert session_model.transition_sim is not None
    assert len(session_model.song_ids) == 11  # s11 never appears in positive events


def test_transition_similarity_learned(session_model):
    """Users repeatedly play electronic tracks together -> high s00->s01 sim."""
    sim = session_model.transition_sim
    i0, i1 = session_model.song_to_idx["s00"], session_model.song_to_idx["s01"]
    i8 = session_model.song_to_idx["s08"]
    assert sim[i0, i1] > 0
    assert sim[i0, i1] >= sim[i0, i8]


def test_recommend_for_session(session_model):
    recs = session_model.recommend_for_session(["s00", "s01"], k=4)
    assert len(recs) > 0
    assert "s00" not in recs.index and "s01" not in recs.index


def test_recency_weighting(session_model):
    """A session ending on jazz tracks should score jazz neighbors higher."""
    jazz_end = session_model.recommend_for_session(["s00", "s04", "s05"], k=3)
    jazz_start = session_model.recommend_for_session(["s04", "s05", "s00"], k=3)
    jazz = {"s04", "s05", "s06", "s07"}
    score_jazz_end = sum(v for k, v in jazz_end.items() if k in jazz)
    score_jazz_start = sum(v for k, v in jazz_start.items() if k in jazz)
    assert score_jazz_end >= score_jazz_start


def test_empty_and_unknown_session(session_model):
    assert session_model.recommend_for_session([]).empty
    assert session_model.recommend_for_session(["unknown_song"]).empty


def test_session_scores_for_candidates(session_model):
    scores = session_model.session_scores(["s00"], ["s01", "s08", "unknown"])
    assert list(scores.index) == ["s01", "s08", "unknown"]
    assert scores["unknown"] == 0.0


def test_recommend_for_user_uses_cached_session(session_model):
    recs = session_model.recommend_for_user("u1", k=3)
    assert isinstance(recs, pd.Series)


def test_dataframe_without_timestamps_or_sessions():
    """Chunked-session fallback when the log has no time info (MSD case)."""
    rows = [{"user_id": "u", "song_id": f"s{i % 5}", "event_type": "play"} for i in range(30)]
    model = SessionRecommender().fit(pd.DataFrame(rows))
    recs = model.recommend_for_session(["s0", "s1"], k=3)
    assert len(recs) > 0
