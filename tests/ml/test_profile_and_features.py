"""Tests for user profiles, ranking features and the evaluation harness."""
from __future__ import annotations

import pandas as pd
import pytest

from ml.evaluation.evaluator import RecommenderEvaluator
from ml.features.ranking_features import RANKING_FEATURES, RankingFeatureExtractor
from ml.features.user_profile import UserProfileService
from ml.recommenders.popularity import PopularityRecommender


@pytest.fixture(scope="module")
def profile_service(tiny_songs):
    return UserProfileService(tiny_songs)


def test_user_profile_affinities(profile_service, tiny_interactions):
    events = tiny_interactions[tiny_interactions["user_id"] == "u1"]
    profile = profile_service.build("u1", events)
    assert not profile.is_cold_start
    assert profile.top_genres[0] == "electronic"
    assert "ar1" in profile.artist_affinity
    assert abs(sum(profile.genre_affinity.values()) - 1.0) < 1e-6
    assert profile.audio_feature_means["energy"] > 0.6
    assert profile.listening_stats["positive_events"] > 0
    assert len(profile.recent_songs) > 0


def test_user_profile_cold_start(profile_service):
    profile = profile_service.build("ghost", pd.DataFrame())
    assert profile.is_cold_start
    assert profile.top_genres == []


def test_user_profile_sessions(profile_service, tiny_interactions):
    events = tiny_interactions[tiny_interactions["user_id"] == "u6"]
    profile = profile_service.build("u6", events)
    assert len(profile.recent_sessions) > 0


def test_ranking_feature_extraction(tiny_songs, tiny_interactions, profile_service):
    pop = PopularityRecommender(recent_window_days=365).fit(tiny_interactions, tiny_songs)
    extractor = RankingFeatureExtractor(songs=tiny_songs, popularity=pop.global_popularity)
    events = tiny_interactions[tiny_interactions["user_id"] == "u1"]
    profile = profile_service.build("u1", events)

    candidates = ["s04", "s05", "s08"]
    source_scores = {"content": pd.Series({"s04": 0.8, "s05": 0.6}),
                     "session": pd.Series({"s04": 0.3})}
    features = extractor.extract(profile, candidates, source_scores)

    assert list(features.columns) == RANKING_FEATURES
    assert features.loc["s04", "content_score"] > features.loc["s05", "content_score"]
    assert features.loc["s08", "content_score"] == 0.0
    # u1 loves electronic -> jazz affinity should be 0
    assert features.loc["s04", "genre_affinity"] == 0.0
    assert 0.0 <= features.loc["s04", "audio_match"] <= 1.0


def test_ranking_features_cold_start_user(tiny_songs, tiny_interactions, profile_service):
    extractor = RankingFeatureExtractor(songs=tiny_songs)
    profile = profile_service.build("ghost", pd.DataFrame())
    features = extractor.extract(profile, ["s00", "s01"], {})
    assert (features["genre_affinity"] == 0).all()
    assert (features["audio_match"] == 0).all()


def test_evaluator_end_to_end(tiny_songs, tiny_interactions):
    """The harness must produce finite metrics for a trivial recommender."""
    from ml.preprocessing.splits import split_interactions

    splits = split_interactions(tiny_interactions, train_frac=0.6, val_frac=0.2, min_events_for_split=5)
    pop = PopularityRecommender(recent_window_days=365).fit(splits["train"], tiny_songs)
    evaluator = RecommenderEvaluator(
        songs=tiny_songs, train=splits["train"], test=splits["test"],
        popularity=pop.global_popularity)
    result = evaluator.evaluate(pop, "popularity", k=5)
    assert result.n_users_evaluated >= 0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.coverage <= 1.0
    assert result.latency_ms_per_user >= 0.0
