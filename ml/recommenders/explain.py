"""Recommendation explanation generation.

Explanations are derived from the *actual* ranking features that drove
each item's score (weighted contributions), never hardcoded per song.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ml.features.user_profile import UserProfile

FEATURE_PHRASES: dict[str, str] = {
    "genre_affinity": "matches your top genres ({detail})",
    "artist_affinity": "you often listen to {detail}",
    "tag_affinity": "fits moods you enjoy ({detail})",
    "content_score": "sounds similar to tracks you liked",
    "collaborative_score": "listeners with similar taste also play this",
    "session_score": "fits what you are listening to right now",
    "popularity_score": "trending among listeners recently",
    "song_popularity": "popular on TuneFlow right now",
    "audio_match": "matches your preferred energy and mood",
    "novelty": "a fresh discovery outside your usual rotation",
    "artist_popularity": "from a widely followed artist",
    "release_recency": "a recent release",
}

FEATURE_SIGNALS: dict[str, str] = {
    "genre_affinity": "genre_affinity",
    "artist_affinity": "artist_affinity",
    "tag_affinity": "tag_affinity",
    "content_score": "content_similarity",
    "collaborative_score": "collaborative_filtering",
    "session_score": "recent_session",
    "popularity_score": "popularity",
    "song_popularity": "popularity",
    "audio_match": "audio_profile_match",
    "novelty": "novelty",
}


@dataclass
class Explanation:
    reason: str
    signals: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)


def explain_item(
    song_id: str,
    song_row: pd.Series,
    feature_row: pd.Series,
    ranker_weights: dict[str, float],
    profile: UserProfile | None = None,
    max_phrases: int = 3,
) -> Explanation:
    """Compose a human-readable reason from the strongest weighted signals."""
    contributions: dict[str, float] = {}
    for feature, weight in ranker_weights.items():
        if weight == 0 or feature not in feature_row.index:
            continue
        value = float(feature_row[feature])
        if value > 0:
            contributions[feature] = weight * value

    ranked = sorted(contributions.items(), key=lambda kv: -kv[1])
    signals: list[str] = []
    phrases: list[str] = []
    for feature, _ in ranked[:max_phrases]:
        signal = FEATURE_SIGNALS.get(feature)
        if signal and signal not in signals:
            signals.append(signal)
        phrase = FEATURE_PHRASES.get(feature)
        if not phrase:
            continue
        detail = _detail_for(feature, song_row, profile)
        phrases.append(phrase.format(detail=detail) if "{detail}" in phrase else phrase)

    if not phrases:
        reason = "Recommended for you based on overall listening activity."
        signals = signals or ["popularity"]
    else:
        reason = "Recommended because " + _join_phrases(phrases) + "."

    if profile is not None and profile.is_cold_start:
        reason = ("Welcome to TuneFlow! Here are popular tracks to get you started — "
                  "like or skip a few and we will personalize quickly.")
        signals = ["cold_start", "popularity"]

    return Explanation(reason=reason, signals=signals, contributions=contributions)


def _detail_for(feature: str, song_row: pd.Series, profile: UserProfile | None) -> str:
    if feature == "genre_affinity":
        genres = song_row.get("genres") or []
        if isinstance(genres, (list, tuple)) and genres:
            top = [g for g in genres if profile is None or g in profile.genre_affinity] or list(genres)
            return ", ".join(str(g) for g in top[:3])
        return "your genres"
    if feature == "artist_affinity":
        return str(song_row.get("artist_name", "this artist"))
    if feature == "tag_affinity":
        tags = song_row.get("tags") or []
        if isinstance(tags, (list, tuple)) and tags:
            return ", ".join(str(t) for t in tags[:3])
        return "your moods"
    return ""


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f" and {phrases[-1]}"
