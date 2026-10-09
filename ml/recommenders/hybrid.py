"""Hybrid recommendation engine.

Pipeline (mirrors the architecture diagram):

    candidate generation (content + CF + session + popularity)
      -> merge
      -> ranking feature extraction
      -> ranking (BaseRanker)
      -> filtering (seen / disliked / saved-hidden)
      -> diversification (MMR)
      -> explanations
      -> top-k

Everything is configurable through ``configs/*.yaml`` (hybrid section);
nothing is hardcoded.  This class orchestrates trained artifacts -- it
never trains anything itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.features.ranking_features import RankingFeatureExtractor
from ml.features.user_profile import UserProfile, UserProfileService
from ml.ranking.base import BaseRanker
from ml.ranking.diversity import mmr_rerank
from ml.recommenders.base import BaseRecommender
from ml.recommenders.collaborative import CollaborativeRecommender
from ml.recommenders.content import ContentRecommender
from ml.recommenders.explain import explain_item
from ml.recommenders.popularity import PopularityRecommender
from ml.recommenders.session import SessionRecommender
from ml.utils.logging import get_logger

logger = get_logger(__name__)

MODES = ("personalized", "discover", "recent", "trending")


@dataclass
class Recommendation:
    song_id: str
    score: float
    reason: str
    signals: list[str] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class HybridConfig:
    weights: dict[str, float]
    ranker_weights: dict[str, float]
    candidate_counts: dict[str, int]
    diversity: dict
    modes: dict
    recent_session_limit: int = 12


class HybridRecommender(BaseRecommender):
    name = "hybrid"
    family = "ranking"

    def __init__(
        self,
        songs: pd.DataFrame,
        content: ContentRecommender | None,
        collaborative: CollaborativeRecommender | None,
        session: SessionRecommender | None,
        popularity: PopularityRecommender,
        ranker: BaseRanker,
        hybrid_config: dict,
        profile_service: UserProfileService | None = None,
    ):
        self.songs = songs.set_index("song_id", drop=False)
        self.content = content
        self.collaborative = collaborative
        self.session = session
        self.popularity = popularity
        self.ranker = ranker
        self.profile_service = profile_service or UserProfileService(songs)

        hybrid_config = hybrid_config or {}
        self.config = HybridConfig(
            weights=hybrid_config.get("weights", {}) or {},
            ranker_weights=hybrid_config.get("ranker_weights", {}) or {},
            candidate_counts=hybrid_config.get("candidate_counts",
                                               {"content": 100, "collaborative": 100, "session": 60, "popularity": 60}),
            diversity=hybrid_config.get("diversity", {"relevance_weight": 0.7, "diversity_weight": 0.3,
                                                      "novelty_weight": 0.05, "enabled": True}),
            modes=hybrid_config.get("modes", {}),
            recent_session_limit=int(hybrid_config.get("recent_session_limit", 12)),
        )
        counts = self.config.candidate_counts
        popularity_counts = popularity.global_popularity
        self.feature_extractor = RankingFeatureExtractor(
            songs=songs,
            popularity=popularity_counts,
            artist_popularity=self._artist_popularity(songs, popularity_counts),
        )
        self._fitted = all(x is not None for x in (content, collaborative, session, popularity, ranker))

    @staticmethod
    def _artist_popularity(songs: pd.DataFrame, song_pop: pd.Series) -> pd.Series:
        if song_pop is None or len(song_pop) == 0 or "artist_id" not in songs.columns:
            return pd.Series(dtype=float)
        artist_of = songs.set_index("song_id")["artist_id"].astype(str).to_dict()
        mapped = song_pop.groupby([artist_of.get(str(sid), "") for sid in song_pop.index]).sum()
        return mapped.drop(index="", errors="ignore")

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        mode: str = "personalized",
        interactions: pd.DataFrame | None = None,
        profile: UserProfile | None = None,
        session_items: list[str] | None = None,
        preferred_genres: list[str] | None = None,
        exclude_songs: set[str] | None = None,
    ) -> list[Recommendation]:
        """Produce top-k explained recommendations.

        ``interactions`` is the user's history (needed for profile-based
        sources when ``profile`` is not supplied).  ``session_items``
        overrides the offline session for session-aware ranking.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown recommendation mode {mode!r}; expected one of {MODES}")
        exclude = {str(s) for s in (exclude_songs or set())}

        if profile is None:
            profile = self.profile_service.build(user_id, interactions if interactions is not None
                                                 else pd.DataFrame())
        exclude |= set(profile.history_songs) | set(profile.disliked_songs)

        if session_items is None:
            session_items = profile.recent_songs[: self.config.recent_session_limit][::-1]  # oldest -> newest

        cold_start = profile.is_cold_start

        # ---------------- candidate generation ----------------
        counts = self.config.candidate_counts
        source_scores: dict[str, pd.Series] = {}
        n_pool = max(k * 5, 50)

        if cold_start:
            genres = preferred_genres or []
            pop_candidates = self._cold_start_candidates(k=n_pool * 2, genres=genres)
            source_scores["popularity"] = pop_candidates
        else:
            if self.content is not None:
                source_scores["content"] = self.content.recommend_for_user(
                    user_id, k=counts.get("content", 100), exclude=exclude, interactions=interactions)
            if self.collaborative is not None:
                source_scores["collaborative"] = self.collaborative.recommend_for_user(
                    user_id, k=counts.get("collaborative", 100), exclude=exclude)
            if self.session is not None and session_items:
                source_scores["session"] = self.session.recommend_for_session(
                    session_items, k=counts.get("session", 60), exclude=exclude)
            source_scores["popularity"] = self._popularity_candidates(
                k=counts.get("popularity", 60), profile=profile, mode=mode, exclude=exclude)

        candidates = sorted(set().union(*[set(s.index) for s in source_scores.values()])) if source_scores else []
        candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            logger.info("No candidates for user=%s mode=%s; falling back to global popularity", user_id, mode)
            fallback = self.popularity.top(k=k, exclude=exclude)
            candidates = list(fallback.index)
            source_scores = {"popularity": fallback}
        if not candidates:
            return []

        # ---------------- features + ranking ----------------
        features = self.feature_extractor.extract(profile, candidates, source_scores)
        ranker = self._mode_ranker(mode)
        scores = ranker.rank(features)

        # ---------------- filtering ----------------
        order = np.argsort(-scores)
        candidate_ids = [candidates[i] for i in order]
        features = features.loc[candidate_ids]
        scores = scores[order]

        # ---------------- diversification ----------------
        diversity = self.config.diversity
        pool_size = min(len(candidate_ids), n_pool)
        if diversity.get("enabled", True) and self.content is not None and pool_size > 1:
            pool = candidate_ids[:pool_size]
            pool_scores = scores[:pool_size]
            sim = self.content.similarity(pool)
            if sim is not None:
                selected = mmr_rerank(
                    pool_scores, sim, k=min(k, pool_size),
                    relevance_weight=float(diversity.get("relevance_weight", 0.7)),
                    diversity_weight=float(diversity.get("diversity_weight", 0.3)),
                    novelty=features["novelty"].to_numpy()[:pool_size],
                    novelty_weight=float(diversity.get("novelty_weight", 0.0)),
                )
                candidate_ids = [pool[i] for i in selected]
                scores = pool_scores[selected]
            else:
                candidate_ids, scores = pool[:k], pool_scores[:k]
        else:
            candidate_ids, scores = candidate_ids[:k], scores[:k]

        # ---------------- explanations ----------------
        results: list[Recommendation] = []
        for song_id, score in zip(candidate_ids[:k], scores[:k]):
            if song_id not in self.songs.index:
                continue
            song_row = self.songs.loc[song_id]
            feature_row = features.loc[song_id] if song_id in features.index else pd.Series(dtype=float)
            explanation = explain_item(
                song_id, song_row, feature_row,
                getattr(ranker, "weights", self.config.ranker_weights), profile,
            )
            sources = {name: float(series.get(song_id, 0.0)) for name, series in source_scores.items()}
            results.append(Recommendation(
                song_id=str(song_id), score=float(score), reason=explanation.reason,
                signals=explanation.signals, source_scores=sources,
            ))
        return results

    # ------------------------------------------------------------------
    def _reorder_scores(self, pool: list[str], selected: list[int],
                        score_map: dict[str, float]) -> tuple[list[str], np.ndarray]:
        ids = [pool[i] for i in selected]
        return ids, np.array([score_map[i] for i in ids], dtype=float)

    def recommend_for_user(self, user_id: str, k: int = 10, exclude: set[str] | None = None,
                           interactions: pd.DataFrame | None = None) -> pd.Series:
        """BaseRecommender-compatible simple interface."""
        recs = self.recommend(user_id, k=k, exclude_songs=exclude, interactions=interactions)
        return pd.Series({r.song_id: r.score for r in recs}, dtype=float)

    # ------------------------------------------------------------------
    def _mode_ranker(self, mode: str) -> BaseRanker:
        mode_cfg = self.config.modes.get(mode, {}) or {}
        overrides = mode_cfg.get("ranker_weight_overrides")
        if overrides:
            return self.ranker.with_weight_overrides(overrides)
        return self.ranker

    def _cold_start_candidates(self, k: int, genres: list[str]) -> pd.Series:
        """New users: onboarding genre picks + global/recent popularity."""
        parts: list[pd.Series] = []
        for genre in (genres or [])[:3]:
            parts.append(self.popularity.top(k=k // max(len(genres), 1) + 5, genre=genre))
        parts.append(self.popularity.top(k=k, recent=True))
        parts.append(self.popularity.top(k=k))
        merged = pd.concat(parts) if parts else pd.Series(dtype=float)
        if merged.empty:
            return merged
        merged = merged.groupby(level=0).max().sort_values(ascending=False)
        return merged.head(k)

    def _popularity_candidates(self, k: int, profile: UserProfile, mode: str, exclude: set[str]) -> pd.Series:
        parts = [self.popularity.top(k=k, exclude=exclude, recent=(mode in ("discover", "trending")))]
        for genre in profile.top_genres[:2]:
            parts.append(self.popularity.top(k=max(k // 3, 10), genre=genre, exclude=exclude))
        merged = pd.concat(parts).groupby(level=0).max().sort_values(ascending=False)
        return merged.head(k)
