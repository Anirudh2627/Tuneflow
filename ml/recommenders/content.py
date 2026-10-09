"""Content-based recommender.

Song feature space (audio features + genres + tags + TF-IDF text, all
optional/graceful) with cosine similarity:

* ``similar_songs`` / ``recommend_for_song``: item-item cosine over the
  row-normalized feature matrix.
* ``recommend_for_user``: builds a taste vector as an event-weighted
  centroid of the user's liked/played songs minus a penalty centroid of
  disliked songs, then ranks by cosine.

An optional SentenceTransformer embedding block can be blended in via
config for stronger semantic matching.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from ml.embeddings.provider import EmbeddingProvider, build_embedding_provider
from ml.features.song_features import SongFeatureBuilder, song_text
from ml.recommenders.base import BaseRecommender
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class ContentRecommender(BaseRecommender):
    name = "content"
    family = "content"

    def __init__(self, embedding_config: dict | None = None, embedding_weight: float = 0.3,
                 like_weight: float = 2.0, play_weight: float = 1.0, dislike_penalty: float = 1.5,
                 profile_history_limit: int = 60):
        self.embedding_config = embedding_config or {"provider": "none"}
        self.embedding_weight = embedding_weight
        self.like_weight = like_weight
        self.play_weight = play_weight
        self.dislike_penalty = dislike_penalty
        self.profile_history_limit = profile_history_limit

        self.feature_builder = SongFeatureBuilder()
        self.song_ids: list[str] = []
        self.song_to_idx: dict[str, int] = {}
        self.X: sparse.csr_matrix | None = None          # row-normalized
        self.embeddings: np.ndarray | None = None        # row-normalized, optional
        self._embed_provider: EmbeddingProvider | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, songs: pd.DataFrame, interactions: pd.DataFrame | None = None) -> "ContentRecommender":
        songs = songs.sort_values("song_id").reset_index(drop=True)
        self.song_ids = songs["song_id"].astype(str).tolist()
        self.song_to_idx = {s: i for i, s in enumerate(self.song_ids)}

        self.feature_builder = SongFeatureBuilder().fit(songs)
        X = self.feature_builder.transform(songs)
        self.X = normalize(X, norm="l2")

        provider = build_embedding_provider(self.embedding_config)
        if provider is not None:
            try:
                texts = songs.apply(song_text, axis=1).tolist()
                emb = provider.embed(texts)
                self.embeddings = normalize(emb, norm="l2")
                logger.info("Content embeddings built with %s (dim=%d)", provider.name(), emb.shape[1])
            except Exception as exc:  # keep content CF working without embeddings
                logger.warning("Embedding provider failed (%s); continuing without embeddings", exc)
                self.embeddings = None

        if interactions is not None:
            logger.debug("ContentRecommender.fit received %d interactions (user profiles are computed on demand)", len(interactions))
        self._fitted = True
        logger.info("ContentRecommender fitted on %d songs (%d features%s)",
                    len(self.song_ids), self.X.shape[1],
                    f", +{self.embeddings.shape[1]}d embeddings" if self.embeddings is not None else "")
        return self

    # ------------------------------------------------------------------
    def add_songs(self, songs: pd.DataFrame) -> None:
        """Register brand-new songs (cold-start items) without refitting."""
        new = songs[~songs["song_id"].astype(str).isin(self.song_to_idx)].sort_values("song_id")
        if new.empty:
            return
        X_new = normalize(self.feature_builder.transform(new), norm="l2")
        self.X = sparse.vstack([self.X, X_new], format="csr")  # type: ignore[arg-type]
        if self.embeddings is not None:
            provider = self._get_embed_provider()
            emb_new = normalize(provider.embed(new.apply(song_text, axis=1).tolist()), norm="l2")
            self.embeddings = np.vstack([self.embeddings, emb_new])
        for song_id in new["song_id"].astype(str):
            self.song_to_idx[song_id] = len(self.song_ids)
            self.song_ids.append(song_id)
        logger.info("ContentRecommender: added %d new songs (catalog=%d)", len(new), len(self.song_ids))

    def _get_embed_provider(self) -> EmbeddingProvider:
        if self._embed_provider is None:
            provider = build_embedding_provider(self.embedding_config)
            if provider is None:
                raise RuntimeError("Embedding provider is disabled")
            self._embed_provider = provider
        return self._embed_provider

    # ------------------------------------------------------------------
    def similarity(self, song_ids: list[str]) -> np.ndarray | None:
        """Pairwise cosine similarity between the given catalog songs."""
        idxs = [self.song_to_idx.get(s) for s in song_ids]
        if any(i is None for i in idxs):
            return None
        idx_arr = np.asarray(idxs, dtype=int)
        sim = (self.X[idx_arr] @ self.X[idx_arr].T).toarray()  # type: ignore[index]
        if self.embeddings is not None:
            emb_sim = self.embeddings[idx_arr] @ self.embeddings[idx_arr].T
            sim = (1 - self.embedding_weight) * sim + self.embedding_weight * emb_sim
        np.fill_diagonal(sim, 1.0)
        return sim

    def _similarity_to_all(self, idx: int) -> np.ndarray:
        sims = np.asarray((self.X[idx] @ self.X.T).todense()).ravel()  # type: ignore[index]
        if self.embeddings is not None:
            emb_sims = self.embeddings @ self.embeddings[idx]
            sims = (1 - self.embedding_weight) * sims + self.embedding_weight * emb_sims
        sims[idx] = -np.inf
        return sims

    # ------------------------------------------------------------------
    def similar_songs(self, song_id: str, k: int = 10, exclude: set[str] | None = None) -> pd.Series:
        idx = self.song_to_idx.get(str(song_id))
        if idx is None or self.X is None:
            logger.warning("similar_songs: unknown song_id %r", song_id)
            return pd.Series(dtype=float)
        exclude = {str(s) for s in (exclude or set())} | {str(song_id)}
        sims = self._similarity_to_all(idx)
        return self._topk(sims, k, exclude)

    def recommend_for_song(self, song_id: str, k: int = 10, exclude: set[str] | None = None) -> pd.Series:
        """Alias used by 'more like this song' surfaces."""
        return self.similar_songs(song_id, k=k, exclude=exclude)

    def recommend_for_user(self, user_id: str, k: int = 10, exclude: set[str] | None = None,
                           interactions: pd.DataFrame | None = None) -> pd.Series:
        """Taste-profile ranking.  Requires the user's interactions."""
        if interactions is None or len(interactions) == 0 or self.X is None:
            return pd.Series(dtype=float)
        df = interactions
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            df = df.sort_values("timestamp")
        df = df.tail(self.profile_history_limit * 3)

        taste = np.zeros(self.X.shape[1], dtype=np.float64)
        emb_taste = np.zeros(self.embeddings.shape[1], dtype=np.float64) if self.embeddings is not None else None
        seen: set[str] = set()
        for row in df.itertuples(index=False):
            idx = self.song_to_idx.get(str(row.song_id))
            seen.add(str(row.song_id))
            if idx is None:
                continue
            event = getattr(row, "event_type", "play")
            if event in ("like", "save"):
                w = self.like_weight
            elif event in ("skip", "dislike"):
                w = -self.dislike_penalty
            else:
                w = self.play_weight
            taste += w * np.asarray(self.X[idx].todense()).ravel()
            if emb_taste is not None:
                emb_taste += w * self.embeddings[idx]

        norm = np.linalg.norm(taste)
        if norm == 0:
            return pd.Series(dtype=float)
        taste /= norm
        scores = np.asarray(self.X @ taste).ravel()
        if emb_taste is not None:
            e_norm = np.linalg.norm(emb_taste)
            if e_norm > 0:
                scores = (1 - self.embedding_weight) * scores + self.embedding_weight * (self.embeddings @ (emb_taste / e_norm))

        exclude = {str(s) for s in (exclude or set())} | seen
        sims = np.where(np.isin(np.array(self.song_ids), list(exclude)), -np.inf, scores)
        order = np.argsort(-sims)[:k]
        order = [i for i in order if np.isfinite(sims[i])]
        return pd.Series(sims[order], index=[self.song_ids[i] for i in order], dtype=float)

    def profile_scores(self, taste_vector: np.ndarray) -> np.ndarray:
        """Cosine of every catalog song against an externally built taste vector."""
        assert self.X is not None
        return np.asarray(self.X @ taste_vector).ravel()

    # ------------------------------------------------------------------
    def _topk(self, sims: np.ndarray, k: int, exclude: set[str]) -> pd.Series:
        if exclude:
            mask = np.isin(np.array(self.song_ids), list(exclude))
            sims = np.where(mask, -np.inf, sims)
        order = np.argsort(-sims)[:k]
        order = [i for i in order if np.isfinite(sims[i])]
        return pd.Series(sims[order], index=[self.song_ids[i] for i in order], dtype=float)
