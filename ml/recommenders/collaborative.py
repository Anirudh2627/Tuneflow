"""Collaborative filtering recommender.

Primary backend: implicit-feedback ALS via the ``implicit`` library.
Fallback backend (always available, pure scipy/sklearn): item-item
cosine CF on the confidence matrix.  The backend is selected by config
(``collaborative.backend: auto | als | item_item``) so the class API is
identical either way.

Never train this inside the API -- use ``scripts/train_collaborative.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from ml.features.interaction_matrix import InteractionIndex, build_interaction_matrix
from ml.recommenders.base import BaseRecommender
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class CollaborativeRecommender(BaseRecommender):
    name = "collaborative"
    family = "collaborative"

    def __init__(self, backend: str = "auto", factors: int = 64, iterations: int = 15,
                 regularization: float = 0.05, alpha: float = 20.0, n_neighbors: int = 100):
        self.backend_pref = backend
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.n_neighbors = n_neighbors

        self.backend: str = "item_item"
        self.index: InteractionIndex | None = None
        self.matrix: sparse.csr_matrix | None = None      # positive confidence
        self.model = None                                  # implicit ALS model
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_similarity: sparse.csr_matrix | None = None  # top-k neighbors
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, interactions: pd.DataFrame, index: InteractionIndex | None = None) -> "CollaborativeRecommender":
        matrix, index = build_interaction_matrix(interactions, index=index, positive_only=True)
        self.matrix = matrix
        self.index = index
        logger.info("CF matrix: %d users x %d songs, nnz=%d, density=%.4f%%",
                    index.n_users, index.n_songs, matrix.nnz,
                    100.0 * matrix.nnz / max(index.n_users * index.n_songs, 1))

        backend = self.backend_pref
        if backend == "auto":
            backend = "als" if _implicit_available() else "item_item"
            if backend == "item_item":
                logger.info("implicit library unavailable -> falling back to item-item CF")

        if backend == "als":
            self._fit_als(matrix)
        else:
            self._fit_item_item(matrix)
        self.backend = backend
        self._fitted = True
        return self

    def _fit_als(self, matrix: sparse.csr_matrix) -> None:
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=42,
            use_gpu=False,
        )
        confidence = (matrix * self.alpha).astype(np.float32)
        model.fit(confidence, show_progress=False)
        self.model = model
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        logger.info("ALS fitted: factors=%d iterations=%d", self.factors, self.iterations)

    def _fit_item_item(self, matrix: sparse.csr_matrix) -> None:
        binary = (matrix > 0).astype(np.float32)
        norms = np.sqrt(np.asarray(binary.multiply(binary).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        normed = binary.multiply(1.0 / norms).tocsr()
        sim = (normed.T @ normed).tocsr()          # song x song cosine
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.item_similarity = _keep_topk_per_row(sim, self.n_neighbors)
        # cheap latent factors for user/item similarity endpoints (TruncatedSVD)
        try:
            from sklearn.utils.extmath import randomized_svd

            k = min(self.factors, min(sim.shape) - 1, 32)
            if k >= 4:
                u, s, vt = randomized_svd(binary, n_components=k, random_state=42)
                self.user_factors = (u * s).astype(np.float32)
                self.item_factors = vt.T.astype(np.float32)
        except Exception as exc:  # pragma: no cover
            logger.warning("SVD factors unavailable (%s); similar_* endpoints use raw co-occurrence", exc)
        logger.info("Item-item CF fitted: %d neighbors/item", self.n_neighbors)

    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id: str, k: int = 10, exclude: set[str] | None = None) -> pd.Series:
        if not self._fitted or self.index is None:
            return pd.Series(dtype=float)
        uidx = self.index.user_index(str(user_id))
        if uidx is None:
            logger.debug("recommend_for_user: unknown user %r (cold start)", user_id)
            return pd.Series(dtype=float)

        exclude = {str(s) for s in (exclude or set())}
        exclude_idx = [self.index.song_to_idx[s] for s in exclude if s in self.index.song_to_idx]

        if self.backend == "als" and self.model is not None:
            ids, scores = self.model.recommend(
                uidx, self.matrix[uidx], N=k + len(exclude_idx), filter_already_liked_items=True,
            )
            pairs = [(self.index.song_ids[i], float(s)) for i, s in zip(ids, scores)
                     if self.index.song_ids[i] not in exclude]
            pairs = pairs[:k]
            return pd.Series({s: v for s, v in pairs}, dtype=float)

        scores = self._item_item_scores(uidx)
        return self._series_from_scores(scores, k, exclude_idx)

    def _item_item_scores(self, uidx: int) -> np.ndarray:
        assert self.matrix is not None and self.item_similarity is not None
        user_row = self.matrix[uidx]
        return np.asarray((user_row @ self.item_similarity).todense()).ravel()

    def _series_from_scores(self, scores: np.ndarray, k: int, exclude_idx: list[int]) -> pd.Series:
        assert self.index is not None
        scores = scores.copy()
        scores[exclude_idx] = -np.inf
        # the user's own history is always excluded
        order = np.argsort(-scores)[:k]
        order = [i for i in order if np.isfinite(scores[i]) and scores[i] > 0]
        return pd.Series(scores[order], index=[self.index.song_ids[i] for i in order], dtype=float)

    # ------------------------------------------------------------------
    def similar_users(self, user_id: str, k: int = 10) -> pd.Series:
        if not self._fitted or self.index is None or self.user_factors is None:
            return pd.Series(dtype=float)
        uidx = self.index.user_index(str(user_id))
        if uidx is None:
            return pd.Series(dtype=float)
        factors = normalize(self.user_factors, norm="l2")
        sims = factors @ factors[uidx]
        sims[uidx] = -np.inf
        order = np.argsort(-sims)[:k]
        return pd.Series(sims[order], index=[self.index.user_ids[i] for i in order], dtype=float)

    def similar_items(self, song_id: str, k: int = 10, exclude: set[str] | None = None) -> pd.Series:
        if not self._fitted or self.index is None:
            return pd.Series(dtype=float)
        sidx = self.index.song_index(str(song_id))
        if sidx is None:
            return pd.Series(dtype=float)
        exclude = {str(s) for s in (exclude or set())} | {str(song_id)}
        exclude_idx = [self.index.song_to_idx[s] for s in exclude if s in self.index.song_to_idx]

        if self.item_factors is not None and self.backend == "als":
            factors = normalize(self.item_factors, norm="l2")
            scores = factors @ factors[sidx]
        elif self.item_similarity is not None:
            scores = np.asarray(self.item_similarity[sidx].todense()).ravel()
        else:  # pragma: no cover
            return pd.Series(dtype=float)
        return self._series_from_scores(scores, k, exclude_idx)

    def scores_for_candidates(self, user_id: str, song_ids: list[str]) -> pd.Series:
        """Raw CF scores for a specific candidate list (used by the ranker)."""
        if not self._fitted or self.index is None:
            return pd.Series(0.0, index=song_ids)
        uidx = self.index.user_index(str(user_id))
        if uidx is None:
            return pd.Series(0.0, index=song_ids)
        if self.backend == "als" and self.user_factors is not None and self.item_factors is not None:
            idxs = [self.index.song_to_idx.get(s) for s in song_ids]
            scores = np.array([
                float(self.user_factors[uidx] @ self.item_factors[i]) if i is not None else 0.0
                for i in idxs
            ])
        else:
            full = self._item_item_scores(uidx)
            scores = np.array([full[self.index.song_to_idx[s]] if s in self.index.song_to_idx else 0.0
                               for s in song_ids])
        return pd.Series(scores, index=song_ids, dtype=float)


def _implicit_available() -> bool:
    try:
        import implicit  # noqa: F401
        return True
    except Exception:
        return False


def _keep_topk_per_row(sim: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Zero out all but the top-k similarities in each row."""
    n_rows = sim.shape[0]
    out_rows, out_cols, out_data = [], [], []
    csr = sim.tocsr()
    for i in range(n_rows):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        if end - start <= k:
            out_rows.extend([i] * (end - start))
            out_cols.extend(csr.indices[start:end])
            out_data.extend(csr.data[start:end])
        else:
            row_data = csr.data[start:end]
            top = np.argpartition(-row_data, k)[:k]
            out_rows.extend([i] * k)
            out_cols.extend(csr.indices[start:end][top])
            out_data.extend(row_data[top])
    return sparse.csr_matrix(
        (np.asarray(out_data, dtype=np.float32), (out_rows, out_cols)), shape=csr.shape
    )
