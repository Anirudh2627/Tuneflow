"""Session-based recommender (sequence-aware baseline).

Model: item-item transition similarity learned from sessionized
listening logs, combined with exponential recency weighting over the
current session:

    score(candidate) = sum_i  decay^(n-1-i) * trans_sim(session[i], candidate)

This mirrors the item-KNN family of session-based baselines (cf. GRU4Rec
paper baselines).  The public API (``recommend_for_session``) is what a
future GRU4Rec / Transformer implementation should also satisfy, so it
can be swapped without touching callers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ml.recommenders.base import BaseRecommender
from ml.utils.logging import get_logger

logger = get_logger(__name__)

POSITIVE_EVENTS = ("play", "like", "save")


class SessionRecommender(BaseRecommender):
    name = "session"
    family = "session"

    def __init__(self, decay: float = 0.6, n_neighbors: int = 100, session_gap_minutes: int = 30,
                 max_session_len: int = 20):
        self.decay = decay
        self.n_neighbors = n_neighbors
        self.session_gap_minutes = session_gap_minutes
        self.max_session_len = max_session_len

        self.song_ids: list[str] = []
        self.song_to_idx: dict[str, int] = {}
        self.transition_sim: sparse.csr_matrix | None = None
        self.user_last_sessions: dict[str, list[str]] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, interactions: pd.DataFrame) -> "SessionRecommender":
        positive = interactions[interactions["event_type"].isin(POSITIVE_EVENTS)].copy()
        positive["song_id"] = positive["song_id"].astype(str)
        self.song_ids = sorted(positive["song_id"].unique())
        self.song_to_idx = {s: i for i, s in enumerate(self.song_ids)}
        n = len(self.song_ids)
        if n == 0:
            raise ValueError("No positive interactions for session model")

        sessions = self._sessionize(positive)
        cooc = sparse.lil_matrix((n, n), dtype=np.float32)
        for seq in sessions:
            idxs = [self.song_to_idx[s] for s in dict.fromkeys(seq) if s in self.song_to_idx]
            for pos, a in enumerate(idxs):
                for b in idxs[pos + 1:pos + 11]:  # transitions within a 10-item window
                    cooc[a, b] += 1.0
                    cooc[b, a] += 0.5  # weaker backward edge
        cooc = cooc.tocsr()
        self.transition_sim = self._normalize_and_prune(cooc)

        # remember each user's most recent session for offline recommend_for_user
        for seq, (user_id, _) in zip(sessions, self._session_meta):
            self.user_last_sessions[user_id] = seq[-self.max_session_len:]

        self._fitted = True
        logger.info("SessionRecommender fitted: %d songs, %d sessions, decay=%.2f",
                    n, len(sessions), self.decay)
        return self

    def _sessionize(self, positive: pd.DataFrame) -> list[list[str]]:
        """Split the log into sessions using explicit session_id when present,
        otherwise a time-gap heuristic."""
        sessions: list[list[str]] = []
        self._session_meta: list[tuple[str, pd.Timestamp | None]] = []
        has_explicit = "session_id" in positive.columns and positive["session_id"].notna().any()
        has_time = "timestamp" in positive.columns and positive["timestamp"].notna().any()

        for user_id, group in positive.groupby("user_id", sort=False):
            if has_explicit:
                for session_id, sess in group.groupby("session_id", sort=False):
                    if has_time:
                        sess = sess.sort_values("timestamp", kind="mergesort")
                    seq = sess["song_id"].tolist()
                    if seq:
                        sessions.append(seq[-self.max_session_len:])
                        ts = sess["timestamp"].max() if has_time else None
                        self._session_meta.append((user_id, ts))
            elif has_time:
                group = group.sort_values("timestamp", kind="mergesort")
                gap = pd.Timedelta(minutes=self.session_gap_minutes)
                current: list[str] = []
                last_ts = None
                for ts, song in zip(group["timestamp"], group["song_id"]):
                    if last_ts is not None and ts - last_ts > gap and current:
                        sessions.append(current[-self.max_session_len:])
                        self._session_meta.append((user_id, last_ts))
                        current = []
                    current.append(song)
                    last_ts = ts
                if current:
                    sessions.append(current[-self.max_session_len:])
                    self._session_meta.append((user_id, last_ts))
            else:
                seq = group["song_id"].tolist()
                for i in range(0, len(seq), self.max_session_len):
                    chunk = seq[i:i + self.max_session_len]
                    sessions.append(chunk)
                    self._session_meta.append((user_id, None))
        return sessions

    def _normalize_and_prune(self, cooc: sparse.csr_matrix) -> sparse.csr_matrix:
        """Cosine-normalize co-occurrences and keep top-k neighbors per row."""
        counts = np.sqrt(np.asarray(cooc.power(2).sum(axis=1)).ravel())
        counts[counts == 0] = 1.0
        normed = cooc.multiply(1.0 / counts[:, None]).tocsr()

        k = min(self.n_neighbors, normed.shape[1] - 1)
        if k <= 0:
            return normed
        rows, cols, data = [], [], []
        for i in range(normed.shape[0]):
            start, end = normed.indptr[i], normed.indptr[i + 1]
            if end <= start:
                continue
            row_data = normed.data[start:end]
            if end - start > k:
                top = np.argpartition(-row_data, k)[:k]
            else:
                top = np.arange(len(row_data))
            rows.extend([i] * len(top))
            cols.extend(normed.indices[start:end][top])
            data.extend(row_data[top])
        return sparse.csr_matrix((data, (rows, cols)), shape=normed.shape, dtype=np.float32)

    # ------------------------------------------------------------------
    def recommend_for_session(self, session_items: list[str], k: int = 10,
                              exclude: set[str] | None = None) -> pd.Series:
        """Score catalog items for an ordered (oldest -> newest) session."""
        if not self._fitted or self.transition_sim is None or not session_items:
            return pd.Series(dtype=float)
        known = [s for s in session_items if s in self.song_to_idx][-self.max_session_len:]
        if not known:
            return pd.Series(dtype=float)
        exclude = {str(s) for s in (exclude or set())} | set(session_items)

        scores = np.zeros(self.transition_sim.shape[1], dtype=np.float64)
        n = len(known)
        for pos, song in enumerate(known):
            weight = self.decay ** (n - 1 - pos)
            row = self.transition_sim[self.song_to_idx[song]]
            scores += weight * np.asarray(row.todense()).ravel()

        mask = np.isin(np.array(self.song_ids), list(exclude))
        scores[mask] = -np.inf
        order = np.argsort(-scores)[:k]
        order = [i for i in order if np.isfinite(scores[i]) and scores[i] > 0]
        return pd.Series(scores[order], index=[self.song_ids[i] for i in order], dtype=float)

    def session_scores(self, session_items: list[str], candidate_ids: list[str]) -> pd.Series:
        """Raw session scores for an arbitrary candidate list (ranker input)."""
        if not self._fitted or self.transition_sim is None or not session_items:
            return pd.Series(0.0, index=candidate_ids)
        known = [s for s in session_items if s in self.song_to_idx][-self.max_session_len:]
        scores = np.zeros(self.transition_sim.shape[1], dtype=np.float64)
        n = len(known)
        for pos, song in enumerate(known):
            scores += self.decay ** (n - 1 - pos) * np.asarray(
                self.transition_sim[self.song_to_idx[song]].todense()).ravel()
        values = [scores[self.song_to_idx[s]] if s in self.song_to_idx else 0.0 for s in candidate_ids]
        return pd.Series(values, index=candidate_ids, dtype=float)

    def recommend_for_user(self, user_id: str, k: int = 10, exclude: set[str] | None = None) -> pd.Series:
        """Offline fallback: use the user's last known training session."""
        session = self.user_last_sessions.get(str(user_id), [])
        return self.recommend_for_session(session, k=k, exclude=exclude)

    def add_session_data(self, interactions: pd.DataFrame) -> None:
        """Incrementally refresh cached user sessions (cheap; matrix stays)."""
        positive = interactions[interactions["event_type"].isin(POSITIVE_EVENTS)]
        sessions = self._sessionize(positive)
        for seq, (user_id, _) in zip(sessions, self._session_meta):
            self.user_last_sessions[user_id] = seq[-self.max_session_len:]
