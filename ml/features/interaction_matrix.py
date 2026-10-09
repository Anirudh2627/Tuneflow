"""User-item interaction matrix construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class InteractionIndex:
    """Bidirectional user/song id <-> matrix index maps + CSR matrix."""

    def __init__(self, user_ids: list[str], song_ids: list[str]):
        self.user_ids = list(user_ids)
        self.song_ids = list(song_ids)
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.song_to_idx = {s: i for i, s in enumerate(self.song_ids)}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_songs(self) -> int:
        return len(self.song_ids)

    def user_index(self, user_id: str) -> int | None:
        return self.user_to_idx.get(user_id)

    def song_index(self, song_id: str) -> int | None:
        return self.song_to_idx.get(song_id)

    def to_dict(self) -> dict:
        return {"user_ids": self.user_ids, "song_ids": self.song_ids}

    @classmethod
    def from_dict(cls, data: dict) -> "InteractionIndex":
        return cls(data["user_ids"], data["song_ids"])


def build_interaction_matrix(
    interactions: pd.DataFrame,
    index: InteractionIndex | None = None,
    weight_column: str = "weight",
    positive_only: bool = True,
) -> tuple[sparse.csr_matrix, InteractionIndex]:
    """Build a CSR user x song confidence matrix.

    Repeated events sum.  ``positive_only`` clips negative weights
    (skip/dislike) to zero because implicit-feedback factorization
    assumes non-negative confidence; negatives are handled separately
    by profile/ranking logic.
    """
    df = interactions
    if index is None:
        user_ids = sorted(df["user_id"].astype(str).unique())
        song_ids = sorted(df["song_id"].astype(str).unique())
        index = InteractionIndex(user_ids, song_ids)

    rows = df["user_id"].astype(str).map(index.user_to_idx)
    cols = df["song_id"].astype(str).map(index.song_to_idx)
    keep = rows.notna() & cols.notna()
    rows, cols = rows[keep].to_numpy(), cols[keep].to_numpy()
    if weight_column in df.columns:
        weights = df.loc[keep, weight_column].to_numpy(dtype=np.float32)
    else:
        weights = np.ones(int(keep.sum()), dtype=np.float32)
    if positive_only:
        weights = np.clip(weights, 0, None)

    matrix = sparse.coo_matrix((weights, (rows, cols)), shape=(index.n_users, index.n_songs))
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    return matrix, index
