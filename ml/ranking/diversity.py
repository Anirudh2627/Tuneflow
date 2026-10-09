"""Diversity-aware reranking (MMR).

Maximal Marginal Relevance greedily picks the item maximizing

    relevance_weight * relevance(i)
      - diversity_weight * max_{j in selected} similarity(i, j)

Similarity comes from the content feature space; a novelty bonus can
further demote globally popular (already well-known) items.
"""
from __future__ import annotations

import numpy as np


def mmr_rerank(
    scores: np.ndarray,
    similarity: np.ndarray,
    k: int,
    relevance_weight: float = 0.7,
    diversity_weight: float = 0.3,
    novelty: np.ndarray | None = None,
    novelty_weight: float = 0.0,
) -> list[int]:
    """Return indices of the selected items in selection order.

    Parameters
    ----------
    scores : (n,) relevance scores in [0, 1]-ish scale.
    similarity : (n, n) pairwise item similarity in [0, 1].
    novelty : optional (n,) novelty scores in [0, 1].
    """
    n = len(scores)
    if n == 0:
        return []
    k = min(k, n)
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        norm_scores = (scores - lo) / (hi - lo)
    else:
        norm_scores = np.full(n, 0.5)
    if novelty is not None and novelty_weight > 0:
        norm_scores = (1 - novelty_weight) * norm_scores + novelty_weight * np.asarray(novelty, dtype=np.float64)

    selected: list[int] = []
    remaining = set(range(n))
    max_sim = np.zeros(n, dtype=np.float64)

    for _ in range(k):
        best_idx, best_value = -1, -np.inf
        for idx in remaining:
            penalty = max_sim[idx] if selected else 0.0
            value = relevance_weight * norm_scores[idx] - diversity_weight * penalty
            if value > best_value:
                best_value, best_idx = value, idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
        if remaining:
            rem_list = list(remaining)
            sim_row = similarity[best_idx, rem_list]
            for pos, idx in enumerate(rem_list):
                max_sim[idx] = max(max_sim[idx], sim_row[pos])
    return selected
