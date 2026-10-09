"""Recommendation evaluation metrics.

All ranking metrics take a recommendation list (ordered song_ids) and a
ground-truth relevance set/weights for one user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits, sum_precision = 0, 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / min(len(relevant), k)


def map_at_k(recommendations: list[list[str]], relevants: list[set[str]], k: int) -> float:
    values = [average_precision_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevants) if rel]
    return float(np.mean(values)) if values else 0.0


def dcg_at_k(recommended: list[str], relevance_weights: dict[str, float], k: int) -> float:
    gains = [relevance_weights.get(item, 0.0) for item in recommended[:k]]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(np.asarray(gains) / discounts))


def ndcg_at_k(recommended: list[str], relevance_weights: dict[str, float], k: int) -> float:
    dcg = dcg_at_k(recommended, relevance_weights, k)
    ideal = dcg_at_k(
        sorted(relevance_weights, key=lambda s: -relevance_weights[s]), relevance_weights, k)
    return float(dcg / ideal) if ideal > 0 else 0.0


def mrr(recommendations: list[list[str]], relevants: list[set[str]]) -> float:
    values = []
    for rec, rel in zip(recommendations, relevants):
        if not rel:
            continue
        for rank, item in enumerate(rec, start=1):
            if item in rel:
                values.append(1.0 / rank)
                break
        else:
            values.append(0.0)
    return float(np.mean(values)) if values else 0.0


# ----------------------------------------------------------------------
# Beyond-accuracy metrics
# ----------------------------------------------------------------------
def coverage(all_recommendations: list[list[str]], catalog_size: int) -> float:
    """Fraction of the catalog that ever appears in recommendations."""
    if catalog_size <= 0:
        return 0.0
    recommended_items = {item for rec in all_recommendations for item in rec}
    return len(recommended_items) / catalog_size


def catalog_coverage(all_recommendations: list[list[str]], catalog_ids: set[str]) -> float:
    return coverage(all_recommendations, len(catalog_ids))


def intra_list_diversity(recommended: list[str], similarity_lookup) -> float:
    """Mean pairwise *distance* (1 - cosine similarity) within a list.

    ``similarity_lookup(song_a, song_b)`` returns similarity in [0, 1].
    """
    if len(recommended) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            total += similarity_lookup(recommended[i], recommended[j])
            pairs += 1
    return float(1.0 - total / pairs) if pairs else 0.0


def novelty(recommended: list[str], popularity: pd.Series, n_users: int) -> float:
    """Self-information based novelty: mean -log2(p(item)) (higher = newer)."""
    if n_users <= 0:
        return 0.0
    values = []
    for item in recommended:
        count = float(popularity.get(item, 0.0))
        prob = count / n_users
        if prob > 0:
            values.append(-np.log2(prob))
    return float(np.mean(values)) if values else 0.0


def gini_index(counts: np.ndarray) -> float:
    """Inequality of recommendation frequency across items (0 = uniform)."""
    counts = np.sort(np.asarray(counts, dtype=float))
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    cumulative = np.cumsum(counts)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)
