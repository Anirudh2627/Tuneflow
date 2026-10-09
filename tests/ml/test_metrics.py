"""Tests for evaluation metrics (hand-computed expected values)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation import metrics as M


def test_precision_and_recall():
    rec = ["a", "b", "c", "d"]
    rel = {"a", "c", "e"}
    assert M.precision_at_k(rec, rel, 4) == pytest.approx(0.5)
    assert M.recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)
    assert M.precision_at_k(rec, rel, 2) == pytest.approx(0.5)
    assert M.recall_at_k(rec, rel, 2) == pytest.approx(1 / 3)


def test_hit_rate():
    assert M.hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert M.hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_average_precision():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 3 relevant
    ap = M.average_precision_at_k(["a", "x", "b", "y"], {"a", "b", "c"}, 4)
    assert ap == pytest.approx((1.0 + 2 / 3) / 3)


def test_ndcg():
    gains = {"a": 3.0, "b": 1.0}
    perfect = M.ndcg_at_k(["a", "b"], gains, 2)
    swapped = M.ndcg_at_k(["b", "a"], gains, 2)
    assert perfect == pytest.approx(1.0)
    assert swapped < perfect
    assert M.ndcg_at_k(["x", "y"], gains, 2) == 0.0


def test_mrr():
    recs = [["x", "a"], ["a", "x"], ["x", "y"]]
    rels = [{"a"}, {"a"}, {"a"}]
    assert M.mrr(recs, rels) == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_coverage():
    recs = [["a", "b"], ["b", "c"]]
    assert M.coverage(recs, catalog_size=10) == pytest.approx(0.3)
    assert M.catalog_coverage(recs, {"a", "b", "c", "d"}) == pytest.approx(0.75)


def test_intra_list_diversity():
    identical = lambda a, b: 1.0  # noqa: E731
    distinct = lambda a, b: 0.0   # noqa: E731
    assert M.intra_list_diversity(["a", "b"], identical) == pytest.approx(0.0)
    assert M.intra_list_diversity(["a", "b"], distinct) == pytest.approx(1.0)
    assert M.intra_list_diversity(["a"], distinct) == 0.0


def test_novelty():
    pop = pd.Series({"a": 50.0, "b": 1.0})
    # rare items are more novel
    assert M.novelty(["b"], pop, n_users=100) > M.novelty(["a"], pop, n_users=100)
    assert M.novelty(["a"], pop, n_users=100) == pytest.approx(1.0)  # -log2(0.5)


def test_gini():
    uniform = M.gini_index(np.ones(10))
    skewed = M.gini_index(np.array([100, 0, 0, 0, 0]))
    assert uniform < 0.15
    assert skewed > 0.6
    assert M.gini_index(np.array([])) == 0.0
