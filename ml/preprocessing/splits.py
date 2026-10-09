"""Train/validation/test splitting for recommender data.

Chronological splitting (when timestamps exist) prevents future
information leaking into training.  Each user's events are sorted by
time; the earliest ``train_frac`` go to train, the next slice to
validation, the remainder to test.  Users with too few events would
end up with empty splits, so they are kept whole in train and excluded
from per-user evaluation.

When no timestamps exist (e.g. MSD triplets), fall back to per-user
random splitting, which is still leakage-safe at the user level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.utils.logging import get_logger

logger = get_logger(__name__)


def split_interactions(
    interactions: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    min_events_for_split: int = 5,
) -> dict[str, pd.DataFrame]:
    """Split interactions into train/val/test dicts."""
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"Invalid split fractions: train={train_frac}, val={val_frac}")

    has_time = "timestamp" in interactions.columns and interactions["timestamp"].notna().any()
    mode = "chronological" if has_time else "random_per_user"
    logger.info("Split mode: %s", mode)

    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for user_id, group in interactions.groupby("user_id", sort=False):
        if has_time:
            group = group.sort_values("timestamp", kind="mergesort")
        n = len(group)
        if n < min_events_for_split:
            train_parts.append(group)  # too small to split safely
            continue
        if mode == "random_per_user":
            order = rng.permutation(n)
            group = group.iloc[order]
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
        n_val = min(n_val, n - n_train - 1) if n - n_train > 1 else n_val
        n_val = max(0, min(n_val, n - n_train))
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    out = {
        "train": pd.concat(train_parts).reset_index(drop=True) if train_parts else interactions.iloc[0:0],
        "val": pd.concat(val_parts).reset_index(drop=True) if val_parts else interactions.iloc[0:0],
        "test": pd.concat(test_parts).reset_index(drop=True) if test_parts else interactions.iloc[0:0],
    }
    logger.info("Split sizes: train=%d val=%d test=%d", len(out["train"]), len(out["val"]), len(out["test"]))
    return out
