"""Tests for preprocessing: cleaning, filtering, splitting."""
from __future__ import annotations

import pandas as pd
import pytest

from ml.preprocessing.cleaning import attach_event_weights, dedupe_interactions, normalize_song_metadata
from ml.preprocessing.filtering import compute_statistics, drop_negative_only_users, k_core_filter
from ml.preprocessing.splits import split_interactions


def test_dedupe_removes_exact_and_key_duplicates(tiny_interactions):
    df = tiny_interactions.copy()
    dup = df.iloc[[0, 0, 1]]
    combined = pd.concat([df, dup])
    out = dedupe_interactions(combined)
    assert len(out) < len(combined)
    keys = out[["user_id", "song_id", "event_type", "session_id"]]
    assert not keys.duplicated().any()


def test_event_weights_mapping(tiny_interactions, event_weights):
    df = attach_event_weights(tiny_interactions.drop(columns=["weight"]), event_weights)
    assert (df.loc[df["event_type"] == "like", "weight"] == 3.0).all()
    assert (df.loc[df["event_type"] == "skip", "weight"] == -0.5).all()


def test_normalize_song_metadata_handles_missing(tiny_songs):
    df = tiny_songs.copy()
    df.loc[0, "title"] = None
    df.loc[1, "artist_id"] = ""
    df.loc[2, "duration_ms"] = -5
    out = normalize_song_metadata(df)
    assert out.loc[0, "title"] == "Unknown Title"
    assert out.loc[1, "artist_id"].startswith("ar_unknown_")
    assert pd.isna(out.loc[2, "duration_ms"])


def test_k_core_filter(tiny_interactions):
    df = tiny_interactions.copy()
    # add a one-off user that should be dropped with min_user=2
    extra = df.iloc[[0]].copy()
    extra["user_id"] = "ghost"
    out = k_core_filter(pd.concat([df, extra]), min_user=2, min_item=1)
    assert "ghost" not in set(out["user_id"])


def test_drop_negative_only_users(tiny_interactions, event_weights):
    df = tiny_interactions.copy()
    neg = df.iloc[[0]].copy()
    neg["user_id"] = "hater"
    neg["event_type"] = "dislike"
    neg["weight"] = -2.0
    out = drop_negative_only_users(pd.concat([df, neg]))
    assert "hater" not in set(out["user_id"])


def test_chronological_split_no_leakage(tiny_interactions):
    splits = split_interactions(tiny_interactions, train_frac=0.6, val_frac=0.2, min_events_for_split=5)
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert len(train) > 0 and len(test) > 0
    # every test timestamp must be >= every train timestamp *for the same user*
    for user, group in train.groupby("user_id"):
        if user in set(test["user_id"]):
            user_test = test[test["user_id"] == user]
            assert group["timestamp"].max() <= user_test["timestamp"].min()
    # no duplicate events across splits
    def keys(df):
        return set(zip(df["user_id"], df["song_id"], df["timestamp"], df["event_type"]))
    assert not (keys(train) & keys(test))
    assert not (keys(train) & keys(val))


def test_split_invalid_fractions(tiny_interactions):
    with pytest.raises(ValueError):
        split_interactions(tiny_interactions, train_frac=0.9, val_frac=0.2)


def test_compute_statistics(tiny_songs, tiny_artists, tiny_interactions):
    stats = compute_statistics(tiny_songs, tiny_artists, tiny_interactions)
    assert stats["n_users"] == 6
    assert stats["n_songs_in_catalog"] == 12
    assert 0 <= stats["sparsity"] <= 1
    assert stats["n_genres"] == 3
    assert abs(sum(stats["event_type_distribution"].values()) - 1.0) < 1e-6
