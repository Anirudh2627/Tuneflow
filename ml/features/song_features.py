"""Song feature engineering.

Builds a combined song feature matrix from whatever is available:

* numeric audio features (imputed + scaled, with a presence mask so the
  model can tell "missing" from "low value")
* multi-hot genre encoding
* multi-hot tag encoding
* TF-IDF over title + artist text

Datasets missing audio features or genres degrade gracefully: those
blocks simply contribute zero columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ml.utils.logging import get_logger

logger = get_logger(__name__)

AUDIO_COLUMNS = [
    "energy", "danceability", "valence", "acousticness", "instrumentalness",
    "speechiness", "loudness", "tempo", "duration_ms",
]

GENRE_TOP_K = 60
TAG_TOP_K = 40


def song_text(song: pd.Series) -> str:
    """Text representation of a song for TF-IDF / embeddings."""
    parts = [str(song.get("title", "")), str(song.get("artist_name", ""))]
    genres = song.get("genres") or []
    tags = song.get("tags") or []
    if isinstance(genres, (list, tuple)):
        parts.extend(str(g) for g in genres)
    if isinstance(tags, (list, tuple)):
        parts.extend(str(t) for t in tags)
    year = song.get("release_year")
    if year is not None and not pd.isna(year):
        parts.append(str(int(year)))
    return " ".join(p for p in parts if p and p.lower() != "nan")


class SongFeatureBuilder:
    """Fit-once / transform-many song featurizer (persisted as artifact)."""

    def __init__(self, use_audio: bool = True, use_genres: bool = True, use_tags: bool = True,
                 use_text: bool = True, text_max_features: int = 20000, text_ngram: int = 2):
        self.use_audio = use_audio
        self.use_genres = use_genres
        self.use_tags = use_tags
        self.use_text = use_text
        self.text_max_features = text_max_features
        self.text_ngram = text_ngram
        self._fitted = False
        self.audio_columns: list[str] = []
        self.scaler: StandardScaler | None = None
        self.genre_vocab: dict[str, int] = {}
        self.tag_vocab: dict[str, int] = {}
        self.tfidf: TfidfVectorizer | None = None
        self.n_features: int = 0

    # ------------------------------------------------------------------
    def fit(self, songs: pd.DataFrame) -> "SongFeatureBuilder":
        self.audio_columns = [c for c in AUDIO_COLUMNS if self.use_audio and c in songs.columns
                              and pd.to_numeric(songs[c], errors="coerce").notna().any()]
        if self.audio_columns:
            audio = self._audio_matrix(songs)
            self.scaler = StandardScaler().fit(audio)
        if self.use_genres:
            self.genre_vocab = self._build_vocab(songs, "genres", GENRE_TOP_K)
        if self.use_tags:
            self.tag_vocab = self._build_vocab(songs, "tags", TAG_TOP_K)
        if self.use_text:
            texts = songs.apply(song_text, axis=1)
            if texts.str.strip().eq("").all():
                self.use_text = False
            else:
                self.tfidf = TfidfVectorizer(
                    max_features=self.text_max_features, ngram_range=(1, self.text_ngram),
                    sublinear_tf=True, strip_accents="unicode", lowercase=True,
                    min_df=2 if len(songs) > 200 else 1,
                )
                self.tfidf.fit(texts.tolist())
        self.n_features = self._block_sizes(songs)
        self._fitted = True
        logger.info(
            "SongFeatureBuilder fitted: audio=%d(+mask) genres=%d tags=%d tfidf=%d -> total=%d",
            len(self.audio_columns), len(self.genre_vocab), len(self.tag_vocab),
            len(self.tfidf.vocabulary_) if self.tfidf else 0, sum(self.n_features.values()),
        )
        return self

    def _block_sizes(self, songs: pd.DataFrame) -> dict[str, int]:
        return {
            "audio": len(self.audio_columns) * 2 if self.audio_columns else 0,
            "genres": len(self.genre_vocab),
            "tags": len(self.tag_vocab),
            "text": len(self.tfidf.vocabulary_) if self.tfidf else 0,
        }

    @property
    def total_dim(self) -> int:
        return sum(self.n_features.values())

    # ------------------------------------------------------------------
    def transform(self, songs: pd.DataFrame) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("SongFeatureBuilder.transform called before fit()")
        blocks: list[sparse.csr_matrix] = []
        if self.audio_columns:
            audio = self._audio_matrix(songs)
            scaled = self.scaler.transform(audio)  # type: ignore[union-attr]
            mask = (~np.isnan(self._raw_audio(songs))).astype(np.float32)
            scaled = np.nan_to_num(scaled, nan=0.0)
            blocks.append(sparse.csr_matrix(np.hstack([scaled, mask])))
        if self.genre_vocab:
            blocks.append(self._multihot(songs, "genres", self.genre_vocab))
        if self.tag_vocab:
            blocks.append(self._multihot(songs, "tags", self.tag_vocab))
        if self.tfidf is not None:
            texts = songs.apply(song_text, axis=1).tolist()
            blocks.append(self.tfidf.transform(texts))
        if not blocks:
            raise RuntimeError("No feature blocks available; check dataset columns")
        return sparse.hstack(blocks, format="csr")

    def fit_transform(self, songs: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(songs).transform(songs)

    # ------------------------------------------------------------------
    def _raw_audio(self, songs: pd.DataFrame) -> np.ndarray:
        values = np.column_stack([
            pd.to_numeric(songs[c], errors="coerce").to_numpy(dtype=np.float64)
            for c in self.audio_columns
        ])
        return values

    def _audio_matrix(self, songs: pd.DataFrame) -> np.ndarray:
        values = self._raw_audio(songs)
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        return np.where(np.isnan(values), medians, values)

    @staticmethod
    def _as_token_list(values) -> list:
        if values is None or isinstance(values, (str, float)):
            return [] if values is None or (isinstance(values, float) and pd.isna(values)) else [values]
        if isinstance(values, (list, tuple, np.ndarray)):
            return list(values)
        return []

    @staticmethod
    def _build_vocab(songs: pd.DataFrame, column: str, top_k: int) -> dict[str, int]:
        if column not in songs.columns:
            return {}
        counts: dict[str, int] = {}
        for values in songs[column]:
            for v in SongFeatureBuilder._as_token_list(values):
                counts[str(v)] = counts.get(str(v), 0) + 1
        top = sorted(counts, key=lambda k: (-counts[k], k))[:top_k]
        return {token: i for i, token in enumerate(top)}

    @staticmethod
    def _multihot(songs: pd.DataFrame, column: str, vocab: dict[str, int]) -> sparse.csr_matrix:
        rows, cols = [], []
        for i, values in enumerate(songs[column] if column in songs.columns else []):
            for v in SongFeatureBuilder._as_token_list(values):
                idx = vocab.get(str(v))
                if idx is not None:
                    rows.append(i)
                    cols.append(idx)
        matrix = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(songs), len(vocab)),
        )
        return matrix
