"""Built-in sample dataset generator.

Produces a small but *realistic* synthetic music catalogue and listening
log so the whole system (pipeline, training, API, frontend) runs without
downloading anything.  Generated files are cached under ``data/sample``.

Generation model:
* artists are assigned 1-3 genres, songs inherit artist genres plus noise
  on audio features (correlated with genre, e.g. electronic -> high energy)
* users have latent genre preferences + a favorite-artist bias
* interactions are sampled from the user's preference distribution with
  Zipf-style popularity skew, grouped into chronological sessions
* event types are sampled conditioned on preference match (liked genres
  produce more LIKE/SAVE, mismatched produce SKIP/DISLIKE)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets.adapters.base import BaseAdapter
from ml.datasets.schema import Dataset
from ml.utils.logging import get_logger
from ml.utils.paths import SAMPLE_DATA_DIR

logger = get_logger(__name__)

GENRES = [
    "electronic", "rock", "pop", "hip hop", "jazz", "classical", "metal",
    "r&b", "folk", "indie", "ambient", "latin",
]

# genre -> mean shifts for (energy, danceability, valence, acousticness,
# instrumentalness, speechiness, loudness_db, tempo)
GENRE_PROFILE: dict[str, tuple[float, ...]] = {
    "electronic": (0.80, 0.75, 0.55, 0.10, 0.35, 0.06, -6.0, 126),
    "rock":       (0.78, 0.45, 0.50, 0.08, 0.05, 0.05, -5.5, 120),
    "pop":        (0.68, 0.68, 0.68, 0.12, 0.02, 0.06, -6.5, 115),
    "hip hop":    (0.70, 0.80, 0.55, 0.10, 0.02, 0.25, -5.8, 98),
    "jazz":       (0.35, 0.50, 0.55, 0.65, 0.35, 0.05, -12.0, 110),
    "classical":  (0.15, 0.25, 0.35, 0.92, 0.75, 0.04, -18.0, 90),
    "metal":      (0.93, 0.35, 0.30, 0.03, 0.10, 0.07, -4.0, 145),
    "r&b":        (0.55, 0.70, 0.60, 0.25, 0.03, 0.10, -8.0, 100),
    "folk":       (0.40, 0.45, 0.55, 0.70, 0.10, 0.04, -11.0, 105),
    "indie":      (0.60, 0.50, 0.55, 0.35, 0.08, 0.04, -8.5, 118),
    "ambient":    (0.20, 0.30, 0.40, 0.80, 0.85, 0.03, -20.0, 85),
    "latin":      (0.75, 0.78, 0.72, 0.20, 0.04, 0.09, -6.0, 102),
}

WORD_A = ["Neon", "Silent", "Golden", "Midnight", "Electric", "Fading", "Crystal", "Velvet", "Broken", "Distant", "Burning", "Paper", "Iron", "Ocean", "Static", "Wild", "Hollow", "Silver", "Lost", "Cosmic"]
WORD_B = ["Dreams", "Rain", "Light", "Echoes", "Hearts", "Shadows", "Waves", "Skies", "Fire", "Mirror", "Roads", "Nights", "Ghost", "Sun", "Tides", "Cities", "Stars", "Wind", "Rivers", "Pulse"]
FIRST = ["Ava", "Liam", "Noah", "Mia", "Ethan", "Zoe", "Leo", "Ivy", "Kai", "Nora", "Eli", "Ruby", "Max", "Luna", "Jude", "Iris", "Theo", "Sage", "Axel", "Wren"]
LAST = ["Kite", "Vale", "Moon", "Frost", "Rivers", "Stone", "Marlow", "Ash", "Crown", "Vega", "Onyx", "Hale", "Wolf", "Lane", "Pike", "Reyes", "North", "Quinn", "Fox", "Delgado"]


def _song_title(rng: np.random.Generator) -> str:
    return f"{rng.choice(WORD_A)} {rng.choice(WORD_B)}"


class SampleAdapter(BaseAdapter):
    """Deterministic synthetic dataset (cached as CSV under data/sample)."""

    name = "sample"

    def __init__(
        self,
        n_users: int = 300,
        n_artists: int = 80,
        n_songs: int = 1200,
        n_interactions: int = 15000,
        seed: int = 42,
        cache_dir: Path | None = None,
        regenerate: bool = False,
    ):
        self.n_users = n_users
        self.n_artists = n_artists
        self.n_songs = n_songs
        self.n_interactions = n_interactions
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else SAMPLE_DATA_DIR
        self.regenerate = regenerate

    # ------------------------------------------------------------------
    def load(self) -> Dataset:
        cached = self._cached_frames()
        if cached is not None and not self.regenerate:
            songs, artists, interactions = cached
            logger.info("Loaded cached sample dataset from %s", self.cache_dir)
        else:
            songs, artists, interactions = self.generate()
            self._write_cache(songs, artists, interactions)
            logger.info("Generated and cached sample dataset in %s", self.cache_dir)
        return Dataset(songs=songs, artists=artists, interactions=interactions, name=self.name)

    # ------------------------------------------------------------------
    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)

        artists = self._make_artists(rng)
        songs = self._make_songs(rng, artists)
        interactions = self._make_interactions(rng, songs, artists)
        return songs, artists, interactions

    # ------------------------------------------------------------------
    def _make_artists(self, rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for i in range(self.n_artists):
            n_genres = int(rng.integers(1, 4))
            genres = list(rng.choice(GENRES, size=n_genres, replace=False))
            rows.append({
                "artist_id": f"ar_{i:05d}",
                "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "genres": genres,
                "followers": float(int(rng.pareto(1.6) * 5000 + 500)),
            })
        return pd.DataFrame(rows)

    def _make_songs(self, rng: np.random.Generator, artists: pd.DataFrame) -> pd.DataFrame:
        artist_ids = artists["artist_id"].to_numpy()
        # Zipf-ish: a few prolific artists
        artist_pop = rng.pareto(1.2, size=len(artist_ids)) + 1.0
        artist_pop /= artist_pop.sum()
        chosen = rng.choice(artist_ids, size=self.n_songs, p=artist_pop)
        artist_genres = {r.artist_id: r.genres for r in artists.itertuples()}
        artist_names = dict(zip(artists["artist_id"], artists["name"]))

        rows = []
        for i, aid in enumerate(chosen):
            genres = list(artist_genres[aid])
            primary = genres[0]
            profile = np.array(GENRE_PROFILE[primary], dtype=float)
            noise = rng.normal(0, 0.08, size=8)
            feat = np.clip(profile + noise, 0, 1)
            feat[6] = float(np.clip(profile[6] + rng.normal(0, 2.0), -40, 0))   # loudness dB
            feat[7] = float(np.clip(profile[7] + rng.normal(0, 12), 50, 210))    # tempo
            row_genres = list(genres)
            if rng.random() < 0.2:  # genre crossover noise
                extra = str(rng.choice(GENRES))
                if extra not in row_genres:
                    row_genres.append(extra)
            tags = [t for t in rng.choice(
                ["chill", "workout", "focus", "party", "roadtrip", "sleep", "melancholy", "uplifting"],
                size=int(rng.integers(0, 3)), replace=False)]
            rows.append({
                "song_id": f"so_{i:06d}",
                "title": _song_title(rng),
                "artist_id": aid,
                "artist_name": artist_names[aid],
                "album": f"Album {(i % 7) + 1} - {artist_names[aid]}",
                "release_year": int(rng.integers(1995, 2025)),
                "duration_ms": float(int(rng.normal(220000, 45000))),
                "genres": row_genres,
                "tags": tags,
                "preview_url": None,
                "artwork_url": None,
                "energy": float(feat[0]),
                "danceability": float(feat[1]),
                "valence": float(feat[2]),
                "acousticness": float(feat[3]),
                "instrumentalness": float(feat[4]),
                "speechiness": float(feat[5]),
                "loudness": float(feat[6]),
                "tempo": float(feat[7]),
                "popularity": float(rng.integers(0, 100)),
            })
        return pd.DataFrame(rows)

    def _make_interactions(
        self, rng: np.random.Generator, songs: pd.DataFrame, artists: pd.DataFrame
    ) -> pd.DataFrame:
        song_ids = songs["song_id"].to_numpy()
        song_genres = songs.set_index("song_id")["genres"].to_dict()
        song_artist = songs.set_index("song_id")["artist_id"].to_dict()
        song_energy = songs.set_index("song_id")["energy"].to_dict()

        # global popularity skew so popularity models have signal
        song_pop = rng.pareto(1.1, size=len(song_ids)) + 1.0
        song_pop /= song_pop.sum()

        base_ts = pd.Timestamp("2025-01-01")
        rows: list[dict] = []
        user_ids = [f"us_{u:05d}" for u in range(self.n_users)]

        for uid in user_ids:
            preferred = set(rng.choice(GENRES, size=int(rng.integers(1, 4)), replace=False))
            fav_artists = set(rng.choice(list(artists["artist_id"]), size=3, replace=False))
            energy_pref = rng.uniform(0.3, 0.9)
            n_events = int(rng.integers(15, 120))
            n_sessions = int(rng.integers(3, 12))
            user_start = int(rng.integers(0, 60))  # day offset within window

            # per-user sampling distribution biased towards preferred genres
            weights = np.ones(len(song_ids), dtype=float)
            for j, sid in enumerate(song_ids):
                w = song_pop[j]
                if preferred & set(song_genres[sid]):
                    w *= 6.0
                if song_artist[sid] in fav_artists:
                    w *= 3.0
                w *= 1.0 - 0.7 * abs(song_energy[sid] - energy_pref)
                weights[j] = max(w, 1e-6)
            weights /= weights.sum()

            picks = rng.choice(song_ids, size=n_events, p=weights)
            for s in range(n_sessions):
                session_len = int(rng.integers(3, 12))
                session_slice = picks[s * session_len:(s + 1) * session_len]
                if len(session_slice) == 0:
                    continue
                session_id = f"{uid}_s{s:03d}"
                ts = base_ts + pd.Timedelta(days=user_start + s, hours=int(rng.integers(8, 23)),
                                            minutes=int(rng.integers(0, 60)))
                for k, sid in enumerate(session_slice):
                    match = bool(preferred & set(song_genres[sid]))
                    r = rng.random()
                    if match:
                        event = "like" if r < 0.18 else ("save" if r < 0.26 else ("skip" if r < 0.36 else "play"))
                    else:
                        event = "dislike" if r < 0.08 else ("skip" if r < 0.42 else "play")
                    rows.append({
                        "user_id": uid,
                        "song_id": sid,
                        "event_type": event,
                        "timestamp": ts + pd.Timedelta(minutes=int(k * rng.integers(2, 6))),
                        "session_id": session_id,
                    })
        interactions = pd.DataFrame(rows)
        interactions = interactions.sort_values("timestamp").reset_index(drop=True)
        return interactions

    # ------------------------------------------------------------------
    def _cache_paths(self) -> dict[str, Path]:
        return {
            "songs": self.cache_dir / "songs.csv",
            "artists": self.cache_dir / "artists.csv",
            "interactions": self.cache_dir / "interactions.csv",
        }

    def _cached_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        paths = self._cache_paths()
        if not all(p.exists() for p in paths.values()):
            return None

        def _read_lists(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda v: [] if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "")
                        else str(v).split("|")
                    )
            return df

        songs = _read_lists(pd.read_csv(paths["songs"]), ["genres", "tags"])
        artists = _read_lists(pd.read_csv(paths["artists"]), ["genres"])
        interactions = pd.read_csv(paths["interactions"], parse_dates=["timestamp"])
        return songs, artists, interactions

    def _write_cache(self, songs: pd.DataFrame, artists: pd.DataFrame, interactions: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        def _join_lists(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
            out = df.copy()
            for col in columns:
                if col in out.columns:
                    out[col] = out[col].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
            return out

        _join_lists(songs, ["genres", "tags"]).to_csv(self._cache_paths()["songs"], index=False)
        _join_lists(artists, ["genres"]).to_csv(self._cache_paths()["artists"], index=False)
        interactions.to_csv(self._cache_paths()["interactions"], index=False)
