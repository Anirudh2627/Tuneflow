# 🎧 TuneFlow — Full-Stack Personalized Music Recommendation System

A complete, production-shaped recommendation platform: data pipeline → four
recommendation strategies → hybrid ranking → diversification → explained
top-K → FastAPI → React/TypeScript UI → feedback loop → evaluation &
retraining. Everything runs on a normal laptop.

```
                    MUSIC DATA
                        │
         ┌──────────────┼──────────────┐
      Metadata    Audio Features  Interactions
         └──────────────┼──────────────┘
                        ▼
             DATA INGESTION (adapters)
                        ▼
             PREPROCESSING / CLEANING / SPLITS
                        ▼
              FEATURE ENGINEERING
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
     Content      Collaborative    Session
         └──────────────┼──────────────┘
                        ▼
              CANDIDATE GENERATION
                        ▼
                 HYBRID RANKER  (WeightedRanker / LearnedRanker / LightGBM)
                        ▼
              DIVERSIFICATION (MMR + novelty)
                        ▼
              TOP-K + EXPLANATIONS
                        ▼
        FastAPI  ──▶  React UI  ──▶  Feedback  ──▶  PostgreSQL  ──▶  Future training
```

---

## 1. Project overview

TuneFlow recommends songs from:

* **listening history** (plays, weighted by event type)
* **likes / dislikes / saves** (configurable feedback weights)
* **artists & genres** (affinity profiles derived from history)
* **audio features** (energy, danceability, valence, tempo, … — all optional)
* **collaborative behavior** (implicit-feedback matrix factorization / item-item CF)
* **recent listening sessions** (recency-weighted transition model)

Every recommendation ships a **generated explanation** built from the actual
ranking signals that fired (never hardcoded per song), and the platform
measures itself with a proper **offline evaluation harness**
(Precision/Recall/MAP/NDCG/HitRate/MRR + Coverage/Novelty/Diversity).

## 2. Problem statement

Given a catalog of songs with (optionally sparse) metadata and a growing log
of user interactions, produce for each user a *diverse, explainable,
fresh-yet-relevant* top-K list that:

1. works for **cold-start users** (onboarding genres + popularity),
2. works for **cold-start songs** (metadata/audio-feature content model),
3. adapts within a **listening session**,
4. learns from **explicit feedback** (like/dislike/save) with configurable weights,
5. can be **evaluated offline** on a leakage-safe chronological split,
6. and is served through a **clean API + polished web UI** — with training
   strictly offline (the API never trains on request).

## 3. Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline (your laptop)"]
        RAW[(raw data\nMSD / sample / CSV)] --> PREP[prepare_data.py\nvalidate·clean·filter·split]
        PREP --> PROC[(data/processed)]
        PROC --> TC[train_content.py]
        PROC --> TF[train_collaborative.py]
        PROC --> TS[train_session.py]
        PROC --> TR[train_ranker.py]
        TC & TF & TS & TR --> ART[(models/* artifacts)]
        PROC --> EVAL[evaluate.py] --> REP[(reports/evaluation)]
    end

    subgraph Online["Online (serving)"]
        ART -->|load once at startup| ENG[HybridRecommender\nEngineHolder]
        ENG --> API[FastAPI]
        API <--> PG[(PostgreSQL\nusers·songs·events·sessions·recommendations)]
        UI[React + Vite + Tailwind] <--> API
        UI -->|plays·likes·skips·saves| API
    end

    PG -.->|export interactions| PREP
```

* **`ml/`** — dataset-agnostic ML library (preprocessing, features,
  recommenders, ranking, evaluation). Pure Python; no web framework imports.
* **`backend/`** — FastAPI app. Loads trained artifacts once at startup via
  `EngineHolder`; degrades gracefully to a DB-popularity fallback when no
  artifacts exist yet.
* **`frontend/`** — React 18 + TypeScript + Vite + Tailwind SPA.
* **`scripts/`** — every offline step is a CLI script (download → prepare →
  EDA → train × 4 → evaluate → seed DB).
* **`configs/`** — all hyperparameters/weights live here (YAML), never in code.

## 4. Recommendation pipeline

Implemented in `ml/recommenders/hybrid.py`:

1. **Candidate generation** — four parallel sources with configurable pool
   sizes: content (taste-centroid cosine), collaborative (ALS or item-item
   CF), session (recency-decayed transition similarity), popularity
   (global / recent / per-genre).
2. **Merge** — union of candidate ids, de-duplicated.
3. **Feature extraction** — 14 ranking features per candidate
   (`ml/features/ranking_features.py`): per-source scores, genre/artist/tag
   affinity, audio-profile match, song & artist popularity, user-artist /
   user-genre interaction strength, novelty, release recency.
4. **Ranking** — pluggable `BaseRanker`:
   * `WeightedRanker` (config weights, min-max normalized per request)
   * `LearnedRanker` (logistic regression trained on validation-split labels)
   * `LightGBMRanker` (optional, `pip install lightgbm`)
   Per-mode weight overrides (discover boosts novelty, trending boosts
   popularity, recent boosts session) come from `configs/default.yaml`.
5. **Filtering** — remove seen / disliked / unknown-to-catalog items.
6. **Diversification** — MMR over content-space similarity with a novelty
   term (`relevance_weight`, `diversity_weight`, `novelty_weight` all
   configurable).
7. **Explanations** — top weighted contributing features are mapped to
   natural-language reasons + machine-readable `signals`.

## 5. Dataset

| Source | Command | Notes |
|---|---|---|
| **Built-in sample** (default) | `python scripts/download_data.py --source sample` | Deterministic synthetic catalog: 300 users · 80 artists · 1 200 songs · ~15 000 sessionized events with genres, tags and audio features. Lets the whole system run offline immediately. |
| **Million Song Dataset** (taste-profile triplets) | `python scripts/download_data.py --source msd` | Auto-downloads `train_triplets.txt.zip` (~500 MB) and `unique_tracks.txt.gz` (~200 MB) into `data/raw/msd/`. If the mirrors are unreachable, download manually and drop the files there — exact filenames are printed by `python scripts/download_data.py --source msd --check`. No audio features/genres → the content model automatically falls back to artist/title/year TF-IDF. |
| **Any CSV dataset** | `python scripts/prepare_data.py --source csv --csv-songs … --csv-interactions … [--csv-artists …]` | Documented schema in `ml/datasets/schema.py`; list columns are pipe-separated (`rock\|pop`). No code changes needed. |

Expected raw layout for MSD (`data/raw/msd/`):

```
train_triplets.txt(.gz)   user_id <TAB> song_id <TAB> play_count
unique_tracks.txt(.gz)    track_id <SEP> song_id <SEP> artist <SEP> title <SEP> year
```

## 6. Data preprocessing

`python scripts/prepare_data.py --config configs/sample.yaml`

1. **Ingestion** through a dataset adapter (`ml/datasets/adapters/`).
2. **Schema validation** (`ml/datasets/schema.py`) — required columns,
   unique ids, valid event types, referential integrity.
3. **Cleaning** — exact & key duplicate collapsing, title/artist
   normalization, absurd duration/year masking, list-column parsing.
4. **Weighting** — `weight` column from configurable `feedback.event_weights`.
5. **Filtering** — iterative k-core (`min_interactions_per_user/song`),
   drop negative-only users.
6. **Splitting** — **chronological per user** (train ≤ val ≤ test in time)
   when timestamps exist, else leakage-safe per-user random split. Users
   with < 5 events stay whole in train.
7. **Statistics** — `data/processed/stats.json` (users/songs/artists,
   sparsity, event mix, cold-start counts, genre distribution).

Outputs (parquet, CSV fallback): `songs`, `artists`, `interactions`,
`train`, `val`, `test`.

## 7. Content recommendation

`ml/recommenders/content.py` — `ContentRecommender`

* Feature space (`SongFeatureBuilder`): scaled audio features **+ missingness
  mask**, multi-hot genres & tags, TF-IDF over title/artist/genre/year text.
  Any block absent from the dataset is simply skipped (MSD-safe).
* Optional `SentenceTransformer` embedding block blended by
  `content.embedding_weight` (config `content.text_embedding.provider`).
* `recommend_for_song(song_id, k)` / `similar_songs(song_id, k)` — item-item
  cosine; `recommend_for_user(user_id, k)` — event-weighted taste centroid
  with **dislike penalty**; `add_songs()` registers brand-new catalog items
  (new-song cold start) without refitting.
* Artifacts persist vectorizer + scaler + matrices (`models/content/`).

## 8. Collaborative filtering

`ml/recommenders/collaborative.py` — `CollaborativeRecommender`

* Backend `auto` → **implicit ALS** when the `implicit` library imports,
  otherwise a pure-scipy **item-item cosine CF** fallback with the identical
  API (`fit`, `recommend_for_user`, `similar_users`, `similar_items`).
* Confidence matrix from weighted interactions (negative events clipped to 0
  for factorization; dislikes are used by profile/ranking layers instead).
* Hyperparameters in `configs/*.yaml` (`collaborative.factors`,
  `iterations`, `regularization`, `alpha`, `n_neighbors`).
* Trained offline only — `scripts/train_collaborative.py` (also fits the
  `PopularityRecommender` used for cold start & exploration).

## 9. Session recommendation

`ml/recommenders/session.py` — `SessionRecommender`

* Sessions come from explicit `session_id`s or a time-gap heuristic.
* Item-item **transition similarity** (windowed co-occurrence, cosine
  normalized, top-k pruned) + **exponential recency decay** over the current
  session: `score(c) = Σ decay^(n-1-i) · sim(session[i], c)`.
* The public surface (`fit`, `recommend_for_session`, `session_scores`) is
  the contract a GRU4Rec/Transformer implementation should satisfy later —
  swap the class, keep the pipeline.

## 10. Hybrid ranking

`HybridRecommender` composes the four sources exactly as in §4. Weights,
candidate counts, diversity parameters and per-mode overrides all live in
`configs/default.yaml` (or env overrides — no source edits):

```yaml
ranker:
  weights: { content_score: 0.30, collaborative_score: 0.35, session_score: 0.20, ... }
hybrid:
  candidate_counts: { content: 100, collaborative: 100, session: 60, popularity: 60 }
  diversity: { relevance_weight: 0.7, diversity_weight: 0.3, novelty_weight: 0.05 }
```

## 11. Feedback loop

* UI buttons → `POST /feedback` / `POST /events` → `user_song_events`
  (user, song, event_type, timestamp, session_id, weight).
* Sessions auto-close after inactivity; the active session feeds the
  SessionRecommender on the very next request.
* Every served list is persisted in `recommendations` (mode + payload) for
  later impression-based training.
* To retrain from production data, export `user_song_events` to the CSV
  schema and run `prepare_data.py --source csv` → train scripts.

## 12. Evaluation

`python scripts/evaluate.py --config configs/sample.yaml`

* Protocol: positives = held-out **test** split events; history (train) is
  excluded from candidates; graded NDCG gains (like 3 > save 2 > play 1).
* Metrics: **Precision@K, Recall@K, MAP@K, NDCG@K, HitRate@K, MRR,
  Coverage, Catalog coverage, Novelty, Intra-list Diversity, latency**.
* Models: popularity, content, collaborative, session, hybrid — same
  protocol for all; persisted artifacts are used when present, otherwise
  models are fit on the train split inside the script.
* Results: `reports/evaluation/<run>.json` + `.md`. Computed live — nothing
  is pre-baked. (Numbers below are only produced when **you** run it.)

## 13. Project structure

```
tuneflow/
├── backend/
│   ├── app/
│   │   ├── api/            # routers (users, catalog, events, sessions, health) + deps
│   │   ├── core/           # settings, logging, path bootstrap
│   │   ├── db/             # engine/session, declarative Base, seeder
│   │   ├── models/         # SQLAlchemy ORM (users, songs, artists, albums,
│   │   │                   #   genres, user_song_events, sessions, recommendations)
│   │   ├── schemas/        # Pydantic request/response models
│   │   ├── services/       # engine holder, catalog, events, sessions,
│   │   │                   #   profiles, recommendations, search
│   │   └── main.py         # FastAPI app factory + lifespan
│   ├── alembic/            # migrations (0001_initial)
│   └── tests/              # API tests (SQLite) + hybrid integration test
├── ml/
│   ├── datasets/           # canonical schema + adapters (sample/msd/csv)
│   ├── preprocessing/      # cleaning, filtering, splits, pipeline
│   ├── features/           # song features, interaction matrix, user profile,
│   │                       #   ranking features
│   ├── embeddings/         # EmbeddingProvider (+SentenceTransformer impl)
│   ├── recommenders/       # base, content, collaborative, session,
│   │                       #   popularity, hybrid, explain, factory
│   ├── ranking/            # BaseRanker, WeightedRanker, Learned/LightGBM, MMR
│   ├── evaluation/         # metrics + evaluator harness
│   └── utils/              # config, logging, artifacts, paths
├── scripts/                # download_data, prepare_data, eda, train_*, evaluate, seed_db
├── frontend/               # React 18 + TS + Vite + Tailwind
│   └── src/{components,pages,hooks,services,store,types}
├── configs/                # default.yaml, sample.yaml, msd.yaml
├── data/{raw,processed,sample}
├── models/{content,collaborative,session,ranking,metadata}
├── reports/{figures,evaluation}
├── notebooks/01_eda.ipynb
├── docker/                 # Dockerfiles, nginx, entrypoint
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── .env.example
```

## 14. Installation

```bash
git clone <your-fork> tuneflow && cd tuneflow
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add: pip install implicit lightgbm sentence-transformers (optional extras)
cp .env.example .env                     # adjust DATABASE_URL if needed
```

Node 20+ is required for the frontend (`cd frontend && npm install`).

## 15. Dataset preparation

```bash
# Option A — built-in sample (instant, no download)
python scripts/download_data.py --source sample
python scripts/prepare_data.py --config configs/sample.yaml

# Option B — Million Song Dataset
python scripts/download_data.py --source msd          # ~700 MB download
python scripts/prepare_data.py --config configs/msd.yaml

# Option C — your own CSVs (schema in ml/datasets/schema.py)
python scripts/prepare_data.py --source csv \
    --csv-songs data/raw/my/songs.csv \
    --csv-artists data/raw/my/artists.csv \
    --csv-interactions data/raw/my/interactions.csv

# EDA (figures -> reports/figures, stats -> reports/eda_stats.json)
python scripts/eda.py --config configs/sample.yaml
# or open notebooks/01_eda.ipynb
```

## 16. Model training

Run in order (each reads `data/processed` and writes to `models/`):

```bash
python scripts/train_content.py        --config configs/sample.yaml
python scripts/train_collaborative.py  --config configs/sample.yaml   # + popularity tables
python scripts/train_session.py        --config configs/sample.yaml
python scripts/train_ranker.py         --config configs/sample.yaml --ranker weighted
# supervised alternatives (train on validation-split labels):
python scripts/train_ranker.py         --config configs/sample.yaml --ranker learned
python scripts/train_ranker.py         --config configs/sample.yaml --ranker lightgbm   # needs lightgbm
```

Every knob (ALS factors/iterations/regularization/alpha, session decay,
embedding model, ranker weights, candidate counts, diversity, feedback
weights) is in `configs/*.yaml`; layer experiment files:

```bash
python scripts/train_collaborative.py --config configs/msd.yaml --override-config configs/exp_als256.yaml
```

Artifacts land in `models/<family>/<name>/{model.joblib,metadata.json}` and
are loaded **read-only** by the API — no request ever trains.

## 17. Evaluation

```bash
python scripts/evaluate.py --config configs/sample.yaml            # all 5 models, K = 5/10/20
python scripts/evaluate.py --recommenders hybrid,content --k 10 --run-name exp7
cat reports/evaluation/exp7.md
```

## 18. Backend setup

```bash
# 1) database — either docker or local Postgres:
docker compose up -d db
export DATABASE_URL=postgresql+psycopg2://tuneflow:tuneflow@localhost:5432/tuneflow
#    (quick experiments can use SQLite: DATABASE_URL=sqlite:///./data/tuneflow_dev.db)

# 2) migrations + seed catalog & demo user:
cd backend && alembic upgrade head && cd ..
python scripts/seed_db.py

# 3) serve:
uvicorn app.main:app --app-dir backend --reload --port 8000
#    → http://localhost:8000/docs (OpenAPI), /health shows engine status
```

## 19. Frontend setup

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies API calls to :8000)
npm run build        # production bundle in frontend/dist
```

Pages: **Home** (Made For You / Recently Played / Because You Liked… /
Discover / Trending / Session-similar shelves) · **Discover** (mode + genre
exploration) · **Search** (songs/artists/albums/genres) · **Song**
(features, similar, more-from-artist, because-you-listened) · **Artist** ·
**Library** (liked/saved/disliked) · **History** (filterable event log) ·
**Profile** (genre/artist affinity, audio-profile means, stats). A bottom
player plays `preview_url`s when the dataset provides legal previews —
recommendations work fully independently of playback.

## 20. Docker setup

```bash
# after training artifacts exist locally (they are volume-mounted read-only):
docker compose up --build
# frontend  -> http://localhost:8080
# backend   -> http://localhost:8000/docs
# postgres  -> localhost:5432
```

The backend container waits for Postgres, runs `alembic upgrade head`,
auto-seeds the sample catalog (idempotent) and serves the API. Training is
deliberately *not* containerized — you train on your laptop, the containers
only serve artifacts.

## 21. API documentation

Interactive docs at `/docs` (Swagger) and `/redoc`. Summary:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + engine/components status |
| POST | `/users` | Register user (display name, onboarding genres) |
| GET | `/users/{id}` | User record |
| GET | `/users/{id}/profile` | Derived taste profile (affinities, stats, liked/disliked/saved) |
| GET | `/users/{id}/recommendations?limit=&mode=&genre=` | Modes: `personalized`, `discover`, `recent`, `trending` |
| GET | `/users/{id}/history?event_type=&limit=` | Event log with embedded song payloads |
| GET | `/users/{id}/recent` | Recently played songs |
| GET | `/users/{id}/liked` | liked / saved / disliked song lists |
| GET | `/songs/{id}` | Song detail (metadata + audio features) |
| GET | `/songs/{id}/similar` | Content-similarity shelf with explanations |
| GET | `/artists/{id}` | Artist + top songs + albums |
| GET | `/search?q=&limit=` | Songs / artists / albums / genres |
| GET | `/genres`, `/genres/{id}/songs` | Genre browse |
| POST | `/events`, `/events/batch` | PLAY / SKIP / LIKE / DISLIKE / SAVE ingestion |
| POST | `/feedback` | Single feedback event (convenience) |
| POST | `/sessions`, GET `/sessions/{id}`, POST `/sessions/{id}/end` | Session lifecycle |

## 22. Example requests

```bash
# register a user with onboarding preferences (cold start)
curl -s localhost:8000/users -H 'content-type: application/json' \
  -d '{"display_name":"Alice","preferred_genres":["electronic","jazz"]}'
# → {"id":"usr_8f1c…", …}

# personalized recommendations with explanations
curl -s 'localhost:8000/users/usr_8f1c…/recommendations?limit=3&mode=personalized' | jq '.items[0]'
# → { "song_id": "so_000431", "title": "Neon Tides", "artist": "Ava Moon",
#     "score": 0.83, "signals": ["content_similarity","genre_affinity","recent_session"],
#     "reason": "Recommended because sounds similar to tracks you liked and matches your top genres (electronic)." }

# feedback (feeds the loop)
curl -s localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"user_id":"usr_8f1c…","song_id":"so_000431","feedback":"like"}'

# session-aware exploration
curl -s localhost:8000/sessions -H 'content-type: application/json' -d '{"user_id":"usr_8f1c…"}'
curl -s 'localhost:8000/users/usr_8f1c…/recommendations?mode=recent&limit=5'
```

## 23. Testing

```bash
python -m pytest -q          # 85 tests: ML (preprocessing, content, CF, session,
                             # hybrid, cold start, diversity, metrics, profiles)
                             # + backend API (users, songs, recs, feedback, sessions)
```

ML tests use tiny in-memory fixtures (12 songs / 6 users). Backend tests run
on throwaway SQLite databases with `TUNEFLOW_DISABLE_ENGINE=1`; the hybrid
integration test activates when sample artifacts are present.

## 24. Future improvements

* **Sequential models**: GRU4Rec / SASRec behind the existing
  `SessionRecommender` contract (fit / recommend_for_session / session_scores).
* **Learned rankers**: LambdaMART with impression data from the
  `recommendations` table; A/B via `ARTIFACT_NAME`.
* **Real audio**: FMA adapter (legal preview clips → playable UI) and
  audio-embedding features (e.g. MERT/CLAP) behind `EmbeddingProvider`.
* **Online learning**: nightly incremental ALS updates; near-real-time
  session model refresh from `user_song_events`.
* **Vector search**: pgvector for semantic search over song/text embeddings.
* **Bandit exploration**: ε-greedy / Thompson sampling on the discover shelf.
* **MLOps**: artifact registry, training pipelines (you said you'd own this —
  the artifact/metadata structure in `models/` is designed for it).

## 25. Known limitations (honest ones)

* The built-in sample dataset is synthetic — metrics on it measure plumbing,
  not real-world quality. Train on MSD/your own data for meaningful numbers.
* MSD has no timestamps → splits fall back to per-user random; sessions are
  approximated by chunking.
* `implicit` ALS needs `libgomp` on Linux (`apt install libgomp1`; the Docker
  image includes it). Without it the scipy item-item fallback is used.
* Search is SQL `ILIKE`-based; semantic search activates only if you install
  `sentence-transformers` and enable the provider in config.
* Playback requires the dataset to provide legal `preview_url`s; the sample
  dataset intentionally ships none.
