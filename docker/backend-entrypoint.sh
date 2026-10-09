#!/usr/bin/env bash
# Backend container entrypoint: wait for Postgres, migrate, seed, serve.
set -euo pipefail

echo "[tuneflow] waiting for database..."
python - <<'PY'
import os, time
from sqlalchemy import create_engine, text

url = os.environ.get("DATABASE_URL", "")
for attempt in range(30):
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[tuneflow] database ready (attempt {attempt + 1})")
        break
    except Exception as exc:  # noqa: BLE001
        if attempt == 29:
            raise SystemExit(f"database never became ready: {exc}")
        time.sleep(2)
PY

echo "[tuneflow] running migrations..."
cd /app/backend && alembic upgrade head && cd /app

echo "[tuneflow] seeding database (no-op when catalog exists)..."
python scripts/seed_db.py || echo "[tuneflow] seed skipped/failed (continuing)"

echo "[tuneflow] starting API: $*"
exec "$@"
