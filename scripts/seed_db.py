#!/usr/bin/env python3
"""Seed the database with the sample catalog + demo listening history.

Usage::

    python scripts/seed_db.py                      # uses DATABASE_URL from env/.env
    python scripts/seed_db.py --no-demo-history    # catalog only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.logging import setup_api_logging  # noqa: E402
from app.db import session as db_session  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.seed import seed_database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--no-demo-history", action="store_true")
    parser.add_argument("--history-limit", type=int, default=1500)
    args = parser.parse_args()

    setup_api_logging()
    db_session.init_engine(args.database_url)
    Base.metadata.create_all(bind=db_session.engine)
    db = db_session.SessionLocal()
    try:
        result = seed_database(db, with_demo_history=not args.no_demo_history,
                               history_limit=args.history_limit)
        print(f"Seed result: {result}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
