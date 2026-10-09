"""Make the monorepo root importable so the backend can use the ``ml`` package."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/app/core -> project root


def ensure_ml_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
