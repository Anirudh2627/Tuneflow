"""Canonical project path resolution.

Every module/script obtains paths from here so nothing is hardcoded
across the codebase.  The project root can be overridden with the
``TUNEFLOW_ROOT`` environment variable (useful in Docker).
"""
from __future__ import annotations

import os
from pathlib import Path

# ml/utils/paths.py -> ml/utils -> ml -> <project root>
_PROJECT_ROOT_ENV = os.environ.get("TUNEFLOW_ROOT")
PROJECT_ROOT: Path = Path(_PROJECT_ROOT_ENV).resolve() if _PROJECT_ROOT_ENV else Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_DIR = REPORTS_DIR / "evaluation"


def ensure_dirs() -> None:
    """Create all data/model/report directories if they do not exist."""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        EVALUATION_DIR,
        *(MODELS_DIR / sub for sub in ("content", "collaborative", "session", "ranking", "metadata")),
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the project root when not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
