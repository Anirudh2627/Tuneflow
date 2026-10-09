"""Model artifact persistence.

Artifacts are stored under ``models/<family>/``.  Each artifact is a
directory containing ``model.joblib`` plus ``metadata.json`` describing
when/how it was trained, the config used, and dataset statistics.  The
inference code only ever *loads* from here; it never trains.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ml.utils.paths import MODELS_DIR

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def artifact_dir(family: str, name: str = "latest") -> Path:
    """Return ``models/<family>/<name>`` creating parents as needed."""
    path = MODELS_DIR / family / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifact(obj: Any, family: str, name: str = "latest", metadata: dict | None = None,
                  config: dict | None = None) -> Path:
    """Persist ``obj`` plus metadata.  Returns the artifact directory."""
    directory = artifact_dir(family, name)
    joblib.dump(obj, directory / MODEL_FILENAME, compress=3)
    meta: dict[str, Any] = {
        "family": family,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
    }
    if metadata:
        meta["metadata"] = metadata
    if config:
        meta["config"] = config
    with open(directory / METADATA_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return directory


def load_artifact(family: str, name: str = "latest") -> Any:
    """Load a previously saved artifact.  Raises ``FileNotFoundError``."""
    path = MODELS_DIR / family / name / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No artifact for family={family!r} name={name!r} at {path}. "
            f"Run the corresponding scripts/train_*.py script first."
        )
    return joblib.load(path)


def load_metadata(family: str, name: str = "latest") -> dict:
    path = MODELS_DIR / family / name / METADATA_FILENAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def artifact_exists(family: str, name: str = "latest") -> bool:
    return (MODELS_DIR / family / name / MODEL_FILENAME).exists()


def clear_artifact(family: str, name: str = "latest") -> None:
    directory = MODELS_DIR / family / name
    if directory.exists():
        shutil.rmtree(directory)
