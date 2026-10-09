"""Configuration management.

Configuration lives in ``configs/*.yaml``.  Files are deep-merged in the
order given (later files win), then a small set of environment variables
may override scalar values using dotted keys, e.g.::

    TUNEFLOW_OVERRIDE__database__url=postgresql://...

Usage::

    cfg = Config.load("default.yaml", "sample.yaml")
    cfg.get("hybrid.weights.content")
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from ml.utils.paths import CONFIG_DIR, resolve


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


class Config:
    """Thin dict wrapper with dotted-path access and env overrides."""

    ENV_PREFIX = "TUNEFLOW_OVERRIDE__"

    def __init__(self, data: dict[str, Any]):
        self._data = self._apply_env_overrides(data)

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(data)
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(Config.ENV_PREFIX):
                continue
            dotted = env_key[len(Config.ENV_PREFIX):].lower().replace("__", ".")
            parts = dotted.split(".")
            node = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = _coerce(env_value)
        return out

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, *files: str | None) -> "Config":
        """Load and deep-merge one or more YAML config files.

        ``None`` entries are skipped, so callers can pass optional
        experiment configs safely.  Relative names resolve against
        ``configs/``.
        """
        data: dict[str, Any] = {}
        for name in files:
            if not name:
                continue
            path = resolve(name) if os.path.isabs(name) or "/" in name else CONFIG_DIR / name
            with open(path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            data = deep_merge(data, loaded)
        return cls(data)

    # ------------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"Missing required config key: {dotted_key}")
        return value

    def section(self, dotted_key: str, default: dict | None = None) -> dict:
        value = self.get(dotted_key, default if default is not None else {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section {dotted_key!r} is not a mapping")
        return value

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(keys={sorted(self._data)})"


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()
