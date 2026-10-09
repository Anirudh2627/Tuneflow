"""API logging configuration."""
from __future__ import annotations

import logging


def setup_api_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    for noisy in ("uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING if not debug else logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
