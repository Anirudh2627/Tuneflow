"""Dataset adapters.

An adapter converts a specific raw dataset layout into the canonical
:class:`ml.datasets.schema.Dataset`.  Adding a new dataset means adding
an adapter here -- no other module changes.
"""
from __future__ import annotations

from ml.datasets.adapters.base import BaseAdapter
from ml.datasets.adapters.csv_adapter import CsvAdapter
from ml.datasets.adapters.msd import MsdAdapter
from ml.datasets.adapters.sample import SampleAdapter

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "sample": SampleAdapter,
    "msd": MsdAdapter,
    "csv": CsvAdapter,
}


def get_adapter(name: str, **kwargs) -> BaseAdapter:
    try:
        return ADAPTERS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset source {name!r}. Available: {sorted(ADAPTERS)}") from exc


__all__ = ["BaseAdapter", "CsvAdapter", "MsdAdapter", "SampleAdapter", "get_adapter", "ADAPTERS"]
