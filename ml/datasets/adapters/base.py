"""Adapter base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ml.datasets.schema import Dataset, validate_dataset


class BaseAdapter(ABC):
    """Converts a raw dataset directory into the canonical Dataset."""

    name: str = "base"

    @abstractmethod
    def load(self) -> Dataset:
        """Read raw files and return a validated Dataset."""

    def load_validated(self) -> Dataset:
        return validate_dataset(self.load())
