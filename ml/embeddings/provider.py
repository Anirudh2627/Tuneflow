"""Embedding providers.

``EmbeddingProvider`` is the stable interface; swap implementations via
config (``content.text_embedding.provider``) without touching callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ml.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Maps text to dense vectors."""

    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array."""

    def name(self) -> str:
        return type(self).__name__


class SentenceTransformerProvider(EmbeddingProvider):
    """Lightweight sentence-transformers embeddings.

    The model is loaded lazily and cached on disk by sentence-transformers
    itself.  Used for semantic search / text similarity where it adds
    genuine value; the content recommender works without it.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is not installed. Install with "
                    "`pip install sentence-transformers` or switch "
                    "content.text_embedding.provider to 'none'."
                ) from exc
            logger.info("Loading sentence-transformer model %r", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class NoopEmbeddingProvider(EmbeddingProvider):
    """Placeholder when embeddings are disabled."""

    dim = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise RuntimeError("Embeddings are disabled (provider='none').")


def build_embedding_provider(config: dict | None) -> EmbeddingProvider | None:
    """Factory driven by config ``{provider, model_name, device, batch_size}``."""
    config = config or {}
    provider = str(config.get("provider", "none")).lower()
    if provider in ("none", "", "disabled"):
        return None
    if provider in ("sentence-transformer", "sentence_transformers", "sbert"):
        return SentenceTransformerProvider(
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            device=config.get("device"),
            batch_size=int(config.get("batch_size", 64)),
        )
    raise ValueError(f"Unknown embedding provider: {provider!r}")
