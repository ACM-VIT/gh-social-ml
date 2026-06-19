"""Text embedding helpers for repository documents."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import README_CHUNK_CHARS, README_CHUNK_OVERLAP_CHARS, REPOSITORY_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


Vector = list[float]


@dataclass(slots=True)
class TextChunk:
    """A chunk of source text prepared for embedding."""

    text: str
    index: int
    start_char: int
    end_char: int


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers wrapper used by repository embedding stages."""

    def __init__(self, model_name: str = REPOSITORY_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        # The below lazy load is for keeping normal imports lightweight; the
        # transformer model is initialized only when embeddings are requested.
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for repository embeddings. "
                    "Install dependencies from requirements.txt."
                ) from exc
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: Sequence[str], *, normalize: bool = True) -> list[Vector]:
        """Embed a sequence of texts and return JSON-serializable vectors."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors]

    def embed_text(self, text: str, *, normalize: bool = True) -> Vector:
        """Embed one text value."""
        vectors = self.embed_texts([text], normalize=normalize)
        return vectors[0] if vectors else []


def chunk_text(
    text: str,
    *,
    max_chars: int = README_CHUNK_CHARS,
    overlap_chars: int = README_CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Split text into overlapping chunks suitable for README embeddings."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be greater than or equal to 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [TextChunk(text=clean, index=0, start_char=0, end_char=len(clean))]

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(clean):
        hard_end = min(start + max_chars, len(clean))
        # The below boundary selection is for keeping README chunks readable by
        # avoiding paragraph or sentence splits when a nearby boundary exists.
        end = _best_chunk_boundary(clean, start, hard_end)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(TextChunk(text=chunk, index=index, start_char=start, end_char=end))
            index += 1
        if end >= len(clean):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def aggregate_vectors(vectors: Sequence[Sequence[float]], *, weights: Sequence[float] | None = None) -> Vector:
    """Average and L2-normalize vectors."""
    if not vectors:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    if weights is not None:
        weight_array = np.asarray(weights, dtype=np.float32)
        if len(weight_array) != len(matrix):
            raise ValueError("weights length must match vectors length")
        total = float(weight_array.sum())
        if total <= 0:
            weight_array = np.ones(len(matrix), dtype=np.float32)
            total = float(weight_array.sum())
        vector = np.average(matrix, axis=0, weights=weight_array / total)
    else:
        vector = np.mean(matrix, axis=0)

    # The below normalization is for cosine-distance storage in Qdrant.
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return vector.astype(np.float32).tolist()


def _best_chunk_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    for marker in ("\n\n", "\n", ". "):
        idx = window.rfind(marker)
        if idx > int(len(window) * 0.55):
            return start + idx + len(marker)
    return hard_end
