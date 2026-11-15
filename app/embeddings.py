"""Embedding backends behind one small interface."""
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings (no GPU, no API key). The model downloads once on first use."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words hashing vectors. For tests and offline dev only:
    it captures word overlap, not meaning."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def build_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(model_name)
