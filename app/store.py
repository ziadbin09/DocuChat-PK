"""Qdrant vector store.

Runs embedded (on-disk, no server) by default and against a Qdrant server or
Qdrant Cloud when QDRANT_URL is set. The code path is identical.
"""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


class VectorStore:
    def __init__(self, collection: str, dim: int, *, path: Path | None = None,
                 url: str | None = None, api_key: str | None = None):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        else:
            existing = self.client.get_collection(collection).config.params.vectors.size
            if existing != dim:
                raise RuntimeError(
                    f"Collection '{collection}' holds {existing}-dim vectors but the embedding "
                    f"model produces {dim}. Delete the data directory or change COLLECTION."
                )

    @staticmethod
    def _doc_filter(doc_ids: set[str] | None) -> models.Filter | None:
        if not doc_ids:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=sorted(doc_ids)))]
        )

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        for start in range(0, len(ids), 128):
            end = start + 128
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=i, vector=v.tolist(), payload=p)
                    for i, v, p in zip(ids[start:end], vectors[start:end], payloads[start:end])
                ],
            )

    def search(self, vector: np.ndarray, limit: int, doc_ids: set[str] | None = None) -> list[tuple[str, float, dict]]:
        result = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
            query_filter=self._doc_filter(doc_ids),
        )
        return [(str(p.id), float(p.score), p.payload or {}) for p in result.points]

    def get(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        points = self.client.retrieve(self.collection, ids=ids, with_payload=True)
        return {str(p.id): p.payload or {} for p in points}

    def scroll_all(self) -> Iterator[tuple[str, dict]]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                yield str(p.id), p.payload or {}
            if offset is None:
                return

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._doc_filter({doc_id})),
        )

    def close(self) -> None:
        self.client.close()
