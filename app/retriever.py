"""Hybrid retrieval: dense + BM25 -> reciprocal rank fusion -> cross-encoder rerank."""
from dataclasses import dataclass, field
from typing import Protocol

from .bm25 import BM25Index
from .embeddings import Embedder
from .store import VectorStore

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk_id: str
    payload: dict
    score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Cross-encoder reranker via fastembed (ONNX, CPU). Loaded lazily on first query."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self.model_name)
        return [float(s) for s in self._model.rerank(query, passages)]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Rank-based, so it needs no score normalisation
    between BM25 scores and cosine similarities."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, store: VectorStore, bm25: BM25Index, embedder: Embedder,
                 reranker: Reranker | None = None, *, dense_candidates: int = 30,
                 bm25_candidates: int = 30, rerank_candidates: int = 20, rrf_k: int = 60):
        self.store, self.bm25, self.embedder, self.reranker = store, bm25, embedder, reranker
        self.dense_candidates = dense_candidates
        self.bm25_candidates = bm25_candidates
        self.rerank_candidates = rerank_candidates
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 6, doc_ids: set[str] | None = None,
               mode: str = "hybrid_rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        hits: dict[str, Hit] = {}
        dense_ids: list[str] = []
        bm25_ids: list[str] = []

        if mode != "bm25":
            vector = self.embedder.embed_query(query)
            for rank, (cid, score, payload) in enumerate(
                self.store.search(vector, self.dense_candidates, doc_ids), start=1
            ):
                hits[cid] = Hit(cid, payload, dense_rank=rank, extra={"dense_score": score})
                dense_ids.append(cid)

        if mode != "dense":
            results = self.bm25.search(query, self.bm25_candidates, doc_ids)
            missing = [cid for cid, _ in results if cid not in hits]
            payloads = self.store.get(missing)
            for rank, (cid, score) in enumerate(results, start=1):
                if cid not in hits:
                    if cid not in payloads:
                        continue
                    hits[cid] = Hit(cid, payloads[cid])
                hits[cid].bm25_rank = rank
                hits[cid].extra["bm25_score"] = score
                bm25_ids.append(cid)

        fused = reciprocal_rank_fusion([r for r in (dense_ids, bm25_ids) if r], self.rrf_k)
        ordered: list[Hit] = []
        for cid, score in fused:
            hit = hits[cid]
            hit.rrf_score = hit.score = score
            ordered.append(hit)

        if mode == "hybrid_rerank" and self.reranker and ordered:
            pool = ordered[: self.rerank_candidates]
            scores = self.reranker.score(query, [h.text for h in pool])
            for hit, score in zip(pool, scores):
                hit.rerank_score = hit.score = score
            ordered = sorted(pool, key=lambda h: h.rerank_score, reverse=True)

        return ordered[:top_k]
