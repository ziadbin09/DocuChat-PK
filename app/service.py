"""The RAG pipeline: ingest documents, retrieve, and stream grounded answers."""
import asyncio
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator

from . import loaders
from .bm25 import BM25Index
from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .llm import LLM, LLMError
from .registry import Registry
from .retriever import Hit, Reranker, Retriever
from .store import VectorStore

log = logging.getLogger("docuchat")
_CHUNK_NAMESPACE = uuid.UUID("6f0c2b1e-5a3d-4c7e-9b1a-2d4f6a8c0e11")


class DuplicateDocument(Exception):
    def __init__(self, existing: dict):
        self.existing = existing


class RagService:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLM,
                 reranker: Reranker | None = None, *, persist: bool = True):
        self.s = settings
        self.embedder = embedder
        self.llm = llm
        self.registry = Registry(settings.data_dir / "registry.sqlite" if persist else None)
        self.store = VectorStore(
            settings.collection,
            embedder.dim,
            path=settings.data_dir / "qdrant" if persist and not settings.qdrant_url else None,
            url=settings.qdrant_url if persist else None,
            api_key=settings.qdrant_api_key,
        )
        self.bm25 = BM25Index()
        for chunk_id, payload in self.store.scroll_all():
            self.bm25.add(chunk_id, payload["doc_id"], self._keyword_text(payload))
        self.retriever = Retriever(
            self.store, self.bm25, embedder, reranker,
            dense_candidates=settings.dense_candidates, bm25_candidates=settings.bm25_candidates,
            rerank_candidates=settings.rerank_candidates, rrf_k=settings.rrf_k,
        )
        # Embedded Qdrant and the BM25 index are not safe for concurrent writers.
        self._lock = threading.Lock()

    @staticmethod
    def _keyword_text(payload: dict) -> str:
        return f"{payload.get('heading', '')}\n{payload['text']}"

    # ---------------------------------------------------------------- ingest
    def ingest(self, filename: str, data: bytes) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        if existing := self.registry.by_hash(sha):
            raise DuplicateDocument(existing)

        pages = loaders.load(filename, data)
        chunks = chunk_pages(pages, self.s.chunk_chars, self.s.chunk_overlap_chars)
        doc_id = uuid.uuid4().hex[:12]
        title = filename.rsplit(".", 1)[0]

        payloads = [
            {
                "doc_id": doc_id, "doc_name": filename, "chunk_index": i, "text": c.text,
                "heading": c.heading, "page_start": c.page_start, "page_end": c.page_end,
            }
            for i, c in enumerate(chunks)
        ]
        ids = [str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_id}:{i}")) for i in range(len(chunks))]
        # Contextual chunk headers: embed each chunk together with its document title
        # and section so "what is the limit?"-style chunks keep their context.
        vectors = self.embedder.embed_documents(
            [f"{title} | {c.heading}\n{c.text}" if c.heading else f"{title}\n{c.text}" for c in chunks]
        )
        with self._lock:
            self.store.upsert(ids, vectors, payloads)
            for chunk_id, payload in zip(ids, payloads):
                self.bm25.add(chunk_id, doc_id, self._keyword_text(payload))
            return self.registry.add(doc_id, filename, sha, len(pages), len(chunks), len(data))

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if not self.registry.delete(doc_id):
                return False
            self.store.delete_document(doc_id)
            self.bm25.remove_document(doc_id)
            return True

    # ---------------------------------------------------------------- search
    def search(self, query: str, top_k: int | None = None, doc_ids: set[str] | None = None,
               mode: str | None = None) -> list[Hit]:
        mode = mode or ("hybrid_rerank" if self.retriever.reranker else "hybrid")
        with self._lock:
            return self.retriever.search(query, top_k or self.s.top_k, doc_ids, mode)

    @staticmethod
    def source_view(n: int, hit: Hit) -> dict:
        p = hit.payload
        pages = f"p.{p['page_start']}" if p["page_start"] == p["page_end"] else f"pp.{p['page_start']}-{p['page_end']}"
        location = f"{pages}, {p['heading']}" if p.get("heading") else pages
        return {
            "n": n,
            "chunk_id": hit.chunk_id,
            "doc_id": p["doc_id"],
            "doc_name": p["doc_name"],
            "title": f"{p['doc_name']} ({location})",
            "uri": f"doc://{p['doc_id']}#chunk-{p['chunk_index']}",
            "location": location,
            "text": p["text"],
            "scores": {
                "dense_rank": hit.dense_rank, "bm25_rank": hit.bm25_rank,
                "rrf": hit.rrf_score, "rerank": hit.rerank_score,
            },
        }

    def _clean_history(self, history: list[dict]) -> list[dict]:
        turns = [
            {"role": m["role"], "content": m["content"].strip()}
            for m in history
            if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
        ][-2 * self.s.history_turns :]
        # The Messages API needs alternating roles that start with the user.
        cleaned: list[dict] = []
        for m in turns:
            if not cleaned and m["role"] != "user":
                continue
            if cleaned and cleaned[-1]["role"] == m["role"]:
                cleaned[-1] = m
            else:
                cleaned.append(m)
        if cleaned and cleaned[-1]["role"] == "user":
            cleaned.pop()
        return cleaned

    # ------------------------------------------------------------------ chat
    async def chat(self, question: str, history: list[dict] | None = None,
                   doc_ids: set[str] | None = None, top_k: int | None = None) -> AsyncIterator[tuple[str, dict]]:
        """Yield (event_name, data) pairs for the SSE stream."""
        started = time.perf_counter()
        history = self._clean_history(history or [])
        try:
            query = question
            if history:
                yield "status", {"message": "Reading the conversation"}
                try:
                    query = await self.llm.rewrite(question, history)
                except Exception:
                    query = question  # a failed rewrite should never block the answer

            yield "status", {"message": "Searching your documents"}
            t_search = time.perf_counter()
            hits = await asyncio.to_thread(self.search, query, top_k, doc_ids)
            retrieval_ms = round((time.perf_counter() - t_search) * 1000)
            sources = [self.source_view(i, h) for i, h in enumerate(hits, start=1)]
            yield "sources", {"query": query, "sources": sources, "retrieval_ms": retrieval_ms}

            if not sources:
                yield "token", {"text": "I couldn't find anything in your documents for that. "
                                        "Upload a document first, or check the document filter."}
                yield "done", {"stop_reason": "no_sources", "retrieval_ms": retrieval_ms,
                               "total_ms": round((time.perf_counter() - started) * 1000)}
                return

            yield "status", {"message": "Writing the answer"}
            async for event in self.llm.stream_answer(question, sources, history):
                if event["type"] == "text":
                    yield "token", {"text": event["text"]}
                elif event["type"] == "citation":
                    yield "citation", {"n": event["n"], "quote": event["quote"]}
                elif event["type"] == "end":
                    if event["stop_reason"] == "refusal":
                        yield "token", {"text": "\n\nThe model declined to answer this request."}
                    elif event["stop_reason"] == "max_tokens":
                        yield "token", {"text": "\n\n_(The answer was cut off at the length limit.)_"}
                    yield "done", {
                        "stop_reason": event["stop_reason"], "model": event["model"],
                        "usage": event["usage"], "retrieval_ms": retrieval_ms,
                        "total_ms": round((time.perf_counter() - started) * 1000),
                    }
        except LLMError as e:
            yield "error", {"message": str(e)}
        except Exception as e:  # keep the stream well-formed whatever happens
            log.exception("chat failed")
            yield "error", {"message": f"Unexpected error: {type(e).__name__}"}

    def close(self) -> None:
        self.store.close()
