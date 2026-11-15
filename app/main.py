"""FastAPI app. Run with:  uvicorn app.main:create_app --factory --reload"""
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .llm import LLM, build_llm
from .loaders import UnsupportedFile
from .retriever import MODES, CrossEncoderReranker, Reranker
from .service import DuplicateDocument, RagService

STATIC = Path(__file__).parent / "static"


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] | None = None
    top_k: int = Field(default=6, ge=1, le=20)
    mode: str | None = None


def create_app(settings: Settings | None = None, *, embedder: Embedder | None = None,
               llm: LLM | None = None, reranker: Reranker | None = None,
               persist: bool = True) -> FastAPI:
    s = settings or get_settings()
    embedder = embedder or build_embedder(s.embed_provider, s.embed_model)
    if reranker is None and s.rerank_enabled and persist:
        reranker = CrossEncoderReranker(s.rerank_model)
    service = RagService(s, embedder, llm or build_llm(s), reranker, persist=persist)

    app = FastAPI(title="DocuChat", version="1.0.0",
                  description="Hybrid-search RAG with reranking and cited, streamed answers.")
    app.state.service = service

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "documents": len(service.registry.list()),
            "chunks": len(service.bm25),
            "llm": {"provider": s.llm_provider, "model": service.llm.model},
            "embedding_model": s.embed_model if s.embed_provider != "hash" else "hash (dev only)",
            "reranker": s.rerank_model if service.retriever.reranker else None,
        }

    @app.get("/api/documents")
    def list_documents() -> list[dict]:
        return service.registry.list()

    @app.post("/api/documents")
    async def upload_documents(files: list[UploadFile] = File(...)) -> list[dict]:
        limit = s.max_upload_mb * 1024 * 1024
        results = []
        for f in files:
            name = Path(f.filename or "document").name
            data = await f.read(limit + 1)
            if len(data) > limit:
                results.append({"name": name, "status": "error",
                                "detail": f"File is larger than {s.max_upload_mb} MB."})
                continue
            try:
                doc = await asyncio.to_thread(service.ingest, name, data)
                results.append({"name": name, "status": "indexed", "document": doc})
            except DuplicateDocument as e:
                results.append({"name": name, "status": "duplicate", "document": e.existing})
            except UnsupportedFile as e:
                results.append({"name": name, "status": "error", "detail": str(e)})
            except Exception as e:
                results.append({"name": name, "status": "error",
                                "detail": f"Could not read this file ({type(e).__name__})."})
        return results

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: str) -> dict:
        if not await asyncio.to_thread(service.delete, doc_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id}

    @app.post("/api/search")
    async def search(req: SearchRequest) -> dict:
        """Retrieval only, with per-stage scores. Useful for debugging and demos."""
        if req.mode and req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        hits = await asyncio.to_thread(
            service.search, req.query, req.top_k, set(req.doc_ids) if req.doc_ids else None, req.mode
        )
        return {"query": req.query, "results": [service.source_view(i, h) for i, h in enumerate(hits, 1)]}

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        async def events():
            async for name, data in service.chat(
                req.question, [t.model_dump() for t in req.history],
                set(req.doc_ids) if req.doc_ids else None, req.top_k,
            ):
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
