"""FastAPI service exposing the search pipeline.

Run:
    uv run uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app.retrieve.service import MODES, SearchService
from app.schemas.api import AnswerResponse, Hit, SearchResponse

app = FastAPI(title="Semantic Search + RAG")
_service: SearchService | None = None
_generator = None
# Embedded (local-mode) Qdrant and the shared ST models are not thread-safe, and
# FastAPI runs sync endpoints in a threadpool — serialize retrieval with a lock.
# (The store also locks to one process: don't run the API and `make eval` at once.)
_retrieval_lock = threading.Lock()
_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


def get_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


def get_generator():
    global _generator
    if _generator is None:  # lazy — only needs the LLM key when /answer is called
        from app.generate.generator import LLMGenerator

        _generator = LLMGenerator()
    return _generator


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_FRONTEND)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="query text"),
    mode: str = Query("hybrid", description=f"one of {MODES}"),
    top_k: int = Query(8, ge=1, le=100),
) -> SearchResponse:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    with _retrieval_lock:
        hits = get_service().retrieve(q, mode=mode, top_k=top_k)
    return SearchResponse(
        query=q,
        mode=mode,
        hits=[
            Hit(doc_id=h.doc_id, score=h.score, title=h.metadata.get("title", ""), text=h.text)
            for h in hits
        ],
    )


@app.get("/answer", response_model=AnswerResponse)
def answer(
    q: str = Query(..., description="question"),
    mode: str = Query("hybrid", description=f"retrieval mode, one of {MODES}"),
    top_k: int = Query(5, ge=1, le=20),
) -> AnswerResponse:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    with _retrieval_lock:
        hits = get_service().retrieve(q, mode=mode, top_k=top_k)
    ans = get_generator().generate(q, hits)  # network call — safe outside the lock
    return AnswerResponse(
        query=q,
        answer=ans.text,
        citations=ans.citations,
        hits=[
            Hit(doc_id=h.doc_id, score=h.score, title=h.metadata.get("title", ""), text=h.text)
            for h in hits
        ],
    )
