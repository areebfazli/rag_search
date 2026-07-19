"""FastAPI service exposing the search pipeline.

Run:
    uv run uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import ipaddress
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from openai import OpenAIError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.retrieve.service import MODES, SearchService
from app.schemas.api import AnswerResponse, Hit, SearchResponse

app = FastAPI(title="Semantic Search + RAG")

# The API is unauthenticated; without a limit, a public deploy could be spammed to
# burn the LLM token budget or pin the CPU. Per-IP rate limiting mitigates that.
def _forwarded_ip(value: str) -> str | None:
    """Normalise one X-Forwarded-For entry to a bare IP, or None if it isn't one.

    Some proxies append ``client:port`` (Azure App Service, HAProxy's forwardfor with
    port). ``ipaddress`` rejects those outright, and falling back to the peer address
    would put every client behind the proxy in ONE bucket — silently turning per-IP
    limiting into a single global limit that one client can exhaust for everyone.
    """
    v = value.strip()
    if v.startswith("["):  # [2001:db8::5]:443
        v = v[1:].partition("]")[0]
    elif v.count(":") == 1:  # 1.2.3.4:5678 — exactly one colon can't be IPv6
        v = v.partition(":")[0]
    try:
        return str(ipaddress.ip_address(v))
    except ValueError:
        return None


def _client_ip(request: Request) -> str:
    # Behind a trusted proxy, count in from the RIGHT of X-Forwarded-For: each proxy
    # appends the peer it saw, so our own proxies own the last `trusted_proxy_hops`
    # entries and the client sits just before them. Leftmost entries are
    # client-supplied — keying on them would let a client spoof a fresh IP per request
    # and evade the limit. (get_remote_address alone would return the proxy IP → one
    # shared bucket for everyone.)
    if settings.trust_proxy:
        # RFC 7230 treats repeated header lines as one comma-joined list, but
        # Starlette's .get() returns only the FIRST line. Proxies that append XFF as a
        # new line (Envoy, some ingresses) would then let a client-supplied first line
        # win — so join every line before indexing from the right.
        parts = [
            piece.strip()
            for value in request.headers.getlist("x-forwarded-for")
            for piece in value.split(",")
            if piece.strip()
        ]
        hops = max(1, settings.trusted_proxy_hops)
        if len(parts) >= hops:
            # Normalising also collapses equivalent spellings of one address
            # (e.g. ::1 vs 0:0:0:0:0:0:0:1) into a single limiter bucket.
            ip = _forwarded_ip(parts[-hops])
            if ip is not None:
                return ip
            # Not an address — fall back to the real peer rather than trust it.
    return get_remote_address(request)


# NOTE: slowapi's default storage is in-memory / per-process — limits are not shared
# across workers or instances. Set a storage_uri (e.g. Redis) to scale out.
limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_service: SearchService | None = None
_generator = None
# Embedded (local-mode) Qdrant and the shared ST models are not thread-safe, and
# FastAPI runs sync endpoints in a threadpool — serialize retrieval with a lock.
# (The store also locks to one process: don't run the API and `make eval` at once.)
_retrieval_lock = threading.Lock()
_gen_lock = threading.Lock()
_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


def get_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


def get_generator():
    global _generator
    if _generator is None:  # double-checked lock — avoid two concurrent inits
        with _gen_lock:
            if _generator is None:
                from app.generate.generator import LLMGenerator

                _generator = LLMGenerator()
    return _generator


def _to_hits(hits) -> list[Hit]:
    return [
        Hit(doc_id=h.doc_id, score=h.score, title=h.metadata.get("title", ""), text=h.text)
        for h in hits
    ]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_FRONTEND)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=1000, description="query text"),
    mode: str = Query("hybrid", description=f"one of {MODES}"),
    top_k: int = Query(8, ge=1, le=100),
) -> SearchResponse:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    with _retrieval_lock:
        hits = get_service().retrieve(q, mode=mode, top_k=top_k)
    return SearchResponse(query=q, mode=mode, hits=_to_hits(hits))


@app.get("/answer", response_model=AnswerResponse)
@limiter.limit("10/minute")
def answer(
    request: Request,
    q: str = Query(..., min_length=1, max_length=1000, description="question"),
    mode: str = Query("hybrid", description=f"retrieval mode, one of {MODES}"),
    top_k: int = Query(5, ge=1, le=20),
) -> AnswerResponse:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM not configured (set SSR_LLM_API_KEY)")
    with _retrieval_lock:
        hits = get_service().retrieve(q, mode=mode, top_k=top_k)
    try:
        ans = get_generator().generate(q, hits)  # network call — safe outside the lock
    except OpenAIError as e:  # bad key, model gone, provider down — not a server bug
        raise HTTPException(status_code=502, detail=f"LLM backend error: {type(e).__name__}")
    return AnswerResponse(query=q, answer=ans.text, citations=ans.citations, hits=_to_hits(hits))
