"""FastAPI service exposing the search pipeline.

Run:
    uv run uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import ipaddress
import math
import re
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from limits import parse as parse_rate_limit
from openai import OpenAIError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.llm_endpoints import (
    EmptyCompletionError,
    EndpointConfigError,
    MissingApiKey,
    SpendPolicyError,
    resolve_endpoint,
)
from app.retrieve.pubmed import quiet_http_logs
from app.retrieve.semantic_scholar import S2Error, safe_http_url
from app.retrieve.service import MODES, WEB_MODES, SearchService
from app.schemas.api import AnswerResponse, Hit, SearchResponse

# httpx logs every outbound request's full URL at INFO, and the optional NCBI api_key is
# a URL parameter: keep those loggers at WARNING (and redacting) in the API process.
quiet_http_logs()

app = FastAPI(title="Semantic Search with RAG")

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
    out = []
    for h in hits:
        year = h.metadata.get("year")
        out.append(
            Hit(
                doc_id=h.doc_id,
                score=h.score,
                title=h.metadata.get("title", ""),
                text=h.text,
                # Web hits only; re-checked here so a non-http(s) URL can never reach a
                # client, whatever produced the hit.
                url=safe_http_url(h.metadata.get("url")) or None,
                year=year if isinstance(year, int) and not isinstance(year, bool) else None,
                source=h.metadata.get("source") or "local",
            )
        )
    return out


# --- Web-mode guards ------------------------------------------------------------------
# A web-mode request (web / hybrid_web) fans out to several Semantic Scholar + PubMed
# calls, so on top of the per-endpoint limits it passes two extra guards. Local modes
# never touch either: they cost no outbound requests and keep exactly their old limits.
#
# 1. A stricter per-IP rate limit (settings.web_rate_limit). It is checked by hand
#    rather than with @limiter.limit, because a decorator limit cannot see the `mode`
#    query parameter (exempt_when gets no request). It goes through the SAME slowapi
#    backend and storage (so limiter.reset() clears it and a storage_uri scales it out),
#    keyed on the same _client_ip under a "web-modes" namespace, which no endpoint-limit
#    key can collide with. One bucket is shared by /search and /answer: the budget is
#    outbound web load per client, whichever endpoint it arrives through.
# 2. A process-wide cap on concurrent web retrievals (settings.web_max_concurrent).
#    Acquisition never blocks: with every slot busy the request gets 503 + Retry-After
#    at once, rather than parking a worker thread behind a slow upstream. A counter under
#    a lock (not a Semaphore built at import) reads the setting live, so changing it
#    never strands a held slot on a discarded semaphore.
#
# Order: endpoint limit (decorator) -> mode check -> web rate limit -> web slot. A 503 for
# "busy" therefore still spends one web token, so a client retrying in a tight loop is
# throttled by its own bucket rather than by everyone else's capacity.
_WEB_RATE_SCOPE = "web-modes"
_WEB_BUSY_RETRY_AFTER_S = 5
_web_slots_lock = threading.Lock()
_web_active = 0


def _check_web_rate_limit(request: Request) -> None:
    """Spend one token of the client's web-mode bucket, or raise 429 with Retry-After."""
    if not limiter.enabled:
        return
    item = parse_rate_limit(settings.web_rate_limit)  # read live: tests/ops can change it
    key = _client_ip(request)
    backend = limiter.limiter
    if backend.hit(item, _WEB_RATE_SCOPE, key):
        return
    reset_at = backend.get_window_stats(item, _WEB_RATE_SCOPE, key).reset_time
    retry_after = max(1, math.ceil(reset_at - time.time()))
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded for web search modes: {item}",
        headers={"Retry-After": str(retry_after)},
    )


def _try_acquire_web_slot() -> bool:
    global _web_active
    with _web_slots_lock:
        if _web_active >= settings.web_max_concurrent:
            return False
        _web_active += 1
        return True


def _release_web_slot() -> None:
    global _web_active
    with _web_slots_lock:
        _web_active = max(0, _web_active - 1)


@contextmanager
def _web_guard(request: Request, mode: str) -> Iterator[None]:
    """Hold a web slot for the block (web modes only); a no-op for local modes."""
    if mode not in WEB_MODES:
        yield
        return
    _check_web_rate_limit(request)
    if not _try_acquire_web_slot():
        raise HTTPException(
            status_code=503,
            detail="Web search is busy; try again shortly.",
            headers={"Retry-After": str(_WEB_BUSY_RETRY_AFTER_S)},
        )
    try:
        yield
    finally:
        _release_web_slot()


def _add_warnings(warnings: list[str], extra: Iterable | None) -> list[str]:
    """Append `extra` strings to `warnings`, de-duplicated, first occurrence kept."""
    if isinstance(extra, str):  # one bare string, not an iterable of characters
        extra = [extra]
    merged = list(dict.fromkeys([*warnings, *(w for w in extra or [] if isinstance(w, str) and w)]))
    warnings[:] = merged
    return warnings


# An exception CLASS name (what generator.reask_error records) — anything else is dropped
# from the warning text, so a raw message or a key fragment can never reach a client.
_ERROR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def _reask_warning(ans) -> str | None:
    err = getattr(ans, "reask_error", None)
    if not err:
        return None
    name = f" ({err})" if isinstance(err, str) and _ERROR_NAME.fullmatch(err) else ""
    return f"Verdict check failed{name}; showing the first answer without a verdict."


def _retrieve(q: str, mode: str, top_k: int) -> tuple[list, list[str]]:
    """Retrieval for one request, returning (hits, warnings).

    Local modes run entirely under the retrieval lock, exactly as before. Web modes hold
    it only around the local part (SearchService's local_lock), so a slow or throttled
    Semantic Scholar never blocks other users' local searches. `web` has no local
    fallback, so an S2 failure is a 503; hybrid_web degrades to local results + warning.
    Any `warnings` attribute the returned hits carry (a hook for the retrieval layer) is
    merged in, de-duplicated. Callers hold the web slot (_web_guard) around this.
    """
    warnings: list[str] = []
    if mode not in WEB_MODES:
        with _retrieval_lock:
            hits = get_service().retrieve(q, mode=mode, top_k=top_k)
        return hits, _add_warnings(warnings, getattr(hits, "warnings", None))
    with _retrieval_lock:  # first-call construction opens embedded Qdrant: never twice
        service = get_service()
    try:
        hits = service.retrieve(
            q, mode=mode, top_k=top_k, warnings=warnings, local_lock=_retrieval_lock
        )
    except S2Error as e:
        raise HTTPException(status_code=503, detail=f"Semantic Scholar search unavailable ({e})")
    return hits, _add_warnings(warnings, getattr(hits, "warnings", None))


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
    with _web_guard(request, mode):  # web modes: stricter limit + concurrency slot
        hits, warnings = _retrieve(q, mode, top_k)
    return SearchResponse(query=q, mode=mode, hits=_to_hits(hits), warnings=warnings)


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
    try:  # provider-aware: names the missing env var (never a value)
        resolve_endpoint("generator")
    except (MissingApiKey, EndpointConfigError, SpendPolicyError) as e:
        raise HTTPException(status_code=503, detail=f"LLM not configured ({e})")
    # Web hits go to the generator unchanged: the same grounded prompt (context as
    # data, question delimited and sanitised) applies, and their text was already
    # normalised as untrusted by the S2 retriever.
    # The web slot covers RETRIEVAL only. It exists to bound concurrent outbound
    # S2/PubMed load, and generation makes none of that; holding it across an LLM call
    # (tens of seconds with reasoning + a re-ask) would let two slow answers lock web
    # search out for everyone. Generation has its own guards: this endpoint's per-IP
    # limit, the OpenRouter spend policy and the key's credit limit.
    with _web_guard(request, mode):
        hits, warnings = _retrieve(q, mode, top_k)
    try:
        ans = get_generator().generate(q, hits)  # network call — safe outside the lock
    except OpenAIError as e:  # bad key, model gone, provider down — not a server bug
        raise HTTPException(status_code=502, detail=f"LLM backend error: {type(e).__name__}")
    except EmptyCompletionError:  # HTTP 200 with no choices: an upstream failure too
        raise HTTPException(status_code=502, detail="LLM backend returned no completion")
    reask_warning = _reask_warning(ans)
    _add_warnings(warnings, [reask_warning] if reask_warning else None)
    _add_warnings(warnings, getattr(ans, "warnings", None))
    return AnswerResponse(
        query=q,
        answer=ans.text,
        citations=ans.citations,
        hits=_to_hits(hits),
        verdict=ans.verdict,
        verdict_source=getattr(ans, "verdict_source", None),  # GeneratedAnswer side channel
        warnings=warnings,
    )
