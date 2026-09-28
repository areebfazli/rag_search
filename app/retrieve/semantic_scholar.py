"""Semantic Scholar (S2) paper search as an optional, networked Retriever.

Adapts ``GET https://api.semanticscholar.org/graph/v1/paper/search`` (relevance search;
``limit`` must be <= 100 — Academic Graph API spec, api.semanticscholar.org/api-docs/graph)
to the Retriever Protocol, so the SearchService can offer it as the ``web`` and
``hybrid_web`` modes. It is NOT part of the default pipeline: hybrid (RRF over the local
index) stays the default, and nothing here runs unless one of those modes is asked for.

Why SciFact and S2 line up: SciFact's doc ids are S2ORC ids (allenai/scifact
doc/data.md: "The document's S2ORC ID"), i.e. S2 corpus ids — so ``doc_id`` here is
``str(corpusId)``, and a paper present in both the local corpus and S2 dedupes to one hit.

Everything S2 returns is UNTRUSTED text (anyone can get an abstract into S2). It is
normalised to a single line with no quote runs or control characters (so it cannot forge
the grounded prompt's ``\"\"\"`` delimiter or its line structure), length-capped, and
URLs are kept only if they are plain http(s).

Rate limits (semanticscholar.org/product/api): without a key, requests share a public
pool ("1000 requests per second shared among all unauthenticated users", "may also be
further throttled during periods of heavy use" — in practice it 429s often); with a key
("x-api-key" header) "the introductory rate limit ... is 1 RPS on all endpoints". A
process-wide limiter spaces every request (retries included) to settings.s2_rate_per_s,
so the unauthenticated public /search endpoint cannot be used to hammer S2 or burn the
key. Responses can be cached on disk (data/s2_cache/), keyed by (query, limit, fields,
offset), so evals are reproducible and re-runs are free.
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.core.config import settings
from app.core.interfaces import SearchHit

# Fixed, not a setting: the S2 key must only ever travel to Semantic Scholar.
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_PATH = "/paper/search"
BATCH_PATH = "/paper/batch"
SEARCH_FIELDS = "title,abstract,corpusId,url,year"
S2_MAX_LIMIT = 100  # paper/search: "limit ... Must be <= 100"
S2_BATCH_MAX_IDS = 500  # paper/batch: "Can only process 500 paper ids at a time"
SOURCE = "semantic_scholar"
USER_AGENT = "semantic-search-rag (github.com/areebfazli/rag_search)"

MAX_QUERY_CHARS = 1000
MAX_TITLE_CHARS = 500
MAX_ABSTRACT_CHARS = 4000
BACKOFF_BASE_S = 2.0

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ‪-‮⁦-⁩]")
_QUOTE_RUN = re.compile(r'"{2,}')
_WHITESPACE_RUN = re.compile(r"\s+")
_WS = re.compile(r"\s")


class S2Error(RuntimeError):
    """Any Semantic Scholar failure after retries (network, HTTP status, bad payload,
    rate-limit budget exhausted). Messages carry a status/reason only — never headers,
    so the API key cannot end up in a log or an HTTP response."""


class S2RateLimited(S2Error):
    """429s (or our own limiter) did not clear within the retry/wait budget."""


def clean_untrusted(text: object, max_chars: int) -> str:
    """Normalise third-party text to one bounded line that cannot forge prompt structure.

    Mirrors prompts._sanitize_question for passages that come from the open web: control
    and bidi characters dropped, whitespace (newlines included) collapsed to one space,
    and runs of 2+ double quotes collapsed to one so the ``\"\"\"`` question delimiter
    cannot be reconstructed. Non-strings become "".
    """
    if not isinstance(text, str):
        return ""
    s = _CONTROL.sub("", text)
    s = _QUOTE_RUN.sub('"', _WHITESPACE_RUN.sub(" ", s)).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s


def safe_http_url(url: object) -> str:
    """The URL if it is an absolute http(s) URL with a host, else "" (drops
    javascript:, data:, protocol-relative and malformed values)."""
    if not isinstance(url, str) or len(url) > 2048 or _CONTROL.search(url) or _WS.search(url):
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    return url if parts.scheme in ("http", "https") and parts.netloc else ""


def normalize_query(query: str) -> str:
    """S2 relevance search: "No special query syntax is supported" and "Hyphenated query
    terms yield no matches (replace it with space to find matches)" — so hyphens become
    spaces, whitespace collapses, and the length is capped."""
    q = _WHITESPACE_RUN.sub(" ", _CONTROL.sub("", query).replace("-", " ")).strip()
    return q[:MAX_QUERY_CHARS]


def parse_retry_after(value: str | None, now: Callable[[], float] = time.time) -> float | None:
    """Retry-After as seconds (delta-seconds or an HTTP-date), or None if absent/garbage."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, when.timestamp() - now())


class RateLimiter:
    """Thread-safe request spacer: at most `rate_per_s` requests per second, no bursts.

    Each acquire() reserves the next free slot under a lock and sleeps outside it, so
    concurrent callers queue in order. With `timeout`, a caller whose slot is further
    away than that gives up immediately (returns False) WITHOUT reserving — the API uses
    this so a flood of web-mode requests degrades instead of parking threadpool threads.
    """

    def __init__(
        self,
        rate_per_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be > 0")
        self.interval = 1.0 / rate_per_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_free = float("-inf")

    def acquire(self, timeout: float | None = None) -> bool:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_free)
            wait = slot - now
            if timeout is not None and wait > timeout:
                return False
            self._next_free = slot + self.interval
        if wait > 0:
            self._sleep(wait)
        return True


# One limiter per process, shared by every retriever instance (API and eval alike), so
# the documented per-key rate holds no matter how many retrievers exist. It is
# per-PROCESS: several uvicorn workers each get their own, so divide the rate by the
# worker count if you run more than one.
_SHARED_LIMITER: RateLimiter | None = None
_SHARED_LOCK = threading.Lock()


def shared_limiter() -> RateLimiter:
    global _SHARED_LIMITER
    with _SHARED_LOCK:
        if _SHARED_LIMITER is None:
            _SHARED_LIMITER = RateLimiter(settings.s2_rate_per_s)
        return _SHARED_LIMITER


class ResponseCache:
    """On-disk JSON cache of raw S2 responses, one file per request key.

    Raw payloads (not parsed hits) are stored, so a parsing change never needs a
    re-fetch, and each entry records when it was fetched — S2's index changes over time,
    so an eval built from the cache is a dated snapshot. Writes are atomic
    (temp file + os.replace); an unreadable entry is treated as a miss.
    """

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    @staticmethod
    def key(endpoint: str, params: dict) -> str:
        blob = json.dumps({"endpoint": endpoint, **params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def path(self, endpoint: str, params: dict) -> Path:
        return self.dir / f"{self.key(endpoint, params)}.json"

    def get(self, endpoint: str, params: dict) -> dict | None:
        p = self.path(endpoint, params)
        try:
            entry = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(entry, dict) or "response" not in entry:
            return None
        return entry

    def put(self, endpoint: str, params: dict, response: object) -> dict:
        entry = {
            "endpoint": endpoint,
            "params": params,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "response": response,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(entry, f)
            os.replace(tmp, self.path(endpoint, params))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return entry


def search_params(query: str, limit: int) -> dict:
    """The exact paper/search query parameters — also the cache key."""
    limit = max(1, min(int(limit), S2_MAX_LIMIT))
    return {"query": normalize_query(query), "limit": limit, "fields": SEARCH_FIELDS, "offset": 0}


def parse_search(payload: object, top_k: int, keep_no_abstract: bool = False) -> list[SearchHit]:
    """S2 paper/search payload -> SearchHits, best first, deduped by corpusId.

    Papers without a usable corpusId are dropped (no stable id to fuse/score on), and so
    are papers without an abstract unless `keep_no_abstract` (the eval uses that to ask
    whether S2 *found* the paper, abstract or not). Scores are rank-derived (1/position),
    strictly decreasing, like the eval harness's runs.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for paper in data:
        if len(hits) >= top_k:
            break
        if not isinstance(paper, dict):
            continue
        cid = paper.get("corpusId")
        if isinstance(cid, bool) or not isinstance(cid, (int, str)) or not str(cid).isdigit():
            continue
        doc_id = str(int(cid))
        if doc_id in seen:
            continue
        abstract = clean_untrusted(paper.get("abstract"), MAX_ABSTRACT_CHARS)
        if not abstract and not keep_no_abstract:
            continue
        year = paper.get("year")
        seen.add(doc_id)
        hits.append(
            SearchHit(
                doc_id=doc_id,
                score=1.0 / (len(hits) + 1),
                text=abstract,
                metadata={
                    "title": clean_untrusted(paper.get("title"), MAX_TITLE_CHARS),
                    "url": safe_http_url(paper.get("url")),
                    "year": year if isinstance(year, int) and not isinstance(year, bool) else None,
                    "source": SOURCE,
                },
            )
        )
    return hits


class SemanticScholarRetriever:
    """Retriever over S2 paper relevance search (see module docstring).

    `cache=None` disables caching. `max_wait_s` bounds how long a call waits for a
    rate-limiter slot (None = wait as long as needed, the eval's choice);
    `max_backoff_s` bounds any single retry sleep, Retry-After included — a server
    asking for longer ends the call rather than holding it.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        max_retries: int | None = None,
        max_wait_s: float | None = None,
        max_backoff_s: float = 30.0,
        timeout_s: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._api_key = (settings.s2_api_key if api_key is None else api_key).strip()
        self._client = client or httpx.Client(
            timeout=settings.s2_timeout_s if timeout_s is None else timeout_s
        )
        self.limiter = limiter or shared_limiter()
        self.cache = cache
        self.max_retries = settings.s2_max_retries if max_retries is None else max_retries
        self.max_wait_s = max_wait_s
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        self.requests_sent = 0  # network attempts, retries included (for eval accounting)

    @classmethod
    def for_api(cls) -> "SemanticScholarRetriever":
        """The API's instance: short limiter wait and few retries, so a slow or
        throttled S2 degrades a request quickly instead of pinning a worker thread.
        Caching is opt-in (SSR_S2_API_CACHE): arbitrary user queries would otherwise
        grow the cache without bound on an unauthenticated service."""
        return cls(
            cache=ResponseCache(settings.s2_cache_dir) if settings.s2_api_cache else None,
            max_retries=min(settings.s2_max_retries, 1),
            max_wait_s=settings.s2_max_wait_s,
            max_backoff_s=settings.s2_max_wait_s,
        )

    @property
    def authenticated(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self._api_key:  # header only when a key is configured — never an empty one
            headers["x-api-key"] = self._api_key
        return headers

    def _request(self, method: str, path: str, params: dict, json_body: object = None) -> object:
        """One logical call: limiter slot per attempt, bounded retries on 429/5xx and
        transport errors, Retry-After honoured (capped by max_backoff_s)."""
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            if not self.limiter.acquire(timeout=self.max_wait_s):
                raise S2RateLimited("local rate limit: no request slot within the wait budget")
            self.requests_sent += 1
            try:
                r = self._client.request(
                    method, S2_BASE_URL + path, params=params, json=json_body, headers=self._headers()
                )
            except httpx.HTTPError as e:  # timeout, connect error, protocol error
                last = f"network error ({type(e).__name__})"
                delay = BACKOFF_BASE_S * 2**attempt
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as e:
                        raise S2Error("malformed JSON response") from e
                last = f"HTTP {r.status_code}"
                if r.status_code != 429 and r.status_code < 500:
                    raise S2Error(last)  # 400/401/403/404: retrying cannot help
                retry_after = parse_retry_after(r.headers.get("retry-after"))
                delay = retry_after if retry_after is not None else BACKOFF_BASE_S * 2**attempt
            if attempt == self.max_retries:
                break
            if delay > self.max_backoff_s:
                last += f"; server asked to wait {delay:.0f}s, over the {self.max_backoff_s:.0f}s budget"
                break
            self._sleep(delay)
        if last.startswith("HTTP 429"):
            raise S2RateLimited(last)
        raise S2Error(last)

    def fetch(self, query: str, limit: int) -> dict:
        """Raw paper/search payload for (query, limit), from the cache when present.
        Returns the cache-entry shape: {"response": ..., "fetched_at": ..., ...}."""
        params = search_params(query, limit)
        if not params["query"]:
            return {"response": {"data": []}, "fetched_at": None, "params": params}
        if self.cache is not None and (hit := self.cache.get(SEARCH_PATH, params)) is not None:
            return hit
        payload = self._request("GET", SEARCH_PATH, params)
        if not isinstance(payload, dict) or not isinstance(payload.get("data", []), list):
            raise S2Error("unexpected response shape")
        payload.setdefault("data", [])
        if self.cache is not None:
            return self.cache.put(SEARCH_PATH, params, payload)
        return {"response": payload, "fetched_at": None, "params": params}

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        limit = max(1, min(int(top_k), S2_MAX_LIMIT))
        return parse_search(self.fetch(query, limit)["response"], limit)

    def batch(self, ids: list[str], fields: str) -> list:
        """POST paper/batch (<= 500 ids per call, one request). Cached like search.
        Unknown ids come back as null entries, in input order."""
        if len(ids) > S2_BATCH_MAX_IDS:
            raise ValueError(f"paper/batch takes at most {S2_BATCH_MAX_IDS} ids")
        params = {"fields": fields, "ids": list(ids)}
        if self.cache is not None and (hit := self.cache.get(BATCH_PATH, params)) is not None:
            return hit["response"]
        payload = self._request("POST", BATCH_PATH, {"fields": fields}, {"ids": list(ids)})
        if not isinstance(payload, list):
            raise S2Error("unexpected response shape")
        if self.cache is not None:
            self.cache.put(BATCH_PATH, params, payload)
        return payload
