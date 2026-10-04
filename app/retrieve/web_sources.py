"""Shared plumbing for the extra open-web candidate sources (PubMed, OpenAlex, S2 snippet
search, S2 citation expansion) that feed the `web` pipeline's candidate pool.

Every source returns ExternalPaper records — whatever identifiers it knows (PMID, DOI,
S2 corpus id) plus untrusted title/abstract text, already flattened by
semantic_scholar.clean_untrusted. They are then resolved to S2 corpus ids (one S2
paper/batch call; app/retrieve/s2_extra.py), because S2 corpus ids are what SciFact's doc
ids are and what the local index, fusion and the gold qrels use.

HttpFetcher is the non-S2 counterpart of SemanticScholarRetriever._request: a host-pinned
GET client with a process-wide per-host RateLimiter (no bursts, retries included), bounded
retries on 429/5xx/transport errors honouring Retry-After (capped), a response-size cap,
and the same on-disk ResponseCache (raw payload + fetch time, keyed by URL + params).
It adds no credentials itself; a caller that needs one (OpenAlex's optional key)
supplies a host-checked httpx auth hook on its own client.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.retrieve.semantic_scholar import (
    BACKOFF_BASE_S,
    USER_AGENT,
    RateLimiter,
    ResponseCache,
    parse_retry_after,
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024  # an abstract page is ~10-500 KB; refuse anything absurd


class SourceError(RuntimeError):
    """A non-S2 source failed after retries. Messages carry a status/reason only."""


class SourceRateLimited(SourceError):
    """429s (or our own limiter) did not clear within the retry/wait budget."""


@dataclass
class ExternalPaper:
    """One candidate from an external source, before S2 id resolution.

    `rank` is the 0-based position in that source's own result list. Text fields are
    already cleaned (single line, bounded). Ids are bare: pmid "12345", doi lower-cased
    without a resolver prefix ("10.1000/xyz"), corpus_id "215416146".
    """

    source: str
    rank: int
    title: str = ""
    abstract: str = ""
    year: int | None = None
    url: str = ""
    pmid: str | None = None
    doi: str | None = None
    corpus_id: str | None = None


def normalize_doi(doi: object) -> str | None:
    """Bare lower-case DOI ("https://doi.org/10.1/X" -> "10.1/x"), or None if not a DOI."""
    if not isinstance(doi, str):
        return None
    d = doi.strip()
    low = d.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if low.startswith(prefix):
            d = d[len(prefix):]
            break
    d = d.strip().lower()
    if not d.startswith("10.") or "/" not in d or len(d) > 300 or any(c.isspace() for c in d):
        return None
    return d


def normalize_pmid(pmid: object) -> str | None:
    s = str(pmid).strip() if isinstance(pmid, (int, str)) and not isinstance(pmid, bool) else ""
    return s if s.isdigit() and 0 < len(s) <= 10 else None


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def host_limiter(name: str, rate_per_s: float) -> RateLimiter:
    """One RateLimiter per source name per process (shared by every instance), so a
    host's documented rate holds however many clients exist. The first caller's rate
    wins for the process lifetime."""
    with _LIMITERS_LOCK:
        if name not in _LIMITERS:
            _LIMITERS[name] = RateLimiter(rate_per_s)
        return _LIMITERS[name]


class HttpFetcher:
    """Host-pinned GET client: limiter slot per attempt, bounded retries, cache.

    `base_url` is fixed per source (never a setting). `default_params` are merged into
    every request and are part of the cache key. `parse` is "json" or "text".
    """

    def __init__(
        self,
        base_url: str,
        *,
        limiter: RateLimiter,
        cache: ResponseCache | None = None,
        client: httpx.Client | None = None,
        timeout_s: float = 10.0,
        max_retries: int = 3,
        max_wait_s: float | None = None,
        max_backoff_s: float = 30.0,
        default_params: dict | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.limiter = limiter
        self.cache = cache
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.max_backoff_s = max_backoff_s
        self.default_params = dict(default_params or {})
        self._sleep = sleep
        self.requests_sent = 0

    def get(
        self, path: str, params: dict, parse: str = "json", validate: Callable[[object], bool] | None = None
    ) -> dict:
        """Cache-entry shape {"response": payload, "fetched_at": ..., "params": ...}.
        With `validate`, a fresh payload that fails it raises SourceError and is NOT
        cached (so a wrong-shaped 200 cannot poison every later read)."""
        full = {**self.default_params, **params}
        endpoint = self.base_url + path
        if self.cache is not None and (hit := self.cache.get(endpoint, full)) is not None:
            return hit
        payload = self._request(endpoint, full, parse)
        if validate is not None and not validate(payload):
            raise SourceError("unexpected response shape")
        if self.cache is not None:
            return self.cache.put(endpoint, full, payload)
        return {"response": payload, "fetched_at": None, "params": full}

    def _request(self, url: str, params: dict, parse: str) -> object:
        last = "no attempt made"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json" if parse == "json" else "*/*"}
        for attempt in range(self.max_retries + 1):
            if not self.limiter.acquire(timeout=self.max_wait_s):
                raise SourceRateLimited("local rate limit: no request slot within the wait budget")
            self.requests_sent += 1
            try:
                r = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                last = f"network error ({type(e).__name__})"
                delay = BACKOFF_BASE_S * 2**attempt
            else:
                if r.status_code == 200:
                    if len(r.content) > MAX_RESPONSE_BYTES:
                        raise SourceError("response too large")
                    if parse == "text":
                        return r.text
                    try:
                        return r.json()
                    except ValueError as e:
                        raise SourceError("malformed JSON response") from e
                last = f"HTTP {r.status_code}"
                if r.status_code != 429 and r.status_code < 500:
                    raise SourceError(last)
                retry_after = parse_retry_after(r.headers.get("retry-after"))
                delay = retry_after if retry_after is not None else BACKOFF_BASE_S * 2**attempt
            if attempt == self.max_retries:
                break
            if delay > self.max_backoff_s:
                last += f"; server asked to wait {delay:.0f}s, over the {self.max_backoff_s:.0f}s budget"
                break
            self._sleep(delay)
        if last.startswith("HTTP 429"):
            raise SourceRateLimited(last)
        raise SourceError(last)
