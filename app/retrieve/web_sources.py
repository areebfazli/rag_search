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
the same on-disk ResponseCache (raw payload + fetch time, keyed by URL + params), and the
current web request's RequestBudget (semantic_scholar.request_budget: no attempt after the
deadline or once the request is abandoned; waits and timeouts capped at what is left).
It adds no credentials itself; a caller that needs one (OpenAlex's optional key)
supplies a host-checked httpx auth hook on its own client.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.retrieve.semantic_scholar import (
    BACKOFF_BASE_S,
    USER_AGENT,
    BudgetSpent,
    RateLimiter,
    ResponseCache,
    ResponseTooLarge,
    budget_timeout,
    budget_wait,
    current_budget,
    parse_retry_after,
    read_capped,
)

# An abstract page is ~10-500 KB; refuse anything absurd. Enforced while streaming.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


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
    every request and are part of the cache key. `wire_params` (e.g. NCBI's email /
    api_key) are sent with every request but are NOT part of the cache key and never
    written to a cache entry, so adding them cannot invalidate a cache or put a key on
    disk. `parse` is "json" or "text". Bodies are streamed and cut at `max_bytes`.
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
        wire_params: dict | None = None,
        max_bytes: int = MAX_RESPONSE_BYTES,
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
        self._wire_params = {k: v for k, v in (wire_params or {}).items() if v}
        self.max_bytes = max_bytes
        self._sleep = sleep
        self.requests_sent = 0

    def get(
        self, path: str, params: dict, parse: str = "json", validate: Callable[[object], bool] | None = None
    ) -> dict:
        """Cache-entry shape {"response": payload, "fetched_at": ..., "params": ...}.
        With `validate`, a fresh payload that fails it (returns False or raises
        SourceError) raises SourceError and is NOT cached, so a wrong-shaped 200 cannot
        poison every later read; a cached entry that fails it is treated as a miss."""
        full = {**self.default_params, **params}
        endpoint = self.base_url + path
        if self.cache is not None and (hit := self.cache.get(endpoint, full)) is not None:
            if validate is None or _valid(validate, hit.get("response")):
                return hit
        payload = self._request(endpoint, {**full, **self._wire_params}, parse)
        if validate is not None and not validate(payload):
            raise SourceError("unexpected response shape")
        if self.cache is not None:
            return self.cache.put(endpoint, full, payload)
        return {"response": payload, "fetched_at": None, "params": full}

    def _request(self, url: str, params: dict, parse: str) -> object:
        last = "no attempt made"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json" if parse == "json" else "*/*"}
        budget = current_budget()  # the web request's deadline / abandon flag, if any
        for attempt in range(self.max_retries + 1):
            if budget is not None and budget.expired():
                raise SourceError(budget.reason())
            rem = budget.remaining() if budget is not None else None
            if not self.limiter.acquire(timeout=budget_wait(self.max_wait_s, rem)):
                if rem is not None and (self.max_wait_s is None or rem < self.max_wait_s):
                    raise SourceError(f"{budget.reason()} (no request slot in time)")
                raise SourceRateLimited("local rate limit: no request slot within the wait budget")
            if budget is not None and budget.expired():
                raise SourceError(budget.reason())
            timeout = budget_timeout(self._client, budget.remaining() if budget is not None else None)
            self.requests_sent += 1
            body = b""
            try:
                with self._client.stream("GET", url, params=params, headers=headers, **timeout) as r:
                    status, rheaders, encoding = r.status_code, r.headers, r.encoding or "utf-8"
                    if status == 200:
                        body = read_capped(r, self.max_bytes)
            except ResponseTooLarge as e:
                raise SourceError("response too large") from e
            except BudgetSpent as e:
                raise SourceError(str(e)) from e
            except httpx.HTTPError as e:
                last = f"network error ({type(e).__name__})"
                delay = BACKOFF_BASE_S * 2**attempt
            else:
                if status == 200:
                    if parse == "text":
                        try:
                            return body.decode(encoding, errors="replace")
                        except LookupError:  # unknown charset label
                            return body.decode("utf-8", errors="replace")
                    try:
                        return json.loads(body)
                    except ValueError as e:
                        raise SourceError("malformed JSON response") from e
                last = f"HTTP {status}"
                if status != 429 and status < 500:
                    raise SourceError(last)
                retry_after = parse_retry_after(rheaders.get("retry-after"))
                delay = retry_after if retry_after is not None else BACKOFF_BASE_S * 2**attempt
            if attempt == self.max_retries:
                break
            if delay > self.max_backoff_s:
                last += f"; server asked to wait {delay:.0f}s, over the {self.max_backoff_s:.0f}s budget"
                break
            if budget is not None and (r_left := budget.remaining()) is not None and delay >= r_left:
                last += f"; {budget.reason()} before the retry"
                break
            self._sleep(delay)
        if last.startswith("HTTP 429"):
            raise SourceRateLimited(last)
        raise SourceError(last)


def _valid(validate: Callable[[object], bool], payload: object) -> bool:
    try:
        return bool(validate(payload))
    except SourceError:
        return False
