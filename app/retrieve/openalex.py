"""OpenAlex works search as an extra open-web CANDIDATE SOURCE for claim -> paper retrieval.

``GET https://api.openalex.org/works?search=<query>&per_page=<n>&select=<fields>`` returns
works in relevance order; each becomes an ExternalPaper (DOI, PMID, title, abstract, year)
that app/retrieve/s2_extra.py later resolves to an S2 corpus id. Nothing here runs unless
the web pipeline asks for it, and no setting is read here (the lead wires settings later).

What the official docs say (help.openalex.org; docs.openalex.org 301-redirects there),
checked 2026-10-03:

* Access (/how-to-use-the-api/rate-limits-and-authentication, /api/authentication):
  "OpenAlex data is free, and so is casual use of the API — you can make basic queries
  with no key at all." A free key "raises your daily budget 10×"; it is sent "as an
  ``api_key`` query parameter" or "as a bearer token in the ``Authorization`` header".
  429 "Too Many Requests" on an exhausted daily budget or "more than 100 requests per
  second". So NO KEY IS REQUIRED for this module's volume. Budget headers on every
  response: X-RateLimit-Limit / -Remaining / -Credits-Used / -Reset ("seconds to midnight
  UTC").
* Cost (/access/example-costs): without a key "$0.10/day — a tenth of the above, enough
  to try the API"; with a free key "$1 of usage per day". Singleton gets are "Free",
  list+filter "$0.10" per 1,000 calls, search and semantic search "$1" per 1,000 calls
  ("A search costs 10 credits; with ``rerank=true`` it costs 20"). Observed live
  (keyless, 2026-10-03): ``x-ratelimit-limit: 1000``, ``x-ratelimit-credits-used: 10``
  per search — i.e. ~100 keyless searches per day, per IP. Evals over hundreds of claims
  need either the response cache or a key.
* mailto (/api/deprecations): "Before February 2026, OpenAlex used a 'polite pool'
  system ... ``mailto=you@example.com``"; now "The ``mailto`` parameter is ignored". This
  module never sends an email address.
* Search (/api/searching, /api-entities/works/search-works): works ``search`` covers
  "title, abstract, fulltext, and keywords" (live ``meta.x_query`` shows it runs as the
  ``fulltext.search`` filter); "OpenAlex uses stemming and removes stop words"; "Words not
  separated by boolean operators" are ANDed, with relevance ranking that favours words
  "close together". Special syntax: "AND, OR, NOT (uppercase)", parentheses, double
  quotes for phrases, wildcards ``*`` / ``?``, fuzzy/proximity ``~N``; "the whole request
  URL is limited to about 4 KB" (a 400 past it). Scoped variants: ``search.title_and_abstract``
  ("Title and abstract text only"), ``search.title_abstract_keywords``, ``search.exact``.
  Results carry ``relevance_score`` and are "sorted by it (descending) by default".
* Semantic search (/api/semantic-search): ``search.semantic`` embeds the input (GTE Large
  EN); "2,000 characters are used for matching", "50 per query" max results, "1 request
  per second"; "Only one search parameter is allowed per request". The docs pitch it for
  long inputs ("an abstract, a grant aim, or a paragraph"), so it is offered as
  ``mode="semantic"`` — same cost as keyword search, but at most 50 results.
* Paging/select (/api/paging, /api/selecting-fields): per_page "Default: 25", "100 is the
  supported maximum"; ``select`` takes "top-level fields but not nested properties".
* Work object (/api-entities/works/work-object): "OpenAlex does not ship plaintext
  abstracts for legal reasons; reconstruct the abstract from the index"
  (``abstract_inverted_index``: word -> positions). ``ids`` holds external ids "as URIs
  where possible" (pmid as ``https://pubmed.ncbi.nlm.nih.gov/<n>``); ``display_name`` is
  "Identical to ``title``".

Everything OpenAlex returns is UNTRUSTED text: flattened with clean_untrusted, bounded,
and URLs kept only via safe_http_url. The optional API key is sent only as an
``Authorization: Bearer`` header, only to https://api.openalex.org (an httpx auth hook
checks the host per request; redirects are not followed), and never as a query
parameter — so it is never part of a cache key, a cache file or an error message.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Generator
from urllib.parse import quote

import httpx

from app.retrieve.semantic_scholar import (
    MAX_ABSTRACT_CHARS,
    MAX_TITLE_CHARS,
    RateLimiter,
    ResponseCache,
    clean_untrusted,
    safe_http_url,
)
from app.retrieve.web_sources import (
    ExternalPaper,
    HttpFetcher,
    SourceError,
    host_limiter,
    normalize_doi,
    normalize_pmid,
)

# Fixed, not a setting: an OpenAlex key must only ever travel to OpenAlex.
OPENALEX_HOST = "api.openalex.org"
OPENALEX_BASE_URL = f"https://{OPENALEX_HOST}"
WORKS_PATH = "/works"
SOURCE = "openalex"
# Top-level fields only (select cannot name nested properties). title == display_name;
# both are selected so either one being absent still yields a title.
SELECT_FIELDS = "id,doi,title,display_name,publication_year,ids,abstract_inverted_index"
OPENALEX_MAX_PER_PAGE = 100  # "100 is the supported maximum"
SEMANTIC_MAX_RESULTS = 50  # search.semantic: "50 per query"

# mode -> the one search parameter allowed per request.
SEARCH_PARAMS = {
    "search": "search",  # title + abstract + fulltext + keywords (default)
    "title_and_abstract": "search.title_and_abstract",
    "semantic": "search.semantic",
}

# Defaults (module-level; not settings). 100 req/s is the hard ceiling and semantic search
# is 1 req/s, so 1 req/s holds for every mode and keeps a keyless day's ~100 searches
# from being burnt in a burst.
DEFAULT_RATE_PER_S = 1.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_BACKOFF_S = 30.0
DEFAULT_TIMEOUT_S = 20.0  # semantic search took ~4 s server-side on the smoke check

MAX_QUERY_CHARS = 1000  # claims are short; semantic matching uses only the first 2,000
MAX_QUERY_URL_BYTES = 2500  # percent-encoded query, well inside the ~4 KB URL limit

# reconstruct_abstract bounds: positions beyond MAX_ABSTRACT_POSITIONS are ignored, at
# most MAX_INDEX_ENTRIES (word, position) pairs are read, overlong "words" are dropped.
MAX_ABSTRACT_POSITIONS = 3000
MAX_INDEX_ENTRIES = 20000
MAX_WORD_CHARS = 100

_OPENALEX_WORK_ID = re.compile(r"https://openalex\.org/W\d{1,15}")
_PUBMED_PREFIXES = ("https://pubmed.ncbi.nlm.nih.gov/", "http://pubmed.ncbi.nlm.nih.gov/")
_BOOLEAN_OPS = re.compile(r"\b(AND|OR|NOT)\b")
_WHITESPACE_RUN = re.compile(r"\s+")


def openalex_query(query: object) -> str:
    """Neutralise OpenAlex search syntax in untrusted input, then bound its length.

    Keeps letters, digits and whitespace; a hyphen survives only inside a word
    ("COVID-19", "IL-6") and a period only inside a number ("2.5"). Everything else —
    double quotes (phrases), parentheses, wildcards ``*`` ``?``, ``~`` (fuzzy/proximity),
    ``:``, ``|``, ``,``, ``/``, ``\\``, ``+``, ``!``, apostrophes, control characters —
    becomes a space. Uppercase AND/OR/NOT are lower-cased so they are ordinary (stop)
    words, not operators. The result is cut at a word boundary to MAX_QUERY_CHARS and so
    its percent-encoding stays within MAX_QUERY_URL_BYTES. Non-strings become "".
    """
    if not isinstance(query, str):
        return ""
    chars = []
    n = len(query)
    for i, c in enumerate(query):
        if c.isalnum() or c.isspace():
            chars.append(c)
        elif c in "-." and 0 < i < n - 1:
            prev, nxt = query[i - 1], query[i + 1]
            ok = prev.isalnum() and nxt.isalnum() if c == "-" else prev.isdigit() and nxt.isdigit()
            chars.append(c if ok else " ")
        else:
            chars.append(" ")
    q = _WHITESPACE_RUN.sub(" ", "".join(chars)).strip()
    q = _BOOLEAN_OPS.sub(lambda m: m.group(1).lower(), q)
    if len(q) > MAX_QUERY_CHARS:
        head = q[:MAX_QUERY_CHARS]
        q = head.rsplit(" ", 1)[0] if " " in head else head
    while q and len(quote(q, safe="")) > MAX_QUERY_URL_BYTES:
        q = q.rsplit(" ", 1)[0] if " " in q else q[: len(q) // 2]
    return q


def reconstruct_abstract(inv: object, max_chars: int = MAX_ABSTRACT_CHARS) -> str:
    """Plain abstract from an ``abstract_inverted_index`` ({word: [positions]}).

    Words are placed at their positions and joined in position order; missing positions
    (gaps) are skipped, a position claimed twice keeps the first word read. Malformed
    input degrades instead of raising: non-dict -> "", non-string words, non-list
    position lists, non-int/bool/negative positions are ignored; positions past
    MAX_ABSTRACT_POSITIONS, pairs past MAX_INDEX_ENTRIES and words longer than
    MAX_WORD_CHARS are dropped, so a huge or hostile index costs bounded work. The text
    is then flattened and capped by clean_untrusted.
    """
    if not isinstance(inv, dict):
        return ""
    at: dict[int, str] = {}
    budget = MAX_INDEX_ENTRIES
    for word, positions in inv.items():
        if budget <= 0:
            break
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        if not word or len(word) > MAX_WORD_CHARS:
            budget -= 1
            continue
        for pos in positions[:budget]:
            budget -= 1
            if (
                isinstance(pos, int)
                and not isinstance(pos, bool)
                and 0 <= pos < MAX_ABSTRACT_POSITIONS
                and pos not in at
            ):
                at[pos] = word
    if not at:
        return ""
    out: list[str] = []
    total = 0
    for pos in sorted(at):
        out.append(at[pos])
        total += len(at[pos]) + 1
        if total > max_chars + MAX_WORD_CHARS:  # clean_untrusted cuts the rest
            break
    return clean_untrusted(" ".join(out), max_chars)


def _pmid_from(ids: object) -> str | None:
    raw = ids.get("pmid") if isinstance(ids, dict) else None
    if isinstance(raw, str):
        s = raw.strip()
        for prefix in _PUBMED_PREFIXES:
            if s.lower().startswith(prefix):
                s = s[len(prefix):]
                break
        return normalize_pmid(s.rstrip("/"))
    return normalize_pmid(raw)


def _work_url(doi: str | None, work_id: object) -> str:
    if doi:
        if url := safe_http_url("https://doi.org/" + quote(doi, safe="/:;()._-")):
            return url
    if isinstance(work_id, str) and _OPENALEX_WORK_ID.fullmatch(work_id):
        return safe_http_url(work_id)
    return ""


def parse_works(payload: object, limit: int) -> list[ExternalPaper]:
    """OpenAlex /works list payload -> ExternalPapers in OpenAlex's rank order.

    Raises SourceError when the payload is not ``{"results": [...]}``. Non-dict entries
    are skipped; a work id seen twice keeps its first (better-ranked) occurrence. `rank`
    is the 0-based position in OpenAlex's own list (gaps where entries were skipped).
    Works with neither DOI nor PMID are kept (title/abstract may still be useful); the
    resolver decides what it can map.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise SourceError("unexpected response shape")
    papers: list[ExternalPaper] = []
    seen: set[str] = set()
    for rank, work in enumerate(results):
        if len(papers) >= limit:
            break
        if not isinstance(work, dict):
            continue
        work_id = work.get("id")
        if isinstance(work_id, str):
            if work_id in seen:
                continue
            seen.add(work_id)
        doi = normalize_doi(work.get("doi"))
        if doi is None and isinstance(work.get("ids"), dict):
            doi = normalize_doi(work["ids"].get("doi"))
        title = clean_untrusted(work.get("display_name") or work.get("title"), MAX_TITLE_CHARS)
        year = work.get("publication_year")
        papers.append(
            ExternalPaper(
                source=SOURCE,
                rank=rank,
                title=title,
                abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
                year=year if isinstance(year, int) and not isinstance(year, bool) else None,
                url=_work_url(doi, work_id),
                pmid=_pmid_from(work.get("ids")),
                doi=doi,
            )
        )
    return papers


class _OpenAlexKeyAuth(httpx.Auth):
    """Adds ``Authorization: Bearer <key>`` only to https requests for api.openalex.org."""

    def __init__(self, key: str):
        self._key = key

    def __repr__(self) -> str:  # never show the key
        return "_OpenAlexKeyAuth(<redacted>)"

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if request.url.scheme == "https" and request.url.host == OPENALEX_HOST:
            request.headers["Authorization"] = f"Bearer {self._key}"
        yield request


def search_params(query: str, limit: int, mode: str = "search") -> dict:
    """The exact /works query parameters for (query, limit, mode) — also the cache key."""
    if mode not in SEARCH_PARAMS:
        raise ValueError(f"unknown OpenAlex search mode {mode!r}; one of {sorted(SEARCH_PARAMS)}")
    cap = SEMANTIC_MAX_RESULTS if mode == "semantic" else OPENALEX_MAX_PER_PAGE
    return {
        SEARCH_PARAMS[mode]: openalex_query(query),
        "per_page": max(1, min(int(limit), cap)),
        "select": SELECT_FIELDS,
    }


class OpenAlexSource:
    """OpenAlex works search -> ExternalPapers (see module docstring).

    One request per search (a single page: at most 100 results, 50 for semantic).
    `cache=None` disables caching. `max_wait_s` bounds the wait for a limiter slot (None =
    wait as long as needed); `max_backoff_s` bounds any retry sleep, Retry-After included
    (an exhausted daily budget asks for hours, so it ends the call with
    SourceRateLimited). `limiter` defaults to the process-wide ``host_limiter("openalex")``
    — the first caller's `rate` wins for the process. `api_key` is optional (keyless works,
    at a tenth of the daily budget) and is sent only as a bearer header to OpenAlex; when
    given with a `client`, that client's ``auth`` is set. `fetcher` replaces the whole
    HTTP layer (tests) and cannot be combined with `api_key`.
    """

    def __init__(
        self,
        *,
        fetcher: HttpFetcher | None = None,
        cache: ResponseCache | None = None,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        rate: float = DEFAULT_RATE_PER_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_wait_s: float | None = None,
        max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key: str | None = None,
        mode: str = "search",
        sleep: Callable[[float], None] = time.sleep,
    ):
        if mode not in SEARCH_PARAMS:
            raise ValueError(f"unknown OpenAlex search mode {mode!r}; one of {sorted(SEARCH_PARAMS)}")
        self.mode = mode
        key = (api_key or "").strip()
        if fetcher is not None:
            if key:
                raise ValueError("pass api_key or fetcher, not both (the fetcher owns the client)")
            self.fetcher = fetcher
            self._authenticated = False
            return
        if key:
            client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
            client.auth = _OpenAlexKeyAuth(key)
        self._authenticated = bool(key)
        self.fetcher = HttpFetcher(
            OPENALEX_BASE_URL,
            limiter=limiter or host_limiter(SOURCE, rate),
            cache=cache,
            client=client,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_wait_s=max_wait_s,
            max_backoff_s=max_backoff_s,
            sleep=sleep,
        )

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    @property
    def requests_sent(self) -> int:
        """Network attempts, retries included (cache hits are not requests)."""
        return self.fetcher.requests_sent

    def fetch(self, query: str, limit: int = OPENALEX_MAX_PER_PAGE, mode: str | None = None) -> dict:
        """Raw /works payload in the cache-entry shape {"response": ..., "fetched_at": ...}.
        An empty query (after sanitising) makes no request."""
        params = search_params(query, limit, mode or self.mode)
        if not next(iter(params.values())):
            return {"response": {"results": []}, "fetched_at": None, "params": params}
        return self.fetcher.get(
            WORKS_PATH, params, validate=lambda p: isinstance(p, dict) and isinstance(p.get("results"), list)
        )

    def search(self, query: str, limit: int = 100, mode: str | None = None) -> list[ExternalPaper]:
        """Up to `limit` works for `query`, best first (capped at one page: 100, or 50
        for semantic). Raises SourceError / SourceRateLimited on failure."""
        params = search_params(query, limit, mode or self.mode)
        return parse_works(self.fetch(query, limit, mode)["response"], params["per_page"])
