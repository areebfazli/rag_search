"""Extra Semantic Scholar candidate sources for the `web` pipeline, plus S2 id resolution.

Three pieces, all of which send their requests through a SemanticScholarRetriever's
``_request`` (shared process-wide limiter, bounded retries, Retry-After honoured, the API
key only ever sent to the fixed S2 base URL) — there is no second S2 client here:

A. ``snippet_search`` — GET /graph/v1/snippet/search (full-text snippet search).
B. ``expand_citations`` — references / citations of seed papers (citation-graph neighbours).
C. ``S2IdResolver`` — ExternalPaper (PMID / DOI / corpus id) -> S2 paper via POST
   /paper/batch, with a per-id on-disk cache, and -> SearchHit for the local re-ranker.

Documented API facts (Academic Graph API OpenAPI spec, https://api.semanticscholar.org/
graph/v1/swagger.json, rendered at https://api.semanticscholar.org/api-docs/graph; read
2026-10-03):

* /snippet/search: "Return the text snippets that most closely match the query. Text
  snippets are excerpts of approximately 500 words, drawn from a paper's title, abstract,
  and body text, but excluding figure captions and the bibliography." ``query`` is
  required ("No special query syntax is supported"); ``limit`` defaults to 10 and "The
  max limit allowed is 1000". ``fields`` selects only fields under ``snippet`` (e.g.
  ``snippet.text,snippet.snippetKind``) — "Paper info and the score are currently always
  returned". Filters: ``paperIds`` (~100 ids), ``authors``, ``minCitationCount``,
  ``insertedBefore``, ``publicationDateOrYear``, ``year``, ``venue``, ``fieldsOfStudy``
  (comma-separated, e.g. ``Medicine,Biology``). Response: ``{"data": [{"score": float,
  "paper": {"corpusId": str, "title", "authors", "openAccessInfo"}, "snippet": {"text",
  "snippetKind": title|abstract|body, "section", ...}}], "retrievalVersion": str}``. No
  abstract/year/url comes back for the paper — the resolver (C) fills those. No
  endpoint-specific rate limit is documented (the key's 1 RPS applies to all endpoints).
* /paper/batch: ``fields`` is a query parameter, ``{"ids": [...]}`` the JSON body; nested
  ``references.<f>`` / ``citations.<f>`` fields are supported ("When requesting
  citations and references, the paperId and title subfields are returned by default. To
  request other subfields, use the format citations.title,citations.abstract"). Limits:
  "Can only process 500 paper ids at a time", "Can only return up to 10 MB of data at a
  time", "Can only return up to 9999 citations at a time". No per-paper cap on nested
  lists is documented — the whole list comes back (observed: 355 citations for one seed).
* /paper/{paper_id}/references and /citations: ``limit`` (default 100, "Must be <= 1000")
  and ``offset``; items are ``{"citedPaper": {...}}`` / ``{"citingPaper": {...}}``.
* Paper id prefixes (paper/{paper_id}, which batch defers to): ``CorpusId:<id>``,
  ``DOI:<doi>``, ``PMID:<id>``, ``PMCID:<id>``, ``ARXIV:``, ``MAG:``, ``ACL:``, ``URL:``,
  or a bare sha paperId. Unknown ids come back as ``null`` in batch, in input order.

Observed behaviour (live probe 2026-10-03) that shapes the code:

* Nested ``citations.corpusId`` on a very highly cited seed (CorpusId:13756489, >100k
  citations) made the batch fail with HTTP 504, so citations are only requested nested
  for seeds with citationCount <= NESTED_CITES_MAX; bigger seeds use the bounded
  per-seed GET /citations?limit=.
* ``references`` can be ``null`` when "elided by the publisher" (SciFact doc 5152028), and
  the same paper's ``abstract`` was null in batch too — hence the text fallbacks in C.
* Nested ``corpusId`` values are strings; top-level batch ``corpusId`` is an int (and a
  string in snippet results). All are normalised to digit strings here.
* Citations come back newest-first (highest corpus ids first), so a per-seed cap keeps the
  most recent citing papers.
* DOIs resolve case-insensitively (lower-cased DOI:10.18653/v1/n18-3011 resolved).

CONTAMINATION. Snippet search covers full text, and NLP papers about claim verification /
retrieval quote SciFact claims verbatim (probe: "A Case Study of Enhancing Sparse
Retrieval using LLMs" quotes "0-dimensional biomaterials show inductive properties." as a
query; "What Do Claim Verification Datasets Actually Test?" prints a claim followed by
"Ground Truth: YES"). Such a paper must never reach the re-ranker or the generator, so
``exclude_quoting`` (default on) drops every paper that has a BODY snippet containing the
whole normalised query, or that mentions "SciFact" in a snippet or its title. Body-only
on purpose: a gold abstract may legitimately share a sentence with a claim. Passing
``fields_of_study="Medicine,Biology"`` also removed the CS paper in the probe. A fuzzier
second guard, ``DatasetSnippetFilter`` (``dataset_filter=``; on in the API), also drops a
paper whose snippet carries dataset-dump formatting ("Claim: ...", "Evidence: ...",
"Ground Truth: ...") where the text after such a label overlaps a SciFact claim (or the
query) with token Jaccard >= 0.6 — a lightly reworded or re-cased claim printed as a
dataset example, which the verbatim check misses.

All S2 text is untrusted and goes through semantic_scholar.clean_untrusted.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.interfaces import SearchHit
from app.retrieve.semantic_scholar import (
    BATCH_PATH,
    MAX_ABSTRACT_CHARS,
    MAX_TITLE_CHARS,
    S2_BATCH_MAX_IDS,
    ResponseCache,
    S2Error,
    SemanticScholarRetriever,
    clean_untrusted,
    is_error_body,
    normalize_query,
    payload_error,
    safe_http_url,
)
from app.retrieve.web_sources import ExternalPaper, normalize_doi, normalize_pmid

SNIPPET_PATH = "/snippet/search"
SNIPPET_FIELDS = "snippet.text,snippet.snippetKind"
SNIPPET_MAX_LIMIT = 1000  # snippet/search: "The max limit allowed is 1000"
MAX_SNIPPET_CHARS = 1000
SOURCE_SNIPPET = "s2_snippet"
SOURCE_REFS = "s2_refs"
SOURCE_CITES = "s2_cites"

CITATIONS_PAGE_MAX = 1000  # paper/{id}/citations: "limit ... Must be <= 1000"
# Nested citations are requested in a batch only for seeds at or under this count: a
# >100k-citation seed made the nested batch 504 (probe), and the batch cap is "up to 9999
# citations at a time", so a chunk's summed citationCount stays under NESTED_CITES_BUDGET.
NESTED_CITES_MAX = 1000
NESTED_CITES_BUDGET = 9000
# Seeds per reference-list batch: references are not count-checked first, so keep the
# chunk small enough that ~50-90 refs/seed stays well inside the 9999 / 10 MB caps.
NESTED_REFS_MAX_IDS = 100

RESOLVE_FIELDS = "corpusId,title,abstract,url,year,externalIds"
_ID_CACHE_FILE = "s2_ids.sqlite3"
MAX_MEM_IDS = 50_000  # in-memory id cache cap (cleared when exceeded)
_UNRESOLVED = object()  # _batch_split marker: lookup failed (persistent 400), not "unknown"

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_QUOTE_MIN_TOKENS = 4  # a verbatim match of a 1-3 word query says nothing


def _digits(value: object) -> str | None:
    """A corpus id as a canonical digit string (int or digit str; bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    s = str(value).strip()
    return str(int(s)) if s.isdigit() and len(s) <= 20 else None


def _cached_request(
    s2: SemanticScholarRetriever,
    method: str,
    path: str,
    params: dict,
    body: dict | None,
    cache: ResponseCache | None,
    validate: Callable[[object], bool] | None = None,
) -> object:
    """s2._request through a ResponseCache keyed by endpoint + params (+ JSON body).
    With `validate`, a fresh payload that fails it raises S2Error BEFORE it is cached (a
    wrong-shaped 200 must not be replayed forever), and a cached one that fails it is
    treated as a miss."""
    key_params = {**params, **({"body": body} if body is not None else {})}
    if cache is not None and (hit := cache.get(path, key_params)) is not None:
        if validate is None or validate(hit["response"]):
            return hit["response"]
    payload = s2._request(method, path, params, body)
    if validate is not None and not validate(payload):
        raise S2Error(payload_error(payload))
    if cache is not None:
        cache.put(path, key_params, payload)
    return payload


def _data_list(payload: object) -> bool:
    """{"data": [...]} — the snippet search / citations page shape. A body without a
    "data" list (e.g. an {"error": ...} object served with HTTP 200) is refused, so it
    is never cached or read as "no results"."""
    return isinstance(payload, dict) and not is_error_body(payload) and isinstance(payload.get("data"), list)


def _batch(
    s2: SemanticScholarRetriever, ids: list[str], fields: str, cache: ResponseCache | None
) -> list:
    if len(ids) > S2_BATCH_MAX_IDS:
        raise ValueError(f"paper/batch takes at most {S2_BATCH_MAX_IDS} ids")
    return _cached_request(
        s2, "POST", BATCH_PATH, {"fields": fields}, {"ids": list(ids)}, cache,
        validate=lambda p: isinstance(p, list) and len(p) == len(ids),
    )


# --------------------------------------------------------------------------- A. snippets


@dataclass
class SnippetPaper(ExternalPaper):
    """An ExternalPaper found by snippet search. ``abstract`` stays "" (snippets are not
    abstracts; the resolver supplies the S2 abstract). ``snippet`` is the paper's best
    (first-ranked) snippet, cleaned and capped at MAX_SNIPPET_CHARS; S2IdResolver.to_hits
    uses it as the hit text only when neither S2 nor the source has an abstract."""

    snippet: str = ""
    snippet_kind: str = ""
    score: float | None = None


def _norm_tokens(text: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


def snippet_params(query: str, limit: int, fields_of_study: str | None = None) -> dict:
    """The exact snippet/search query parameters — also the cache key."""
    params = {
        "query": normalize_query(query),
        "limit": max(1, min(int(limit), SNIPPET_MAX_LIMIT)),
        "fields": SNIPPET_FIELDS,
    }
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    return params


_DATASET_LABEL = re.compile(
    r"\b(claims?|evidence|rationale|ground\s+truth|verdict|labels?|gold\s+label)\s*:", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)|\n")
DATASET_JACCARD = 0.6
_SEGMENT_MAX_TOKENS = 80


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_norm_tokens(text).split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class DatasetSnippetFilter:
    """Flags snippets that print a claim-verification dataset example: a dataset label
    ("Claim:", "Evidence:", "Ground Truth:", "Rationale:", "Label:", "Verdict:") followed
    by text whose token set has Jaccard >= `threshold` with a known claim (SciFact's, plus
    the query being searched). The text compared is what follows each label up to the
    next label or sentence end (<= 80 tokens), so a long snippet is not diluted.

    Both conditions are required: a paper that merely shares words with a claim (any
    gold abstract) has no dataset labels, and a methods section that says "evidence:"
    does not repeat a claim."""

    def __init__(self, claims: Iterable[str] = (), threshold: float = DATASET_JACCARD):
        self.threshold = threshold
        self._claims: list[frozenset[str]] = []
        self._index: dict[str, list[int]] = {}
        for c in claims:
            self._add(_token_set(c))

    def _add(self, toks: frozenset[str]) -> None:
        if not toks:
            return
        i = len(self._claims)
        self._claims.append(toks)
        for t in toks:
            self._index.setdefault(t, []).append(i)

    def __len__(self) -> int:
        return len(self._claims)

    @classmethod
    def from_scifact(cls, dataset: str | None = None) -> "DatasetSnippetFilter":
        """Every SciFact claim (beir/scifact queries: train + test). If the dataset cannot
        be loaded (e.g. an API image without ir_datasets data), the filter still checks
        each query against itself."""
        try:
            import ir_datasets

            from app.core.config import settings

            ds = ir_datasets.load(dataset or settings.corpus_dataset)
            return cls(q.text for q in ds.queries_iter())
        except Exception:  # noqa: BLE001 - best effort; degrade to query-only
            return cls()

    def _segments(self, text: str) -> list[frozenset[str]]:
        out = []
        labels = list(_DATASET_LABEL.finditer(text))
        for i, m in enumerate(labels):
            end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
            seg = text[m.end():end]
            if (stop := _SENTENCE_END.search(seg)) is not None:
                seg = seg[: stop.start()]
            toks = _norm_tokens(seg).split()[:_SEGMENT_MAX_TOKENS]
            if toks:
                out.append(frozenset(toks))
        return out

    def flags(self, text: object, query: str = "") -> bool:
        """True if `text` looks like a dataset example of a known claim (or `query`)."""
        if not isinstance(text, str) or not _DATASET_LABEL.search(text):
            return False
        q = _token_set(query)
        for seg in self._segments(text):
            if q and _jaccard(seg, q) >= self.threshold:
                return True
            cands = {i for t in seg for i in self._index.get(t, ())}
            if any(_jaccard(seg, self._claims[i]) >= self.threshold for i in cands):
                return True
        return False


def parse_snippets(
    payload: object,
    query: str = "",
    *,
    exclude_quoting: bool = True,
    dataset_filter: DatasetSnippetFilter | None = None,
) -> tuple[list[SnippetPaper], list[str]]:
    """snippet/search payload -> (papers deduped by corpusId in first-appearance order,
    corpus ids excluded as quoting the query / mentioning SciFact / (with
    `dataset_filter`) printing a dataset example of a claim)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return [], []
    q = _norm_tokens(query)
    check_quote = exclude_quoting and len(q.split()) >= _QUOTE_MIN_TOKENS
    entries: list[tuple[str, dict, dict, object]] = []
    excluded: list[str] = []  # first-seen order, for reporting
    excluded_set: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        snip = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
        cid = _digits(paper.get("corpusId"))
        if cid is None:
            continue
        entries.append((cid, paper, snip, item.get("score")))
        if exclude_quoting and cid not in excluded_set:
            raw_text = snip.get("text") if isinstance(snip.get("text"), str) else ""
            raw_title = paper.get("title") if isinstance(paper.get("title"), str) else ""
            norm = _norm_tokens(raw_text)
            if "scifact" in norm or "scifact" in _norm_tokens(raw_title) or (
                check_quote and snip.get("snippetKind") == "body" and f" {q} " in f" {norm} "
            ):
                excluded.append(cid)
                excluded_set.add(cid)
        if dataset_filter is not None and cid not in excluded_set:
            raw_text = snip.get("text") if isinstance(snip.get("text"), str) else ""
            if dataset_filter.flags(raw_text, query):
                excluded.append(cid)
                excluded_set.add(cid)
    papers: list[SnippetPaper] = []
    seen: set[str] = set()
    for cid, paper, snip, score in entries:
        if cid in seen or cid in excluded_set:
            continue
        seen.add(cid)
        kind = snip.get("snippetKind")
        papers.append(
            SnippetPaper(
                source=SOURCE_SNIPPET,
                rank=len(papers),
                title=clean_untrusted(paper.get("title"), MAX_TITLE_CHARS),
                corpus_id=cid,
                snippet=clean_untrusted(snip.get("text"), MAX_SNIPPET_CHARS),
                snippet_kind=kind if kind in ("title", "abstract", "body") else "",
                score=float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
            )
        )
    return papers, excluded


def snippet_search(
    s2: SemanticScholarRetriever,
    query: str,
    limit: int = 100,
    cache: ResponseCache | None = None,
    *,
    fields_of_study: str | None = None,
    exclude_quoting: bool = True,
    dataset_filter: DatasetSnippetFilter | None = None,
) -> list[SnippetPaper]:
    """S2 full-text snippet search -> SnippetPapers (one per paper, best snippet kept).

    One request (``limit`` capped at 1000; 1000 snippets was ~2.6 MB / ~5 s in the probe,
    100 is ~250 KB). The raw response is cached in ``cache`` (falling back to
    ``s2.cache``; both None = no caching) under endpoint + params, only once its shape
    checks out. See the module docstring for ``exclude_quoting``, ``dataset_filter`` and
    ``fields_of_study``.
    """
    params = snippet_params(query, limit, fields_of_study)
    if not params["query"]:
        return []
    cache = cache if cache is not None else s2.cache
    payload = _cached_request(s2, "GET", SNIPPET_PATH, params, None, cache, validate=_data_list)
    return parse_snippets(payload, query, exclude_quoting=exclude_quoting, dataset_filter=dataset_filter)[0]


# ------------------------------------------------------------------ B. citation graph


def _nested_ids(items: object, inner: str | None, cap: int) -> list[str]:
    """First `cap` valid corpus ids from a references/citations list (null-safe)."""
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if len(out) >= cap:
            break
        paper = item.get(inner) if inner and isinstance(item, dict) else item
        cid = _digits(paper.get("corpusId")) if isinstance(paper, dict) else None
        if cid is not None:
            out.append(cid)
    return out


def expand_citations(
    s2: SemanticScholarRetriever,
    seed_corpus_ids: Iterable[object],
    *,
    max_refs_per_seed: int = 50,
    max_cites_per_seed: int = 50,
    cache: ResponseCache | None = None,
) -> list[ExternalPaper]:
    """Papers each seed cites (source "s2_refs") and papers citing it ("s2_cites").

    Request pattern (cheapest bounded one, see module docstring): one paper/batch for all
    seeds with ``citationCount`` + nested ``references.corpusId`` (chunks of
    NESTED_REFS_MAX_IDS); then, if citations are wanted, one more batch with nested
    ``citations.corpusId`` for the seeds with citationCount <= NESTED_CITES_MAX (chunked
    so a chunk's citations stay <= NESTED_CITES_BUDGET), and one GET
    /paper/CorpusId:<id>/citations?limit=max_cites_per_seed per bigger seed. Typical
    cost: 2 requests per call. Requests are cached like snippet_search.

    Output: per-seed lists truncated to the caps (API order), then interleaved
    round-robin — i-th ref of seed 1, i-th cite of seed 1, i-th ref of seed 2, ... —
    deduped by corpus id (first wins), seeds themselves dropped, papers without a
    corpusId dropped. ``rank`` counts per source in that order. Only corpus_id (and
    nothing else) is set: S2IdResolver fills title/abstract.
    """
    seeds: list[str] = []
    for s in seed_corpus_ids:
        cid = _digits(s)
        if cid is not None and cid not in seeds:
            seeds.append(cid)
    max_refs = max(0, int(max_refs_per_seed))
    max_cites = max(0, int(max_cites_per_seed))
    if not seeds or (max_refs == 0 and max_cites == 0):
        return []
    cache = cache if cache is not None else s2.cache

    refs: dict[str, list[str]] = {}
    counts: dict[str, int | None] = {}
    fields = "corpusId,citationCount" + (",references.corpusId" if max_refs else "")
    for i in range(0, len(seeds), NESTED_REFS_MAX_IDS):
        chunk = seeds[i : i + NESTED_REFS_MAX_IDS]
        for cid, paper in zip(chunk, _batch(s2, [f"CorpusId:{c}" for c in chunk], fields, cache)):
            if not isinstance(paper, dict):
                continue  # unknown seed: nothing to expand
            refs[cid] = _nested_ids(paper.get("references"), None, max_refs)
            n = paper.get("citationCount")
            counts[cid] = n if isinstance(n, int) and not isinstance(n, bool) else None

    cites: dict[str, list[str]] = {}
    if max_cites:
        nested: list[list[str]] = []
        budget = 0
        for cid in seeds:
            if cid not in counts or counts[cid] == 0:
                continue  # unknown seed or no citations: no request needed
            n = counts[cid]
            if n is None or n > NESTED_CITES_MAX:
                page = _cached_request(
                    s2,
                    "GET",
                    f"/paper/CorpusId:{cid}/citations",
                    {"fields": "corpusId", "limit": min(max_cites, CITATIONS_PAGE_MAX), "offset": 0},
                    None,
                    cache,
                    validate=_data_list,
                )
                data = page["data"]
                cites[cid] = _nested_ids(data, "citingPaper", max_cites)
                continue
            if not nested or budget + n > NESTED_CITES_BUDGET or len(nested[-1]) >= S2_BATCH_MAX_IDS:
                nested.append([])
                budget = 0
            nested[-1].append(cid)
            budget += n
        for chunk in nested:
            batch = _batch(s2, [f"CorpusId:{c}" for c in chunk], "corpusId,citations.corpusId", cache)
            for cid, paper in zip(chunk, batch):
                if isinstance(paper, dict):
                    cites[cid] = _nested_ids(paper.get("citations"), None, max_cites)

    out: list[ExternalPaper] = []
    seen = set(seeds)
    rank = {SOURCE_REFS: 0, SOURCE_CITES: 0}
    for i in range(max(max_refs, max_cites)):
        for seed in seeds:
            for source, lists in ((SOURCE_REFS, refs), (SOURCE_CITES, cites)):
                lst = lists.get(seed, [])
                if i < len(lst) and lst[i] not in seen:
                    seen.add(lst[i])
                    out.append(ExternalPaper(source=source, rank=rank[source], corpus_id=lst[i]))
                    rank[source] += 1
    return out


# ---------------------------------------------------------------------- C. resolver


def id_candidates(paper: ExternalPaper) -> list[str]:
    """S2 id strings for a paper, best first: CorpusId, then PMID, then DOI."""
    out = []
    if (cid := _digits(paper.corpus_id)) is not None:
        out.append(f"CorpusId:{cid}")
    if (pmid := normalize_pmid(paper.pmid)) is not None:
        out.append(f"PMID:{pmid}")
    if (doi := normalize_doi(paper.doi)) is not None:
        out.append(f"DOI:{doi}")
    return out


def canonical_id(id_string: str) -> str | None:
    """Validate/canonicalise a "CorpusId:" / "PMID:" / "DOI:" id string (else None)."""
    if not isinstance(id_string, str) or ":" not in id_string:
        return None
    prefix, _, rest = id_string.partition(":")
    p = prefix.strip().lower()
    if p == "corpusid":
        cid = _digits(rest)
        return f"CorpusId:{cid}" if cid else None
    if p == "pmid":
        pmid = normalize_pmid(rest)
        return f"PMID:{pmid}" if pmid else None
    if p == "doi":
        doi = normalize_doi(rest)
        return f"DOI:{doi}" if doi else None
    return None


def _slim(paper: object) -> dict | None:
    """The fields we keep from a batch entry (raw, untrusted — cleaned at to_hits)."""
    if not isinstance(paper, dict) or (cid := _digits(paper.get("corpusId"))) is None:
        return None
    ext = paper.get("externalIds")
    return {
        "corpusId": int(cid),
        "title": paper.get("title") if isinstance(paper.get("title"), str) else None,
        "abstract": paper.get("abstract") if isinstance(paper.get("abstract"), str) else None,
        "url": paper.get("url") if isinstance(paper.get("url"), str) else None,
        "year": paper.get("year") if isinstance(paper.get("year"), int) and not isinstance(paper.get("year"), bool) else None,
        "externalIds": ext if isinstance(ext, dict) else None,
    }


class S2IdResolver:
    """ExternalPaper -> S2 paper (corpusId, title, abstract, url, year, externalIds).

    Lookups go through POST /paper/batch in chunks of S2_BATCH_MAX_IDS, and every id's
    result — including null ("S2 does not know this id") — is cached per id: in memory,
    and on disk in one sqlite file under ``cache_dir`` (None = memory only). Each chunk's
    results are written in one transaction (atomic); an unreadable/corrupt entry or file
    is a miss. So a bulk ``prefetch`` in the eval fills the cache and later per-claim
    calls send zero requests. A paper whose first id (CorpusId > PMID > DOI) resolves to
    null is retried with its next id. A persistent HTTP 400 on a minimal chunk leaves
    those ids unresolved (None, not cached); other batch failures raise S2Error (caller
    decides).
    """

    def __init__(self, s2: SemanticScholarRetriever, cache_dir: str | Path | None = None):
        self.s2 = s2
        self._db = Path(cache_dir) / _ID_CACHE_FILE if cache_dir is not None else None
        self._mem: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    @property
    def requests_sent(self) -> int:
        return self.s2.requests_sent

    @staticmethod
    def paper_key(paper: ExternalPaper) -> str | None:
        """The key ``resolve`` reports a paper under: its best id string."""
        ids = id_candidates(paper)
        return ids[0] if ids else None

    # -- per-id disk cache -------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        assert self._db is not None
        self._db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db, timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS ids (id TEXT PRIMARY KEY, value TEXT NOT NULL, fetched_at TEXT)")
        return conn

    def _disk_get(self, ids: list[str]) -> dict[str, dict | None]:
        if self._db is None or not ids or not self._db.exists():
            return {}
        found: dict[str, dict | None] = {}
        try:
            conn = self._connect()
            try:
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    rows = conn.execute(
                        f"SELECT id, value FROM ids WHERE id IN ({','.join('?' * len(chunk))})", chunk
                    ).fetchall()
                    for id_, value in rows:
                        try:
                            entry = json.loads(value)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(entry, dict) or "paper" not in entry:
                            continue
                        paper = entry["paper"]
                        if paper is None or (slim := _slim(paper)) is not None:
                            found[id_] = None if paper is None else slim
            finally:
                conn.close()
        except sqlite3.Error:
            return found  # corrupt/locked file: treat as misses
        return found

    def _disk_put(self, results: dict[str, dict | None]) -> None:
        if self._db is None or not results:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            conn = self._connect()
            try:
                with conn:  # one transaction
                    conn.executemany(
                        "INSERT OR REPLACE INTO ids (id, value, fetched_at) VALUES (?, ?, ?)",
                        [(k, json.dumps({"paper": v}), now) for k, v in results.items()],
                    )
            finally:
                conn.close()
        except sqlite3.Error:
            pass  # caching is best-effort; the memory cache still holds the results

    # -- lookups -----------------------------------------------------------------------

    def lookup(self, id_strings: Iterable[str]) -> dict[str, dict | None]:
        """Canonical id string -> slim S2 paper or None. Invalid ids are skipped."""
        ids: list[str] = []
        for s in id_strings:
            c = canonical_id(s)
            if c is not None and c not in ids:
                ids.append(c)
        # The lock guards the in-memory dict only — never held across disk or network
        # I/O, so concurrent requests resolving different ids do not serialise. Two
        # requests missing the same id may both fetch it; the results are identical.
        with self._lock:
            if len(self._mem) > MAX_MEM_IDS:  # long-running API process: stay bounded
                self._mem.clear()
            out = {i: self._mem[i] for i in ids if i in self._mem}
        missing = [i for i in ids if i not in out]
        disk = self._disk_get(missing)
        out.update(disk)
        with self._lock:
            self._mem.update(disk)
        missing = [i for i in missing if i not in disk]
        for start in range(0, len(missing), S2_BATCH_MAX_IDS):
            chunk = missing[start : start + S2_BATCH_MAX_IDS]
            batch = self._batch_split(chunk)
            # Cached: None (S2's documented "not found") and a valid slimmed paper. Anything
            # else is unresolved for this call only (a persistent HTTP 400, or an entry
            # like {"error": ...} with no corpusId): reported as None, never cached,
            # retried next time.
            got: dict[str, dict | None] = {}
            for i, p in zip(chunk, batch):
                slim = None if p is None or p is _UNRESOLVED else _slim(p)
                if p is None or slim is not None:
                    got[i] = slim
                else:
                    out[i] = None
            self._disk_put(got)
            with self._lock:
                self._mem.update(got)
            out.update(got)
        return out

    def _batch_split(self, chunk: list[str], depth: int = 0) -> list:
        """paper/batch for `chunk`; on HTTP 400 (seen live, intermittently, on a ~430-id
        mixed PMID/DOI batch that succeeded on retry and in halves) split in two and
        retry each half, at most 3 levels deep, so one bad request cannot drop a whole
        query's external candidates. A chunk that still gets 400 at the bottom (or a
        single id) comes back as _UNRESOLVED entries instead of raising: the rest of the
        lookup carries on. Other failures (429, 5xx, network, limiter) still raise."""
        try:
            return _batch(self.s2, chunk, RESOLVE_FIELDS, None)
        except S2Error as e:
            if "HTTP 400" not in str(e):
                raise
            if len(chunk) < 2 or depth >= 3:
                return [_UNRESOLVED] * len(chunk)
            mid = len(chunk) // 2
            return self._batch_split(chunk[:mid], depth + 1) + self._batch_split(chunk[mid:], depth + 1)

    def resolve(self, papers: Iterable[ExternalPaper]) -> dict[str, dict | None]:
        """paper_key(p) -> slim S2 paper or None, trying each paper's ids in turn
        (one batch round per id rank, only for still-unresolved papers)."""
        pending: dict[str, list[str]] = {}
        for p in papers:
            ids = id_candidates(p)
            if ids and ids[0] not in pending:
                pending[ids[0]] = ids
        result: dict[str, dict | None] = {k: None for k in pending}
        for depth in range(3):
            todo = {k: ids[depth] for k, ids in pending.items() if result[k] is None and depth < len(ids)}
            if not todo:
                break
            found = self.lookup(todo.values())
            for k, id_ in todo.items():
                result[k] = found.get(id_)
        return result

    def prefetch(self, items: Iterable[ExternalPaper | str]) -> int:
        """Fill the cache for papers and/or raw id strings; returns requests sent."""
        before = self.requests_sent
        papers, ids = [], []
        for item in items:
            (papers if isinstance(item, ExternalPaper) else ids).append(item)
        if ids:
            self.lookup(ids)
        if papers:
            self.resolve(papers)
        return self.requests_sent - before

    def to_hits(self, papers: Iterable[ExternalPaper], *, keep_no_abstract: bool = False) -> list[SearchHit]:
        """Resolved papers -> SearchHits, input order, deduped by doc_id.

        doc_id = str(corpusId) when resolved (also for an unresolved paper that carries a
        corpus id — that id is already in S2's namespace), else "pmid:<n>" / "doi:<doi>",
        which can never equal a numeric SciFact/S2 id. Papers with no id are dropped.
        Duplicates merge: the first occurrence fixes position and ``source``; text is
        the S2 abstract, else the first source abstract, else the first snippet
        (SnippetPaper), all via clean_untrusted; ``metadata["text_source"]`` says which.
        Papers with no text are dropped unless ``keep_no_abstract``. Scores are 1/rank.
        """
        papers = list(papers)
        resolved = self.resolve(papers)
        groups: dict[str, list[tuple[ExternalPaper, dict | None]]] = {}
        for p in papers:
            key = self.paper_key(p)
            if key is None:
                continue
            s2p = resolved.get(key)
            if s2p is not None:
                doc_id = str(s2p["corpusId"])
            elif (cid := _digits(p.corpus_id)) is not None:
                doc_id = cid
            elif (pmid := normalize_pmid(p.pmid)) is not None:
                doc_id = f"pmid:{pmid}"
            else:
                doc_id = f"doi:{normalize_doi(p.doi)}"
            groups.setdefault(doc_id, []).append((p, s2p))

        hits: list[SearchHit] = []
        for doc_id, members in groups.items():
            s2p = next((m for _, m in members if m is not None), None)
            first = members[0][0]
            text, text_source = "", ""
            for cand, kind in [((s2p or {}).get("abstract"), "s2_abstract")] + [
                (p.abstract, "source_abstract") for p, _ in members
            ] + [(getattr(p, "snippet", ""), "snippet") for p, _ in members]:
                text = clean_untrusted(cand, MAX_ABSTRACT_CHARS)
                if text:
                    text_source = kind
                    break
            if not text and not keep_no_abstract:
                continue
            title = clean_untrusted((s2p or {}).get("title"), MAX_TITLE_CHARS) or next(
                (t for p, _ in members if (t := clean_untrusted(p.title, MAX_TITLE_CHARS))), ""
            )
            url = safe_http_url((s2p or {}).get("url")) or next(
                (u for p, _ in members if (u := safe_http_url(p.url))), ""
            )
            year = (s2p or {}).get("year")
            if year is None:
                year = next((p.year for p, _ in members if isinstance(p.year, int) and not isinstance(p.year, bool)), None)
            sources = list(dict.fromkeys(p.source for p, _ in members))
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    score=1.0 / (len(hits) + 1),
                    text=text,
                    metadata={
                        "title": title,
                        "url": url,
                        "year": year,
                        "source": first.source,
                        "sources": sources,
                        "text_source": text_source,
                    },
                )
            )
        return hits
