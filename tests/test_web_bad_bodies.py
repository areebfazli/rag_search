"""Error / invalid bodies served with HTTP 200 are never cached and surface as errors, and
the NCBI api_key never reaches a log record or an error message (offline: MockTransport)."""
import json
import logging

import httpx
import pytest

from app.retrieve import pubmed as P
from app.retrieve.pubmed import PubMedSource
from app.retrieve.s2_extra import S2IdResolver, _batch, expand_citations, snippet_search
from app.retrieve.semantic_scholar import RateLimiter, ResponseCache, S2Error, SemanticScholarRetriever
from app.retrieve.web_search import WebSearch
from app.retrieve.web_sources import SourceError

EFETCH_OK = (
    "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>"
    "<ArticleTitle>T</ArticleTitle><Abstract><AbstractText>A</AbstractText></Abstract>"
    "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
)


def _files(d):
    return sorted(p for p in d.rglob("*") if p.is_file())


def _s2(handler, cache=None):
    return SemanticScholarRetriever(api_key="", client=httpx.Client(transport=httpx.MockTransport(handler)),
                                    limiter=RateLimiter(1e6), cache=cache, max_retries=0)


# --- Semantic Scholar -----------------------------------------------------------------

S2_ERROR_BODIES = [
    {"error": "Internal error", "message": "x"},
    {"message": "Internal Server Error"},
    {"code": "429", "message": "Too Many Requests"},
    {"data": "not-a-list"},
    {},  # neither results nor the no-results form ({"total": 0, ...})
]


@pytest.mark.parametrize("body", S2_ERROR_BODIES)
def test_s2_paper_search_error_body_is_an_error_and_not_cached(tmp_path, body):
    s2 = _s2(lambda req: httpx.Response(200, json=body), ResponseCache(tmp_path))
    with pytest.raises(S2Error) as e:
        s2.fetch("vitamin d", 10)
    assert "Internal" not in str(e.value)  # the body's own text is never echoed
    assert _files(tmp_path) == []
    with pytest.raises(S2Error):  # still not an empty success on the next call
        s2.search("vitamin d", 10)


def test_s2_paper_search_no_results_form_is_still_a_valid_empty_page(tmp_path):
    s2 = _s2(lambda req: httpx.Response(200, json={"total": 0, "offset": 0}), ResponseCache(tmp_path))
    assert s2.search("vitamin d", 10) == []
    assert len(_files(tmp_path)) == 1


def test_a_cached_s2_error_body_from_older_code_is_a_miss(tmp_path):
    cache = ResponseCache(tmp_path)
    from app.retrieve.semantic_scholar import SEARCH_PATH, search_params

    cache.put(SEARCH_PATH, search_params("vitamin d", 10), {"error": "old poison"})
    page = {"data": [{"corpusId": 7, "title": "t", "abstract": "a"}], "total": 1}
    s2 = _s2(lambda req: httpx.Response(200, json=page), cache)
    assert [h.doc_id for h in s2.search("vitamin d", 10)] == ["7"]
    assert s2.requests_sent == 1


@pytest.mark.parametrize("body", [{"error": "Internal error"}, {"message": "x"}, {"retrievalVersion": "1"}])
def test_s2_snippet_error_body_is_an_error_and_not_cached(tmp_path, body):
    s2 = _s2(lambda req: httpx.Response(200, json=body))
    with pytest.raises(S2Error):
        snippet_search(s2, "some claim words here", 100, ResponseCache(tmp_path))
    assert _files(tmp_path) == []


@pytest.mark.parametrize("body", [{"error": "Internal error"}, {"data": None}])
def test_s2_batch_and_citation_error_bodies_are_errors_and_not_cached(tmp_path, body):
    s2 = _s2(lambda req: httpx.Response(200, json=body))
    cache = ResponseCache(tmp_path)
    with pytest.raises(S2Error):
        _batch(s2, ["CorpusId:1"], "title", cache)
    with pytest.raises(S2Error):
        s2.batch(["CorpusId:1"], "title")
    with pytest.raises(S2Error):
        expand_citations(s2, ["1"], cache=cache)
    assert _files(tmp_path) == []
    resolver = S2IdResolver(s2, cache_dir=tmp_path / "ids")
    with pytest.raises(S2Error):
        resolver.lookup(["CorpusId:1"])
    assert resolver._mem == {} and not (tmp_path / "ids").exists()


def test_a_bad_snippet_body_becomes_a_source_warning_not_an_empty_success():
    def handler(req):
        if req.url.path.endswith("/snippet/search"):
            return httpx.Response(200, json={"error": "Internal error"})
        return httpx.Response(200, json={"data": [{"corpusId": 1, "title": "t", "abstract": "a"}]})

    s2 = _s2(handler)
    res = WebSearch(s2, snippets=True, snippet_queries=("claim",), resolver=S2IdResolver(s2)).run("claim", 10)
    assert [h.doc_id for h in res.hits] == ["1"]
    assert res.errors == ["snippets: error object in a 200 response"]
    with pytest.raises(S2Error):  # the eval (strict) must not score a thinner pool
        WebSearch(s2, snippets=True, snippet_queries=("claim",), resolver=S2IdResolver(s2), strict=True).run("claim", 10)


# --- PubMed -----------------------------------------------------------------------------

ESEARCH_ERROR_BODIES = [
    {"esearchresult": {"ERROR": "Search Backend failed"}},
    {"esearchresult": {"idlist": "notalist"}},
    {"error": "API rate limit exceeded", "count": "11"},
    {"header": {}},
]


def _pm(handler, cache=None, **kw):
    return PubMedSource(client=httpx.Client(transport=httpx.MockTransport(handler)), limiter=RateLimiter(1e6),
                        cache=cache, max_retries=0, **kw)


@pytest.mark.parametrize("body", ESEARCH_ERROR_BODIES)
def test_pubmed_esearch_error_body_is_an_error_and_not_cached(tmp_path, body):
    src = _pm(lambda req: httpx.Response(200, json=body), ResponseCache(tmp_path), email="", api_key="")
    for _ in range(2):  # each call goes to the network again: nothing was cached
        with pytest.raises(SourceError):
            src.search("vitamin b12", 10)
    assert src.requests_sent == 2
    assert _files(tmp_path) == []


@pytest.mark.parametrize("text", [
    "<eFetchResult><ERROR>temporary backend failure</ERROR></eFetchResult>",
    "<html>busy</html>",
    "not xml",
])
def test_pubmed_efetch_error_body_is_an_error_and_not_cached(tmp_path, text):
    def handler(req):
        if "esearch" in req.url.path:
            return httpx.Response(200, json={"esearchresult": {"idlist": ["123"]}})
        return httpx.Response(200, text=text)

    src = _pm(handler, ResponseCache(tmp_path), email="", api_key="")
    with pytest.raises(SourceError):
        src.search("vitamin b12", 10)
    cached = [json.loads(p.read_text())["response"] for p in _files(tmp_path)]
    assert cached == [{"esearchresult": {"idlist": ["123"]}}]  # only the valid esearch


def test_a_bad_pubmed_body_becomes_a_source_warning():
    class S2:
        cache = None

        def fetch(self, q, limit):
            return {"response": {"data": [{"corpusId": 1, "title": "t", "abstract": "a"}]}}

    class Resolver:
        def prefetch(self, items):
            return 0

        def to_hits(self, papers, keep_no_abstract=False):
            return []

    pm = _pm(lambda req: httpx.Response(200, json={"esearchresult": {"ERROR": "x"}}), email="", api_key="")
    res = WebSearch(S2(), pubmed=pm, resolver=Resolver(), pubmed_queries=("claim",)).run("claim", 10)
    assert [h.doc_id for h in res.hits] == ["1"] and res.errors == ["pubmed: esearch reported an error"]


# --- NCBI api_key never in logs or errors ---------------------------------------------

KEY = "FAKEKEY_c0ffee1234567890"


def _all_text(records):
    out = []
    for r in records:
        out += [r.getMessage(), str(r.msg), repr(r.args)]
    return "\n".join(out)


def test_the_ncbi_key_never_reaches_a_log_record(caplog, monkeypatch):
    seen = []
    for name in P.HTTP_LOGGERS:  # another test's logging config may have disabled them
        monkeypatch.setattr(logging.getLogger(name), "disabled", False)

    def handler(req):
        seen.append(str(req.url))
        if "esearch" in req.url.path:
            return httpx.Response(200, json={"esearchresult": {"idlist": ["123"]}})
        return httpx.Response(200, text=EFETCH_OK)

    for name in P.HTTP_LOGGERS:
        caplog.set_level(logging.DEBUG, logger=name)
    caplog.set_level(logging.DEBUG)
    src = _pm(handler, email="dev@example.org", api_key=KEY, rate=8.0)
    assert [p.pmid for p in src.search("vitamin b12", 10)] == ["123"]
    assert all(f"api_key={KEY}" in u for u in seen)  # the key IS sent (to NCBI only)
    httpx_records = [r for r in caplog.records if r.name.startswith(("httpx", "httpcore"))]
    assert httpx_records  # httpx did log the requests at DEBUG/INFO...
    assert KEY not in _all_text(caplog.records)  # ...but never the key
    assert any("api_key=[REDACTED]" in r.getMessage() for r in httpx_records)
    # Idempotent: a second source does not stack filters.
    _pm(handler, email="", api_key=KEY, rate=8.0)
    assert all(logging.getLogger(n).filters.count(P._REDACTOR) == 1 for n in P.HTTP_LOGGERS)


def test_a_record_with_the_key_in_its_args_is_redacted_too(caplog):
    P.redact_key_in_http_logs(KEY)
    caplog.set_level(logging.DEBUG, logger="httpx")
    logging.getLogger("httpx").info("HTTP Request: %s %s", "GET", f"https://x/?a=1&api_key={KEY}&b=2")
    logging.getLogger("httpx").debug("raw %s", KEY)
    assert KEY not in _all_text(caplog.records)


def test_quiet_http_logs_pins_the_http_loggers_at_warning():
    before = {n: logging.getLogger(n).level for n in P.HTTP_LOGGERS}
    try:
        P.quiet_http_logs()
        assert all(logging.getLogger(n).level == logging.WARNING for n in P.HTTP_LOGGERS)
    finally:
        for n, lvl in before.items():
            logging.getLogger(n).setLevel(lvl)


@pytest.mark.parametrize("kind", ["status", "network"])
def test_errors_returned_to_callers_never_carry_the_key(kind):
    def handler(req):
        if kind == "network":
            raise httpx.ConnectError(f"cannot reach {req.url}", request=req)
        return httpx.Response(503, text=f"overloaded {req.url}")

    src = _pm(handler, email="", api_key=KEY, rate=8.0)
    with pytest.raises(SourceError) as e:
        src.search("vitamin b12", 10)
    chain, exc = [], e.value
    while exc is not None:
        chain.append(f"{exc!s} {exc!r}")
        exc = exc.__cause__ or exc.__context__
    assert KEY not in " ".join(chain)

    class S2:
        cache = None

        def fetch(self, q, limit):
            return {"response": {"data": []}}

    class Resolver:
        def prefetch(self, items):
            return 0

        def to_hits(self, papers, keep_no_abstract=False):
            return []

    res = WebSearch(S2(), pubmed=src, resolver=Resolver(), pubmed_queries=("claim",)).run("claim", 10)
    assert res.errors and KEY not in " ".join(res.errors)  # the API's warnings come from these

