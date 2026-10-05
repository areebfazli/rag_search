"""Per-request state, deadlines and lock scope of the shared web pipeline (offline fakes).

The API serves concurrent web requests from ONE WebSearch / S2IdResolver instance, so
nothing per-request may live on it, a failed base search must not wait for a slow extra
source, and no lock may be held across network I/O.
"""
import threading
import time

import numpy as np
import pytest

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.retrieve.s2_extra import S2IdResolver
from app.retrieve.semantic_scholar import S2Error, S2RateLimited
from app.retrieve.service import SearchService, api_dense_cap, web_extras
from app.retrieve.web_search import WebResult, WebSearch
from app.retrieve.web_sources import ExternalPaper, SourceError


class SlowS2:
    """One paper with an abstract and one title-only, after a short random-ish sleep."""

    cache = None

    def __init__(self, delay=0.002, fail=False):
        self.delay, self.fail = delay, fail

    def fetch(self, query, limit):
        time.sleep(self.delay)
        if self.fail:
            raise S2RateLimited("local rate limit: no request slot within the wait budget")
        return {"response": {"data": [
            {"corpusId": 1, "title": "has abstract", "abstract": "alpha text"},
            {"corpusId": 2, "title": "TITLE ONLY", "abstract": None},
        ]}, "fetched_at": "2026-10-01"}


class Resolver:
    def prefetch(self, items):
        return 0

    def to_hits(self, papers, keep_no_abstract=False):
        return [SearchHit(f"pmid:{p.pmid}", 1.0, p.abstract, {"title": p.title, "source": p.source})
                for p in papers if p.abstract or keep_no_abstract]


def test_concurrent_pooled_searches_never_leak_the_title_only_view():
    # Reviewer repro (C2): the pooled path used to flip self.keep_no_abstract to True
    # around the base fetch and restore it after; interleaved threads restored each
    # other's True, leaving it stuck and serving title-only hits to everyone.
    web = WebSearch(SlowS2(), multi_query=1)  # pooled; served view (no title-only)
    leaked = []

    def go():
        for _ in range(40):
            if any(not h.text for h in web.search("alpha claim", 10)):
                leaked.append(1)

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert leaked == []
    assert web.keep_no_abstract is False  # configuration is never mutated per request
    assert [h.doc_id for h in web.search("alpha claim", 10)] == ["1"]
    assert not hasattr(web, "last_errors") and not hasattr(web, "last_entries")


def test_source_errors_belong_to_the_request_that_hit_them():
    # Request A's PubMed fails slowly while request B (clean) runs: B must not report
    # A's error, and A must still see it.
    started = threading.Event()

    class PubMed:
        def search(self, q, limit):
            if q.startswith("A"):
                started.set()
                time.sleep(0.2)
                raise SourceError("HTTP 503")
            return [ExternalPaper("pubmed", 0, title="t", abstract="pm text", pmid="7")]

    web = WebSearch(SlowS2(), pubmed=PubMed(), resolver=Resolver(), pubmed_queries=("claim",))
    seen: dict[str, WebResult] = {}

    def a():
        seen["A"] = web.run("A claim", 10)

    def b():
        started.wait(2)
        seen["B"] = web.run("B claim", 10)

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    for t in (ta, tb):
        t.start()
    for t in (ta, tb):
        t.join()
    assert seen["A"].errors == ["pubmed: HTTP 503"]
    assert seen["B"].errors == []
    assert "pmid:7" in {h.doc_id for h in seen["B"].hits}
    assert all(e.get("fetched_at") for e in seen["A"].entries + seen["B"].entries)


class BlockingPubMed:
    """Blocks until released (or 5 s), like a slow upstream."""

    def __init__(self):
        self.release = threading.Event()
        self.calls = 0

    def search(self, q, limit):
        self.calls += 1
        self.release.wait(5)
        return [ExternalPaper("pubmed", 0, title="late", abstract="late text", pmid="9")]


def test_a_failing_base_search_does_not_wait_for_pubmed():
    pm = BlockingPubMed()
    web = WebSearch(SlowS2(delay=0, fail=True), pubmed=pm, resolver=Resolver())
    t0 = time.monotonic()
    with pytest.raises(S2Error):
        web.search("claim", 10)
    elapsed = time.monotonic() - t0
    pm.release.set()
    assert elapsed < 1.0  # was: the PubMed worker was joined first (its full latency)


def test_the_deadline_abandons_a_pending_extra_source_and_serves_the_base_pool():
    pm = BlockingPubMed()
    web = WebSearch(SlowS2(delay=0), pubmed=pm, resolver=Resolver(), deadline_s=0.2)
    t0 = time.monotonic()
    res = web.run("claim", 10)
    elapsed = time.monotonic() - t0
    pm.release.set()
    assert elapsed < 1.5
    assert [h.doc_id for h in res.hits] == ["1"]
    assert res.errors and res.errors[0].startswith("pubmed: abandoned")
    # No deadline (the eval): the worker is waited for and its papers are pooled.
    pm2 = BlockingPubMed()
    threading.Timer(0.3, pm2.release.set).start()
    res = WebSearch(SlowS2(delay=0), pubmed=pm2, resolver=Resolver()).run("claim", 10)
    assert {h.doc_id for h in res.hits} == {"1", "pmid:9"} and res.errors == []


def test_an_expired_deadline_skips_later_steps_with_a_reason(monkeypatch):
    from app.retrieve import s2_extra

    calls = []
    monkeypatch.setattr(s2_extra, "snippet_search", lambda *a, **k: calls.append(a) or [])
    web = WebSearch(SlowS2(delay=0.05), snippets=True, resolver=Resolver(), deadline_s=0.01)
    res = web.run("claim", 10)
    assert calls == [] and [h.doc_id for h in res.hits] == ["1"]
    assert res.errors == ["snippets: skipped, the 0.01s web deadline was reached"]


class FakeEmbedder:
    def __init__(self):
        self.documents = 0
        self.lock_seen: list[bool] = []
        self.lock = None

    def encode_query(self, text):
        self.lock_seen.append(self.lock.locked() if self.lock else False)
        return np.ones(4, dtype=np.float32) / 2

    def encode_documents(self, texts, show_progress=False):
        self.documents += len(texts)
        self.lock_seen.append(self.lock.locked() if self.lock else False)
        return np.ones((len(texts), 4), dtype=np.float32) / 2


class ManyS2:
    cache = None

    def fetch(self, query, limit):
        return {"response": {"data": [{"corpusId": i, "title": f"t{i}", "abstract": f"abs {i}"}
                                      for i in range(1, 81)]}}


def test_api_embedding_cap_and_rerank_under_the_lock(monkeypatch):
    monkeypatch.setattr(settings, "web_dense_cap", 50)
    monkeypatch.setattr(settings, "web_api_dense_cap", 30)
    assert api_dense_cap() == 30
    monkeypatch.setattr(settings, "web_api_dense_cap", 0)
    assert api_dense_cap() == 50
    monkeypatch.setattr(settings, "web_dense_cap", 0)
    assert api_dense_cap() is None
    monkeypatch.setattr(settings, "web_api_dense_cap", 30)
    assert api_dense_cap() == 30

    lock = threading.Lock()
    for pooled in (True, False):
        emb = FakeEmbedder()
        emb.lock = lock
        web = WebSearch(ManyS2(), rerank=True, embedder=emb, multi_query=int(pooled), dense_cap=30)
        hits = web.search("claim", 100, lock=lock)
        assert emb.documents == 30 and len(hits) == 80  # whole pool back, 30 embedded
        assert emb.lock_seen and all(emb.lock_seen)  # models only ever used under the lock


def test_web_extras_sets_the_api_bounds(monkeypatch):
    for name, value in [("web_deadline_s", 20.0), ("web_dense_cap", 50), ("web_api_dense_cap", 30),
                        ("web_pubmed", False), ("web_snippets", False), ("web_citation_seeds", 0)]:
        monkeypatch.setattr(settings, name, value)
    kw = web_extras(object())
    assert kw["deadline_s"] == 20.0 and kw["dense_cap"] == 30 and "snippet_filter" not in kw
    monkeypatch.setattr(settings, "web_deadline_s", 0.0)
    assert web_extras(object())["deadline_s"] is None


# --- SearchService: source errors reach the response warnings ------------------------


class Local:
    def search(self, query, top_k):
        return [SearchHit("l1", 1.0, "local", {"title": "L"})][:top_k]


def _service(web):
    svc = SearchService(dense=Local(), lexical=Local())
    svc._web = web
    return svc


@pytest.mark.parametrize("mode", ["web", "hybrid_web"])
def test_failed_extra_sources_become_response_warnings(mode):
    class FailingPubMed:
        def search(self, q, limit):
            raise SourceError("HTTP 503")

    web = WebSearch(SlowS2(delay=0), pubmed=FailingPubMed(), resolver=Resolver())
    svc = _service(web)
    warnings: list[str] = []
    hits = svc.retrieve("claim", mode=mode, top_k=5, warnings=warnings)
    assert "1" in {h.doc_id for h in hits}
    assert warnings == ["Web source skipped (pubmed: HTTP 503); results may be incomplete."]
    res = svc.retrieve_web("claim", mode=mode, top_k=5)
    assert res.errors == ["pubmed: HTTP 503"] and res.warnings == warnings and res.entries
    # A clean request on the same instance reports nothing.
    clean = _service(WebSearch(SlowS2(delay=0)))
    w2: list[str] = []
    clean.retrieve("claim", mode=mode, top_k=5, warnings=w2)
    assert w2 == []


def test_retrieve_web_reports_the_s2_fallback_and_rejects_local_modes():
    svc = _service(WebSearch(SlowS2(delay=0, fail=True)))
    res = svc.retrieve_web("claim", mode="hybrid_web", top_k=5)
    assert [h.doc_id for h in res.hits] == ["l1"]
    assert res.warnings and res.warnings[0].startswith("Semantic Scholar unavailable")
    with pytest.raises(S2Error):
        svc.retrieve_web("claim", mode="web")
    with pytest.raises(ValueError):
        svc.retrieve_web("claim", mode="hybrid")


# --- S2IdResolver: the lock is never held across network I/O --------------------------


class BarrierS2:
    """paper/batch fake whose requests must overlap: each waits at a 2-party barrier, so
    if the resolver serialised lookups the barrier would time out (BrokenBarrierError)."""

    cache = None
    requests_sent = 0

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=3)

    def _request(self, method, path, params, body):
        self.barrier.wait()
        return [{"corpusId": int(i.split(":")[1]), "abstract": "a"} for i in body["ids"]]


def test_concurrent_lookups_of_different_ids_overlap():
    r = S2IdResolver(BarrierS2())
    out: dict = {}
    errors: list = []

    def go(i):
        try:
            out.update(r.lookup([f"CorpusId:{i}"]))
        except Exception as e:  # noqa: BLE001 - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=go, args=(i,)) for i in (1, 2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert out["CorpusId:1"]["corpusId"] == 1 and out["CorpusId:2"]["corpusId"] == 2


# --- The deadline covers steps already running, and abandoned work stops (C6) ----------


class SlowResolver(Resolver):
    """prefetch blocks until released (or 5 s): a resolver batch stuck upstream."""

    def __init__(self):
        self.release = threading.Event()

    def prefetch(self, items):
        self.release.wait(5)
        return 0


def test_a_resolver_step_running_at_the_deadline_is_abandoned(monkeypatch):
    from app.retrieve import s2_extra

    monkeypatch.setattr(s2_extra, "snippet_search",
                        lambda *a, **k: [ExternalPaper("s2_snippet", 0, abstract="s", corpus_id="5")])
    r = SlowResolver()
    web = WebSearch(SlowS2(delay=0), snippets=True, snippet_queries=("claim",), resolver=r, deadline_s=0.3)
    t0 = time.monotonic()
    res = web.run("claim", 10)
    elapsed = time.monotonic() - t0
    r.release.set()
    assert elapsed < 1.5  # was: the full prefetch latency (it started before the deadline)
    assert [h.doc_id for h in res.hits] == ["1"]
    assert res.errors == ["resolve: abandoned, the 0.3s web deadline was reached"]


def test_without_a_deadline_steps_run_inline_and_are_waited_for(monkeypatch):
    from app.retrieve import s2_extra

    monkeypatch.setattr(s2_extra, "snippet_search",
                        lambda *a, **k: [ExternalPaper("s2_snippet", 0, abstract="s", pmid="5")])
    r = SlowResolver()
    threading.Timer(0.2, r.release.set).start()
    res = WebSearch(SlowS2(delay=0), snippets=True, snippet_queries=("claim",), resolver=r).run("claim", 10)
    assert {h.doc_id for h in res.hits} == {"1", "pmid:5"} and res.errors == []


def _pubmed_over(handler):
    import httpx

    from app.retrieve.pubmed import PubMedSource
    from app.retrieve.semantic_scholar import RateLimiter

    return PubMedSource(client=httpx.Client(transport=httpx.MockTransport(handler)), limiter=RateLimiter(1e6),
                        max_retries=0, email="", api_key="")


def test_an_abandoned_pubmed_worker_sends_no_more_requests():
    import httpx

    sent, done = [], threading.Event()

    def handler(req):
        sent.append(req.url.path)
        if "esearch" in req.url.path:
            time.sleep(0.4)  # outlives the 0.1 s deadline
            return httpx.Response(200, json={"esearchresult": {"idlist": ["123"]}})
        return httpx.Response(200, text="<PubmedArticleSet/>")

    pm = _pubmed_over(handler)
    real_search = pm.search

    def search(q, limit):
        try:
            return real_search(q, limit)
        finally:
            done.set()

    pm.search = search
    web = WebSearch(SlowS2(delay=0), pubmed=pm, resolver=Resolver(), pubmed_queries=("claim",), deadline_s=0.1)
    res = web.run("claim", 10)
    assert res.errors == ["pubmed: abandoned, the 0.1s web deadline was reached"]
    assert done.wait(3)
    time.sleep(0.05)
    # The worker's esearch was in flight; its efetch never started: the request's budget
    # was cancelled when it returned, and the next HTTP check refused.
    assert sent == ["/entrez/eutils/esearch.fcgi"]


def test_abandoned_work_never_grows_the_thread_count():
    from app.retrieve import web_search as WSmod

    pms = []
    before = threading.active_count()
    for _ in range(3 * WSmod.WEB_EXTRA_WORKERS):
        pm = BlockingPubMed()
        pms.append(pm)
        res = WebSearch(SlowS2(delay=0), pubmed=pm, resolver=Resolver(), deadline_s=0.01).run("claim", 10)
        assert res.errors and res.errors[0].startswith("pubmed: abandoned")
    grown = threading.active_count() - before
    for pm in pms:
        pm.release.set()
    assert grown <= WSmod.WEB_EXTRA_WORKERS  # bounded pool; was: one new thread per request
    assert sum(pm.calls for pm in pms) <= WSmod.WEB_EXTRA_WORKERS  # queued ones were cancelled


# --- RequestBudget in the HTTP layers ----------------------------------------------------


def test_an_expired_budget_refuses_new_http_attempts():
    import httpx

    from app.retrieve.semantic_scholar import RateLimiter, RequestBudget, SemanticScholarRetriever, request_budget
    from app.retrieve.web_sources import HttpFetcher

    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"data": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    s2 = SemanticScholarRetriever(api_key="", client=client, limiter=RateLimiter(1e6), max_retries=0)
    f = HttpFetcher("https://example.org", limiter=RateLimiter(1e6), client=client, max_retries=0)
    spent = RequestBudget(0.0)
    cancelled = RequestBudget(None)
    cancelled.cancel()
    for budget, reason in ((spent, "web deadline reached"), (cancelled, "request abandoned")):
        with request_budget(budget):
            with pytest.raises(S2Error, match=reason):
                s2.fetch("q", 10)
            with pytest.raises(SourceError, match=reason):
                f.get("/x", {})
    assert calls == [] and s2.requests_sent == 0 and f.requests_sent == 0
    # Outside any budget (the eval): unchanged.
    assert s2.fetch("q", 10)["response"] == {"data": []}


def test_waits_retries_and_timeouts_are_capped_by_the_budget():
    import httpx

    from app.retrieve.semantic_scholar import (
        RateLimiter,
        RequestBudget,
        SemanticScholarRetriever,
        budget_timeout,
        budget_wait,
        request_budget,
    )
    from app.retrieve.web_sources import HttpFetcher

    assert budget_wait(5.0, None) == 5.0 and budget_wait(5.0, 0.5) == 0.5 and budget_wait(None, 2.0) == 2.0
    client = httpx.Client(timeout=10.0)
    assert budget_timeout(client, None) == {}
    t = budget_timeout(client, 0.25)["timeout"]
    assert (t.connect, t.read, t.write, t.pool) == (0.25, 0.25, 0.25, 0.25)
    assert budget_timeout(httpx.Client(timeout=httpx.Timeout(0.1, read=3.0)), 1.0)["timeout"].connect == 0.1

    # A retry whose backoff would overrun the budget ends the call without sleeping.
    slept, seen_timeouts = [], []

    def handler(req):
        seen_timeouts.append(req.extensions.get("timeout"))
        return httpx.Response(503, headers={"retry-after": "5"})

    mock = httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)
    s2 = SemanticScholarRetriever(api_key="", client=mock, limiter=RateLimiter(1e6), max_retries=3,
                                  sleep=slept.append)
    f = HttpFetcher("https://example.org", limiter=RateLimiter(1e6), client=mock, max_retries=3,
                    sleep=slept.append)
    with request_budget(RequestBudget(2.0)):
        with pytest.raises(S2Error, match="web deadline reached before the retry"):
            s2.fetch("q", 10)
        with pytest.raises(SourceError, match="web deadline reached before the retry"):
            f.get("/x", {})
    assert slept == [] and s2.requests_sent == 1 and f.requests_sent == 1
    assert all(0 < t["read"] <= 2.0 and 0 < t["connect"] <= 2.0 for t in seen_timeouts)

    # A limiter slot further away than the budget is refused at once (not waited for).
    slow = RateLimiter(0.1)  # one slot per 10 s
    assert slow.acquire()
    s2b = SemanticScholarRetriever(api_key="", client=mock, limiter=slow, max_retries=0, max_wait_s=30)
    t0 = time.monotonic()
    with request_budget(RequestBudget(0.2)), pytest.raises(S2Error, match="no request slot in time"):
        s2b.fetch("q", 10)
    assert time.monotonic() - t0 < 0.5


def test_a_slow_drip_body_is_cut_when_the_budget_runs_out():
    import httpx

    from app.retrieve.semantic_scholar import RateLimiter, RequestBudget, request_budget
    from app.retrieve.web_sources import HttpFetcher

    def drip():
        for _ in range(100):
            time.sleep(0.02)
            yield b" "

    mock = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=drip())))
    f = HttpFetcher("https://example.org", limiter=RateLimiter(1e6), client=mock, max_retries=0)
    t0 = time.monotonic()
    with request_budget(RequestBudget(0.15)), pytest.raises(SourceError, match="web deadline reached"):
        f.get("/x", {}, parse="text")
    assert time.monotonic() - t0 < 1.0  # the full body would take 2 s
