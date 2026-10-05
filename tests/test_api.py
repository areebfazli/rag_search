"""Rate-limit keying tests for _client_ip — the security-sensitive bit of the API.

The bucket key decides who shares a rate limit, so every case here is really asking:
can a client choose its own key? If it can, the limit is decorative.
"""
from starlette.requests import Request

from app.api.main import _client_ip
from app.core.config import settings


def make_request(peer: str, xff: str | list[str] | None = None) -> Request:
    """`xff` as a list produces REPEATED header lines, not one comma-joined value."""
    values = [] if xff is None else ([xff] if isinstance(xff, str) else xff)
    headers = [(b"x-forwarded-for", v.encode()) for v in values]
    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": headers, "client": (peer, 1234)}
    )


def test_no_proxy_ignores_forwarded_header(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", False)
    # Directly exposed: a client-sent XFF must never influence the bucket key.
    assert _client_ip(make_request("198.51.100.9", xff="1.2.3.4")) == "198.51.100.9"


def test_trust_proxy_uses_rightmost_entry(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # Append-style proxies put the real peer LAST; leftmost entries are attacker-
    # controlled, so keying on them would grant a fresh bucket per request.
    req = make_request("10.0.0.1", xff="6.6.6.6, 7.7.7.7, 203.0.113.7")
    assert _client_ip(req) == "203.0.113.7"


def test_trust_proxy_joins_repeated_header_lines(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # Proxies that APPEND a new X-Forwarded-For line rather than extending the
    # existing one (Envoy, several ingresses) mean the client's own line arrives
    # first. Reading only the first line hands the attacker the key — and lets them
    # rotate it per request to evade the limit entirely.
    req = make_request("10.0.0.1", xff=["1.2.3.4", "203.0.113.7"])
    assert _client_ip(req) == "203.0.113.7"


def test_trust_proxy_multi_hop_skips_own_proxies(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 2)
    # Two proxies of our own (e.g. CDN -> load balancer): the rightmost entry is the
    # CDN, so keying on it would collapse every user behind that CDN into one bucket.
    req = make_request("10.0.0.1", xff="203.0.113.7, 172.16.0.5")
    assert _client_ip(req) == "203.0.113.7"


def test_trust_proxy_handles_entries_with_ports(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # Azure App Service and HAProxy's forwardfor append `client:port`. Rejecting those
    # would fall back to the peer — i.e. the proxy — collapsing EVERY client into one
    # bucket, so a single user could exhaust the limit for everyone.
    assert _client_ip(make_request("10.0.0.1", xff="198.51.100.5:51423")) == "198.51.100.5"
    assert _client_ip(make_request("10.0.0.1", xff="[2001:db8::5]:443")) == "2001:db8::5"
    # A bare IPv6 address has more than one colon and must NOT be truncated.
    assert _client_ip(make_request("10.0.0.1", xff="2001:db8::5")) == "2001:db8::5"


def test_trust_proxy_rejects_non_ip_value(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # Garbage would otherwise become a literal bucket key — a free fresh bucket.
    assert _client_ip(make_request("10.0.0.1", xff="not-an-ip")) == "10.0.0.1"
    assert _client_ip(make_request("10.0.0.1", xff=" , ")) == "10.0.0.1"


def test_trust_proxy_normalises_equivalent_addresses(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # Equivalent spellings of one address must land in ONE bucket, or a client can
    # mint new buckets by rewriting its own IPv6 form.
    a = _client_ip(make_request("10.0.0.1", xff="0:0:0:0:0:0:0:1"))
    b = _client_ip(make_request("10.0.0.1", xff="::1"))
    # Assert the VALUE, not just a == b: if IPv6 were rejected outright both calls would
    # fall back to the peer and compare equal, so the equality alone passes vacuously.
    assert a == b == "::1"


def test_trust_proxy_too_few_entries_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 2)
    # Fewer entries than configured hops means the chain isn't what we think it is;
    # trusting the only entry would trust a client-supplied value.
    assert _client_ip(make_request("10.0.0.1", xff="203.0.113.7")) == "10.0.0.1"


def test_trust_proxy_without_header_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    assert _client_ip(make_request("10.0.0.1")) == "10.0.0.1"


def test_answer_returns_502_when_the_llm_returns_no_completion(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import main
    from app.core.llm_endpoints import EmptyCompletionError
    from app.core.interfaces import SearchHit

    class _Service:
        def retrieve(self, q, mode, top_k):
            return [SearchHit("d1", 1.0, "text")]

    class _Generator:
        def generate(self, q, hits):
            raise EmptyCompletionError("LLM backend returned no completion (choices: null)")

    monkeypatch.setattr(main, "resolve_endpoint", lambda role: None)
    monkeypatch.setattr(main, "get_service", lambda: _Service())
    monkeypatch.setattr(main, "get_generator", lambda: _Generator())
    r = TestClient(main.app).get("/answer", params={"q": "does it?"})
    assert r.status_code == 502
    assert r.json()["detail"] == "LLM backend returned no completion"


def test_answer_serves_the_reask_verdict_through_the_real_generator(monkeypatch):
    # /answer -> LLMGenerator.generate: a claim truncated to nothing gets one verdict-only
    # re-ask; the response keeps its shape, with the verdict and its source filled in.
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from app.api import main
    from app.core.interfaces import SearchHit
    from app.generate.generator import REASK_NOTE, LLMGenerator

    class _Service:
        def retrieve(self, q, mode, top_k):
            return [SearchHit("d1", 1.0, "text", {"title": "T"})]

    replies = iter([("", "length"), ("", "length"), ("Verdict: REFUTED [1]", "stop")])
    sent = []

    def create(**kw):
        sent.append(kw)
        content, finish = next(replies)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish)], usage=None)

    monkeypatch.setattr(settings, "llm_reask", True)
    gen = LLMGenerator(model="m", base_url="http://localhost:1", api_key="k", reasoning_effort="")
    gen.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(main, "resolve_endpoint", lambda role: None)
    monkeypatch.setattr(main, "get_service", lambda: _Service())
    monkeypatch.setattr(main, "get_generator", lambda: gen)
    r = TestClient(main.app).get("/answer", params={"q": "Aspirin cures stroke."})
    assert r.status_code == 200 and len(sent) == 3
    body = r.json()
    assert (body["verdict"], body["verdict_source"]) == ("REFUTED", "reask")
    assert body["answer"] == REASK_NOTE and body["citations"] == []
    assert set(body) == {"query", "answer", "citations", "hits", "verdict", "verdict_source", "warnings"}


# --- web modes -------------------------------------------------------------------------


class _WebService:
    """Fake SearchService: a web hit with a URL, plus optional S2 failure/warning."""

    def __init__(self, error=None, warning=None, url="https://www.semanticscholar.org/paper/abc"):
        self.error, self.warning, self.url = error, warning, url
        self.calls: list[dict] = []

    def retrieve(self, q, mode, top_k, **kw):
        from app.core.interfaces import SearchHit

        self.calls.append({"mode": mode, **kw})
        if self.error:
            raise self.error
        if self.warning and kw.get("warnings") is not None:
            kw["warnings"].append(self.warning)
        return [
            SearchHit(
                "4983",
                0.5,
                "abstract",
                {"title": "T", "url": self.url, "year": 1998, "source": "semantic_scholar"},
            ),
            SearchHit("d1", 0.4, "local text", {"title": "L"}),
        ]


def _client(monkeypatch, service):
    from fastapi.testclient import TestClient

    from app.api import main

    main.limiter.reset()
    monkeypatch.setattr(main, "get_service", lambda: service)
    return TestClient(main.app)


def test_search_accepts_web_modes_and_returns_source_url_year(monkeypatch):
    from app.api import main

    svc = _WebService()
    client = _client(monkeypatch, svc)
    for mode in ("web", "hybrid_web"):
        r = client.get("/search", params={"q": "white matter", "mode": mode})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == mode and body["warnings"] == []
        web, local = body["hits"]
        assert web == {
            "doc_id": "4983", "score": 0.5, "title": "T", "text": "abstract",
            "url": "https://www.semanticscholar.org/paper/abc", "year": 1998,
            "source": "semantic_scholar",
        }
        assert local["source"] == "local" and local["url"] is None and local["year"] is None
    # Web modes get the retrieval lock passed down (held around local retrieval only).
    assert all(c["local_lock"] is main._retrieval_lock for c in svc.calls)


def test_search_still_rejects_unknown_modes(monkeypatch):
    r = _client(monkeypatch, _WebService()).get("/search", params={"q": "x", "mode": "internet"})
    assert r.status_code == 422


def test_web_mode_s2_failure_is_503_not_500(monkeypatch):
    from app.retrieve.semantic_scholar import S2RateLimited

    client = _client(monkeypatch, _WebService(error=S2RateLimited("HTTP 429")))
    r = client.get("/search", params={"q": "x", "mode": "web"})
    assert r.status_code == 503
    assert "Semantic Scholar" in r.json()["detail"]


def test_hybrid_web_degradation_warning_reaches_the_response(monkeypatch):
    client = _client(monkeypatch, _WebService(warning="Semantic Scholar unavailable (HTTP 429)"))
    r = client.get("/search", params={"q": "x", "mode": "hybrid_web"})
    assert r.status_code == 200 and r.json()["warnings"] == ["Semantic Scholar unavailable (HTTP 429)"]


def test_non_http_urls_never_reach_the_client(monkeypatch):
    client = _client(monkeypatch, _WebService(url="javascript:alert(1)"))
    r = client.get("/search", params={"q": "x", "mode": "web"})
    assert r.json()["hits"][0]["url"] is None


def test_answer_passes_web_hits_to_the_generator_unchanged(monkeypatch):
    from app.api import main
    from app.core.interfaces import Answer

    seen = {}

    class _Gen:
        def generate(self, q, hits):
            seen["hits"] = hits
            return Answer(text="It does [1].", citations=[hits[0].doc_id], hits=hits)

    svc = _WebService()
    client = _client(monkeypatch, svc)
    monkeypatch.setattr(main, "resolve_endpoint", lambda role: None)
    monkeypatch.setattr(main, "get_generator", lambda: _Gen())
    r = client.get("/answer", params={"q": "does it?", "mode": "hybrid_web"})
    assert r.status_code == 200, r.text
    assert [h.doc_id for h in seen["hits"]] == ["4983", "d1"]
    assert seen["hits"][0].metadata["source"] == "semantic_scholar"
    assert r.json()["citations"] == ["4983"] and r.json()["hits"][0]["source"] == "semantic_scholar"


# --- web-mode guards: stricter shared rate limit + non-blocking concurrency cap ---------


def _answer_gen(**extra):
    """Fake generator: an Answer with optional side-channel attributes (reask_error...)."""
    from app.core.interfaces import Answer

    class _Gen:
        def generate(self, q, hits):
            ans = Answer(text="It does [1].", citations=[hits[0].doc_id], hits=hits)
            for k, v in extra.items():
                setattr(ans, k, v)
            return ans

    return _Gen()


def _guarded_client(monkeypatch, service, **settings_overrides):
    from app.api import main

    client = _client(monkeypatch, service)
    monkeypatch.setattr(main, "_web_active", 0)
    monkeypatch.setattr(main, "resolve_endpoint", lambda role: None)
    monkeypatch.setattr(main, "get_generator", lambda: _answer_gen())
    for k, v in settings_overrides.items():
        monkeypatch.setattr(settings, k, v)
    return client


def test_web_modes_have_a_stricter_limit_shared_by_search_and_answer(monkeypatch):
    client = _guarded_client(monkeypatch, _WebService(), web_rate_limit="2/minute")
    assert client.get("/search", params={"q": "x", "mode": "web"}).status_code == 200
    assert client.get("/answer", params={"q": "x", "mode": "hybrid_web"}).status_code == 200
    # Third web request from this IP, on either endpoint: over the shared web bucket,
    # though far under /search's own 30/minute.
    for path in ("/search", "/answer"):
        r = client.get(path, params={"q": "x", "mode": "hybrid_web"})
        assert r.status_code == 429, r.text
        assert int(r.headers["Retry-After"]) >= 1
    # Local modes for the same IP are untouched by the exhausted web bucket.
    assert client.get("/search", params={"q": "x", "mode": "hybrid"}).status_code == 200
    assert client.get("/answer", params={"q": "x", "mode": "bm25"}).status_code == 200


def test_local_requests_never_spend_the_web_bucket(monkeypatch):
    client = _guarded_client(monkeypatch, _WebService(), web_rate_limit="1/minute")
    for _ in range(5):
        assert client.get("/search", params={"q": "x", "mode": "dense"}).status_code == 200
    assert client.get("/search", params={"q": "x", "mode": "web"}).status_code == 200
    assert client.get("/search", params={"q": "x", "mode": "web"}).status_code == 429


def test_local_search_limit_is_unchanged(monkeypatch):
    client = _guarded_client(monkeypatch, _WebService())
    codes = [client.get("/search", params={"q": "x"}).status_code for _ in range(31)]
    assert codes[:30] == [200] * 30 and codes[30] == 429


def test_web_concurrency_cap_is_503_with_retry_after_and_never_queues(monkeypatch):
    import threading

    from app.api import main

    entered, release = threading.Event(), threading.Event()

    class _Blocking(_WebService):
        def retrieve(self, q, mode, top_k, **kw):
            if mode in ("web", "hybrid_web"):
                entered.set()
                assert release.wait(10)
            return super().retrieve(q, mode, top_k, **kw)

    client = _guarded_client(monkeypatch, _Blocking(), web_max_concurrent=1)
    result = {}
    t = threading.Thread(
        target=lambda: result.setdefault(
            "r", client.get("/search", params={"q": "x", "mode": "web"})
        )
    )
    t.start()
    try:
        assert entered.wait(10)
        r = client.get("/search", params={"q": "x", "mode": "hybrid_web"})
        assert r.status_code == 503 and r.headers["Retry-After"] == "5"
        assert client.get("/answer", params={"q": "x", "mode": "web"}).status_code == 503
        # A local request is not held up by, and does not need, a web slot.
        assert client.get("/search", params={"q": "x", "mode": "hybrid"}).status_code == 200
    finally:
        release.set()
        t.join(10)
    assert result["r"].status_code == 200
    assert main._web_active == 0  # released in finally
    assert client.get("/search", params={"q": "x", "mode": "web"}).status_code == 200


def test_held_web_slots_block_web_only_and_free_up_on_release(monkeypatch):
    from app.api import main

    client = _guarded_client(monkeypatch, _WebService(), web_max_concurrent=2)
    assert main._try_acquire_web_slot() and main._try_acquire_web_slot()
    assert not main._try_acquire_web_slot()  # non-blocking: just says no
    try:
        assert client.get("/answer", params={"q": "x", "mode": "web"}).status_code == 503
        assert client.get("/answer", params={"q": "x", "mode": "hybrid"}).status_code == 200
    finally:
        main._release_web_slot()
    assert client.get("/answer", params={"q": "x", "mode": "web"}).status_code == 200
    main._release_web_slot()
    assert main._web_active == 0


def test_web_slot_is_released_when_retrieval_or_generation_fails(monkeypatch):
    from openai import APITimeoutError

    from app.api import main
    from app.retrieve.semantic_scholar import S2RateLimited

    client = _guarded_client(
        monkeypatch, _WebService(error=S2RateLimited("HTTP 429")), web_max_concurrent=1
    )
    assert client.get("/search", params={"q": "x", "mode": "web"}).status_code == 503
    assert main._web_active == 0

    class _Failing:
        def generate(self, q, hits):
            raise APITimeoutError(request=None)

    client = _guarded_client(monkeypatch, _WebService(), web_max_concurrent=1)
    monkeypatch.setattr(main, "get_generator", lambda: _Failing())
    assert client.get("/answer", params={"q": "x", "mode": "web"}).status_code == 502
    assert main._web_active == 0


def test_answer_generation_runs_outside_the_web_slot(monkeypatch):
    from app.api import main

    seen = {}

    class _Gen:
        def generate(self, q, hits):
            seen["active"] = main._web_active
            return _answer_gen().generate(q, hits)

    client = _guarded_client(monkeypatch, _WebService())
    monkeypatch.setattr(main, "get_generator", lambda: _Gen())
    assert client.get("/answer", params={"q": "x", "mode": "hybrid_web"}).status_code == 200
    assert seen["active"] == 0


# --- warnings: re-ask failures + hooks for the retrieval layer / answer ----------------


def test_failed_reask_surfaces_as_a_warning_without_raw_error_text(monkeypatch):
    from app.api import main

    client = _guarded_client(monkeypatch, _WebService())
    monkeypatch.setattr(main, "get_generator", lambda: _answer_gen(reask_error="APITimeoutError"))
    r = client.get("/answer", params={"q": "Aspirin cures stroke."})
    assert r.status_code == 200
    assert r.json()["warnings"] == [
        "Verdict check failed (APITimeoutError); showing the first answer without a verdict."
    ]
    # Anything that is not a bare class name (a raw message, a key fragment) is dropped.
    monkeypatch.setattr(
        main, "get_generator", lambda: _answer_gen(reask_error="Error: key sk-or-v1-abc rejected")
    )
    w = client.get("/answer", params={"q": "Aspirin cures stroke."}).json()["warnings"]
    assert w == ["Verdict check failed; showing the first answer without a verdict."]
    # No re-ask error -> no warning (backward compatible shape).
    monkeypatch.setattr(main, "get_generator", lambda: _answer_gen(reask_error=None))
    assert client.get("/answer", params={"q": "x"}).json()["warnings"] == []


class _HitsWithWarnings(list):
    warnings: list[str]


class _AnnotatingService(_WebService):
    """Returns hits carrying a `warnings` attribute (the retrieval-layer hook)."""

    def __init__(self, hit_warnings, **kw):
        super().__init__(**kw)
        self.hit_warnings = hit_warnings

    def retrieve(self, q, mode, top_k, **kw):
        hits = _HitsWithWarnings(super().retrieve(q, mode, top_k, **kw))
        hits.warnings = list(self.hit_warnings)
        return hits


def test_hit_warnings_hook_reaches_search_and_answer_deduplicated(monkeypatch):
    from app.api import main

    s2 = "Semantic Scholar unavailable (HTTP 429)"
    svc = _AnnotatingService([s2, "PubMed skipped", "PubMed skipped"], warning=s2)
    client = _guarded_client(monkeypatch, svc)
    for mode in ("hybrid_web", "hybrid"):  # local modes get the hook too
        r = client.get("/search", params={"q": "x", "mode": mode})
        expected = [s2, "PubMed skipped"]  # service list first, hook appended, no repeats
        assert r.status_code == 200 and r.json()["warnings"] == expected
    monkeypatch.setattr(
        main, "get_generator", lambda: _answer_gen(warnings=["PubMed skipped", "Answer note"])
    )
    r = client.get("/answer", params={"q": "x", "mode": "hybrid_web"})
    assert r.json()["warnings"] == [s2, "PubMed skipped", "Answer note"]
