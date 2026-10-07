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


# --- answer warnings: re-ask failures -------------------------------------------------


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


class _LocalService:
    def retrieve(self, q, mode, top_k):
        from app.core.interfaces import SearchHit

        return [SearchHit("d1", 0.4, "local text", {"title": "L"})]


def _client(monkeypatch, service):
    from fastapi.testclient import TestClient

    from app.api import main

    main.limiter.reset()
    monkeypatch.setattr(main, "get_service", lambda: service)
    monkeypatch.setattr(main, "resolve_endpoint", lambda role: None)
    monkeypatch.setattr(main, "get_generator", lambda: _answer_gen())
    return TestClient(main.app)


def test_search_returns_local_hits_and_rejects_unknown_modes(monkeypatch):
    client = _client(monkeypatch, _LocalService())
    r = client.get("/search", params={"q": "x", "mode": "hybrid"})
    assert r.status_code == 200
    assert r.json() == {"query": "x", "mode": "hybrid",
                        "hits": [{"doc_id": "d1", "score": 0.4, "title": "L", "text": "local text"}]}
    for mode in ("web", "hybrid_web", "bogus"):
        assert client.get("/search", params={"q": "x", "mode": mode}).status_code == 422
        assert client.get("/answer", params={"q": "x", "mode": mode}).status_code == 422


def test_failed_reask_surfaces_as_a_warning_without_raw_error_text(monkeypatch):
    from app.api import main

    client = _client(monkeypatch, _LocalService())
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
