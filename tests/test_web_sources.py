"""Shared web-source plumbing (app/retrieve/web_sources.py). Offline, MockTransport fakes."""
import httpx
import pytest

from app.retrieve.semantic_scholar import ResponseCache
from app.retrieve.web_sources import (
    HttpFetcher,
    SourceError,
    SourceRateLimited,
    host_limiter,
    normalize_doi,
    normalize_pmid,
)


class NoWait:
    def acquire(self, timeout=None):
        return True


def fetcher(handler, **kw):
    sleeps: list[float] = []
    f = HttpFetcher(
        "https://example.org/api",
        limiter=NoWait(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        **kw,
    )
    return f, sleeps


def test_ids_are_normalised():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi:10.1/x") == "10.1/x"
    assert normalize_doi("not a doi") is None and normalize_doi("10.1/a b") is None and normalize_doi(None) is None
    assert normalize_pmid("123") == "123" and normalize_pmid(456) == "456"
    assert normalize_pmid("12a") is None and normalize_pmid(True) is None and normalize_pmid("") is None


def test_retry_after_honoured_then_success_and_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "7"})
        return httpx.Response(200, json={"ok": True})

    f, sleeps = fetcher(handler, cache=ResponseCache(tmp_path), default_params={"tool": "t"})
    entry = f.get("/x", {"q": "a"})
    assert entry["response"] == {"ok": True} and sleeps == [7.0] and f.requests_sent == 2
    assert calls[-1].url.host == "example.org" and calls[-1].url.params["tool"] == "t"
    assert f.get("/x", {"q": "a"})["response"] == {"ok": True} and len(calls) == 2  # cache hit


def test_client_errors_are_not_retried_and_429s_exhaust_to_rate_limited():
    f, _ = fetcher(lambda r: httpx.Response(400), max_retries=3)
    with pytest.raises(SourceError):
        f.get("/x", {})
    assert f.requests_sent == 1
    f, _ = fetcher(lambda r: httpx.Response(429), max_retries=2)
    with pytest.raises(SourceRateLimited):
        f.get("/x", {})
    assert f.requests_sent == 3
    f, _ = fetcher(lambda r: httpx.Response(429, headers={"retry-after": "999"}), max_backoff_s=30)
    with pytest.raises(SourceRateLimited):
        f.get("/x", {})
    assert f.requests_sent == 1  # the server asked for longer than the budget: no sleep


def test_invalid_payload_is_refused_and_never_cached(tmp_path):
    f, _ = fetcher(lambda r: httpx.Response(200, json={"unexpected": 1}), cache=ResponseCache(tmp_path))
    with pytest.raises(SourceError):
        f.get("/x", {"q": "a"}, validate=lambda p: "results" in p)
    assert not list(tmp_path.iterdir())
    f, _ = fetcher(lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(SourceError):
        f.get("/x", {})


def test_host_limiter_is_shared_per_name():
    assert host_limiter("test-host-a", 2.0) is host_limiter("test-host-a", 5.0)
    assert host_limiter("test-host-a", 2.0) is not host_limiter("test-host-b", 2.0)
