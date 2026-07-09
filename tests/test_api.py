"""Rate-limit keying tests for _client_ip — the security-sensitive bit of the API."""
from starlette.requests import Request

from app.api.main import _client_ip
from app.core.config import settings


def make_request(peer: str, xff: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": headers, "client": (peer, 1234)}
    )


def test_no_proxy_ignores_forwarded_header(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", False)
    # Directly exposed: a client-sent XFF must never influence the bucket key.
    assert _client_ip(make_request("198.51.100.9", xff="1.2.3.4")) == "198.51.100.9"


def test_trust_proxy_uses_rightmost_entry(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    # Append-style proxies put the real peer LAST; leftmost entries are attacker-
    # controlled, so keying on them would grant a fresh bucket per request.
    req = make_request("10.0.0.1", xff="6.6.6.6, 7.7.7.7, 203.0.113.7")
    assert _client_ip(req) == "203.0.113.7"


def test_trust_proxy_without_header_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    assert _client_ip(make_request("10.0.0.1")) == "10.0.0.1"
