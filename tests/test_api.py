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
    assert a == b


def test_trust_proxy_too_few_entries_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trusted_proxy_hops", 2)
    # Fewer entries than configured hops means the chain isn't what we think it is;
    # trusting the only entry would trust a client-supplied value.
    assert _client_ip(make_request("10.0.0.1", xff="203.0.113.7")) == "10.0.0.1"


def test_trust_proxy_without_header_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    assert _client_ip(make_request("10.0.0.1")) == "10.0.0.1"
