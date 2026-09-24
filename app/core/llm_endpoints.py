"""Where each LLM request goes — provider -> (base URL, API key, model id) — and the
OpenRouter spend policy that every request to openrouter.ai must pass.

One resolver for both roles (the answer generator and the RAG-eval judge), so the API,
rag_eval and judge_agreement cannot disagree about which endpoint, key or model they use.

Precedence — the PROVIDER setting decides everything else:

* ``openrouter``: base URL is the constant ``OPENROUTER_BASE_URL`` (SSR_LLM_BASE_URL is
  never consulted), key is ``openrouter_api_key``, model is ``openrouter_llm_model``
  (generator) / ``openrouter_judge_model`` with ``:free`` appended if missing (judge).
* ``groq`` (any OpenAI-compatible endpoint): ``llm_base_url`` / ``llm_api_key`` /
  ``llm_model`` or ``judge_model``. An openrouter.ai base URL is refused here, so the
  Groq key is never sent to OpenRouter, and the OpenRouter key is only ever paired with
  the constant OpenRouter URL.

Spend policy (``enforce_spend_policy``), checked BEFORE any network call on every
request to an openrouter.ai URL — at resolution, at client construction, on every
``chat.completions.create`` through ``GuardedClient``, and in LLMGenerator's own path:

1. A ``:free`` model id is always allowed, and is sent with ``allow_fallbacks: false``
   and a $0/$0 ``max_price``, so only a zero-priced endpoint can serve it.
2. The judge must be ``:free`` — a paid judge id is refused outright.
3. A paid generator id is allowed only if it is in ``openrouter_paid_model_allowlist``
   (default: exactly ``openai/gpt-oss-120b``), and only if the request carries
   ``PAID_ROUTING``: a pinned provider ``order``, ``allow_fallbacks: false``,
   ``quantizations: ["bf16"]`` and a ``max_price`` no higher than
   ``PAID_MAX_PRICE_USD_PER_M``. With fallbacks off and an explicit order, OpenRouter
   fails the request rather than route to an endpoint outside the list, and max_price
   "will prevent your request from running if the price is not available".
4. Request fields that could add spend on their own (``models`` fallback lists,
   ``route``, ``plugins`` such as web search) are refused on any OpenRouter request.

Provider-routing field names, units and failure semantics are from
https://openrouter.ai/docs/features/provider-routing (``order``, ``allow_fallbacks``,
``quantizations``, ``max_price`` in USD per million tokens).

Secrets: API keys live only in ``LLMEndpoint.api_key``, which is excluded from repr and
from ``metadata()``; error messages name the env var, never its value.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from app.core.config import Settings, settings

Provider = Literal["groq", "openrouter"]
Role = Literal["generator", "judge"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FREE_SUFFIX = ":free"

# --- OpenRouter routing (docs: /docs/features/provider-routing) ------------------------
# allow_fallbacks (default true) lets OpenRouter try backup providers; false pins the
# request to the listed/top endpoint. max_price is USD per 1M tokens.
FREE_ROUTING: dict = {
    "allow_fallbacks": False,
    "max_price": {"prompt": 0, "completion": 0},
}
# Paid generator ceiling, USD per 1M tokens. The cheapest bf16 endpoint of
# openai/gpt-oss-120b (AkashML, $0.03 / $0.17 per M, per GET
# /api/v1/models/openai/gpt-oss-120b/endpoints on 2026-09-24) sits exactly at it.
PAID_MAX_PRICE_USD_PER_M: dict[str, float] = {"prompt": 0.03, "completion": 0.17}
# Endpoint tags (provider/quantization) tried in order, and nothing else. DeepInfra's
# bf16 endpoint is listed second for reproducibility, but at $0.037/M prompt it is ABOVE
# the prompt cap, so max_price filters it out: if AkashML is down the request fails,
# which is the intended behaviour (fail, never route elsewhere). Raising the prompt cap
# to 0.037 would make it a live second choice.
PAID_PROVIDER_ORDER: tuple[str, ...] = ("akashml/bf16", "deepinfra/bf16")
PAID_QUANTIZATIONS: tuple[str, ...] = ("bf16",)  # excludes CoreWeave's fp4 at the same price
PAID_ROUTING: dict = {
    "order": list(PAID_PROVIDER_ORDER),
    "allow_fallbacks": False,
    "quantizations": list(PAID_QUANTIZATIONS),
    "max_price": dict(PAID_MAX_PRICE_USD_PER_M),
}
# Body fields that can add cost independently of the model's price.
FORBIDDEN_OPENROUTER_FIELDS = frozenset({"models", "route", "plugins", "web_search_options"})

# The judge's reply budget is max_tokens=120 (rag_eval.judge). OpenRouter counts reasoning
# tokens against max_tokens (docs: /docs/use-cases/reasoning-tokens), so a reasoning-on
# Qwen3 would spend the budget thinking and return empty content. effort "none" disables
# it — matching the Groq judge the committed numbers came from, which fit its JSON in 120
# tokens on all 50 calls (0 parse failures).
OPENROUTER_JUDGE_REASONING = {"effort": "none"}

KEY_ENV: dict[str, str] = {"groq": "SSR_LLM_API_KEY", "openrouter": "SSR_OPENROUTER_API_KEY"}
PROVIDER_ENV: dict[str, str] = {"generator": "SSR_LLM_PROVIDER", "judge": "SSR_JUDGE_PROVIDER"}
# Settings that only apply to one provider, per role: set-but-ignored ones are reported.
_PROVIDER_ONLY: dict[tuple[str, str], tuple[str, ...]] = {
    ("generator", "groq"): ("llm_base_url", "llm_model"),
    ("generator", "openrouter"): ("openrouter_llm_model",),
    ("judge", "groq"): ("judge_model",),
    ("judge", "openrouter"): ("openrouter_judge_model",),
}


class SpendPolicyError(RuntimeError):
    """A request to OpenRouter would violate the spend policy; nothing was sent."""


class PaidModelRefused(SpendPolicyError):
    """A paid model id that the policy does not allow would have been sent to OpenRouter."""


class MissingApiKey(RuntimeError):
    """The provider in use has no API key configured."""


class EndpointConfigError(ValueError):
    """The settings would pair a key with the wrong provider's endpoint."""


def is_openrouter(url: str) -> bool:
    """Conservative: any URL mentioning openrouter.ai counts (a false positive can only
    cause a refusal, never a spend)."""
    return "openrouter.ai" in (url or "").lower()


def free_id(model: str) -> str:
    return model if model.endswith(FREE_SUFFIX) else model + FREE_SUFFIX


def base_model(model: str | None) -> str | None:
    """The underlying model, without OpenRouter's `:free` variant suffix."""
    return None if model is None else model.removesuffix(FREE_SUFFIX)


def is_free(model: str) -> bool:
    return str(model).endswith(FREE_SUFFIX)


def routing_for(model: str) -> dict:
    """The provider-routing object an OpenRouter request for `model` carries."""
    return _copy(FREE_ROUTING if is_free(model) else PAID_ROUTING)


def _copy(d: dict) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
            for k, v in d.items()}


def _price_ok(max_price: object, cap: Mapping[str, float]) -> bool:
    if not isinstance(max_price, Mapping):
        return False
    for k, limit in cap.items():
        v = max_price.get(k)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            return False
        if v < 0 or v > limit:
            return False
    return True


def enforce_spend_policy(
    role: str,
    base_url: str,
    model: str,
    body: Mapping | None,
    allowlist: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Raise SpendPolicyError unless this request is allowed. `body` is the merged
    ``extra_body`` that will be sent. No-op for non-OpenRouter URLs."""
    if not is_openrouter(base_url):
        return
    body = body or {}
    bad = FORBIDDEN_OPENROUTER_FIELDS & set(body)
    if bad:
        raise SpendPolicyError(f"refusing OpenRouter request with {sorted(bad)}: they can add cost")
    routing = body.get("provider")
    if not isinstance(routing, Mapping) or routing.get("allow_fallbacks") is not False:
        raise SpendPolicyError(
            "refusing OpenRouter request without provider.allow_fallbacks=false"
        )
    if is_free(model):
        if not _price_ok(routing.get("max_price"), {"prompt": 0.0, "completion": 0.0}):
            raise SpendPolicyError("refusing ':free' request without a $0/$0 provider.max_price")
        return
    if role != "generator":
        raise PaidModelRefused(
            f"refusing to send model {model!r} to OpenRouter: the {role} must be a ':free' id"
        )
    allowed = settings.openrouter_paid_model_allowlist if allowlist is None else allowlist
    if model not in allowed:
        raise PaidModelRefused(
            f"refusing to send paid model {model!r} to OpenRouter: not in "
            f"SSR_OPENROUTER_PAID_MODEL_ALLOWLIST (only ':free' ids and allowlisted models)"
        )
    if not _price_ok(routing.get("max_price"), PAID_MAX_PRICE_USD_PER_M):
        raise SpendPolicyError(
            f"refusing paid OpenRouter request: provider.max_price must be <= "
            f"{PAID_MAX_PRICE_USD_PER_M} USD per 1M tokens"
        )
    order = routing.get("order")
    if not isinstance(order, list | tuple) or not order:
        raise SpendPolicyError("refusing paid OpenRouter request without a pinned provider.order")
    quant = routing.get("quantizations")
    if not isinstance(quant, list | tuple) or not quant or not set(quant) <= set(PAID_QUANTIZATIONS):
        raise SpendPolicyError(
            f"refusing paid OpenRouter request: provider.quantizations must be within "
            f"{list(PAID_QUANTIZATIONS)}"
        )


def provider_of(role: Role, s: Settings | None = None) -> Provider:
    s = settings if s is None else s
    return s.llm_provider if role == "generator" else s.judge_provider


def model_id(role: Role, s: Settings | None = None, provider: Provider | None = None) -> str:
    """The model id a role sends, with no key or URL checks (safe at import time)."""
    s = settings if s is None else s
    provider = provider or provider_of(role, s)
    if provider == "openrouter":
        return s.openrouter_llm_model if role == "generator" else free_id(s.openrouter_judge_model)
    return s.llm_model if role == "generator" else base_model(s.judge_model)


@dataclass(frozen=True)
class LLMEndpoint:
    role: Role
    provider: Provider
    base_url: str
    model: str
    api_key: str = field(default="", repr=False, compare=False)
    allowlist: tuple[str, ...] = ()

    @property
    def key_env(self) -> str:
        return KEY_ENV[self.provider]

    @property
    def paid(self) -> bool:
        return self.provider == "openrouter" and not is_free(self.model)

    def extra_body(self) -> dict:
        """Provider-specific request fields every call to this endpoint must carry."""
        if not is_openrouter(self.base_url):
            return {}
        body: dict = {"provider": routing_for(self.model)}
        if self.role == "judge":
            body["reasoning"] = dict(OPENROUTER_JUDGE_REASONING)
        return body

    def check(self, model: str | None = None, body: Mapping | None = None) -> None:
        enforce_spend_policy(
            self.role,
            self.base_url,
            self.model if model is None else model,
            self.extra_body() if body is None else body,
            self.allowlist,
        )

    def metadata(self) -> dict:
        """Provenance for results files. Never includes the key."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "extra_body": self.extra_body(),
        }

    def describe(self) -> str:
        return f"{self.provider} {self.model}{' (paid)' if self.paid else ''} @ {self.base_url}"


def resolve_endpoint(
    role: Role,
    s: Settings | None = None,
    *,
    provider: Provider | None = None,
    require_key: bool = True,
) -> LLMEndpoint:
    """(provider, base URL, key, model id) for `role`. See the module docstring for the
    precedence rules. `provider` overrides the setting (the provider-comparison mode)."""
    s = settings if s is None else s
    provider = provider or provider_of(role, s)
    if provider == "openrouter":
        url, key = OPENROUTER_BASE_URL, s.openrouter_api_key
    elif provider == "groq":
        url, key = s.llm_base_url, s.llm_api_key
        if is_openrouter(url):
            raise EndpointConfigError(
                f"SSR_LLM_BASE_URL points at openrouter.ai but the {role} provider is 'groq'; "
                f"set {PROVIDER_ENV[role]}=openrouter instead (refusing to send "
                f"SSR_LLM_API_KEY to OpenRouter)"
            )
    else:
        raise EndpointConfigError(f"unknown {role} provider {provider!r}")
    ep = LLMEndpoint(
        role=role,
        provider=provider,
        base_url=url,
        model=model_id(role, s, provider),
        api_key=key,
        allowlist=tuple(s.openrouter_paid_model_allowlist),
    )
    ep.check()  # a disallowed model fails at startup, not on the first request
    if require_key and not key:
        raise MissingApiKey(
            f"{KEY_ENV[provider]} is not set; the {provider} {role} needs it (add it to .env)"
        )
    return ep


def endpoint_for_url(
    role: Role, base_url: str, model: str, api_key: str | None = None, s: Settings | None = None
) -> LLMEndpoint:
    """An endpoint from an explicit base URL (code callers, tests). The provider — and
    so the default key — follows the URL, so an explicit URL can't pick up the other
    provider's key."""
    s = settings if s is None else s
    provider: Provider = "openrouter" if is_openrouter(base_url) else "groq"
    key = api_key if api_key is not None else (
        s.openrouter_api_key if provider == "openrouter" else s.llm_api_key
    )
    ep = LLMEndpoint(role, provider, base_url, model, key, tuple(s.openrouter_paid_model_allowlist))
    ep.check()
    return ep


def ignored_settings(role: Role, s: Settings | None = None, provider: Provider | None = None) -> list[str]:
    """Env vars explicitly set (env or .env) that the provider in use ignores — e.g. a
    Groq SSR_LLM_MODEL left in .env under llm_provider=openrouter. Names only."""
    s = settings if s is None else s
    provider = provider or provider_of(role, s)
    other = "groq" if provider == "openrouter" else "openrouter"
    return [f"SSR_{f.upper()}" for f in _PROVIDER_ONLY[(role, other)] if f in s.model_fields_set]


def describe_with_ignored(ep: LLMEndpoint, s: Settings | None = None) -> str:
    """One console line: the effective endpoint, plus any set-but-ignored settings."""
    ignored = ignored_settings(ep.role, s, ep.provider)
    note = f"  [set but ignored under {ep.provider}: {', '.join(ignored)}]" if ignored else ""
    return f"{ep.role}: {ep.describe()}{note}"


# --- response accounting ----------------------------------------------------------------


def response_cost(resp: object) -> float | None:
    """OpenRouter's reported cost of one response, in credits (USD), from usage.cost —
    always included in OpenRouter responses (docs: /docs/use-cases/usage-accounting).
    None when the provider doesn't report it (Groq, Ollama)."""
    usage = getattr(resp, "usage", None)
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    if isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost):
        return None
    return float(cost)


def response_provider(resp: object) -> str | None:
    """The upstream provider OpenRouter actually routed to (top-level `provider`)."""
    p = getattr(resp, "provider", None)
    return p if isinstance(p, str) else None


def cost_upper_bound(prompt_tokens: int | None, completion_tokens: int | None) -> float:
    """What a paid call can have cost at most, at the max_price caps — used when the
    provider reports no cost, so the spend ceiling never counts an unknown as zero."""
    p, c = PAID_MAX_PRICE_USD_PER_M["prompt"], PAID_MAX_PRICE_USD_PER_M["completion"]
    return ((prompt_tokens or 0) * p + (completion_tokens or 0) * c) / 1e6


class GuardedClient:
    """The ``chat.completions.create`` surface with the spend policy enforced on every
    request, the endpoint's provider fields merged into ``extra_body`` (the endpoint's
    fields win over the caller's, so no call site can drop them), and a running tally of
    calls and reported cost."""

    def __init__(self, client: object, endpoint: LLMEndpoint):
        self._client = client
        self.endpoint = endpoint
        self.calls = 0
        self.cost_usd = 0.0
        self.unreported_cost_calls = 0
        self.chat = _Chat(self._create)

    def _create(self, **kwargs):
        body = {**(kwargs.get("extra_body") or {}), **self.endpoint.extra_body()}
        self.endpoint.check(model=kwargs.get("model", ""), body=body)
        if body:
            kwargs["extra_body"] = body
        resp = self._client.chat.completions.create(**kwargs)
        self.calls += 1
        cost = response_cost(resp)
        if cost is None:
            self.unreported_cost_calls += 1
        else:
            self.cost_usd += cost
        return resp


class _Chat:
    def __init__(self, create: Callable):
        self.completions = _Completions(create)


class _Completions:
    def __init__(self, create: Callable):
        self.create = create


def build_client(endpoint: LLMEndpoint, factory: Callable | None = None, **client_kw) -> GuardedClient:
    """An OpenAI-compatible client for `endpoint`, wrapped in GuardedClient. `factory`
    defaults to openai.OpenAI (injectable so tests never touch the network)."""
    endpoint.check()
    if factory is None:
        from openai import OpenAI as factory  # noqa: N813
    kw = {"max_retries": 5, "timeout": 30.0, **client_kw}
    return GuardedClient(
        factory(base_url=endpoint.base_url, api_key=endpoint.api_key, **kw), endpoint
    )
