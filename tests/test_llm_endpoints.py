"""Endpoint resolution, key/URL precedence and the OpenRouter spend policy. All offline:
settings are built with no .env, clients are fakes that record requests."""
from types import SimpleNamespace

import pytest

from app.core import llm_endpoints as le
from app.core.config import Settings
from app.core.llm_endpoints import (
    FREE_ROUTING,
    OPENROUTER_BASE_URL,
    PAID_ROUTES,
    EndpointConfigError,
    GuardedClient,
    LLMEndpoint,
    MissingApiKey,
    PaidModelRefused,
    SpendPolicyError,
    build_client,
    enforce_spend_policy,
    ignored_settings,
    paid_routing,
    resolve_endpoint,
)
from app.core.interfaces import SearchHit
from app.generate.generator import REASK_MAX_TOKENS, LLMGenerator
from app.generate.prompts import REASK_SYSTEM

OR_KEY = "sk-or-v1-FAKE-openrouter-0000"
GROQ_KEY = "gsk_FAKE-groq-1111"
ALLOW = ("openai/gpt-oss-120b",)
PAID = "openai/gpt-oss-120b"
FREE_GEN = "inclusionai/ling-3.0-flash-sante:free"
LUNA = "openai/gpt-6-luna"
PAID_ONLY_FIELDS = {"order", "quantizations"}  # routing fields only paid routing carries
# gpt-oss-120b's routing exactly as the single hard-coded PAID_ROUTING had it before the
# per-model table (same keys, same order, same values), so its runs stay comparable.
GPT_OSS_ROUTING = {
    "order": ["akashml/bf16", "deepinfra/bf16"],
    "allow_fallbacks": False,
    "quantizations": ["bf16"],
    "max_price": {"prompt": 0.03, "completion": 0.17},
}
LUNA_ROUTING = {
    "order": ["openai"],
    "allow_fallbacks": False,
    "max_price": {"prompt": 0.10, "completion": 0.50},
}
PAID_ROUTING = paid_routing(PAID)


def _settings(**kw) -> Settings:
    base = {"openrouter_api_key": OR_KEY, "llm_api_key": GROQ_KEY}
    return Settings(_env_file=None, **{**base, **kw})


class _FakeOpenAI:
    """Records every request; never touches the network."""

    def __init__(self, cost=None, provider="AkashML", content="ok", finish="stop", **client_kw):
        self.client_kw = client_kw
        self.requests: list[dict] = []
        self._cost, self._provider, self._content, self._finish = cost, provider, content, finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, completion_tokens_details=None, cost=self._cost
        )
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=self._finish)],
            usage=usage,
            provider=self._provider,
        )


# --- resolution ---------------------------------------------------------------------------


def test_defaults_are_openrouter_free_generator_and_free_judge():
    s = _settings()
    gen, judge = resolve_endpoint("generator", s), resolve_endpoint("judge", s)
    assert (gen.provider, gen.base_url, gen.model) == ("openrouter", OPENROUTER_BASE_URL, FREE_GEN)
    assert not gen.paid and gen.api_key == OR_KEY
    assert (judge.provider, judge.model, judge.paid) == (
        "openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", False
    )
    # Free routing only: $0 max_price, no fallbacks, none of the paid pinning fields.
    assert gen.extra_body() == {"provider": FREE_ROUTING}
    assert not PAID_ONLY_FIELDS & set(gen.extra_body()["provider"])
    assert judge.extra_body() == {"provider": FREE_ROUTING, "reasoning": {"effort": "none"}}


def test_free_generator_is_never_subject_to_the_paid_allowlist():
    # Even with no paid model allowed at all, the free default resolves and passes policy.
    gen = resolve_endpoint("generator", _settings(openrouter_paid_model_allowlist=()))
    assert gen.model == FREE_GEN and gen.allowlist == () and not gen.paid
    gen.check()
    enforce_spend_policy("generator", OPENROUTER_BASE_URL, FREE_GEN, {"provider": FREE_ROUTING}, ())
    # ... and a free id can never be sent with paid routing's price caps.
    with pytest.raises(SpendPolicyError, match=r"\$0/\$0"):
        enforce_spend_policy("generator", OPENROUTER_BASE_URL, FREE_GEN, _paid_body(), ALLOW)


def test_generator_id_gets_free_suffix_unless_allowlisted():
    for model in ("inclusionai/ling-3.0-flash-sante", FREE_GEN):
        assert resolve_endpoint("generator", _settings(openrouter_llm_model=model)).model == FREE_GEN
    # Not allowlisted -> normalised to its $0 variant, like the judge; never sent paid.
    gpt4o = resolve_endpoint("generator", _settings(openrouter_llm_model="openai/gpt-4o"))
    assert gpt4o.model == "openai/gpt-4o:free" and not gpt4o.paid
    assert gpt4o.extra_body() == {"provider": FREE_ROUTING}
    # An allowlisted paid id is left as-is, and with an empty allowlist it too goes free.
    assert resolve_endpoint("generator", _settings(openrouter_llm_model=PAID)).model == PAID
    none_allowed = _settings(openrouter_llm_model=PAID, openrouter_paid_model_allowlist=())
    assert resolve_endpoint("generator", none_allowed).model == PAID + ":free"


def test_paid_gpt_oss_still_resolves_with_pinned_routing_when_configured():
    gen = resolve_endpoint("generator", _settings(openrouter_llm_model=PAID))
    assert (gen.model, gen.paid, gen.allowlist) == (PAID, True, (PAID, LUNA))
    assert gen.extra_body() == {"provider": PAID_ROUTING}
    assert "(paid)" in gen.describe()
    gen.check()  # passes the spend policy only because of the allowlist + PAID_ROUTING
    with pytest.raises(SpendPolicyError):
        gen.check(body={"provider": FREE_ROUTING})  # paid id without the pinned routing


def test_openrouter_judge_gets_free_suffix_once():
    assert resolve_endpoint("judge", _settings(openrouter_judge_model="qwen/qwen3.8-27b")).model == (
        "qwen/qwen3.8-27b:free"
    )
    assert resolve_endpoint("judge", _settings()).model.count(":free") == 1


def test_groq_resolution_uses_the_generic_endpoint_settings():
    s = _settings(llm_provider="groq", judge_provider="groq", judge_model="qwen/qwen3.8-27b:free")
    gen, judge = resolve_endpoint("generator", s), resolve_endpoint("judge", s)
    assert (gen.base_url, gen.model, gen.api_key) == (s.llm_base_url, s.llm_model, GROQ_KEY)
    assert judge.model == "qwen/qwen3.8-27b"  # Groq has no :free variants
    assert gen.extra_body() == {} and judge.extra_body() == {}  # no OpenRouter fields to Groq


def test_provider_decides_url_and_key_so_keys_never_cross():
    # A Groq base URL and model left in .env are ignored under openrouter ...
    s = _settings(llm_base_url="https://api.groq.com/openai/v1", llm_model="llama-3.3-70b")
    gen = resolve_endpoint("generator", s)
    assert gen.base_url == OPENROUTER_BASE_URL and gen.api_key == OR_KEY
    assert gen.model == FREE_GEN
    assert ignored_settings("generator", s) == ["SSR_LLM_BASE_URL", "SSR_LLM_MODEL"]
    # ... and under groq an openrouter.ai base URL is refused, so the Groq key can't go there.
    bad = _settings(llm_provider="groq", llm_base_url="https://openrouter.ai/api/v1")
    with pytest.raises(EndpointConfigError, match="SSR_LLM_PROVIDER=openrouter") as exc:
        resolve_endpoint("generator", bad)
    assert GROQ_KEY not in str(exc.value) and OR_KEY not in str(exc.value)
    # The groq endpoint only ever carries the Groq key.
    assert resolve_endpoint("generator", _settings(llm_provider="groq")).api_key == GROQ_KEY


def test_explicit_url_picks_that_providers_key():
    s = _settings()
    assert le.endpoint_for_url("generator", "http://localhost:11434/v1", "m", s=s).api_key == GROQ_KEY
    or_ep = le.endpoint_for_url("generator", OPENROUTER_BASE_URL, "openai/gpt-oss-120b", s=s)
    assert or_ep.provider == "openrouter" and or_ep.api_key == OR_KEY


@pytest.mark.parametrize(("provider", "var"), [("openrouter", "SSR_OPENROUTER_API_KEY"),
                                               ("groq", "SSR_LLM_API_KEY")])
def test_missing_key_error_names_the_var_and_no_key(provider, var):
    s = _settings(
        llm_provider=provider,
        **({"openrouter_api_key": ""} if provider == "openrouter" else {"llm_api_key": ""}),
    )
    with pytest.raises(MissingApiKey) as exc:
        resolve_endpoint("generator", s)
    msg = str(exc.value)
    assert var in msg and OR_KEY not in msg and GROQ_KEY not in msg
    assert resolve_endpoint("generator", s, require_key=False).api_key == ""  # dry runs


def test_metadata_and_repr_never_include_the_key():
    ep = resolve_endpoint("generator", _settings())
    assert set(ep.metadata()) == {"provider", "base_url", "model", "extra_body"}
    assert OR_KEY not in repr(ep.metadata()) and OR_KEY not in repr(ep)
    assert OR_KEY not in le.describe_with_ignored(ep, _settings())


# --- spend policy ---------------------------------------------------------------------------


def _paid_body(**routing) -> dict:
    return {"provider": {**PAID_ROUTING, **routing}}


def test_free_ids_pass_with_zero_price_routing():
    enforce_spend_policy("judge", OPENROUTER_BASE_URL, "qwen/qwen3.8-27b:free",
                         {"provider": FREE_ROUTING}, ALLOW)


def test_paid_judge_is_refused_even_if_allowlisted():
    with pytest.raises(PaidModelRefused, match="judge must be a ':free' id"):
        enforce_spend_policy("judge", OPENROUTER_BASE_URL, "openai/gpt-oss-120b", _paid_body(), ALLOW)
    with pytest.raises(PaidModelRefused):
        resolve_endpoint("judge", _settings(openrouter_judge_model="x:free"), provider="openrouter") \
            .check(model="qwen/qwen3.8-27b")


def test_paid_generator_outside_allowlist_is_refused():
    for model in ("anthropic/claude-opus", "openai/gpt-oss-120b:nitro", "openai/gpt-oss-120b:online"):
        with pytest.raises(PaidModelRefused, match="ALLOWLIST"):
            enforce_spend_policy("generator", OPENROUTER_BASE_URL, model, _paid_body(), ALLOW)
    # A paid id reaching a request directly (not via the settings, which normalise it to
    # `:free`) is still refused, even with paid routing.
    ep = LLMEndpoint("generator", "openrouter", OPENROUTER_BASE_URL, "openai/gpt-4o", OR_KEY, ALLOW)
    with pytest.raises(PaidModelRefused):
        ep.check()


def test_allowlisted_paid_generator_passes_only_with_pinned_capped_routing():
    enforce_spend_policy("generator", OPENROUTER_BASE_URL, "openai/gpt-oss-120b", _paid_body(), ALLOW)
    bad_bodies = {
        "no routing": {},
        "fallbacks on": _paid_body(allow_fallbacks=True),
        "fallbacks unset": {"provider": {k: v for k, v in PAID_ROUTING.items() if k != "allow_fallbacks"}},
        "no max_price": {"provider": {k: v for k, v in PAID_ROUTING.items() if k != "max_price"}},
        "prompt over cap": _paid_body(max_price={"prompt": 0.031, "completion": 0.17}),
        "completion over cap": _paid_body(max_price={"prompt": 0.03, "completion": 0.18}),
        "nan price": _paid_body(max_price={"prompt": float("nan"), "completion": 0.17}),
        "no order": _paid_body(order=[]),
        "fp4 allowed": _paid_body(quantizations=["bf16", "fp4"]),
        "no quantization filter": _paid_body(quantizations=[]),
        "models fallback": {**_paid_body(), "models": ["openai/gpt-4o"]},
        "web plugin": {**_paid_body(), "plugins": [{"id": "web"}]},
    }
    for label, body in bad_bodies.items():
        with pytest.raises(SpendPolicyError):
            enforce_spend_policy("generator", OPENROUTER_BASE_URL, "openai/gpt-oss-120b", body, ALLOW)
            pytest.fail(label)


def test_free_request_without_zero_price_is_refused():
    with pytest.raises(SpendPolicyError, match=r"\$0/\$0"):
        enforce_spend_policy("judge", OPENROUTER_BASE_URL, "qwen/qwen3.8-27b:free",
                             {"provider": {"allow_fallbacks": False}}, ALLOW)


def test_policy_is_a_no_op_off_openrouter():
    enforce_spend_policy("judge", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", {}, ())


def test_guarded_client_refuses_before_any_network_call():
    fake = _FakeOpenAI()
    judge_ep = resolve_endpoint("judge", _settings())
    client = GuardedClient(fake, judge_ep)
    with pytest.raises(PaidModelRefused):
        client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[])  # no :free
    gen_client = GuardedClient(fake, resolve_endpoint("generator", _settings()))
    with pytest.raises(PaidModelRefused):
        gen_client.chat.completions.create(model="openai/gpt-4o", messages=[])
    # The free default endpoint can't be used to smuggle out the paid allowlisted id.
    with pytest.raises(SpendPolicyError):
        gen_client.chat.completions.create(model=PAID, messages=[])
    assert fake.requests == [] and client.calls == 0


def test_guarded_client_forces_routing_over_the_callers_and_tallies_cost():
    fake = _FakeOpenAI(cost=0.0002)
    client = GuardedClient(fake, resolve_endpoint("generator", _settings(openrouter_llm_model=PAID)))
    client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[],
        extra_body={"provider": {"allow_fallbacks": True}, "reasoning": {"effort": "medium"}},
    )
    body = fake.requests[0]["extra_body"]
    assert body["provider"] == PAID_ROUTING and body["reasoning"] == {"effort": "medium"}
    client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
    assert client.calls == 2 and client.cost_usd == pytest.approx(0.0004)


def test_build_client_passes_key_only_to_its_own_endpoint():
    made = {}

    def factory(**kw):
        made.update(kw)
        return _FakeOpenAI()

    build_client(resolve_endpoint("judge", _settings()), factory=factory)
    assert made["base_url"] == OPENROUTER_BASE_URL and made["api_key"] == OR_KEY
    with pytest.raises(PaidModelRefused):
        build_client(LLMEndpoint("judge", "openrouter", OPENROUTER_BASE_URL, "qwen/qwen3.8-27b"),
                     factory=factory)


def test_response_accounting_helpers():
    resp = _FakeOpenAI(cost=0.00012).chat.completions.create(model="m", messages=[])
    assert le.response_cost(resp) == pytest.approx(0.00012) and le.response_provider(resp) == "AkashML"
    assert le.response_cost(SimpleNamespace(usage=None)) is None
    assert le.cost_upper_bound(1_000_000, 1_000_000, PAID) == pytest.approx(0.03 + 0.17)
    # Luna bills its prompt up to the $0.125/M cache-write rate (above the $0.10 filter).
    assert le.cost_upper_bound(1_000_000, 1_000_000, LUNA) == pytest.approx(0.125 + 0.50)
    assert le.paid_bill_rate(PAID) == le.paid_max_price(PAID)  # gpt-oss: bound == caps
    with pytest.raises(PaidModelRefused, match="routing table"):
        le.cost_upper_bound(1, 1, "openai/gpt-4o")


# --- the generator on OpenRouter ------------------------------------------------------------


def _hits(n):
    return [SearchHit(f"doc{i}", 1.0, f"text {i}", {"title": f"T{i}"}) for i in range(n)]


def _or_generator(monkeypatch, model=PAID, reask=False, **fake_kw):
    # reask=False: these pin the first generation's requests; the re-ask has its own test.
    monkeypatch.setattr(le.settings, "openrouter_paid_model_allowlist", (PAID, LUNA))
    ep = resolve_endpoint("generator", _settings(openrouter_llm_model=model))
    gen = LLMGenerator(endpoint=ep, reasoning_effort="auto", reask=reask)
    fake = _FakeOpenAI(**fake_kw)
    gen.client = GuardedClient(fake, ep)
    return gen, fake


def test_generator_sends_openrouter_reasoning_object_and_pinned_routing(monkeypatch):
    gen, fake = _or_generator(monkeypatch, cost=0.0003, content="Yes [1].\nVerdict: SUPPORTED")
    ans = gen.generate("claim", _hits(2))
    body = fake.requests[0]["extra_body"]
    assert body == {"reasoning": {"effort": "medium"}, "provider": PAID_ROUTING}
    assert "reasoning_effort" not in body  # Groq's form never goes to OpenRouter
    assert ans.verdict == "SUPPORTED" and ans.cost_usd == pytest.approx(0.0003)
    assert ans.provider == "AkashML"


def test_free_generator_sends_free_routing_and_no_reasoning_or_paid_fields(monkeypatch):
    gen, fake = _or_generator(
        monkeypatch, model=FREE_GEN, cost=0, provider="Novita", content="No [1].\nVerdict: REFUTED"
    )
    assert gen.endpoint.model == FREE_GEN and gen.reasoning_effort is None  # "auto" gate
    ans = gen.generate("claim", _hits(2))
    body = fake.requests[0]["extra_body"]
    assert body == {"provider": FREE_ROUTING}  # no reasoning object, no order/quantizations
    assert "reasoning_effort" not in fake.requests[0]
    assert ans.verdict == "REFUTED" and ans.cost_usd == 0 and ans.provider == "Novita"


def test_free_generator_truncation_retry_stays_free(monkeypatch):
    gen, fake = _or_generator(monkeypatch, model=FREE_GEN, cost=0, finish="length")
    ans = gen.generate("claim", _hits(2))
    assert [r["max_tokens"] for r in fake.requests] == [
        le.settings.llm_max_completion_tokens, 2 * le.settings.llm_max_completion_tokens
    ]
    assert all(r["extra_body"] == {"provider": FREE_ROUTING} for r in fake.requests)
    assert ans.attempts == 2 and ans.truncated and ans.cost_usd == 0


def test_explicit_reasoning_effort_reaches_a_free_generator_only_when_asked(monkeypatch):
    monkeypatch.setattr(le.settings, "openrouter_paid_model_allowlist", ALLOW)
    ep = resolve_endpoint("generator", _settings())
    gen = LLMGenerator(endpoint=ep, reasoning_effort="low")
    fake = _FakeOpenAI()
    gen.client = GuardedClient(fake, ep)
    gen.generate("claim", _hits(1))
    assert fake.requests[0]["extra_body"] == {"reasoning": {"effort": "low"}, "provider": FREE_ROUTING}


def test_generator_cost_sums_the_truncation_retry(monkeypatch):
    gen, fake = _or_generator(monkeypatch, cost=0.0002, finish="length")
    ans = gen.generate("claim", _hits(2))
    assert len(fake.requests) == 2 and ans.cost_usd == pytest.approx(0.0004)
    unreported, _ = _or_generator(monkeypatch, cost=None)
    assert unreported.generate("claim", _hits(2)).cost_usd is None


def test_generator_refuses_a_disallowed_paid_model_before_any_request():
    ep = LLMEndpoint("generator", "openrouter", OPENROUTER_BASE_URL, "openai/gpt-4o", OR_KEY, ALLOW)
    with pytest.raises(PaidModelRefused):
        LLMGenerator(endpoint=ep)
    ok = LLMGenerator(endpoint=LLMEndpoint(
        "generator", "openrouter", OPENROUTER_BASE_URL, "openai/gpt-oss-120b", OR_KEY, ALLOW
    ))
    fake = _FakeOpenAI()
    ok.client = fake  # even an unguarded client: _complete checks the policy itself
    ok.model = "openai/gpt-4o"
    with pytest.raises(PaidModelRefused):
        ok.generate("claim", _hits(1))
    assert fake.requests == []


# --- the per-model paid routing table ---------------------------------------------------------


def test_gpt_oss_routing_is_unchanged_by_the_per_model_table():
    assert paid_routing(PAID) == GPT_OSS_ROUTING
    assert list(paid_routing(PAID)) == list(GPT_OSS_ROUTING)  # key order too (JSON bodies)
    gen = resolve_endpoint("generator", _settings(openrouter_llm_model=PAID))
    assert gen.extra_body() == {"provider": GPT_OSS_ROUTING}
    # A fresh copy each time: mutating one request's routing can't loosen the table.
    paid_routing(PAID)["max_price"]["prompt"] = 9.0
    assert paid_routing(PAID) == GPT_OSS_ROUTING


def test_default_allowlist_is_exactly_gpt_oss_and_luna_and_all_have_routes():
    allow = Settings.model_fields["openrouter_paid_model_allowlist"].default
    assert allow == (PAID, LUNA)
    assert set(allow) <= set(PAID_ROUTES)


def test_luna_resolves_as_paid_with_its_own_pinned_routing():
    gen = resolve_endpoint("generator", _settings(openrouter_llm_model=LUNA))
    assert (gen.model, gen.paid) == (LUNA, True) and LUNA in gen.allowlist
    assert gen.extra_body() == {"provider": LUNA_ROUTING}
    assert "quantizations" not in gen.extra_body()["provider"]  # OpenAI publishes none
    gen.check()
    # The Luna id is allowlisted only as a paid generator: never with gpt-oss's routing,
    # never free routing, never as the judge.
    for body in ({"provider": GPT_OSS_ROUTING}, {"provider": FREE_ROUTING}):
        with pytest.raises(SpendPolicyError):
            gen.check(body=body)
    with pytest.raises(PaidModelRefused, match="judge must be"):
        enforce_spend_policy("judge", OPENROUTER_BASE_URL, LUNA, {"provider": LUNA_ROUTING},
                             (PAID, LUNA))
    # Not allowlisted -> normalised to a `:free` id like any other, never sent paid.
    off = resolve_endpoint("generator", _settings(openrouter_llm_model=LUNA,
                                                  openrouter_paid_model_allowlist=(PAID,)))
    assert off.model == LUNA + ":free" and not off.paid


def test_luna_request_must_carry_exactly_its_routing():
    allow = (PAID, LUNA)
    enforce_spend_policy("generator", OPENROUTER_BASE_URL, LUNA, {"provider": LUNA_ROUTING}, allow)

    def body(**kw):
        return {"provider": {**LUNA_ROUTING, **kw}}

    bad = {
        "fast tier": body(order=["openai/fast"]),
        "flex tier": body(order=["openai/flex"]),
        "azure fallback": body(order=["openai", "azure"]),
        "fallbacks on": body(allow_fallbacks=True),
        "prompt over cap": body(max_price={"prompt": 0.11, "completion": 0.50}),
        "completion over cap": body(max_price={"prompt": 0.10, "completion": 1.0}),
        "extra price key": body(max_price={"prompt": 0.1, "completion": 0.5, "request": 1}),
        "quantization filter": body(quantizations=["bf16"]),
        "sort": body(sort="throughput"),
        "models fallback": {**body(), "models": [PAID]},
        "route": {**body(), "route": "fallback"},
        "plugins": {**body(), "plugins": [{"id": "web"}]},
    }
    for label, b in bad.items():
        with pytest.raises(SpendPolicyError):
            enforce_spend_policy("generator", OPENROUTER_BASE_URL, LUNA, b, allow)
            pytest.fail(label)


def test_allowlisted_paid_id_without_a_routing_table_entry_is_refused():
    # Allowlisting alone is not enough: the model also needs its own pinned routing.
    allow = (PAID, "openai/gpt-6-luna-pro")
    with pytest.raises(PaidModelRefused, match="routing table"):
        enforce_spend_policy("generator", OPENROUTER_BASE_URL, "openai/gpt-6-luna-pro",
                             {"provider": LUNA_ROUTING}, allow)
    with pytest.raises(PaidModelRefused, match="routing table"):
        resolve_endpoint("generator", _settings(openrouter_llm_model="openai/gpt-6-luna-pro",
                                                openrouter_paid_model_allowlist=allow))


def test_luna_generator_sends_no_temperature_and_medium_reasoning(monkeypatch):
    gen, fake = _or_generator(monkeypatch, model=LUNA, cost=0.0004, provider="OpenAI",
                              content="Refuted [1].\nVerdict: REFUTED")
    ans = gen.generate("claim", _hits(2))
    req = fake.requests[0]
    assert "temperature" not in req
    assert req["extra_body"] == {"reasoning": {"effort": "medium"}, "provider": LUNA_ROUTING}
    assert req["max_tokens"] == le.settings.llm_max_completion_tokens
    assert ans.verdict == "REFUTED" and ans.provider == "OpenAI"
    assert ans.cost_usd == pytest.approx(0.0004)


def test_luna_truncation_retry_keeps_its_params(monkeypatch):
    gen, fake = _or_generator(monkeypatch, model=LUNA, cost=0.001, finish="length")
    ans = gen.generate("claim", _hits(2))
    budget = le.settings.llm_max_completion_tokens
    assert [r["max_tokens"] for r in fake.requests] == [budget, 2 * budget]
    assert all("temperature" not in r for r in fake.requests)
    assert all(r["extra_body"]["provider"] == LUNA_ROUTING for r in fake.requests)
    assert ans.attempts == 2 and ans.cost_usd == pytest.approx(0.002)


@pytest.mark.parametrize("model", [PAID, FREE_GEN, LUNA])
def test_reask_goes_through_the_same_spend_policy_routing_and_params(monkeypatch, model):
    # A claim truncated to "ok" twice: the re-ask is the third request, and it carries
    # exactly the first request's routing (pinned paid provider / $0 free routing) and
    # model-aware params (no temperature for Luna, reasoning object per model).
    gen, fake = _or_generator(monkeypatch, model=model, cost=0.0001, finish="length", reask=True)
    ans = gen.generate("Statins lower LDL cholesterol.", _hits(2))
    assert len(fake.requests) == 3
    first, reask = fake.requests[0], fake.requests[2]
    assert reask["max_tokens"] == REASK_MAX_TOKENS and reask["model"] == first["model"]
    assert reask["extra_body"] == first["extra_body"]
    assert reask.get("temperature") == first.get("temperature")
    assert ("temperature" in reask) == (model != LUNA)
    assert reask["messages"][0]["content"] == REASK_SYSTEM
    assert ans.reask_attempted and ans.cost_usd == pytest.approx(0.0003)  # all 3 billed


@pytest.mark.parametrize("model", [PAID, FREE_GEN])
def test_gpt_oss_and_ling_still_get_temperature(monkeypatch, model):
    gen, fake = _or_generator(monkeypatch, model=model, cost=0)
    gen.generate("claim", _hits(1))
    assert fake.requests[0]["temperature"] == 0.1
