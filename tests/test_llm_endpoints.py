"""Endpoint resolution, key/URL precedence and the OpenRouter free-only guard. All
offline: settings are built with no .env, clients are fakes that record requests."""
from types import SimpleNamespace

import pytest

from app.core import llm_endpoints as le
from app.core.config import Settings
from app.core.llm_endpoints import (
    FREE_ROUTING,
    OPENROUTER_BASE_URL,
    EndpointConfigError,
    GuardedClient,
    LLMEndpoint,
    MissingApiKey,
    PaidModelRefused,
    SpendPolicyError,
    build_client,
    enforce_spend_policy,
    ignored_settings,
    resolve_endpoint,
)
from app.core.interfaces import SearchHit
from app.generate.generator import REASK_MAX_TOKENS, LLMGenerator
from app.generate.prompts import REASK_SYSTEM

OR_KEY = "sk-or-v1-FAKE-openrouter-0000"
COMPAT_KEY = "gsk_FAKE-compat-1111"
FREE_GEN = "inclusionai/ling-3.0-flash-sante:free"
FREE_BODY = {"provider": FREE_ROUTING}


def _settings(**kw) -> Settings:
    base = {"openrouter_api_key": OR_KEY, "llm_api_key": COMPAT_KEY}
    return Settings(_env_file=None, **{**base, **kw})


class _FakeOpenAI:
    """Records every request; never touches the network."""

    def __init__(self, cost=None, provider="Novita", content="ok", finish="stop", **client_kw):
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
    assert gen.api_key == OR_KEY
    assert (judge.provider, judge.model) == ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free")
    # Free routing only: $0 max_price, no fallbacks.
    assert gen.extra_body() == FREE_BODY
    assert judge.extra_body() == {"provider": FREE_ROUTING, "reasoning": {"effort": "none"}}


def test_every_openrouter_id_gets_the_free_suffix_once():
    for model in ("inclusionai/ling-3.0-flash-sante", FREE_GEN):
        assert resolve_endpoint("generator", _settings(openrouter_llm_model=model)).model == FREE_GEN
    # Any configured id is normalised to its $0 variant; there is no paid path.
    for model in ("openai/gpt-oss-120b", "openai/gpt-4o"):
        ep = resolve_endpoint("generator", _settings(openrouter_llm_model=model))
        assert ep.model == model + ":free" and ep.extra_body() == FREE_BODY
    assert resolve_endpoint("judge", _settings(openrouter_judge_model="qwen/qwen3.8-27b")).model == (
        "qwen/qwen3.8-27b:free"
    )
    assert resolve_endpoint("judge", _settings()).model.count(":free") == 1


def test_openai_compat_resolution_uses_the_generic_endpoint_settings():
    s = _settings(llm_provider="openai_compat", judge_provider="openai_compat",
                  judge_model="qwen/qwen3.8-27b:free")
    gen, judge = resolve_endpoint("generator", s), resolve_endpoint("judge", s)
    assert (gen.base_url, gen.model, gen.api_key) == (s.llm_base_url, s.llm_model, COMPAT_KEY)
    assert gen.provider == judge.provider == "openai_compat"
    assert judge.model == "qwen/qwen3.8-27b"  # no :free variants off OpenRouter
    assert gen.extra_body() == {} and judge.extra_body() == {}  # no OpenRouter fields


def test_the_old_groq_provider_name_is_rejected():
    with pytest.raises(ValueError):
        _settings(llm_provider="groq")


def test_provider_decides_url_and_key_so_keys_never_cross():
    # A base URL and model left in .env are ignored under openrouter ...
    s = _settings(llm_base_url="https://api.groq.com/openai/v1", llm_model="llama-3.3-70b")
    gen = resolve_endpoint("generator", s)
    assert gen.base_url == OPENROUTER_BASE_URL and gen.api_key == OR_KEY
    assert gen.model == FREE_GEN
    assert ignored_settings("generator", s) == ["SSR_LLM_BASE_URL", "SSR_LLM_MODEL"]
    # ... and under openai_compat an openrouter.ai base URL is refused, so
    # SSR_LLM_API_KEY can't go there.
    bad = _settings(llm_provider="openai_compat", llm_base_url="https://openrouter.ai/api/v1")
    with pytest.raises(EndpointConfigError, match="SSR_LLM_PROVIDER=openrouter") as exc:
        resolve_endpoint("generator", bad)
    assert COMPAT_KEY not in str(exc.value) and OR_KEY not in str(exc.value)
    # The openai_compat endpoint only ever carries its own key.
    assert resolve_endpoint("generator", _settings(llm_provider="openai_compat")).api_key == COMPAT_KEY


def test_explicit_url_picks_that_providers_key():
    s = _settings()
    local = le.endpoint_for_url("generator", "http://localhost:11434/v1", "m", s=s)
    assert local.provider == "openai_compat" and local.api_key == COMPAT_KEY
    or_ep = le.endpoint_for_url("generator", OPENROUTER_BASE_URL, FREE_GEN, s=s)
    assert or_ep.provider == "openrouter" and or_ep.api_key == OR_KEY
    with pytest.raises(PaidModelRefused):  # an explicit OpenRouter URL is still free-only
        le.endpoint_for_url("generator", OPENROUTER_BASE_URL, "openai/gpt-oss-120b", s=s)


def test_keys_are_masked_in_settings_repr_but_reach_the_endpoint():
    s = Settings(_env_file=None, openrouter_api_key="sk-or-SECRET", llm_api_key="gsk-SECRET")
    for text in (repr(s), str(s), repr(s.openrouter_api_key), str(s.llm_api_key)):
        assert "SECRET" not in text
    assert resolve_endpoint("generator", s).api_key == "sk-or-SECRET"
    compat = _settings(llm_provider="openai_compat", llm_api_key="gsk-SECRET")
    assert resolve_endpoint("generator", compat).api_key == "gsk-SECRET"
    ep = le.endpoint_for_url("generator", "http://localhost:11434/v1", "m", s=s)
    assert ep.api_key == "gsk-SECRET" and "SECRET" not in repr(ep)


@pytest.mark.parametrize(("provider", "var"), [("openrouter", "SSR_OPENROUTER_API_KEY"),
                                               ("openai_compat", "SSR_LLM_API_KEY")])
def test_missing_key_error_names_the_var_and_no_key(provider, var):
    s = _settings(
        llm_provider=provider,
        **({"openrouter_api_key": ""} if provider == "openrouter" else {"llm_api_key": ""}),
    )
    with pytest.raises(MissingApiKey) as exc:
        resolve_endpoint("generator", s)
    msg = str(exc.value)
    assert var in msg and OR_KEY not in msg and COMPAT_KEY not in msg
    assert resolve_endpoint("generator", s, require_key=False).api_key == ""  # dry runs


def test_metadata_and_repr_never_include_the_key():
    ep = resolve_endpoint("generator", _settings())
    assert set(ep.metadata()) == {"provider", "base_url", "model", "extra_body"}
    assert OR_KEY not in repr(ep.metadata()) and OR_KEY not in repr(ep)
    assert OR_KEY not in le.describe_with_ignored(ep, _settings())


# --- the free-only guard ----------------------------------------------------------------------


def test_free_ids_pass_with_zero_price_routing():
    for role in ("generator", "judge"):
        enforce_spend_policy(role, OPENROUTER_BASE_URL, "qwen/qwen3.8-27b:free", FREE_BODY)


def test_non_free_ids_are_refused_for_either_role():
    for role in ("generator", "judge"):
        for model in ("openai/gpt-oss-120b", "openai/gpt-6-luna", "anthropic/claude-opus",
                      "openai/gpt-oss-120b:nitro"):
            with pytest.raises(PaidModelRefused, match="only ':free' ids"):
                enforce_spend_policy(role, OPENROUTER_BASE_URL, model, FREE_BODY)
    # A non-free id reaching a request directly (not via the settings, which normalise it
    # to `:free`) is still refused.
    ep = LLMEndpoint("generator", "openrouter", OPENROUTER_BASE_URL, "openai/gpt-4o", OR_KEY)
    with pytest.raises(PaidModelRefused):
        ep.check()


def test_free_requests_need_no_fallbacks_a_zero_price_and_no_spend_fields():
    bad_bodies = {
        "no routing": {},
        "fallbacks on": {"provider": {**FREE_ROUTING, "allow_fallbacks": True}},
        "fallbacks unset": {"provider": {"max_price": {"prompt": 0, "completion": 0}}},
        "no max_price": {"provider": {"allow_fallbacks": False}},
        "prompt priced": {"provider": {**FREE_ROUTING, "max_price": {"prompt": 0.01, "completion": 0}}},
        "nan price": {"provider": {**FREE_ROUTING, "max_price": {"prompt": float("nan"), "completion": 0}}},
        "models fallback": {**FREE_BODY, "models": ["openai/gpt-4o"]},
        "route": {**FREE_BODY, "route": "fallback"},
        "web plugin": {**FREE_BODY, "plugins": [{"id": "web"}]},
    }
    for label, body in bad_bodies.items():
        with pytest.raises(SpendPolicyError):
            enforce_spend_policy("generator", OPENROUTER_BASE_URL, FREE_GEN, body)
            pytest.fail(label)


def test_policy_is_a_no_op_off_openrouter():
    enforce_spend_policy("judge", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", {})


def test_guarded_client_refuses_before_any_network_call():
    fake = _FakeOpenAI()
    client = GuardedClient(fake, resolve_endpoint("judge", _settings()))
    with pytest.raises(PaidModelRefused):
        client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[])  # no :free
    gen_client = GuardedClient(fake, resolve_endpoint("generator", _settings()))
    with pytest.raises(PaidModelRefused):
        gen_client.chat.completions.create(model="openai/gpt-4o", messages=[])
    assert fake.requests == [] and client.calls == 0


def test_guarded_client_forces_routing_over_the_callers_and_tallies_cost():
    fake = _FakeOpenAI(cost=0.0002)
    client = GuardedClient(fake, resolve_endpoint("generator", _settings()))
    client.chat.completions.create(
        model=FREE_GEN,
        messages=[],
        extra_body={"provider": {"allow_fallbacks": True}, "reasoning": {"effort": "medium"}},
    )
    body = fake.requests[0]["extra_body"]
    assert body["provider"] == FREE_ROUTING and body["reasoning"] == {"effort": "medium"}
    client.chat.completions.create(model=FREE_GEN, messages=[])
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
    assert le.response_cost(resp) == pytest.approx(0.00012) and le.response_provider(resp) == "Novita"
    assert le.response_cost(SimpleNamespace(usage=None)) is None


# --- the generator on OpenRouter ------------------------------------------------------------


def _hits(n):
    return [SearchHit(f"doc{i}", 1.0, f"text {i}", {"title": f"T{i}"}) for i in range(n)]


def _or_generator(reask=False, reasoning_effort="auto", **fake_kw):
    # reask=False: these pin the first generation's requests; the re-ask has its own test.
    ep = resolve_endpoint("generator", _settings())
    gen = LLMGenerator(endpoint=ep, reasoning_effort=reasoning_effort, reask=reask)
    fake = _FakeOpenAI(**fake_kw)
    gen.client = GuardedClient(fake, ep)
    return gen, fake


def test_free_generator_sends_free_routing_temperature_and_no_reasoning():
    gen, fake = _or_generator(cost=0, provider="Novita", content="No [1].\nVerdict: REFUTED")
    assert gen.endpoint.model == FREE_GEN and gen.reasoning_effort is None  # "auto": nothing
    ans = gen.generate("claim", _hits(2))
    req = fake.requests[0]
    assert req["extra_body"] == FREE_BODY and req["temperature"] == 0.1
    assert "reasoning_effort" not in req
    assert ans.verdict == "REFUTED" and ans.cost_usd == 0 and ans.provider == "Novita"


def test_free_generator_truncation_retry_stays_free():
    gen, fake = _or_generator(cost=0, finish="length")
    ans = gen.generate("claim", _hits(2))
    assert [r["max_tokens"] for r in fake.requests] == [
        le.settings.llm_max_completion_tokens, 2 * le.settings.llm_max_completion_tokens
    ]
    assert all(r["extra_body"] == FREE_BODY for r in fake.requests)
    assert ans.attempts == 2 and ans.truncated and ans.cost_usd == 0


def test_explicit_reasoning_effort_reaches_a_free_generator_only_when_asked():
    gen, fake = _or_generator(reasoning_effort="low")
    gen.generate("claim", _hits(1))
    assert fake.requests[0]["extra_body"] == {"reasoning": {"effort": "low"}, "provider": FREE_ROUTING}
    assert "reasoning_effort" not in fake.requests[0]["extra_body"]  # the openai_compat form


def test_generator_cost_sums_the_truncation_retry():
    gen, fake = _or_generator(cost=0.0002, finish="length")
    ans = gen.generate("claim", _hits(2))
    assert len(fake.requests) == 2 and ans.cost_usd == pytest.approx(0.0004)
    unreported, _ = _or_generator(cost=None)
    assert unreported.generate("claim", _hits(2)).cost_usd is None


def test_generator_refuses_a_non_free_model_before_any_request():
    ep = LLMEndpoint("generator", "openrouter", OPENROUTER_BASE_URL, "openai/gpt-4o", OR_KEY)
    with pytest.raises(PaidModelRefused):
        LLMGenerator(endpoint=ep)
    ok = LLMGenerator(endpoint=LLMEndpoint(
        "generator", "openrouter", OPENROUTER_BASE_URL, FREE_GEN, OR_KEY
    ))
    fake = _FakeOpenAI()
    ok.client = fake  # even an unguarded client: _complete checks the policy itself
    ok.model = "openai/gpt-4o"
    with pytest.raises(PaidModelRefused):
        ok.generate("claim", _hits(1))
    assert fake.requests == []


def test_reask_goes_through_the_same_guard_routing_and_params():
    # A claim truncated to "ok" twice: the re-ask is the third request, and it carries
    # exactly the first request's routing ($0 free routing) and params.
    gen, fake = _or_generator(cost=0.0001, finish="length", reask=True)
    ans = gen.generate("Statins lower LDL cholesterol.", _hits(2))
    assert len(fake.requests) == 3
    first, reask = fake.requests[0], fake.requests[2]
    assert reask["max_tokens"] == REASK_MAX_TOKENS and reask["model"] == first["model"]
    assert reask["extra_body"] == first["extra_body"] == FREE_BODY
    assert reask["temperature"] == first["temperature"] == 0.1
    assert reask["messages"][0]["content"] == REASK_SYSTEM
    assert ans.reask_attempted and ans.cost_usd == pytest.approx(0.0003)  # all 3 reported
