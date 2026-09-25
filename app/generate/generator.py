"""LLM answer generation over the OpenAI-compatible interface.

Points at the endpoint app.core.llm_endpoints resolves for the generator role
(OpenRouter by default, under its spend policy; Groq, Ollama or another OpenAI-compatible
backend via SSR_LLM_PROVIDER=groq + SSR_LLM_BASE_URL, with no code change). Maps the model's [n] citations back to
document ids so answers are traceable to sources, and parses the optional final
verdict line (claims only) into Answer.verdict.

On a reasoning model (gpt-oss; the default free Ling generator also reasons by default)
the hidden reasoning is billed against the same completion budget as the answer, so a
too-small budget truncates the reply — possibly to nothing. generate() reads finish_reason, retries a truncated reply once with a
larger budget, and flags one that is still cut off instead of passing it off as whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import settings
from app.core.llm_endpoints import (
    EmptyCompletionError,
    LLMEndpoint,
    build_client,
    completion_choice,
    endpoint_for_url,
    resolve_endpoint,
    response_cost,
    response_provider,
)
from app.core.interfaces import Answer, SearchHit
from app.generate.prompts import SYSTEM, VERDICTS, build_user_prompt

_CITE = re.compile(r"\[(\d+)\]")
# gpt-oss sometimes cites in its native CJK-bracket style — 【2】, occasionally with a
# browsing-style tail like 【2†L3-L5】 — or with fullwidth square brackets ［2］. \d also
# matches fullwidth digits, which int() reads, so ［２］ becomes [2] too.
_WIDE_CITE = re.compile(r"[【［]\s*(\d+)\s*(?:†[^】］\n]*)?[】］]")
# A well-formed verdict line: the whole line, tolerating the markdown emphasis, quotes,
# trailing period and trailing [n] citation markers models like to add, but nothing else
# (no prose) — anything looser starts matching sentences that merely mention a verdict.
# Citations are allowed because gpt-oss writes "Verdict: REFUTED[1]" routinely: 5 of 8
# unparsed verdicts in the 2026-09-24 run were exactly that, all matching the gold label.
_MARKUP = r"[\s*_`'\"]*"
_VERDICT_ALT = "|".join(v.replace(" ", r"\s+") for v in VERDICTS)
_VERDICT_LINE = re.compile(
    rf"^{_MARKUP}verdict{_MARKUP}:{_MARKUP}({_VERDICT_ALT}){_MARKUP}(?:\s*\[\d+\])*\.?{_MARKUP}$",
    re.IGNORECASE,
)
# Any line that *starts* like a verdict — used to detect a second, competing one.
_VERDICT_LIKE = re.compile(rf"^{_MARKUP}verdict{_MARKUP}:", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    """Rewrite 【n】 / ［n］ citation markers as plain ASCII [n]. Idempotent."""
    return _WIDE_CITE.sub(lambda m: f"[{int(m.group(1))}]", text)


def map_citations(text: str, hits: list[SearchHit]) -> list[str]:
    """Map [n] markers in the answer to hit doc_ids — 1-based, deduped, ordered,
    ignoring out-of-range indices the model may hallucinate. Fullwidth markers are
    normalised first, so they map (and are range-checked) exactly like [n]."""
    cited = sorted({int(n) for n in _CITE.findall(normalize_citations(text))})
    return [hits[n - 1].doc_id for n in cited if 1 <= n <= len(hits)]


def split_verdict(text: str) -> tuple[str, str | None]:
    """Split the optional trailing ``Verdict: ...`` line off an answer.

    Returns (display_text, verdict). The verdict counts only if the LAST non-empty line
    is a well-formed verdict line and no earlier line also looks like one. That is what
    keeps a verdict string quoted from a passage (or planted in one) from being read as
    the model's own: mid-line mentions never match, and an echoed verdict line followed
    by the real one is ambiguous, so it yields None rather than a guess.

    A valid line is stripped from the display text — it is a machine field, surfaced
    separately, and the prose already says the same thing in words. A malformed one is
    left in place, so nothing the model wrote is silently dropped. If the verdict was
    the ENTIRE reply, the text is kept as-is rather than serving an empty answer.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return text, None
    m = _VERDICT_LINE.match(lines[-1])
    if m is None or any(_VERDICT_LIKE.match(line) for line in lines[:-1]):
        return text, None
    verdict = " ".join(m.group(1).upper().split())
    body = "\n".join(lines[:-1]).rstrip()
    return (body or text.strip()), verdict


# Appended to the display text when the reply is still cut off after the retry: a
# half-sentence must not read as a complete answer (an empty one least of all).
TRUNCATION_NOTE = "[Answer truncated: the model ran out of its token budget.]"
TRUNCATION_RETRY_FACTOR = 2  # one retry, at twice the configured budget


@dataclass
class GeneratedAnswer(Answer):
    """An Answer plus how the generation call ended — a side channel for the eval.

    Every field is defaulted, so this is a drop-in Answer (the API response is built
    from the base fields only). Token counts are those of the FINAL attempt and are
    None when the provider does not report them (reasoning_tokens is only reported by
    reasoning models, under usage.completion_tokens_details)."""

    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    attempts: int = 1
    truncated: bool = False
    # OpenRouter's reported cost (USD), summed over ALL attempts (a truncation retry is
    # billed too) — None if any attempt went unreported — and the upstream provider that
    # served the final attempt (None when the backend doesn't say).
    cost_usd: float | None = None
    provider: str | None = None


def resolve_reasoning_effort(model: str, setting: str) -> str | None:
    """The reasoning_effort to send for `model`, or None to omit the parameter.

    See Settings.llm_reasoning_effort: "auto" sends "medium" to gpt-oss only, because an
    OpenAI-compatible backend serving a non-reasoning model may reject the parameter
    outright; ""/"off" never sends it; anything else is an explicit choice, sent as-is.
    Every other model — the default free Ling generator included, which reasons on its
    own and did not shorten its reasoning for effort=low — gets nothing under "auto".
    """
    value = setting.strip().lower()
    if value in {"", "off"}:
        return None
    if value == "auto":
        return "medium" if "gpt-oss" in model.lower() else None
    return value


def _usage_counts(resp) -> tuple[int | None, int | None]:
    """(completion_tokens, reasoning_tokens) from resp.usage, tolerating providers that
    omit usage or its completion_tokens_details breakdown."""
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(usage, "completion_tokens", None), getattr(details, "reasoning_tokens", None)


def reasoning_params(provider: str, effort: str | None) -> dict:
    """The request field that carries `effort`, in the provider's own form: OpenRouter's
    unified ``reasoning: {"effort": ...}`` object (docs: openrouter.ai/docs/use-cases/
    reasoning-tokens), or the top-level ``reasoning_effort`` Groq and Ollama accept.
    Empty when there is nothing to send."""
    if not effort:
        return {}
    if provider == "openrouter":
        return {"reasoning": {"effort": effort}}
    return {"reasoning_effort": effort}


def _choice_or_raise(resp, costs: list[float | None]):
    """completion_choice(resp), with any EmptyCompletionError carrying what this
    generation's calls so far reported costing (None if any is unknown), so a spend
    ceiling can count a failed paid attempt instead of treating it as free."""
    try:
        return completion_choice(resp)
    except EmptyCompletionError as e:
        e.cost_usd = None if any(c is None for c in costs) else sum(costs)
        raise


class LLMGenerator:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
        endpoint: LLMEndpoint | None = None,
    ):
        # Where requests go. An explicit endpoint wins; an explicit base_url builds one
        # whose provider (and default key) follows that URL; otherwise the settings
        # decide (llm_endpoints.resolve_endpoint). `model` overrides the model id only.
        if endpoint is None:
            if base_url is not None:
                endpoint = endpoint_for_url(
                    "generator", base_url, model or settings.llm_model, api_key
                )
            else:
                endpoint = resolve_endpoint("generator", require_key=False)
        if model is not None or api_key is not None:
            endpoint = LLMEndpoint(
                "generator",
                endpoint.provider,
                endpoint.base_url,
                model or endpoint.model,
                endpoint.api_key if api_key is None else api_key,
                endpoint.allowlist,
            )
        endpoint.check()  # spend policy: refuse a disallowed model before any request
        self.endpoint = endpoint
        self.model = endpoint.model
        self.max_completion_tokens = max_completion_tokens or settings.llm_max_completion_tokens
        self.reasoning_effort = resolve_reasoning_effort(
            self.model, settings.llm_reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        self.client = build_client(
            endpoint,
            factory=OpenAI,
            max_retries=5,  # backoff through transient free-tier 429s
            timeout=30.0,  # don't let a hung request tie up a worker for minutes
        )

    def _complete(self, messages: list[dict], max_tokens: int):
        # Reasoning rides in extra_body and only when resolved, in the provider's own
        # form: it is a passthrough, and a backend that doesn't know it must never see
        # it. OpenRouter requests also carry the endpoint's pinned provider routing, and
        # the spend policy is checked here as well as in the client wrapper, so it holds
        # even if self.client is swapped out.
        body = {**reasoning_params(self.endpoint.provider, self.reasoning_effort),
                **self.endpoint.extra_body()}
        self.endpoint.check(model=self.model, body=body)
        extra = {"extra_body": body} if body else {}
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            **extra,
        )

    def generate(self, query: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return GeneratedAnswer(text="No relevant documents were found.", citations=[], hits=[])
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(query, hits)},
        ]
        # finish_reason "length" means the budget ran out mid-reply (on a reasoning model
        # it can run out before the first answer token). One retry at a larger budget
        # recovers the common case at the cost of one extra call, only when needed;
        # retrying further would just burn quota on a reply that is pathologically long.
        budget = self.max_completion_tokens
        resp = self._complete(messages, budget)
        attempts = 1
        costs = [response_cost(resp)]
        choice = _choice_or_raise(resp, costs)
        if choice.finish_reason == "length":
            resp = self._complete(messages, budget * TRUNCATION_RETRY_FACTOR)
            attempts = 2
            costs.append(response_cost(resp))
            choice = _choice_or_raise(resp, costs)
        truncated = choice.finish_reason == "length"
        raw = normalize_citations((choice.message.content or "").strip())
        # A cut-off reply can't have been cut inside a valid verdict line (the whole line
        # must match a full verdict value), so the parse is safe on a truncated reply.
        text, verdict = split_verdict(raw)
        if truncated:
            text = f"{text}\n\n{TRUNCATION_NOTE}" if text else TRUNCATION_NOTE
        completion_tokens, reasoning_tokens = _usage_counts(resp)
        # Citations come from the displayed text, so every listed source is one the
        # reader can see referenced (a valid verdict line carries no citations anyway).
        return GeneratedAnswer(
            text=text,
            citations=map_citations(text, hits),
            hits=hits,
            verdict=verdict,
            finish_reason=choice.finish_reason,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            attempts=attempts,
            truncated=truncated,
            # All-or-nothing: a partly reported cost would undercount, and callers that
            # enforce a spend ceiling treat None as "unknown, assume the worst".
            cost_usd=None if any(c is None for c in costs) else sum(costs),
            provider=response_provider(resp),
        )
