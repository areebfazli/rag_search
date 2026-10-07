"""RAG answer-quality eval: deterministic verdict scoring + LLM-as-judge.

For a random sample of SciFact claims: retrieve -> generate -> score. Three things are
measured, and each comes from the most deterministic source available:

* **Verdict accuracy** (no judge). The product prompt asks for an optional final
  ``Verdict: SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE`` line when the input is a claim;
  the generator parses it into ``Answer.verdict``. Mapped to SUPPORT / CONTRADICT /
  NEI it is scored against the gold SciFact claim label (3-class accuracy + confusion
  matrix). "Abstained with no verdict line" counts as NEI only for a complete reply (one
  that explicitly abstains in prose); a reply cut off at the token budget, or empty, with
  no verdict is NO_VERDICT — always wrong — however the judge read it.
* **Abstention** (no judge where possible). ``answered`` comes from the verdict when
  one was parsed (anything but NOT ENOUGH EVIDENCE is an answer). The judge's
  ``answered`` field is consulted ONLY for replies with no parseable verdict line.
* **Faithfulness and context relevance** (the judge, on every scored query). These have
  no gold label, so they stay LLM-judged. LLM-as-judge agrees with humans ~85-92% of
  the time — treat as signal, not truth. Faithfulness is averaged over answered rows
  except those answered only via the re-ask after a first reply with no text — it hit
  the token budget (or, rarely, was empty) — so the judge saw a placeholder, not an
  answer: ``faithfulness_n`` rows averaged,
  ``faithfulness_unjudged_reask`` left out. Context relevance is over all rows.

"It abstained" is not by itself a good outcome: an over-conservative model looks
identical on abstention rate alone. So ``answered`` is crossed with whether the
evidence was actually retrieved, and scored under two definitions of evidence:

* **rationale oracle (headline)** — a doc the SciFact annotators cited *with rationale
  sentences* is in the top-k. NEI claims have none, so abstaining on them is correct.
* **qrels oracle (legacy)** — any BEIR qrels-relevant doc is in the top-k. BEIR marks a
  cited abstract relevant for NEI claims too (112 of the 300 test claims), so this
  definition scores abstaining on an NEI claim as a *false* abstention. It is kept only
  so earlier published numbers stay traceable and the two can be compared per run.

Generator and judge use different model families, each on the provider its setting names
(app.core.llm_endpoints; default: free Ling 3.0 Flash Sante generator and free Nemotron 3
Ultra judge on OpenRouter, under the spend policy — a $0 run), with a provider-aware
throttle so the run stays under the free-tier limits and a hard per-run spend ceiling
(SSR_RAG_MAX_SPEND_USD) for when a paid generator is configured.

Alongside the aggregates, rag.json keeps every scored query (gold label, rationale
docs, verdict, both evidence flags, answer text, citations, and how generation ended:
finish_reason, token usage, retries) and the run's provenance (prompt hashes, git SHA,
models, generator budget, seed, oracle definition, label source), so a published number
can be traced back to the exact answers and prompts behind it.

Sample, dataset and output (env, SSR_ prefix like the rest of the repo):

* ``SSR_RAG_N`` — claims to sample (default 50; ``all`` = the whole split). The sample is
  a seeded shuffle (SEED) of the split's sorted query ids, cut to the first N, so samples
  NEST: the first 50 of N=300 are exactly the 50-claim sample, and runs of different N
  can be compared on their common prefix.
* ``SSR_RAG_DATASET`` — the split (default settings.eval_dataset, i.e. the canonical
  ``beir/scifact/test``). ``beir/scifact/train`` (809 claims, disjoint from test) is for
  prompt development: retrieval runs over the same corpus, labels come from the same
  source archive.
* ``SSR_EVAL_LIMIT`` — smoke subset: keep only the first n sampled claims.

Canonical-output rule: eval/results/rag.{md,json} (the committed artifact) is written
ONLY by a run that covers the WHOLE canonical test split (SSR_RAG_N=all, or an N at
least the split size), with no SSR_EVAL_LIMIT, whose generator AND judge are the code
defaults (Settings field defaults for the providers and OpenRouter model ids — not .env).
Every other run — the default SSR_RAG_N=50 sample, any other partial sample, any
train-split run, any limited run, any run with another generator or judge (e.g. a paid
openai/gpt-6-luna comparison on all 300 test claims) — writes to
``data/eval_runs/rag_<dataset-slug>_<n>_<prompt-hash8>/``, with ``_<generator-slug>``
(and ``_judge-<judge-slug>``) appended when the models differ (gitignored), says so up
front, and can never touch eval/results/. A canonical run must also be COMPLETE: if any
sampled claim has no row at the end (a skipped pipeline error, an unparseable judge
reply) or a re-ask failed, it refuses to write eval/results/ (the checkpoint and re-ask
cache are kept, so re-running the same command retries only those claims). A
non-canonical run still writes, with the missing claims named in rag.json and rag.md. A
canonical run whose sample size differs from the one recorded in the committed rag.json
(only possible if the split itself changed) prints a loud notice up front and again when
it writes: it is replacing the headline with a different N.

Checkpoint + resume: every completed row is persisted at once (atomic temp file +
os.replace) to ``data/eval_cache/rag/<signature>.json``, where the signature covers
everything that changes a row (dataset, sample size, seed, both endpoints, both prompt
hashes, top_k, mode, token budget, reasoning setting, retrieval settings) and nothing
git-specific. A re-run of the same command skips completed rows, so a provider's daily
cap (Groq tokens/day, OpenRouter's free requests/day) just pauses the run: it stops with
the checkpoint saved and the output dir untouched, and resumes after the quota resets.
Skipped rows (pipeline errors, unparseable judge replies) are never checkpointed, so a
re-run retries them. ``SSR_EVAL_REFRESH=1`` ignores the checkpoint; a corrupt or
wrong-shaped one degrades to recompute. The verdict parser is NOT in the signature: every
row, resumed or fresh, goes through reparse_row, which re-reads a stored reply that has no
verdict with the current generator.parse_verdict (no LLM call), so a resumed run records
what a fresh one would. Each row says which parse produced its verdict (``verdict_source``:
line / inline / stance / None) and, when the reply quotes its evidence, whether the quote
really occurs in the cited passage (``quote_found``; measured only, never a verdict flip).
``python -m app.eval.rag_rescore`` applies the same re-read to a finished rag.json.

Verdict-only re-ask (the product's, SSR_LLM_REASK, default on): a claim whose reply still
has no verdict gets ONE more call (generator.LLMGenerator.reask_verdict, the prompt and
budget rag_secondlook froze and measured). The eval applies it as a post-step over the
finished first-pass rows, never inside the checkpointed generation: the checkpoint keeps
first-pass rows and its signature does not include the re-ask, so an existing checkpoint
resumes as-is. Each reply is cached in ``data/eval_cache/secondlook/<dataset>.json``, the
file rag_secondlook shares (app.eval.reply_cache): a reply either one fetched is a hit for
the other, and a re-run re-asks from the cache with no LLM call. Keys are versioned: new
replies go under the v2 key (reask_key_v2: claim, budget, exact messages AND the endpoint
fingerprint — provider, base URL, model, resolved reasoning effort, temperature policy,
routing), so a Groq or paid run never reuses a reply another endpoint gave. Migration: a
lookup (reask_lookup) tries v2, then the legacy v1 key (model + budget + messages) but only
for the canonical OpenRouter Ling endpoint every v1 entry was fetched with — which is how
the committed rag.json's 15 re-ask replies stay cache hits with zero LLM calls (whether a reply came from the cache is counted in ``run.reask_replies``, not in
the rows, so a resumed run's rows equal a fresh one's). SSR_EVAL_REFRESH ignores the row
checkpoint only, never this cache: its entries include the replies behind the measured
result. A run with SSR_LLM_REASK=false is never canonical (``..._noreask``).

Before any LLM call the run prints the requests still needed (remaining rows x (1
generation + the expected truncation-retry rate + the expected re-ask rate + 1 judge),
plus the re-asks resumed rows still miss in the cache) and the wall-clock estimate.
``--check-quota`` (or ``SSR_RAG_CHECK_QUOTA=1``) also reads the OpenRouter key's remaining
free requests for today (GET /api/v1/key, which is not an LLM call and is not counted
against the quota) and aborts before any LLM call if they are fewer than needed.

Opt-in experiments (all default off; each makes a run non-canonical and names its run dir):

* ``SSR_LLM_PROMPT_VARIANT=finding`` — the product prompt with prompts.FINDING_RULE added.
  prompt_hash() renders the variant's system prompt, so the hash, the checkpoint signature
  and the run dir (``..._<hash8>``) all move with it; the default variant's hash is
  unchanged. The re-ask prompt is the same for every variant, so a re-ask reply already in
  the re-ask cache for the same claim + passages + endpoint is reused (an identical request).
* ``SSR_RAG_SKIP_JUDGE=1`` (eval only) — no judge call at all: judge_answered, faithfulness
  and context_relevance are None, a reply with no verdict even after the re-ask is
  ``answered`` with answered_source "no_judge" and so scores NO_VERDICT (conservative: no
  judge fallback can credit it as an NEI abstention). The aggregates average the judge
  metrics over rows that have them (None here); the request estimate counts 0 judge
  calls. Its signature gains ``judge_skipped`` (only then), its dir ends in ``_nojudge``.
* ``SSR_LLM_VOTES=k`` (k > 1) — the pre-registered self-consistency vote
  (generator.combine_votes, VOTE_RULE). Sample #1 of each claim is its ordinary row under
  the UNCHANGED signature (existing checkpoints and re-ask replies are reused) after the
  re-ask post-step; samples 2..k are drawn afterwards with the same passages and request,
  each with its own re-ask when it has no verdict, and cached in
  ``CACHE/<signature>.samples.json`` (atomic, after every record; nested across k; resumed
  after a daily-cap stop; SSR_EVAL_REFRESH ignores it). The eval draws all k samples (the
  product's early stop would not change the vote, but the full split is the measurement).
  Each row is then the served sample's row (sample #1's whenever it is served, its judge
  scores included; another sample's reply is judged once, cached in its record) plus a
  ``vote`` block; aggregate() adds ``votes`` (splits, ties, changes vs sample #1, each
  sample's single-sample accuracy, and the EXPLORATORY "any disagreement -> NEI" accuracy).
  A sample that fails (not a daily cap / fatal error) is not cached: the claim is voted over
  the samples it has and listed in ``missing_vote_samples`` until a re-run draws it. The
  row-level counts (truncated, re-asked, verdict sources) then describe the SERVED samples.
  The run dir ends in ``_votes<k>``.

Run:
    uv run python -m app.eval.rag_eval
    SSR_RAG_N=all uv run python -m app.eval.rag_eval --check-quota     # full 300-claim test
    SSR_RAG_DATASET=beir/scifact/train SSR_RAG_N=100 uv run python -m app.eval.rag_eval
    SSR_RAG_DATASET=beir/scifact/train SSR_RAG_N=100 SSR_LLM_VOTES=3 uv run python -m app.eval.rag_eval
    uv run python -m app.eval.rag_compare A.json B.json                 # McNemar, paired
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path

from openai import APIStatusError, OpenAI, OpenAIError, RateLimitError

from app.core.config import Settings, settings
from app.core.llm_endpoints import (
    OPENROUTER_BASE_URL,
    EmptyCompletionError,
    LLMEndpoint,
    SpendPolicyError,
    build_client,
    completion_choice,
    cost_upper_bound,
    describe_with_ignored,
    model_id,
    paid_bill_rate,
    paid_max_price,
    resolve_endpoint,
)
from app.core.interfaces import SearchHit, hit_passage
from app.core.paths import RESULTS, assert_outside
from app.generate.generator import (
    BATCH_MAX_RETRIES,
    REASK_MAX_RETRIES,
    REASK_MAX_TOKENS,
    REASK_NOTE,
    TRUNCATION_NOTE,
    VOTE_RULE,
    LLMGenerator,
    combine_votes,
    generation_temperature,
    looks_like_claim,
    map_citations,
    needs_reask,
    parse_reask,
    parse_verdict,
    reask_display_text,
    reask_prompt_hash,
    resolve_reasoning_effort,
)
from app.generate.prompts import (
    SYSTEM,
    VERDICTS,
    build_user_prompt,
    reask_messages,
    system_prompt,
)
from app.ingest.corpus import (
    ClaimLabel,
    load_claim_labels,
    load_documents,
    load_queries_qrels,
    scifact_source_zip,
)
from app.eval.reply_cache import ReplyCache
from app.eval.retrieval_eval import CANONICAL_DATASET, _index_fingerprint
from app.retrieve.service import SearchService

# Generator and judge are different model families, so the judge isn't grading its own
# or a sibling model's output. The ids are what llm_endpoints resolves for the provider
# in use (main() re-resolves them with key + spend-policy checks).
N = 50  # default sample size; SSR_RAG_N overrides (an int, or "all")
SEED = 13  # fixed sample: reproducible, and not just the first N ids in dataset order
GEN_PROVIDER = settings.llm_provider
JUDGE_PROVIDER = settings.judge_provider
GEN_MODEL = model_id("generator")
JUDGE_MODEL = model_id("judge")
TOP_K = 5
MODE = "hybrid"  # the API's default mode — the eval scores what users actually get
# Seconds slept after each query. SSR_RAG_THROTTLE_S overrides; otherwise it depends on
# the providers (default_throttle_s):
# * Groq's free tier caps each model at 8,000 tokens/minute (x-ratelimit-limit-tokens).
#   With reasoning_effort=medium a generation is ~2-3k tokens (prompt + reasoning +
#   answer) and a judge call ~2k, so one query per 15 s overran the generator's bucket
#   and 4 of 50 queries were skipped. 30 s keeps each model under ~6k tokens/minute.
# * OpenRouter's free tier allows 20 requests/min account-wide across ALL `:free`
#   models (paid models are not under that cap). With the default free generator AND free
#   judge, one query is up to 3 free requests (generation, its truncation retry, the
#   judge). The sleep is 4 s per possible free request (12 s/query), and it comes on top
#   of the calls' own latency, so any 60 s window holds at most 5 query starts: <= 15
#   requests even if every generation retries (~10 typical), vs the cap of 20. A paid
#   generator leaves only the judge on the free cap: the 5 s floor, <= 12/min.
_THROTTLE_ENV = os.environ.get("SSR_RAG_THROTTLE_S")
THROTTLE_S: float | None = float(_THROTTLE_ENV) if _THROTTLE_ENV else None
GROQ_THROTTLE_S = 30.0
OPENROUTER_S_PER_FREE_REQUEST = 4.0
OPENROUTER_MIN_THROTTLE_S = 5.0
OPENROUTER_FREE_REQUESTS_PER_MIN = 20  # account-wide, all `:free` models
# A 429 that outlasts the SDK's own retries waits out a full rate-limit window and
# retries the SAME query, rather than skipping it: a skipped query changes the sample,
# and the sample is fixed (seed) so runs stay comparable.
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_S = 60.0
# Groq also caps tokens per DAY (rolling 24 h; 200k for gpt-oss-120b on the free
# tier), and the rate-limit headers don't expose it. Waiting a minute can't clear that,
# so a daily-cap 429 ends the run at once, before anything is written: a partial run
# must never overwrite the committed artifact. Rough per-query cost (prompt + medium
# reasoning + answer, and the judge call), measured on the 2026-09 runs:
EST_GEN_PROMPT_TOKENS = 2500
EST_GEN_COMPLETION_TOKENS = 1000
EST_GEN_TOKENS_PER_QUERY = EST_GEN_PROMPT_TOKENS + EST_GEN_COMPLETION_TOKENS
EST_JUDGE_TOKENS_PER_QUERY = 2000
# Upper bound on one generation prompt (system + 5 abstracts + claim), for the worst-case
# per-query cost the spend ceiling checks BEFORE each query.
MAX_GEN_PROMPT_TOKENS = 4000
# OpenRouter's free tier caps `:free` requests per day (1,000 with >= $10 of credits ever
# bought, 50 otherwise), account-wide. Its daily-cap 429 names the window
# ("free-models-per-day"); it is treated exactly like Groq's tokens-per-day cap.
OPENROUTER_FREE_REQUESTS_PER_DAY = 1000


class DailyTokenBudgetExhausted(RuntimeError):
    """The provider's per-day cap (Groq: tokens, OpenRouter free tier: requests) is hit;
    retrying within the run is pointless."""


class SpendCeilingReached(RuntimeError):
    """The next query could take the run's reported spend past SSR_RAG_MAX_SPEND_USD."""


def _is_daily_cap(e: Exception) -> bool:
    msg = str(e).lower()
    if "per day" in msg or "(tpd)" in msg or "per-day" in msg:
        return True
    # OpenRouter may put the window only in its rate-limit headers: the per-minute free
    # limit is 20, so an exhausted limit above that is the daily one.
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        limit = int(headers.get("x-ratelimit-limit", 0))
        remaining = int(headers.get("x-ratelimit-remaining", 1))
    except (TypeError, ValueError):
        return False
    return remaining == 0 and limit > 20


def _is_fatal(e: Exception) -> bool:
    """Errors every later query would hit too: skipping them would just write an empty
    or thinned artifact. A spend-policy refusal, bad key (401) or no credits (402)."""
    if isinstance(e, SpendPolicyError):
        return True
    return isinstance(e, APIStatusError) and e.status_code in (401, 402)


def free_requests_per_query(
    gen: LLMEndpoint, judge: LLMEndpoint, judge_skipped: bool = False
) -> tuple[int, int]:
    """(typical, worst) requests one query makes against OpenRouter's free-tier caps:
    the judge's one call if it is a free OpenRouter model (none with the judge skipped),
    plus 1 for a free OpenRouter generator — 2 in the worst case (the truncation retry)."""
    judge_free = int(judge.provider == "openrouter" and not judge.paid and not judge_skipped)
    gen_free = int(gen.provider == "openrouter" and not gen.paid)
    return judge_free + gen_free, judge_free + 2 * gen_free


def default_throttle_s(
    gen: LLMEndpoint | None = None, judge: LLMEndpoint | None = None, judge_skipped: bool = False
) -> float:
    """Per-query sleep for this provider mix (see THROTTLE_S); a skipped judge makes no
    request, so neither its provider nor its free-cap share counts."""
    if THROTTLE_S is not None:
        return THROTTLE_S
    gen = gen or resolve_endpoint("generator", require_key=False)
    judge = judge or resolve_endpoint("judge", require_key=False)
    if "groq" in (gen.provider, *(() if judge_skipped else (judge.provider,))):
        return GROQ_THROTTLE_S
    _, worst = free_requests_per_query(gen, judge, judge_skipped)
    return max(OPENROUTER_MIN_THROTTLE_S, worst * OPENROUTER_S_PER_FREE_REQUEST)


# Free OpenRouter requests one vote sample can make at worst (generation + its truncation
# retry; its own re-ask is rare): the per-sample sleep in the vote-sample pass is that x
# OPENROUTER_S_PER_FREE_REQUEST (8 s), like a query's (SSR_RAG_THROTTLE_S overrides).
VOTE_SAMPLE_WORST_FREE_REQUESTS = 2
# Expected share of claims whose vote serves a sample other than #1, so that sample gets
# its own (single) judge call: for the up-front estimate only.
EXPECTED_VOTE_REJUDGE_RATE = 0.15


def vote_sample_throttle_s(gen: LLMEndpoint) -> float:
    """Sleep after each drawn vote sample (and each vote re-judge)."""
    if THROTTLE_S is not None:
        return THROTTLE_S
    if gen.provider == "groq":
        return GROQ_THROTTLE_S
    if gen.provider == "openrouter" and not gen.paid:
        return VOTE_SAMPLE_WORST_FREE_REQUESTS * OPENROUTER_S_PER_FREE_REQUEST
    return OPENROUTER_MIN_THROTTLE_S


def worst_case_generation_cost(gen: LLMEndpoint) -> float:
    """Most one query's generation can cost: both attempts (the truncation retry runs
    at 2x the budget) at THIS generator's billing bound (llm_endpoints.paid_bill_rate),
    with a generous prompt. 0 unless paid."""
    if not gen.paid:
        return 0.0
    budget = settings.llm_max_completion_tokens
    return cost_upper_bound(MAX_GEN_PROMPT_TOKENS, budget, gen.model) + cost_upper_bound(
        MAX_GEN_PROMPT_TOKENS, budget * 2, gen.model
    )


def typical_generation_cost(gen: LLMEndpoint) -> float:
    """A typical query's generation at this generator's billing bound (an upper-bound
    estimate: the endpoint actually served may bill less)."""
    if not gen.paid:
        return 0.0
    return cost_upper_bound(EST_GEN_PROMPT_TOKENS, EST_GEN_COMPLETION_TOKENS, gen.model)


OUT = RESULTS  # canonical run only — the committed artifact (repo-anchored, not cwd)
RUNS = Path("data/eval_runs")  # every other run (gitignored)
CACHE = Path("data/eval_cache/rag")  # per-row resume checkpoints (gitignored)
# Re-ask replies (one file per dataset), in rag_secondlook's ReplyCache format and keys
# (v2, with the legacy-key migration in reask_lookup): the replies its measured post-hoc
# re-ask fetched are cache hits here, and vice versa.
REASK_CACHE = Path("data/eval_cache/secondlook")
# Expected share of claims the re-ask fires on, for the up-front request estimate only:
# 16 of 300 on the committed test run (15 before 3e360cc's scoring rules). A checkpoint's
# own rate wins once it is big enough.
EXPECTED_REASK_RATE = 0.05

# Expected share of generations that need the truncation retry, for the up-front request
# estimate only: 12 of 50 on the committed free-Ling run (retried_answers in rag.json).
# Once a checkpoint holds at least MIN_ROWS_FOR_OBSERVED_RETRY rows, their own rate wins.
EXPECTED_RETRY_RATE = 0.24
MIN_ROWS_FOR_OBSERVED_RETRY = 10
# Rough allowance for the calls' own latency per claim (generation incl. hidden reasoning,
# plus the judge), on top of the throttle sleep. An assumption for the estimate, not a
# measurement.
EST_QUERY_LATENCY_S = 8.0
# OpenRouter's key-info endpoint: `data.free_model_daily_requests.{used,limit,remaining}`
# for the current UTC day (docs: openrouter.ai/docs/api/reference/limits). Only ever sent
# the OpenRouter key.
OPENROUTER_KEY_URL = OPENROUTER_BASE_URL + "/key"

# What `evidence` means. The headline is the rationale oracle; the qrels one is computed
# alongside it in the same run so the two can be compared on identical answers.
ORACLE = "rationale"
ORACLE_DEFINITIONS = {
    "rationale": (
        "evidence = a doc the SciFact annotators cited with rationale sentences (SUPPORT "
        "or CONTRADICT) is in the retrieved top-k; NEI claims have none, so abstaining "
        "is the correct action for them"
    ),
    "qrels": (
        "legacy: evidence = any BEIR qrels-relevant doc is in the retrieved top-k; BEIR "
        "marks a cited abstract relevant for NEI claims too, so abstaining on an NEI "
        "claim is scored as a false abstention"
    ),
}
LABELS = ("SUPPORT", "CONTRADICT", "NEI")
VERDICT_TO_LABEL = dict(zip(VERDICTS, LABELS, strict=True))  # SUPPORTED->SUPPORT, ...
NO_VERDICT = "NONE"  # answered, but with no parseable verdict line: maps to no label
# How a reply with no verdict is scored (predicted_label), recorded in run metadata so
# rag.md describes the rule its rag.json was scored under. A rag.json without the field
# predates it: an abstaining reply with no verdict counted as NEI even if cut off / empty.
NO_VERDICT_SCORING = "nei_only_if_complete"

# The judge's `answered` field is only used as a fallback (no verdict line parsed), but
# its rubric still has to be unambiguous for claims: a rebuttal ("the context refutes
# this") IS an answer, which the old "false if it states the information is not
# present" wording let a small judge read either way on CONTRADICT claims.
JUDGE_SYSTEM = (
    "You are a strict RAG evaluator. Given a QUESTION (which may be a claim to check), the "
    "CONTEXT passages retrieved, and the ANSWER generated, return ONLY a JSON object:\n"
    '{"answered": <true if the answer attempts to answer the question or assess the claim '
    "using the context, INCLUDING saying the context refutes or contradicts the claim; false "
    'only if it says the context does not contain enough information>, "faithfulness": '
    "<float 0-1, fraction of the answer's factual claims that are directly supported by the "
    'context>, "context_relevance": <float 0-1, how relevant the context is to the question>}'
)


_SCORE_KEYS = frozenset({"answered", "faithfulness", "context_relevance"})


class JudgeParseError(RuntimeError):
    """The judge returned something we could not read as JSON."""


def _parse(s: str) -> dict:
    """Extract the first JSON object in the response that carries a score field.

    A greedy ``\\{.*\\}`` span runs from the first brace to the last, so any prose
    containing another brace after the object makes the whole parse fail and silently
    drops the query. raw_decode stops at the end of the first valid object.

    Requiring an expected key matters as much as the decode: a judge that wraps its
    verdict (``{"result": {"answered": true, ...}}``) would otherwise return the
    *outer* object, whose ``.get("answered")`` is None — scoring a real answer as an
    abstention with zero faithfulness, and doing it silently, since a successful parse
    never increments the failure counter. Skipping wrappers finds the inner object.
    """
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and _SCORE_KEYS & obj.keys():
                return obj
    raise JudgeParseError(s[:120])


def _as_bool(v: object) -> bool:
    """Coerce the judge's ``answered`` field explicitly.

    A small judge model routinely emits ``"answered": "false"`` as a JSON *string*,
    and ``bool("false")`` is True — which would count a correct abstention as an
    answer and corrupt both headline metrics at once.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def _as_score(v: object) -> float:
    """Coerce a 0-1 judge score; ``null`` and out-of-range values are common.

    The rule is that malformed output must never *inflate* a published number, so
    every unusable value resolves to 0.0. Two cases get there by surprising routes:

    * NaN must be rejected before the clamp, not by it: ``min(1.0, nan)`` is ``nan``
      and ``max(0.0, nan)`` is ``1.0``, so a NaN would become a *perfect* score.
      json.loads accepts bare ``NaN``, so the judge can really emit one.
    * ``bool`` is a subclass of ``int`` and ``float(True)`` is ``1.0``. The judge is
      already emitting a boolean for the adjacent ``answered`` field, so
      ``"faithfulness": true`` is a plausible slip — and it would score as perfect.
    """
    if isinstance(v, bool):
        return 0.0
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return max(0.0, min(1.0, f))


def judge(client: OpenAI, model: str, q: str, contexts: list[str], answer: str) -> dict:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"QUESTION: {q}\n\nCONTEXT:\n{ctx}\n\nANSWER: {answer}"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    d = _parse(completion_choice(resp).message.content or "")
    return {
        "answered": _as_bool(d.get("answered", False)),
        "faithfulness": _as_score(d.get("faithfulness", 0.0)),
        "context_relevance": _as_score(d.get("context_relevance", 0.0)),
    }


def _abstention_class(answered: bool, evidence: bool) -> str:
    """Which quadrant of the answered x evidence table a query falls in."""
    if answered:
        return "answered_with_evidence" if evidence else "answered_without_evidence"
    return "false_abstention" if evidence else "correct_abstention"


QUADRANTS = (
    "answered_with_evidence",
    "answered_without_evidence",
    "false_abstention",
    "correct_abstention",
)


def evidence_flags(retrieved: list[str], label: ClaimLabel, gold_qrels: set[str]) -> dict:
    """Both definitions of "the evidence was retrieved", for one claim.

    ``evidence`` (rationale oracle) needs a rationale doc in the top-k, so it is False
    for every NEI claim however much was retrieved. ``evidence_qrels`` is the legacy
    definition: any qrels-relevant doc, which NEI claims have too.
    """
    top = set(retrieved)
    return {
        "evidence": bool(label.rationale_doc_ids & top),
        "evidence_qrels": bool(gold_qrels & top),
    }


def resolve_answered(verdict: str | None, judge_answered: bool | None) -> tuple[bool, str]:
    """(answered, source). A parsed verdict decides deterministically — anything but
    NOT ENOUGH EVIDENCE is an attempt, a rebuttal included. Only a reply with no
    parseable verdict line falls back to the judge's call. With no judge call at all
    (judge_answered None: SSR_RAG_SKIP_JUDGE, or a vote sample nobody judged) such a reply
    counts as answered, source "no_judge" — so predicted_label scores it NO_VERDICT
    (always wrong): conservative, never credited as an abstention on an NEI claim."""
    if verdict is not None:
        return verdict != "NOT ENOUGH EVIDENCE", "verdict"
    if judge_answered is None:
        return True, "no_judge"
    return judge_answered, "judge"


def predicted_label(verdict: str | None, answered: bool, cut_off: bool = False) -> str:
    """Map a verdict onto the gold label space. No verdict + answered is NO_VERDICT,
    which never matches a gold label, so a missing verdict can't be scored as correct.
    No verdict + abstained is NEI (it is exactly "not enough info") ONLY when the reply
    is complete: a reply cut off at the token budget, or empty (``cut_off``), abstained
    from nothing — the judge's "not answered" there describes a broken reply, not the
    model's call — so it is NO_VERDICT too and can never be credited on an NEI claim."""
    if verdict is not None:
        return VERDICT_TO_LABEL[verdict]
    return NO_VERDICT if answered or cut_off else "NEI"


def _broken_reply(truncated: bool, answer: str | None) -> bool:
    """The reply is cut off (hit the token budget after the retry) or has no content:
    predicted_label's ``cut_off``."""
    return truncated or (answer or "").strip().endswith(TRUNCATION_NOTE) or _no_judgeable_reply(answer)


def reparse_row(row: dict, query: str) -> dict:
    """Re-apply the generator's verdict parser to a row that has no verdict.

    A checkpointed row keeps the answer text generate() served: a verdict-line reply
    has its line stripped (and its verdict stored), every other reply is kept whole. So
    a row stored with no verdict can be parsed again with the CURRENT parser, without an
    LLM call — which is what lets a resumed run, or an offline re-score of a committed
    rag.json, reproduce what a fresh generation would now record. A row that already
    has a verdict is left alone (its source is "line" if it predates the field) — except
    a first-sentence "stance" verdict on a TRUNCATED reply, which generate() no longer
    reads (a cut-off opening may have been about to be hedged): that row is re-parsed
    with the stance fallback off, as a fresh generation would be.

    Everything derived from the verdict is recomputed: answered (the verdict decides
    it, else the judge), predicted label (a cut-off or empty reply with no verdict is
    NO_VERDICT), both abstention classes, and the cited doc ids (from the new display
    text, plus the citations on a recovered inline verdict, as generate() does). The
    judge's own scores are kept: they were given for the same reply.
    """
    answer = row.get("answer") or ""
    truncated = bool(row.get("truncated")) or answer.endswith(TRUNCATION_NOTE)
    if row.get("verdict") is not None and not (truncated and row.get("verdict_source") == "stance"):
        return {**row, "verdict_source": row.get("verdict_source") or "line"}
    note = f"\n\n{TRUNCATION_NOTE}"
    truncated_note = answer.endswith(TRUNCATION_NOTE)
    raw = answer[: -len(note)] if answer.endswith(note) else (
        "" if answer == TRUNCATION_NOTE else answer
    )
    text, verdict, source = parse_verdict(raw, allow_stance=looks_like_claim(query) and not truncated)
    answered, answered_source = resolve_answered(verdict, row["judge_answered"])
    derived = {
        "verdict": verdict,
        "verdict_source": source,
        "predicted_label": predicted_label(verdict, answered, _broken_reply(truncated, raw)),
        "answered": answered,
        "answered_source": answered_source,
        "abstention_class": _abstention_class(answered, row["evidence"]),
        "abstention_class_qrels": _abstention_class(answered, row["evidence_qrels"]),
    }
    if verdict is None:  # the text is the reply as stored: answer and citations stay
        if row.get("verdict") is None:  # same reply, same judge call: only the label rule
            return {**row, "verdict_source": None, "predicted_label": derived["predicted_label"]}
        return {**row, **derived}  # a demoted stance verdict: the judge decides again
    if truncated_note:
        text = f"{text}\n\n{TRUNCATION_NOTE}" if text else TRUNCATION_NOTE
    retrieved = [SearchHit(d, 0.0, "") for d in row.get("retrieved_doc_ids") or []]
    cite_text = raw if source in ("line", "inline") else text
    return {
        **row,
        **derived,
        "answer": text,
        "cited_doc_ids": map_citations(cite_text, retrieved) if retrieved else row.get("cited_doc_ids", []),
    }


# --- the verdict-only re-ask, applied to a finished first-pass row -----------------------
#
# The product re-asks inside LLMGenerator.generate(). The eval keeps first-pass rows (what
# the checkpoint stores, so its signature and every stored row are untouched by the
# re-ask) and applies the SAME re-ask — same trigger (generator.needs_reask), same request
# (prompts.reask_messages, REASK_MAX_TOKENS, sent by LLMGenerator.reask_verdict), same
# parse and display rule — as a post-step over the rows, with each reply cached in
# REASK_CACHE. A resumed or re-run eval therefore re-asks from the cache, never twice.

# The fields the re-ask may overwrite; their first-pass values are kept in "first_pass".
FIRST_PASS_KEYS = (
    "verdict", "verdict_source", "predicted_label", "answered", "answered_source",
    "abstention_class", "abstention_class_qrels", "answer", "cited_doc_ids",
)


def reask_cache_path(dataset: str) -> Path:
    return REASK_CACHE / f"{dataset.replace('/', '-')}.json"


def reask_hits(doc_ids: Sequence[str], docs: Mapping[str, Mapping]) -> list[SearchHit]:
    """The passages exactly as the generator got them: corpus title + text, rank order
    (SearchService hits carry the same corpus text and title)."""
    return [SearchHit(d, 0.0, docs[d]["text"], {"title": docs[d]["title"]}) for d in doc_ids]


def reask_key(qid: str, model: str, messages: Sequence[Mapping]) -> str:
    """The LEGACY (v1) re-ask key: model id + budget + messages, nothing about who served
    it. Read only as a migration fallback (reask_lookup); never written any more."""
    return ReplyCache.key("reask", qid, model, REASK_MAX_TOKENS, messages)


def reask_request_fingerprint(gen: LLMEndpoint) -> dict:
    """Everything about the endpoint that can change a re-ask reply besides the messages
    and budget: provider, base URL, model id, the reasoning effort LLMGenerator resolves
    for it (from settings.llm_reasoning_effort, as main() builds it), the temperature
    policy (generator.generation_temperature; None = not sent) and the provider routing
    every call carries (endpoint.extra_body()). Never the key."""
    return {
        "provider": gen.provider,
        "base_url": gen.base_url,
        "model": gen.model,
        "reasoning_effort": resolve_reasoning_effort(gen.model, settings.llm_reasoning_effort),
        "temperature": generation_temperature(gen.model),
        "extra_body": gen.extra_body(),
    }


def reask_key_v2(qid: str, gen: LLMEndpoint, messages: Sequence[Mapping]) -> str:
    """The v2 re-ask key (reply_cache.ReplyCache.key_v2 over reask_request_fingerprint):
    the only key new re-ask replies are written under."""
    return ReplyCache.key_v2("reask", qid, reask_request_fingerprint(gen), REASK_MAX_TOKENS, messages)


# The one endpoint every legacy (v1) re-ask entry was fetched with — frozen here as
# literals, not read from the live code defaults, so a later change to the default
# temperature or free routing stops matching the old entries instead of silently
# inheriting them. The model id is the code-default generator (default_endpoints()).
LEGACY_REASK_TEMPERATURE = 0.1
LEGACY_REASK_EXTRA_BODY = {
    "provider": {"allow_fallbacks": False, "max_price": {"prompt": 0, "completion": 0}},
}


def is_legacy_reask_endpoint(gen: LLMEndpoint) -> bool:
    """True only for the canonical OpenRouter Ling endpoint the v1 re-ask entries came
    from: provider openrouter at OPENROUTER_BASE_URL, the code-default generator id, no
    reasoning field sent, temperature 0.1, the free routing."""
    fp = reask_request_fingerprint(gen)
    return (
        fp["provider"] == "openrouter"
        and fp["base_url"] == OPENROUTER_BASE_URL
        and (fp["provider"], fp["model"]) == default_endpoints()[0]
        and fp["reasoning_effort"] is None
        and fp["temperature"] == LEGACY_REASK_TEMPERATURE
        and fp["extra_body"] == LEGACY_REASK_EXTRA_BODY
    )


def reask_lookup(
    cache: ReplyCache, qid: str, gen: LLMEndpoint, messages: Sequence[Mapping]
) -> dict | None:
    """The cached re-ask reply for this request, or None. The v2 key first; then — key
    migration — the legacy v1 key, but ONLY when ``gen`` is exactly the canonical endpoint
    the v1 entries were fetched with (is_legacy_reask_endpoint). Any other endpoint (a
    Groq or paid run with the same model id and messages) never reads a v1 entry."""
    rec = cache.get(reask_key_v2(qid, gen, messages))
    if rec is None and is_legacy_reask_endpoint(gen):
        rec = cache.get(reask_key(qid, gen.model, messages))
    return rec


def _no_judgeable_reply(answer: str | None) -> bool:
    """The first reply carried nothing the judge could score: empty, or only the
    truncation note (generator.reask_display_text's REASK_NOTE condition)."""
    return (answer or "").strip() in ("", TRUNCATION_NOTE)


def judge_scored_placeholder(row: Mapping) -> bool:
    """A row answered via the re-ask whose FIRST reply had no judgeable content: the judge
    scored a placeholder (faithfulness 0.0 by construction), not the answer shown. Read
    from the flag apply_reask sets, else — rows written before the flag existed — from
    the displayed REASK_NOTE or the first-pass answer under ``first_pass``."""
    if "judge_scored_placeholder" in row:
        return bool(row["judge_scored_placeholder"])
    if row.get("verdict_source") != "reask":
        return False
    if row.get("answer") == REASK_NOTE:
        return True
    fp = row.get("first_pass")
    return isinstance(fp, Mapping) and _no_judgeable_reply(fp.get("answer"))


def apply_reask(row: Mapping, rec: Mapping | None, error: str | None = None) -> dict:
    """The row after the re-ask: every row it fired on records the reply (or the error);
    a parsed verdict replaces the first-pass one (verdict_source "reask", answered
    decided by it), with the displayed text per generator.reask_display_text and the
    first-pass values of every changed field under "first_pass". The judge's scores
    stay those of the first reply (not re-judged); ``judge_scored_placeholder`` marks a
    row whose first reply had no judgeable content (empty / the truncation note), so
    its faithfulness scored a placeholder and aggregate() leaves it out of the mean."""
    out = {**row, "reask_attempted": True}
    if rec is None:
        out.update(reask=None, reask_error=error)
        return out
    verdict, how = parse_reask(rec.get("raw"))
    out["reask"] = {
        "raw": rec.get("raw"), "parsed": verdict, "parse_source": how,
        "finish_reason": rec.get("finish_reason"),
        "completion_tokens": rec.get("completion_tokens"),
        "reasoning_tokens": rec.get("reasoning_tokens"),
        "cost_usd": rec.get("cost_usd"), "provider": rec.get("provider"),
    }
    if verdict is None:
        return out
    answered = verdict != "NOT ENOUGH EVIDENCE"
    first_answer = row.get("answer") or ""
    text = reask_display_text(first_answer)
    # Citations of the DISPLAYED text over the retrieved passages, as reparse_row does
    # (today the same as keeping the first reply's when the text is unchanged, and [] for
    # REASK_NOTE; robust if reask_display_text starts rewriting the prose).
    retrieved = [SearchHit(d, 0.0, "") for d in row.get("retrieved_doc_ids") or []]
    cited = (
        map_citations(text, retrieved) if retrieved
        else (row.get("cited_doc_ids") if text == row.get("answer") else [])
    )
    out.update(
        first_pass={k: row.get(k) for k in FIRST_PASS_KEYS},
        verdict=verdict,
        verdict_source="reask",
        predicted_label=predicted_label(verdict, answered),
        answered=answered,
        answered_source="reask",
        abstention_class=_abstention_class(answered, row["evidence"]),
        abstention_class_qrels=_abstention_class(answered, row["evidence_qrels"]),
        answer=text,
        cited_doc_ids=cited,
        # The judge saw the first reply, which here had nothing to judge: its faithfulness
        # (0.0) scores a placeholder, so aggregate() leaves the row out of the mean.
        judge_scored_placeholder=_no_judgeable_reply(first_answer),
    )
    return out


def first_pass_row(row: Mapping) -> dict:
    """A row as the first pass left it (the re-ask undone): what the checkpoint stores and
    what experiments built on the pre-re-ask answers (rag_secondlook) read."""
    out = {k: v for k, v in row.items() if k not in ("first_pass", "reask", "reask_attempted",
                                                     "reask_error", "judge_scored_placeholder")}
    out.update(row.get("first_pass") or {})
    return out


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def prompt_hash(variant: str | None = None) -> str:
    """sha256 of the generator's system prompt + user-prompt template.

    The template is a function, so it is hashed by *rendering* it on fixed placeholders
    rather than hashing its source: an edit to the wording or layout the model sees
    changes the hash, while a comment or refactor that leaves the prompt identical
    does not. (A change confined to the question sanitizer won't move it — that alters
    only how real queries are cleaned, not the template.)

    The system prompt is the one ``variant`` names (default: settings.llm_prompt_variant,
    as LLMGenerator sends it); the "default" variant is SYSTEM itself, so its hash — and
    every checkpoint signature built on it — is unchanged.
    """
    variant = settings.llm_prompt_variant if variant is None else variant
    system = SYSTEM if variant == "default" else system_prompt(variant)
    ph = SearchHit("{doc_id}", 0.0, "{text}", {"title": "{title}"})
    return _sha256(system + "\n\x00\n" + build_user_prompt("{question}", [ph, ph]))


def _git_sha() -> str | None:
    """HEAD commit, or None outside a git checkout (e.g. the Docker image)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


_DEFAULT_RERANKER = Settings.model_fields["reranker_model"].default


def reranker_in_effect(mode: str) -> str | None:
    """The cross-encoder a run of `mode` reranks with, or None if it doesn't rerank.

    SearchService builds CrossEncoderReranker() lazily from settings.reranker_model,
    and that default (MiniLM) is the reranker that measurably HURTS on SciFact (-0.027
    nDCG@10 vs plain hybrid). A rerank run must record which model it scored.
    """
    return settings.reranker_model if mode == "hybrid_rerank" else None


def run_metadata(
    gen: LLMEndpoint | None = None,
    judge_ep: LLMEndpoint | None = None,
    dataset: str | None = None,
    n_requested: int | str | None = None,
    judge_skipped: bool = False,
    votes: Mapping | None = None,
) -> dict:
    """Provenance for rag.json: enough to tell whether two runs are comparable. Records
    each role's provider, base URL, model id and the provider fields sent — never a key.
    ``votes``: the self-consistency vote block (k > 1 only; absent otherwise)."""
    gen = gen or resolve_endpoint("generator", require_key=False)
    judge_ep = judge_ep or resolve_endpoint("judge", require_key=False)
    effort = resolve_reasoning_effort(gen.model, settings.llm_reasoning_effort)
    dataset = dataset or settings.eval_dataset
    return {
        "git_sha": _git_sha(),
        "dataset": dataset,
        "prompt_hash": prompt_hash(),
        # Which product system prompt (prompts.PROMPT_VARIANTS) the hash above is of.
        "prompt_variant": settings.llm_prompt_variant,
        "judge_prompt_hash": _sha256(JUDGE_SYSTEM),
        # SSR_RAG_SKIP_JUDGE: no judge call at all (no faithfulness / context relevance; a
        # reply with no verdict scores NO_VERDICT).
        "judge_skipped": judge_skipped,
        "generator_model": gen.model,
        "generator_provider": gen.provider,
        "generator_base_url": gen.base_url,
        "generator_extra_body": gen.extra_body(),
        # The generator's budget and reasoning effort decide whether answers truncate
        # (a truncated reply carries no verdict), so they are part of what a number means.
        "generator_max_completion_tokens": settings.llm_max_completion_tokens,
        "generator_reasoning_effort": effort,
        # None when no reasoning field is sent at all (e.g. the default free Ling model).
        "generator_reasoning_param": (
            None if effort is None
            else "reasoning.effort" if gen.provider == "openrouter" else "reasoning_effort"
        ),
        # None when the model is sent no temperature at all (e.g. openai/gpt-6-luna).
        "generator_temperature": generation_temperature(gen.model),
        # Both roles are the code defaults: the only runs that may be canonical.
        "default_models": not non_default_roles(gen, judge_ep),
        "judge_model": judge_ep.model,
        "judge_provider": judge_ep.provider,
        "judge_base_url": judge_ep.base_url,
        "judge_extra_body": judge_ep.extra_body(),
        "mode": MODE,
        "reranker_model": reranker_in_effect(MODE),
        "top_k": TOP_K,
        "n_requested": N if n_requested is None else n_requested,
        "sample_seed": SEED,
        "oracle": ORACLE,
        "oracle_definition": ORACLE_DEFINITIONS[ORACLE],
        "legacy_oracle": "qrels",
        "legacy_oracle_definition": ORACLE_DEFINITIONS["qrels"],
        "label_source": {
            "loader": "app.ingest.corpus.load_claim_labels",
            "file": "scifact/queries.jsonl `metadata` field, BEIR SciFact source.zip",
            "dataset": dataset,
            "source_zip_sha256": _file_sha256(scifact_source_zip()),
        },
        "answered_source": (
            "verdict line when parsed (answered = verdict != NOT ENOUGH EVIDENCE); the "
            "judge's `answered` field only when no verdict line was parsed"
        ),
        "no_verdict_scoring": NO_VERDICT_SCORING,
        "reask": {
            "enabled": settings.llm_reask,
            "prompt_hash": reask_prompt_hash(),
            "max_tokens": REASK_MAX_TOKENS,
            "trigger": "a claim (not a question) whose reply has no parseable verdict",
            "judged": "not re-judged: faithfulness / context relevance are the first reply's",
        },
        **({"votes": dict(votes)} if votes else {}),
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _abstention(rows: list[dict], ev_key: str) -> dict:
    """Abstention scored against one evidence definition (`ev_key`)."""
    answered = [r for r in rows if r["answered"]]
    with_ev = [r for r in rows if r[ev_key]]
    without_ev = [r for r in rows if not r[ev_key]]
    abstained = [r for r in rows if not r["answered"]]
    # The two error quadrants: abstaining despite retrieving the evidence, and
    # answering when no evidence was retrieved (the hallucination risk).
    false_abstentions = [r for r in abstained if r[ev_key]]
    answered_no_ev = [r for r in answered if not r[ev_key]]
    return {
        "evidence_rate": _rate(len(with_ev), len(rows)),
        "abstention_precision": _rate(len(abstained) - len(false_abstentions), len(abstained)),
        "abstention_recall": _rate(
            len([r for r in without_ev if not r["answered"]]), len(without_ev)
        ),
        "false_abstention_rate": _rate(len(false_abstentions), len(with_ev)),
        "answered_without_evidence_rate": _rate(len(answered_no_ev), len(answered)),
        "quadrants": {
            q: sum(_abstention_class(r["answered"], r[ev_key]) == q for r in rows)
            for q in QUADRANTS
        },
    }


def verdict_scores(rows: list[dict]) -> dict:
    """3-class accuracy of the predicted label against gold, plus the confusion matrix
    (gold label -> predicted label -> count; NO_VERDICT is an extra predicted column)."""
    confusion = {g: {p: 0 for p in (*LABELS, NO_VERDICT)} for g in LABELS}
    for r in rows:
        confusion[r["gold_label"]][r["predicted_label"]] += 1
    correct = sum(r["predicted_label"] == r["gold_label"] for r in rows)
    parsed = sum(r["verdict"] is not None for r in rows)
    return {
        "verdict_accuracy": _rate(correct, len(rows)),
        "verdict_parsed_rate": _rate(parsed, len(rows)),
        "confusion": confusion,
    }


_QUOTE_UNKNOWN = object()  # the row has no `quote_found` key at all


def _quote_stats(rows: list[dict]) -> dict:
    """Verdict accuracy split by the evidence-quote check. ``no_quote`` only where the
    check ran and found no quote (key present, None); a row with no ``quote_found`` key
    (resumed / legacy rows written before the check existed) is ``unknown``, not
    no_quote."""
    out = {}
    for key, want in (("found", True), ("not_found", False), ("no_quote", None),
                      ("unknown", _QUOTE_UNKNOWN)):
        sub = [r for r in rows if r.get("quote_found", _QUOTE_UNKNOWN) is want]
        out[key] = {
            "n": len(sub),
            "verdict_accuracy": _rate(sum(r["predicted_label"] == r["gold_label"] for r in sub), len(sub)),
        }
    return out


def aggregate(
    rows: list[dict],
    gen_model: str | None = None,
    judge_model: str | None = None,
    gen_provider: str | None = None,
    judge_provider: str | None = None,
    judge_skipped: bool = False,
) -> dict:
    """Cross the answered/abstained call with whether evidence was actually retrieved,
    so abstention is scored rather than assumed correct — under the rationale oracle
    (headline, top-level keys) and the legacy qrels oracle (``qrels_oracle``).

    Judge fields may be None (SSR_RAG_SKIP_JUDGE, or a vote's chosen sample whose judge
    call failed): the faithfulness / context-relevance means and the judge-verdict
    agreement are then over the rows that have a judge value. ``judge_skipped`` adds the
    keys a no-judge run reports; rows with ``vote`` blocks add ``votes`` (_vote_stats)."""
    headline = _abstention(rows, "evidence")
    answered = [r for r in rows if r["answered"]]
    with_verdict = [r for r in rows if r["verdict"] is not None]
    # Faithfulness is averaged over answered rows whose judged reply is the answer shown.
    # A row answered only via the re-ask whose first reply had no content (empty / the
    # truncation note, displayed as REASK_NOTE) was judged on that placeholder — its 0.0
    # says nothing about the answer — so it is counted apart, not averaged in.
    unjudged = [r for r in answered if judge_scored_placeholder(r)]
    judged = [r for r in answered if not judge_scored_placeholder(r)
              and r.get("faithfulness") is not None]
    ctx_rows = [r for r in rows if r.get("context_relevance") is not None]
    judge_rows = [r for r in with_verdict if r.get("judge_answered") is not None]
    extra: dict = {}
    if judge_skipped:
        extra["judge_skipped"] = True
        # Replies with no verdict even after the re-ask: no judge fallback, scored NO_VERDICT.
        extra["no_judge_no_verdict"] = sum(r.get("answered_source") == "no_judge" for r in rows)
    if (votes := _vote_stats(rows)) is not None:
        extra["votes"] = votes
    return {
        "n": len(rows),
        "evidence_rate": headline.pop("evidence_rate"),
        "answered_rate": _rate(len(answered), len(rows)),
        **{k: v for k, v in headline.items() if k != "quadrants"},
        "oracle": ORACLE,
        "quadrants": headline["quadrants"],
        "qrels_oracle": _abstention(rows, "evidence_qrels"),
        **verdict_scores(rows),
        # How often the judge still decided `answered`, and — where a verdict decided
        # it instead — how often the judge would have agreed. Low agreement is the
        # ambiguity the verdict line exists to remove.
        "judge_answered_fallbacks": sum(r["answered_source"] == "judge" for r in rows),
        # Replies still cut off at the token budget after the generator's one retry, and
        # how many needed that retry. A truncated reply has no verdict line, so a nonzero
        # count here depresses verdict accuracy for reasons unrelated to the model's call.
        "truncated_answers": sum(bool(r.get("truncated")) for r in rows),
        "retried_answers": sum((r.get("generation_attempts") or 1) > 1 for r in rows),
        # Which parse produced each verdict (generator.parse_verdict), and — for replies
        # that quote their evidence — how often the quote is really in the cited passage,
        # with verdict accuracy on each side of that split.
        "verdict_sources": {
            src: sum(r.get("verdict_source") == src for r in rows)
            for src in ("line", "inline", "stance", "reask")
        },
        # The verdict-only re-ask: claims it fired on, and how many it gave a verdict.
        "reasked_answers": sum(bool(r.get("reask_attempted")) for r in rows),
        "reask_verdicts": sum(r.get("verdict_source") == "reask" for r in rows),
        "evidence_quotes": _quote_stats(rows),
        "judge_verdict_agreement": _rate(
            sum(r["judge_answered"] == r["answered"] for r in judge_rows), len(judge_rows)
        ),
        "faithfulness_answered": round(statistics.mean(r["faithfulness"] for r in judged), 4)
        if judged
        else None,
        "faithfulness_n": len(judged),
        "faithfulness_unjudged_reask": len(unjudged),
        # Over ALL rows, re-asked placeholders included, deliberately: context relevance
        # judges the retrieved passages against the question, not the answer, and the
        # passages a re-asked row was judged on are exactly the ones it was answered from.
        "context_relevance": round(statistics.mean(r["context_relevance"] for r in ctx_rows), 4)
        if ctx_rows
        else None,
        "by_label": {
            g: {
                "n": sum(r["gold_label"] == g for r in rows),
                "answered": sum(r["gold_label"] == g and r["answered"] for r in rows),
            }
            for g in LABELS
        },
        "generator_model": gen_model or GEN_MODEL,
        "judge_model": judge_model or JUDGE_MODEL,
        "generator_provider": gen_provider or GEN_PROVIDER,
        "judge_provider": judge_provider or JUDGE_PROVIDER,
        "top_k": TOP_K,
        "sample_seed": SEED,
        **extra,
    }


def _tie_nei_label(row: Mapping) -> str:
    """EXPLORATORY rule (pre-registered as such, never eligible for adoption): any
    disagreement among a vote's non-missing verdicts -> NEI; otherwise the row's label."""
    verdicts = {v for v in row["vote"]["verdicts"] if v is not None}
    return "NEI" if len(verdicts) > 1 else row["predicted_label"]


def _vote_stats(rows: Sequence[Mapping]) -> dict | None:
    """The vote summary for rows carrying ``vote`` blocks (None when none do): split
    distribution, ties, verdicts changed vs sample #1, each sample's own single-sample
    accuracy (its predicted label as a row; a sample a claim lacks is left out of that
    sample's n), and the exploratory tie->NEI rule's accuracy."""
    voted = [r for r in rows if isinstance(r.get("vote"), Mapping)]
    if not voted:
        return None
    k = max(int(r["vote"]["k"]) for r in voted)
    splits: dict[str, int] = {}
    for r in voted:
        splits[r["vote"]["split"]] = splits.get(r["vote"]["split"], 0) + 1
    sample_acc, sample_n = [], []
    for j in range(k):
        have = [r for r in voted if j < len(r["vote"]["predicted_labels"])
                and r["vote"]["predicted_labels"][j] is not None]
        sample_n.append(len(have))
        sample_acc.append(_rate(
            sum(r["vote"]["predicted_labels"][j] == r["gold_label"] for r in have), len(have)
        ))
    return {
        "k": k,
        "n": len(voted),
        "splits": dict(sorted(splits.items(), key=lambda kv: (-kv[1], kv[0]))),
        "ties": sum(bool(r["vote"]["tie"]) for r in voted),
        "changed_vs_sample1": sum(bool(r["vote"]["changed"]) for r in voted),
        "chosen_not_sample1": sum(r["vote"]["chosen"] != 1 for r in voted),
        "claims_missing_samples": sum(bool(r["vote"].get("missing")) for r in voted),
        "sample_accuracy": sample_acc,
        "sample_n": sample_n,
        "tie_nei_accuracy": _rate(sum(_tie_nei_label(r) == r["gold_label"] for r in voted), len(voted)),
        "rule": VOTE_RULE,
    }


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _abstention_md(a: dict, answered_rate: float | None) -> str:
    """Rates table for one oracle: an _abstention() dict, or the aggregate itself (whose
    top-level keys are the headline oracle's)."""
    return (
        f"| Metric | Score |\n|---|---|\n"
        f"| Evidence retrieved | {_fmt(a['evidence_rate'])} |\n"
        f"| Answered (model attempted an answer) | {_fmt(answered_rate)} |\n"
        f"| Abstention precision (abstained & no evidence) | {_fmt(a['abstention_precision'])} |\n"
        f"| Abstention recall (no evidence & abstained) | {_fmt(a['abstention_recall'])} |\n"
        f"| False abstention (had evidence, still abstained) | {_fmt(a['false_abstention_rate'])} |\n"
        f"| Answered without evidence (hallucination risk) | "
        f"{_fmt(a['answered_without_evidence_rate'])} |\n"
    )


def _quadrant_md(q: dict) -> str:
    return (
        "| | Evidence retrieved | No evidence |\n|---|---|---|\n"
        f"| Answered | {q['answered_with_evidence']} | {q['answered_without_evidence']} |\n"
        f"| Abstained | {q['false_abstention']} (false) | {q['correct_abstention']} (correct) |\n"
    )


def _first_reply(row: Mapping) -> str:
    """The generator's first reply as stored: under ``first_pass`` once the re-ask replaced
    the verdict, else the row's own answer."""
    fp = row.get("first_pass")
    return str((fp.get("answer") if isinstance(fp, Mapping) else row.get("answer")) or "")


def _cut_off(row: Mapping) -> bool:
    """The first reply hit the token budget (after the generator's one retry)."""
    return bool(row.get("truncated")) or _first_reply(row).strip().endswith(TRUNCATION_NOTE)


def _truncation_note(n: int, rows: Sequence[Mapping] | None) -> str:
    """The header note on truncated answers. With the rows it says what became of them —
    a verdict from the re-ask, one parsed from the text before the cut-off, or none."""
    note = f"{n} answer{'' if n == 1 else 's'} truncated at the token budget"
    if rows is None:
        return note
    cut = [r for r in rows if r.get("truncated")]
    reasked = sum(r.get("verdict_source") == "reask" for r in cut)
    none = sum(r.get("verdict") is None for r in cut)
    kept = len(cut) - reasked - none
    parts = [
        f"{reasked} got a verdict from the verdict-only re-ask" if reasked else "",
        f"{kept} kept a verdict parsed from the text before the cut-off" if kept else "",
        f"{none} {'was' if none == 1 else 'were'} scored with no verdict" if none else "",
    ]
    return note + f" (of those, {', '.join(x for x in parts if x)})"


def _unjudged_cause(rows: Sequence[Mapping] | None) -> str:
    """Why the judge saw no answer text for the re-ask-only answers left out of the
    faithfulness mean: their first reply hit the token budget, or was empty."""
    if rows is None:
        return "had no judgeable text"
    unjudged = [r for r in rows if r.get("answered") and judge_scored_placeholder(r)]
    cut = sum(_cut_off(r) for r in unjudged)
    empty = len(unjudged) - cut
    if cut and empty:
        return f"hit the token budget with no text left ({cut}) or was empty ({empty})"
    return "hit the token budget with no text left" if cut else "was empty"


def _markdown(
    agg: Mapping,
    skipped: int,
    parse_failures: int,
    dataset: str | None = None,
    rows: Sequence[Mapping] | None = None,
    missing: int = 0,
    no_verdict_scoring: str | None = NO_VERDICT_SCORING,
) -> str:
    """rag.md from the aggregates (and the rows, for the notes that break counts down).
    ``missing``: sampled claims with no row (a non-canonical run only), named up front.
    ``no_verdict_scoring``: the rule the rows were scored under (None: the rule before
    NO_VERDICT_SCORING, for a rag.json that predates it)."""
    # The canonical header stays verbatim; any other split names itself, so a train-split
    # table can't pass for the headline.
    source = "SciFact claims" if dataset in (None, CANONICAL_DATASET) else f"claims from {dataset}"
    notes = []
    if missing:
        notes.append(
            f"**INCOMPLETE: {missing} sampled claim{'' if missing == 1 else 's'} missing — "
            f"every number below is over {agg['n']} of {agg['n'] + missing} claims**"
        )
    if skipped:
        notes.append(f"{skipped} quer{'y' if skipped == 1 else 'ies'} skipped (pipeline errors)")
    if parse_failures:
        notes.append(
            f"{parse_failures} judge repl{'y' if parse_failures == 1 else 'ies'} unparseable"
        )
    if truncated := agg.get("truncated_answers", 0):
        notes.append(_truncation_note(truncated, rows))
    note_line = f"\n{'; '.join(notes)}.\n" if notes else ""
    # Re-ask-only answers whose first reply had no text (cut off at the token budget, or
    # empty) were judged on a placeholder: excluded from the faithfulness mean, and the
    # table says so — and why, from the rows.
    faith_scope = ""
    if unjudged := agg.get("faithfulness_unjudged_reask"):
        faith_scope = (
            f", n={agg.get('faithfulness_n')}; {unjudged} answered only via the re-ask excluded "
            f"— the first reply {_unjudged_cause(rows)}, so the judge saw no answer text"
        )
    if no_verdict_scoring == NO_VERDICT_SCORING:
        no_verdict_rule = (
            f"A reply with no verdict line counts as NEI if it is complete and abstained, and "
            f"as `{NO_VERDICT}` (always wrong) if it answered or was cut off at the token "
            f"budget / empty."
        )
    else:  # the rule a rag.json without the field was scored under
        no_verdict_rule = (
            f"A reply with no verdict line counts as NEI if it abstained and as "
            f"`{NO_VERDICT}` (always wrong) if it answered."
        )
    skipped_judge = bool(agg.get("judge_skipped"))
    if skipped_judge:
        no_verdict_rule += (
            f" The judge was skipped (SSR_RAG_SKIP_JUDGE), so a reply with no verdict even "
            f"after the re-ask has no judge fallback and counts as `{NO_VERDICT}` "
            f"({agg.get('no_judge_no_verdict', 0)} of {agg['n']} here)."
        )
        judge_label = "skipped (SSR_RAG_SKIP_JUDGE)"
        quality = (
            "## Answer quality (LLM judge)\n\nJudge skipped (SSR_RAG_SKIP_JUDGE=1): no "
            "faithfulness or context-relevance scores in this run.\n\n"
        )
    else:
        judge_label = f"{agg['judge_model']} ({agg['judge_provider']})"
        quality = (
            f"## Answer quality (LLM judge)\n\n"
            f"| Metric | Score |\n|---|---|\n"
            f"| Faithfulness (over answered{faith_scope}) | {_fmt(agg['faithfulness_answered'])} |\n"
            f"| Context relevance (all) | {_fmt(agg['context_relevance'])} |\n\n"
        )
    k = agg["top_k"]
    cols = (*LABELS, NO_VERDICT)
    confusion = "".join(
        f"| **{g}** | " + " | ".join(str(agg["confusion"][g][p]) for p in cols) + " |\n"
        for g in LABELS
    )
    by_label = "".join(
        f"| {g} | {agg['by_label'][g]['n']} | {agg['by_label'][g]['answered']} |\n"
        for g in LABELS
    )
    return (
        f"# RAG answer quality — verdicts scored against SciFact labels, LLM-as-judge for "
        f"faithfulness\n\n"
        f"{agg['n']} {source} (random sample, seed={agg['sample_seed']}) · "
        f"top_k={k} · generator={agg['generator_model']} ({agg['generator_provider']}) · "
        f"judge={judge_label}\n"
        f"{note_line}\n"
        f"{quality}"
        f"## Claim verdicts, scored against the gold label (no judge)\n\n"
        f"The final `Verdict:` line maps SUPPORTED→SUPPORT, REFUTED→CONTRADICT, NOT ENOUGH "
        f"EVIDENCE→NEI. {no_verdict_rule}\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| 3-class verdict accuracy | {_fmt(agg['verdict_accuracy'])} |\n"
        f"| Verdict line parsed | {_fmt(agg['verdict_parsed_rate'])} |\n"
        f"| Truncated answers (hit the token budget after 1 retry) | "
        f"{agg['truncated_answers']} of {agg['n']} |\n"
        f"| Re-asked (claim with no verdict: one verdict-only call) | "
        f"{agg.get('reasked_answers', 0)} of {agg['n']} "
        f"({agg.get('reask_verdicts', 0)} gave a verdict) |\n\n"
        f"| Gold \\ predicted | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        f"{confusion}\n"
        f"## Abstention — rationale oracle (headline)\n\n"
        f"`evidence` = a doc the annotators cited **with rationale sentences** was retrieved "
        f"into the top-{k}. NEI claims have no rationale docs, so for them abstaining is the "
        f"correct action. `answered` comes from the verdict line (anything but NOT ENOUGH "
        f"EVIDENCE); the judge decides it only when no verdict was parsed "
        f"({agg['judge_answered_fallbacks']} of {agg['n']} here).\n\n"
        f"{_quadrant_md(agg['quadrants'])}\n"
        f"{_abstention_md(agg, agg['answered_rate'])}\n"
        f"| Gold label | n | Answered |\n|---|---|---|\n{by_label}\n"
        f"## Abstention — legacy qrels oracle\n\n"
        f"`evidence` = any BEIR qrels-relevant doc was retrieved into the top-{k}. BEIR marks a "
        f"cited abstract relevant for NEI claims too, so this definition scores abstaining on "
        f"an NEI claim as a false abstention. Kept so earlier published numbers stay "
        f"traceable; same answers as above, different oracle.\n\n"
        f"{_quadrant_md(agg['qrels_oracle']['quadrants'])}\n"
        f"{_abstention_md(agg['qrels_oracle'], agg['answered_rate'])}"
        f"{_votes_md(agg.get('votes'))}"
    )


def _votes_md(v: Mapping | None) -> str:
    """The self-consistency vote section (SSR_LLM_VOTES > 1); empty without a vote."""
    if not v:
        return ""
    splits = ", ".join(f"{s or 'no verdict'}: {c}" for s, c in v["splits"].items())
    per_sample = " · ".join(
        f"#{j + 1} {_fmt(a)} (n={n})" for j, (a, n) in enumerate(zip(v["sample_accuracy"], v["sample_n"]))
    )
    missing = (
        f" {v['claims_missing_samples']} claim(s) lack a sample and were voted over the "
        f"samples they have." if v.get("claims_missing_samples") else ""
    )
    return (
        f"\n## Self-consistency vote (k={v['k']})\n\n"
        f"Rule: {v['rule']}. The verdict accuracy above is the vote's.{missing}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Splits | {splits} |\n"
        f"| Ties | {v['ties']} of {v['n']} |\n"
        f"| Verdict changed vs sample #1 | {v['changed_vs_sample1']} of {v['n']} |\n"
        f"| Served sample other than #1 | {v['chosen_not_sample1']} of {v['n']} |\n"
        f"| Single-sample verdict accuracy | {per_sample} |\n"
        f"| Exploratory: any disagreement -> NEI | {_fmt(v['tie_nei_accuracy'])} |\n"
    )


def markdown_from_json(blob: Mapping) -> str:
    """rag.md re-rendered offline from a finished rag.json — the aggregates and rows it
    stores — with the current writer: no LLM call, no retrieval, nothing recomputed."""
    run = blob.get("run") if isinstance(blob.get("run"), Mapping) else {}
    rows = blob.get("rows") if isinstance(blob.get("rows"), list) else None
    missing = blob.get("missing_query_ids")
    return _markdown(blob, int(blob.get("skipped") or 0), int(blob.get("judge_parse_failures") or 0),
                     run.get("dataset"), rows, missing=len(missing) if isinstance(missing, list) else 0,
                     no_verdict_scoring=run.get("no_verdict_scoring"))


def render_markdown(path: Path) -> Path:
    """Write rag.md next to the rag.json at `path` (which is only read), from its stored
    contents (markdown_from_json). Returns the written path."""
    md_path = path.parent / "rag.md"
    md_path.write_text(markdown_from_json(json.loads(path.read_text())))
    return md_path


def _estimate_line(
    n: int, gen: LLMEndpoint, judge_ep: LLMEndpoint, throttle: float, judge_skipped: bool = False
) -> str:
    """The up-front spend / request / time estimate, per provider (the first pass only:
    _request_line adds re-asks and vote samples)."""
    parts = []
    for role, ep, toks in (
        ("generator", gen, EST_GEN_TOKENS_PER_QUERY),
        ("judge", judge_ep, EST_JUDGE_TOKENS_PER_QUERY),
    ):
        if role == "judge" and judge_skipped:
            parts.append("  judge: skipped (SSR_RAG_SKIP_JUDGE): 0 requests")
        elif ep.provider == "groq":
            parts.append(
                f"  {role}: ~{n * toks:,} tokens on {ep.model} (Groq free tier caps tokens "
                f"per day: 200k/day for gpt-oss-120b)"
            )
        elif ep.paid:
            parts.append(
                f"  {role}: {n}-{2 * n} paid requests to {ep.model} (1 + the truncation retry "
                f"when needed); est. ${n * typical_generation_cost(gen):.4f}, worst case "
                f"${n * worst_case_generation_cost(gen):.4f} at {ep.model}'s billing bound "
                f"{paid_bill_rate(ep.model)} USD/1M tokens (max_price caps "
                f"{paid_max_price(ep.model)})"
            )
        else:
            count = f"{n}" if role == "judge" else f"{n}-{2 * n}"  # generator may retry once
            parts.append(f"  {role}: {count} free requests to {ep.model} ($0)")
    typical, worst = free_requests_per_query(gen, judge_ep, judge_skipped)
    if worst:
        both = "both roles are free: $0 run; " if typical == 2 else ""
        total = f"{n * typical}" if typical == worst else f"{n * typical}-{n * worst}"
        parts.append(
            f"  total: {total} free OpenRouter requests ({both}free tier: "
            f"{OPENROUTER_FREE_REQUESTS_PER_MIN}/min and {OPENROUTER_FREE_REQUESTS_PER_DAY:,}"
            f"/day account-wide, shared with everything else on the account today)"
        )
    ceiling = f"${settings.rag_max_spend_usd:.2f} (SSR_RAG_MAX_SPEND_USD)"
    if not gen.paid:
        ceiling += ", est. and worst-case spend $0 (no paid role)"
    return (
        "Estimated LLM usage:\n" + "\n".join(parts) + "\n"
        f"  spend ceiling: {ceiling}; ~{n * throttle / 60:.0f} min at a {throttle:.1f}s throttle."
    )


# --- sample, dataset, output guard ------------------------------------------------------


def parse_n(raw: str | None) -> int | None:
    """SSR_RAG_N: a positive int, or "all" (-> None: the whole split). Unset -> N."""
    if raw is None or not raw.strip():
        return N
    if raw.strip().lower() == "all":
        return None
    n = int(raw)
    if n < 1:
        raise ValueError(f"SSR_RAG_N must be a positive integer or 'all', got {raw!r}")
    return n


def rag_dataset(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get("SSR_RAG_DATASET") or settings.eval_dataset


def sample_claims(query_ids: Collection[str], n: int | None, seed: int = SEED) -> list[str]:
    """The first `n` of a seeded shuffle of the sorted ids (all of them for None).

    Shuffle-then-prefix is what makes samples nest: the shuffle doesn't depend on n, so
    sample(ids, 50) == sample(ids, 300)[:50]. This is exactly how the committed 50-claim
    sample was drawn, so it is unchanged.
    """
    qids = sorted(query_ids)
    random.Random(seed).shuffle(qids)  # fixed random sample, not the first N in id order
    return qids if n is None else qids[:n]


def _slug(dataset: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in dataset).strip("-")


def default_endpoints() -> tuple[tuple[str, str], tuple[str, str]]:
    """((provider, model id), (provider, model id)) for generator and judge under the
    CODE defaults — Settings field defaults, not the environment or .env (model_construct
    skips every settings source), resolved exactly as a run would resolve them."""
    d = Settings.model_construct()
    return (
        (d.llm_provider, model_id("generator", d)),
        (d.judge_provider, model_id("judge", d)),
    )


def non_default_roles(gen: LLMEndpoint, judge_ep: LLMEndpoint) -> list[str]:
    """The roles whose provider or model differs from the code defaults ([] = defaults).
    Compared on the RESOLVED ids, so an equivalent spelling (the free id without its
    `:free` suffix) still counts as the default."""
    dgen, djudge = default_endpoints()
    return [
        role
        for role, ep, want in (("generator", gen, dgen), ("judge", judge_ep, djudge))
        if (ep.provider, ep.model) != want
    ]


def output_dir(
    dataset: str,
    limit: int,
    n_sample: int,
    phash: str,
    gen: LLMEndpoint | None = None,
    judge_ep: LLMEndpoint | None = None,
    reask: bool = True,
    split_size: int | None = None,
    prompt_variant: str = "default",
    votes: int = 1,
    judge_skipped: bool = False,
) -> tuple[Path, bool]:
    """Where this run's rag.{md,json} go, and whether it is the canonical run.

    Only a run over the WHOLE canonical test split (``n_sample == split_size``, the
    number of claims in the split), with no SSR_EVAL_LIMIT, AND both roles on the code
    defaults (default_endpoints) may write to eval/results/. Anything else — a partial
    sample such as the default SSR_RAG_N=50 (or an unknown split size), every
    train-split run, every smoke subset, every run with another generator or judge (a
    paid-model comparison, even on the full test split) — goes to
    data/eval_runs/rag_<dataset-slug>_<n>_<prompt-hash8>[_<generator-slug>]
    [_judge-<judge-slug>]/, so experiments under different prompts or models don't
    collide and can never overwrite the committed free-model artifact. (gen/judge_ep
    None = the defaults, for callers that only vary the sample.) A run with the re-ask
    off (SSR_LLM_REASK=false) is not what the product serves: never canonical either,
    and its directory ends in ``_noreask``. Neither is a run with a non-default prompt
    variant (its prompt hash already names the directory apart), a self-consistency vote
    (SSR_LLM_VOTES = k > 1: ``_votes<k>``) or no judge (SSR_RAG_SKIP_JUDGE: ``_nojudge``).
    """
    changed = non_default_roles(gen, judge_ep) if gen and judge_ep else []
    whole_split = split_size is not None and n_sample == split_size
    defaults = reask and prompt_variant == "default" and votes == 1 and not judge_skipped
    if not limit and dataset == CANONICAL_DATASET and whole_split and not changed and defaults:
        return OUT, True
    name = f"rag_{_slug(dataset)}_{n_sample}_{phash[:8]}"
    if changed:
        name += f"_{_slug(gen.model)}"
        if "judge" in changed:
            name += f"_judge-{_slug(judge_ep.model)}"
    if not reask:
        name += "_noreask"
    if votes > 1:
        name += f"_votes{votes}"
    if judge_skipped:
        name += "_nojudge"
    return assert_outside(RUNS / name), False


def _non_default_notice(changed: Sequence[str], gen: LLMEndpoint, judge_ep: LLMEndpoint) -> str:
    dgen, djudge = default_endpoints()
    lines = [
        f"        {role}: {ep.provider} {ep.model} (code default: {want[0]} {want[1]})"
        for role, ep, want in (("generator", gen, dgen), ("judge", judge_ep, djudge))
        if role in changed
    ]
    bar = "=" * 88
    return (
        f"\n{bar}\n"
        f"NON-CANONICAL: {' and '.join(changed)} differ{'s' if len(changed) == 1 else ''} "
        f"from the code defaults, so this run NEVER writes {OUT}\n"
        + "\n".join(lines)
        + f"\n        (the committed rag.{{md,json}} describe the default models only)\n{bar}\n"
    )


def committed_sample_size(path: Path | None = None) -> int | None:
    """Sample size recorded in the committed rag.json (None if absent or unreadable)."""
    path = OUT / "rag.json" if path is None else path
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    run = blob.get("run") if isinstance(blob.get("run"), dict) else {}
    for v in (run.get("n_sample"), run.get("n_requested"), blob.get("n")):
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def _replace_notice(recorded: int, n_sample: int) -> str:
    bar = "!" * 88
    return (
        f"\n{bar}\n"
        f"NOTICE: this canonical run REPLACES {OUT / 'rag.md'} and {OUT / 'rag.json'}\n"
        f"        committed sample: N={recorded} claims -> this run: N={n_sample} claims.\n"
        f"        The headline numbers will then describe a different sample size; update the\n"
        f"        README alongside, or use SSR_RAG_N=<n> / a train-split run to experiment.\n"
        f"{bar}\n"
    )


# --- checkpoint -------------------------------------------------------------------------

# A checkpointed row is only reused if it carries what aggregate() reads.
_CHECKPOINT_ROW_KEYS = (
    "query_id",
    "gold_label",
    "predicted_label",
    "verdict",
    "answered",
    "answered_source",
    "judge_answered",
    "faithfulness",
    "context_relevance",
    "evidence",
    "evidence_qrels",
)


def signature_fields(
    dataset: str, n_sample: int, gen: LLMEndpoint, judge_ep: LLMEndpoint,
    judge_skipped: bool = False,
) -> dict:
    """Everything that changes a row's content. Nothing git-specific: a commit that
    leaves every one of these alone must not invalidate hours of rate-limited work.

    The sample is fixed by (dataset, n_sample, seed); each row by the two endpoints
    (provider, base URL, model id, routing fields), both prompts (the generator's per
    settings.llm_prompt_variant, via prompt_hash), the generator's token budget and
    reasoning setting, and the retrieval that produced its context. ``judge_skipped``
    (SSR_RAG_SKIP_JUDGE: rows carry no judge scores) adds a key ONLY when set, so every
    default signature — and the checkpoint it names — is unchanged. The vote's k is not
    here: sample #1 is the ordinary row, and the extra samples live in their own cache
    (samples_path) under this same signature, so runs with different k share them.
    """
    reranker = reranker_in_effect(MODE)
    fields: dict = {
        "dataset": dataset,
        "n_sample": n_sample,
        "sample_seed": SEED,
        "generator": gen.metadata(),
        "judge": judge_ep.metadata(),
        "prompt_hash": prompt_hash(),
        "judge_prompt_hash": _sha256(JUDGE_SYSTEM),
        "top_k": TOP_K,
        "mode": MODE,
        "reranker": reranker,
        "rerank_candidates": settings.rerank_candidates if reranker else None,
        "max_completion_tokens": settings.llm_max_completion_tokens,
        "reasoning_effort": resolve_reasoning_effort(gen.model, settings.llm_reasoning_effort),
        "rrf_k": settings.rrf_k,
        "candidate_k": settings.dense_top_k,
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
    }
    if (index := _index_fingerprint()) is not None:
        fields["index_manifest"] = index
    if judge_skipped:
        fields["judge_skipped"] = True
    return fields


def rag_signature(fields: Mapping) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def checkpoint_path(sig: str) -> Path:
    return CACHE / f"{sig}.json"


def load_checkpoint(path: Path, sig: str, qids: Collection[str]) -> tuple[dict[str, dict], dict]:
    """(completed rows for this sample, cumulative stats) — ({}, {}) when there is none.

    Like retrieval_eval, a corrupt or wrong-shaped checkpoint degrades to recompute
    rather than aborting the run; a single malformed row is dropped (and redone) on its
    own. SSR_EVAL_REFRESH=1 ignores the file.
    """
    if os.environ.get("SSR_EVAL_REFRESH") or not path.exists():
        return {}, {}
    try:
        blob = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        print(f"  (unreadable checkpoint, recomputing: {type(e).__name__})", flush=True)
        return {}, {}
    if not (isinstance(blob, dict) and isinstance(blob.get("rows"), dict)):
        print("  (malformed checkpoint, recomputing)", flush=True)
        return {}, {}
    if blob.get("signature") != sig:
        return {}, {}
    wanted = set(qids)
    rows = {
        q: r
        for q, r in blob["rows"].items()
        if q in wanted
        and isinstance(r, dict)
        and r.get("query_id") == q
        and all(k in r for k in _CHECKPOINT_ROW_KEYS)
    }
    stats = blob.get("stats") if isinstance(blob.get("stats"), dict) else {}
    return rows, stats


def save_checkpoint(path: Path, sig: str, fields: Mapping, rows: Mapping, stats: Mapping) -> None:
    """Write-then-rename: os.replace is atomic, so a kill mid-write leaves the previous
    good checkpoint intact rather than a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"signature": sig, "fields": fields, "stats": dict(stats), "rows": rows})
    )
    os.replace(tmp, path)


# --- self-consistency vote: extra samples (SSR_LLM_VOTES = k > 1) -------------------------
#
# Sample #1 of every claim is its ordinary row: the checkpointed first pass under the
# UNCHANGED signature, then the re-ask post-step — so existing checkpoints are reused as-is.
# Samples 2..k are drawn afterwards with the same request (same passages, prompt, budget,
# truncation retry) and stored in their own cache next to the checkpoint:
#   CACHE/<signature>.samples.json = {"signature": sig, "samples": {qid: {"2": rec, ...}}}
# keyed by the base signature + sample index, so the samples NEST (k=3 and k=5 share
# samples 2-3) and a stopped run resumes, drawing only what is missing. Each rec holds what
# a row records about its generation (generation_fields), plus "sample", "requests"
# (attempts + its re-ask calls), "fetched_at", and:
#   "reask": the sample's OWN verdict-only re-ask reply (None = not needed; absent = still
#            to fetch). Not the shared re-ask ReplyCache: its key is per claim, which would
#            hand every sample of a claim the same re-ask reply.
#   "judge": the judge's scores, set only once the vote serves this sample (judge on).
# A sample is complete when it has a rec and its re-ask is not pending.


def samples_path(sig: str) -> Path:
    return CACHE / f"{sig}.samples.json"


def load_samples(path: Path, sig: str) -> dict[str, dict[str, dict]]:
    """{qid: {"<j>": rec}} for this signature; {} with SSR_EVAL_REFRESH, no file, or a
    corrupt / wrong-signature / wrong-shaped file (degrades to redraw, like the
    checkpoint). A malformed rec is dropped on its own."""
    if os.environ.get("SSR_EVAL_REFRESH") or not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        print(f"  (unreadable vote-samples cache, redrawing: {type(e).__name__})", flush=True)
        return {}
    if not (isinstance(blob, dict) and blob.get("signature") == sig
            and isinstance(blob.get("samples"), dict)):
        return {}
    out: dict[str, dict[str, dict]] = {}
    for qid, recs in blob["samples"].items():
        if isinstance(recs, dict):
            good = {j: r for j, r in recs.items()
                    if isinstance(r, dict) and str(r.get("sample")) == j
                    and all(key in r for key in GEN_FIELD_KEYS)
                    and isinstance(r.get("retrieved_doc_ids"), list)}
            if good:
                out[qid] = good
    return out


def save_samples(path: Path, sig: str, samples: Mapping) -> None:
    """Atomic like save_checkpoint: a kill mid-write keeps the previous good file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"signature": sig, "samples": samples}))
    os.replace(tmp, path)


def generation_fields(ans, hits: Sequence[SearchHit]) -> dict:
    """What a row records about one generation (a GeneratedAnswer, or a plain Answer from
    another Generator: None / 1 / False for the side-channel fields)."""
    quote = getattr(ans, "evidence_quote", None)
    return {
        "verdict": ans.verdict,
        "answer": ans.text,
        "cited_doc_ids": ans.citations,
        "retrieved_doc_ids": [h.doc_id for h in hits],
        "finish_reason": getattr(ans, "finish_reason", None),
        "truncated": bool(getattr(ans, "truncated", False)),
        "generation_attempts": getattr(ans, "attempts", 1),
        "completion_tokens": getattr(ans, "completion_tokens", None),
        "reasoning_tokens": getattr(ans, "reasoning_tokens", None),
        "generation_cost_usd": getattr(ans, "cost_usd", None),
        "generation_provider": getattr(ans, "provider", None),
        "verdict_source": getattr(ans, "verdict_source", None),
        "evidence_quote": quote.quote if quote else None,
        "quote_passage": quote.passage if quote else None,
        "quote_found": quote.found if quote else None,
    }


GEN_FIELD_KEYS = tuple(generation_fields(
    SimpleNamespace(verdict=None, text="", citations=[]), ()
))
_NO_JUDGE = {"answered": None, "faithfulness": None, "context_relevance": None}


def build_row(
    qid: str, query: str, label: ClaimLabel, qrels_ids: Collection[str], flags: Mapping,
    gen: Mapping, scores: Mapping | None,
) -> dict:
    """One first-pass row from a generation (generation_fields) and the judge's scores
    (None: no judge call — every judge field None, answered_source "no_judge" for a reply
    with no verdict), re-read by reparse_row. The single construction used for the main
    pass and for every vote sample."""
    s = scores or _NO_JUDGE
    answered, answered_source = resolve_answered(gen["verdict"], s["answered"])
    return reparse_row({
        "query_id": qid,
        "mode": MODE,
        "gold_label": label.label,
        "rationale_doc_ids": sorted(label.rationale_doc_ids),
        "qrels_doc_ids": sorted(qrels_ids),
        "verdict": gen["verdict"],
        "predicted_label": predicted_label(
            gen["verdict"], answered, _broken_reply(bool(gen["truncated"]), gen["answer"]),
        ),
        "answered": answered,
        "answered_source": answered_source,
        "judge_answered": s["answered"],
        "faithfulness": s["faithfulness"],
        "context_relevance": s["context_relevance"],
        "evidence": flags["evidence"],
        "evidence_qrels": flags["evidence_qrels"],
        "abstention_class": _abstention_class(answered, flags["evidence"]),
        "abstention_class_qrels": _abstention_class(answered, flags["evidence_qrels"]),
        "answer": gen["answer"],
        "cited_doc_ids": gen["cited_doc_ids"],
        "retrieved_doc_ids": gen["retrieved_doc_ids"],
        # How the generation call ended (GeneratedAnswer side channel; a plain
        # Answer from another Generator records None / 1 / False).
        "finish_reason": gen["finish_reason"],
        "truncated": gen["truncated"],
        "generation_attempts": gen["generation_attempts"],
        "completion_tokens": gen["completion_tokens"],
        "reasoning_tokens": gen["reasoning_tokens"],
        # OpenRouter-reported cost of this query's generation (all attempts) and
        # the upstream provider that served it; None where not reported.
        "generation_cost_usd": gen["generation_cost_usd"],
        "generation_provider": gen["generation_provider"],
        # Which parse produced the verdict ("line" | "inline" | "stance"; None = no
        # verdict), and the evidence-quote check: the reply's first quoted span, the
        # [n] next to it, and whether it really occurs in that passage (None = the
        # reply quotes nothing). Measured only: quote_found never changes a verdict.
        "verdict_source": gen["verdict_source"],
        "evidence_quote": gen["evidence_quote"],
        "quote_passage": gen["quote_passage"],
        "quote_found": gen["quote_found"],
    }, query)


def sample_row(
    sample1: Mapping, query: str, label: ClaimLabel, rec: Mapping, reask_on: bool,
    scores: Mapping | None = None,
) -> dict | None:
    """Vote sample `rec` as a final row, through the main rows' own logic: build_row on the
    claim's sample-#1 passages / evidence flags (judge fields from `scores`, else None),
    then its own re-ask (apply_reask) when it needs one. None while that re-ask is still
    pending (not fetched yet, or it failed): the sample is incomplete."""
    row = build_row(
        sample1["query_id"], query, label, sample1.get("qrels_doc_ids") or (),
        sample1, rec, scores,
    )
    if reask_on and needs_reask(query, row["verdict"]):
        reask = rec.get("reask")
        return apply_reask(row, reask) if isinstance(reask, Mapping) else None
    return {**row, "reask_attempted": False}


def combine_sample_rows(k: int, rows_by_sample: Mapping[int, Mapping]) -> tuple[int, dict]:
    """(chosen sample number, 1-based; the row's ``vote`` block) for one claim, from its
    final per-sample rows ({1: sample #1's row, j: sample j's}; a missing sample is left
    out and listed under "missing"). The rule is generator.combine_votes over the samples
    present, in sample order."""
    present = sorted(rows_by_sample)
    vote = combine_votes([rows_by_sample[j]["verdict"] for j in present])
    chosen = present[vote.chosen]

    def per_sample(key: str) -> list:
        return [rows_by_sample[j].get(key) if j in rows_by_sample else None
                for j in range(1, k + 1)]

    return chosen, {
        "k": k,
        "verdicts": per_sample("verdict"),
        "verdict_sources": per_sample("verdict_source"),
        "predicted_labels": per_sample("predicted_label"),
        "counts": vote.counts,
        "split": vote.split,
        "tie": vote.tie,
        "abstained": vote.abstained,
        "chosen": chosen,
        "changed": vote.verdict != rows_by_sample[1]["verdict"],
        "missing": [j for j in range(1, k + 1) if j not in rows_by_sample],
    }


def _stat(stats: Mapping, key: str, default: float = 0) -> float:
    v = stats.get(key, default)
    return v if isinstance(v, int | float) and not isinstance(v, bool) else default


# --- up-front estimate + optional quota check ---------------------------------------------


def observed_retry_rate(rows: Sequence[Mapping]) -> float | None:
    if len(rows) < MIN_ROWS_FOR_OBSERVED_RETRY:
        return None
    return sum((r.get("generation_attempts") or 1) > 1 for r in rows) / len(rows)


def request_estimate(
    remaining: int, gen: LLMEndpoint, judge_ep: LLMEndpoint, throttle: float, retry_rate: float,
    reask_rate: float = 0.0, reask_pending: int = 0, reask: bool = False,
    judge: bool = True, vote_samples: int = 0, vote_rejudges: int = 0, vote_throttle: float = 0.0,
) -> dict:
    """Requests the remaining rows need: per claim, 1 generation + the expected retry
    rate + 1 judge call (all providers; none with ``judge`` False, SSR_RAG_SKIP_JUDGE),
    plus — with the re-ask on — the expected re-ask rate per new claim and the re-asks
    resumed rows still need (`reask_pending`: their exact count of cache misses); and the
    share of those on OpenRouter's free caps (expected, and worst case with every
    generation retrying and re-asking).

    The self-consistency vote (SSR_LLM_VOTES > 1) adds ``vote_samples`` extra samples
    still missing from the samples cache, each 1 generation + the retry rate + the re-ask
    rate (worst case 2 + 1), and — with the judge on — one judge call for the expected
    EXPECTED_VOTE_REJUDGE_RATE share of the ``vote_rejudges`` claims that have no judged
    vote sample yet (worst case all of them), each sample / re-judge followed by
    ``vote_throttle`` seconds."""
    gen_free = int(gen.provider == "openrouter" and not gen.paid)
    judge_free = int(judge_ep.provider == "openrouter" and not judge_ep.paid and judge)
    _, worst = free_requests_per_query(gen, judge_ep, not judge)
    reask_rate = reask_rate if reask else 0.0
    reask_pending = reask_pending if reask else 0
    rejudges = EXPECTED_VOTE_REJUDGE_RATE * vote_rejudges if judge else 0.0
    per_sample = 1 + retry_rate + reask_rate

    def up(x: float) -> int:  # ceil, without float noise turning 672.0000001 into 673
        return math.ceil(round(x, 6))

    return {
        "rows": remaining,
        "retry_rate": retry_rate,
        "reask_rate": reask_rate,
        "reask_pending": reask_pending,
        "judge": judge,
        "vote_samples": vote_samples,
        "vote_rejudges": rejudges,
        "requests": up(remaining * (1 + int(judge) + retry_rate + reask_rate) + reask_pending
                       + vote_samples * per_sample + rejudges),
        "free_requests": up(
            remaining * (gen_free * (1 + retry_rate + reask_rate) + judge_free)
            + gen_free * reask_pending
            + gen_free * vote_samples * per_sample + judge_free * rejudges
        ),
        "free_requests_worst": remaining * (worst + gen_free * int(reask))
        + gen_free * reask_pending
        + vote_samples * gen_free * (2 + int(reask)) + judge_free * vote_rejudges * int(judge),
        "seconds": remaining * (throttle + EST_QUERY_LATENCY_S) + reask_pending * throttle
        + vote_samples * (vote_throttle + EST_QUERY_LATENCY_S) + rejudges * vote_throttle,
    }


def _request_line(est: dict, n_total: int, n_done: int, throttle: float, retry_src: str) -> str:
    days = ""
    if est["free_requests"] > OPENROUTER_FREE_REQUESTS_PER_DAY:
        days = (
            f"\n  that is more than one day's {OPENROUTER_FREE_REQUESTS_PER_DAY:,} free requests: "
            f"expect ~{math.ceil(est['free_requests'] / OPENROUTER_FREE_REQUESTS_PER_DAY)} "
            f"daily-cap stops, each resumed by re-running the same command"
        )
    judge = " + 1 judge" if est.get("judge", True) else "; judge skipped"
    votes = ""
    if est.get("vote_samples") or est.get("vote_rejudges"):
        votes = (
            f"\n  self-consistency vote: {est['vote_samples']} extra samples still to draw "
            f"(each 1 generation + {est['retry_rate']:.2f} retries + {est['reask_rate']:.2f} "
            f"re-asks expected) + ~{est['vote_rejudges']:.1f} expected re-judges of a served "
            f"sample other than #1 (included above)"
        )
    return (
        f"Remaining work: {est['rows']} of {n_total} claims"
        + (f" ({n_done} resumed from checkpoint)" if n_done else "")
        + f"\n  ~{est['requests']} LLM requests (per claim: 1 generation + "
        f"{est['retry_rate']:.2f} expected truncation retries [{retry_src}]{judge}); "
        f"~{est['free_requests']} on OpenRouter's free caps (worst case "
        f"{est['free_requests_worst']})\n"
        f"  est. wall-clock ~{est['seconds'] / 60:.0f} min ({throttle:.1f}s throttle + "
        f"~{EST_QUERY_LATENCY_S:.0f}s assumed call latency per claim){days}{votes}"
    )


def fetch_key_info(api_key: str, timeout: float = 15.0) -> dict:
    """GET /api/v1/key on OpenRouter (key metadata; not an LLM call). The key goes only
    to the constant OpenRouter URL and is never printed."""
    req = urllib.request.Request(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def free_requests_remaining(info: object) -> tuple[int | None, int | None]:
    """(remaining, daily limit) from a /key response's free_model_daily_requests."""
    data = info.get("data", info) if isinstance(info, dict) else None
    daily = data.get("free_model_daily_requests") if isinstance(data, dict) else None
    if not isinstance(daily, dict):
        return None, None

    def as_int(v: object) -> int | None:
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    return as_int(daily.get("remaining")), as_int(daily.get("limit"))


def check_quota(
    est: dict,
    gen: LLMEndpoint,
    judge_ep: LLMEndpoint,
    fetch: Callable[[str], dict] = fetch_key_info,
) -> None:
    """Abort (SystemExit) before any LLM call if today's remaining free OpenRouter
    requests are fewer than this run needs. Fails closed: an unreadable answer aborts."""
    needed = est["free_requests"]
    ep = next((e for e in (gen, judge_ep) if e.provider == "openrouter"), None)
    if ep is None or not needed:
        print("Quota check: no free OpenRouter requests needed; nothing to check.", flush=True)
        return
    try:
        info = fetch(ep.api_key)
    except Exception as e:  # network, HTTP 401, bad JSON: fail closed
        raise SystemExit(
            f"--check-quota: could not read {OPENROUTER_KEY_URL} ({type(e).__name__}); "
            f"aborting before any LLM call."
        ) from None
    remaining, limit = free_requests_remaining(info)
    if remaining is None:
        raise SystemExit(
            f"--check-quota: {OPENROUTER_KEY_URL} returned no "
            f"free_model_daily_requests.remaining; aborting before any LLM call."
        )
    print(
        f"Quota check: {remaining} free OpenRouter requests left today"
        + (f" (of {limit})" if limit is not None else "")
        + f"; this run needs ~{needed} (worst case {est['free_requests_worst']}).",
        flush=True,
    )
    if remaining < needed:
        raise SystemExit(
            f"--check-quota: {remaining} free requests left today < ~{needed} needed; "
            f"aborting before any LLM call. Re-run after the daily reset (00:00 UTC), or run "
            f"without --check-quota to use what is left and resume tomorrow."
        )


# --- entry point --------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m app.eval.rag_eval", description=__doc__.split("\n")[0])
    p.add_argument(
        "--check-quota",
        action="store_true",
        help="read the OpenRouter key's remaining free requests (GET /api/v1/key) and abort "
        "before any LLM call if they are fewer than needed (also: SSR_RAG_CHECK_QUOTA=1)",
    )
    p.add_argument(
        "--render-md",
        metavar="RAG_JSON",
        type=Path,
        help="only re-render rag.md next to this finished rag.json from what it stores (the "
        "json is read, never written); no retrieval, no LLM call",
    )
    return p.parse_args(list(argv))


def main(argv: Sequence[str] = ()) -> None:
    args = _parse_args(argv)
    if args.render_md is not None:
        print(f"Wrote {render_markdown(args.render_md)}")
        return
    want_quota = args.check_quota or os.environ.get("SSR_RAG_CHECK_QUOTA", "") not in ("", "0")
    dataset = rag_dataset()
    n_req = parse_n(os.environ.get("SSR_RAG_N"))
    limit = int(os.environ.get("SSR_EVAL_LIMIT", "0") or 0)
    if limit < 0:
        raise ValueError("SSR_EVAL_LIMIT must be >= 0")
    # Opt-in experiment switches; every one of them makes the run non-canonical.
    skip_judge = os.environ.get("SSR_RAG_SKIP_JUDGE", "") not in ("", "0")
    k_votes = settings.llm_votes
    variant = settings.llm_prompt_variant

    # Resolve both endpoints first: a missing key, a key/URL mismatch or a model the
    # spend policy refuses fails here, before minutes of retrieval.
    gen_ep = resolve_endpoint("generator")
    judge_ep = resolve_endpoint("judge")
    print(describe_with_ignored(gen_ep), describe_with_ignored(judge_ep), sep="\n", flush=True)
    throttle = default_throttle_s(gen_ep, judge_ep, skip_judge)
    free_lo, free_hi = free_requests_per_query(gen_ep, judge_ep, skip_judge)
    vote_throttle = vote_sample_throttle_s(gen_ep)
    ceiling = settings.rag_max_spend_usd

    queries, qrels = load_queries_qrels(dataset)
    qids = sample_claims(queries, n_req)
    if os.environ.get("SSR_RAG_N") and n_req is not None and n_req > len(qids):
        print(f"SSR_RAG_N={n_req} exceeds the {len(qids)} claims in {dataset}: using all.")
    if limit:
        qids = qids[:limit]
    labels = load_claim_labels(dataset=dataset, query_ids=set(queries))
    phash = prompt_hash()
    reask_on = settings.llm_reask
    out, canonical = output_dir(
        dataset, limit, len(qids), phash, gen_ep, judge_ep, reask=reask_on, split_size=len(queries),
        prompt_variant=variant, votes=k_votes, judge_skipped=skip_judge,
    )
    if changed := non_default_roles(gen_ep, judge_ep):
        print(_non_default_notice(changed, gen_ep, judge_ep), flush=True)
    if not reask_on:
        print(f"SSR_LLM_REASK is off: no verdict-only re-ask, so this run never writes {OUT}",
              flush=True)
    if variant != "default":
        print(f"SSR_LLM_PROMPT_VARIANT={variant} (prompt hash {phash[:8]}): this run never "
              f"writes {OUT}", flush=True)
    if k_votes > 1:
        print(f"SSR_LLM_VOTES={k_votes}: self-consistency vote over {k_votes} samples per claim "
              f"({VOTE_RULE}); this run never writes {OUT}", flush=True)
    if skip_judge:
        print(f"SSR_RAG_SKIP_JUDGE: no judge calls (no faithfulness / context relevance; a reply "
              f"with no verdict scores {NO_VERDICT}); this run never writes {OUT}", flush=True)
    n_label = "all" if n_req is None else n_req
    print(
        f"Sample: {len(qids)} claims from {dataset} (SSR_RAG_N={n_label}, seed={SEED}"
        + (f", SSR_EVAL_LIMIT={limit}" if limit else "")
        + f") -> {out}"
        + ("" if canonical else f" (non-canonical: never writes {OUT})"),
        flush=True,
    )
    if dataset == CANONICAL_DATASET and not limit and len(qids) < len(queries):
        print(
            f"  partial sample: {len(qids)} of {len(queries)} test claims, so this run is not "
            f"canonical; only SSR_RAG_N=all writes {OUT}",
            flush=True,
        )
    replacing = None
    if canonical and (recorded := committed_sample_size()) is not None and recorded != len(qids):
        replacing = _replace_notice(recorded, len(qids))
        print(replacing, flush=True)

    # Resume: rows completed under the same signature are reused, never re-generated.
    sig_fields = signature_fields(dataset, len(qids), gen_ep, judge_ep, judge_skipped=skip_judge)
    sig = rag_signature(sig_fields)
    ckpt = checkpoint_path(sig)
    done, prior = load_checkpoint(ckpt, sig, qids)
    # Rows stored with no verdict are parsed again with the current parser (no LLM
    # call), so a resumed run records exactly what a fresh generation would now.
    done = {q: reparse_row(r, queries[q]) for q, r in done.items()}
    if done:
        print(f"  (resuming — {len(done)}/{len(qids)} rows already done, {ckpt})", flush=True)
    todo_ids = [q for q in qids if q not in done]

    if (reranker := reranker_in_effect(MODE)) is not None:
        warn = ""
        if reranker == _DEFAULT_RERANKER:
            warn = " — the default MiniLM, which measurably HURTS nDCG@10 on SciFact"
        print(f"MODE={MODE}: reranking with {reranker}{warn}", flush=True)

    service = SearchService()
    skipped = parse_failures = 0
    judge_calls = int(_stat(prior, "judge_calls"))
    prior_judge_usd = float(_stat(prior, "judge_reported_usd"))
    skip_reasons: dict[str, int] = {}

    def skip(n: int, qid: str, e: Exception) -> None:
        nonlocal skipped
        skipped += 1
        # Record WHY, not just how many: a run thinned by rate limits and one thinned by
        # a broken index are very different, and the artifact should say which without
        # anyone having to still have the console log.
        skip_reasons[type(e).__name__] = skip_reasons.get(type(e).__name__, 0) + 1
        print(f"  [{n}/{len(qids)}] q{qid:>4} SKIPPED ({type(e).__name__}: {str(e)[:50]})", flush=True)

    # Pass 1 — retrieval only (local, no LLM budget spent). Knowing the oracle split
    # BEFORE the first LLM call means a sample that can't test abstention (say, no
    # claim lacking evidence) is visible up front, not after 15 minutes of throttling.
    # Resumed rows already carry their evidence flags; only the rest are retrieved.
    position = {q: i for i, q in enumerate(qids, start=1)}
    retrieved: dict[str, list[SearchHit]] = {}
    for qid in todo_ids:
        try:
            retrieved[qid] = service.retrieve(queries[qid], mode=MODE, top_k=TOP_K)
        except Exception as e:  # skip a query rather than lose the whole run
            skip(position[qid], qid, e)
    gold = {qid: {d for d, rel in qrels.get(qid, {}).items() if rel > 0} for qid in retrieved}
    flags = {
        qid: evidence_flags([h.doc_id for h in hits], labels[qid], gold[qid])
        for qid, hits in retrieved.items()
    }
    scored = [q for q in qids if q in done or q in flags]
    ev = {q: (done[q] if q in done else flags[q]) for q in scored}
    n_ev = sum(bool(ev[q]["evidence"]) for q in scored)
    n_evq = sum(bool(ev[q]["evidence_qrels"]) for q in scored)
    mix = ", ".join(f"{g} {sum(labels[q].label == g for q in scored)}" for g in LABELS)
    print(
        f"Oracle check (before any LLM call), {len(scored)} claims [{mix}]:\n"
        f"  rationale oracle: {n_ev} have a rationale doc in top-{TOP_K} -> "
        f"{len(scored) - n_ev} should abstain\n"
        f"  qrels oracle (legacy): {n_evq} have a qrels doc in top-{TOP_K} -> "
        f"{len(scored) - n_evq} should abstain",
        flush=True,
    )

    # The re-ask post-step's work, known before any LLM call: resumed rows with no
    # verdict either have their reply in the re-ask cache or still need one call.
    reask_cache = ReplyCache(reask_cache_path(dataset))
    corpus: dict[str, dict] = {}

    def passages(qid: str) -> list[SearchHit]:
        if qid in retrieved:  # this session's retrieval: exactly what the generator saw
            return retrieved[qid]
        if not corpus:
            corpus.update({d["doc_id"]: d for d in load_documents()})
        return reask_hits(done[qid]["retrieved_doc_ids"], corpus)

    def reask_request(qid: str) -> tuple[list[dict], str]:
        """The re-ask messages and the v2 key a fresh reply is written under (lookups go
        through reask_lookup: v2, then the legacy key for the canonical endpoint only)."""
        msgs = reask_messages(queries[qid], passages(qid))
        return msgs, reask_key_v2(qid, gen_ep, msgs)

    def reask_cached(qid: str) -> dict | None:
        return reask_lookup(reask_cache, qid, gen_ep, reask_request(qid)[0])

    reask_done = [q for q in qids if q in done and needs_reask(queries[q], done[q]["verdict"])]
    reask_missing = [q for q in reask_done if reask_cached(q) is None] if reask_on else []
    if len(done) >= MIN_ROWS_FOR_OBSERVED_RETRY:
        reask_rate = len(reask_done) / len(done)
    else:
        reask_rate = EXPECTED_REASK_RATE

    # The vote's extra samples (k > 1): cached ones are reused (a rec drawn on other
    # passages than sample #1's is dropped and redrawn), the rest are counted up front.
    spath = samples_path(sig)
    samples: dict[str, dict[str, dict]] = {}
    vote_missing = vote_reask_pending = vote_rejudge = 0
    if k_votes > 1:
        for q, recs in load_samples(spath, sig).items():
            if q not in ev:
                continue
            ids = done[q]["retrieved_doc_ids"] if q in done else [h.doc_id for h in retrieved[q]]
            if kept_recs := {j: r for j, r in recs.items() if r["retrieved_doc_ids"] == ids}:
                samples[q] = kept_recs
        for q in scored:
            recs = samples.get(q, {})
            vote_missing += sum(str(j) not in recs for j in range(2, k_votes + 1))
            vote_reask_pending += sum(
                str(j) in recs and "reask" not in recs[str(j)] for j in range(2, k_votes + 1)
            )
            vote_rejudge += not skip_judge and not any("judge" in r for r in recs.values())
        n_cached = sum(len(v) for v in samples.values())
        print(f"  vote samples: {n_cached} cached in {spath}, {vote_missing} to draw "
              f"(samples 2..{k_votes} of {len(scored)} claims)", flush=True)

    print(_estimate_line(len(retrieved), gen_ep, judge_ep, throttle, skip_judge), flush=True)
    observed = observed_retry_rate(list(done.values()))
    retry_rate = EXPECTED_RETRY_RATE if observed is None else observed
    retry_src = "committed-run rate" if observed is None else f"observed on {len(done)} rows"
    est = request_estimate(
        len(retrieved), gen_ep, judge_ep, throttle, retry_rate,
        reask_rate=reask_rate, reask_pending=len(reask_missing) + vote_reask_pending,
        reask=reask_on, judge=not skip_judge, vote_samples=vote_missing,
        vote_rejudges=vote_rejudge, vote_throttle=vote_throttle,
    )
    print(_request_line(est, len(qids), len(done), throttle, retry_src), flush=True)
    if reask_on:
        print(
            f"  re-ask (SSR_LLM_REASK): {len(reask_done)} resumed claims have no verdict -> "
            f"{len(reask_done) - len(reask_missing)} replies cached in "
            f"{reask_cache_path(dataset)}, {len(reask_missing)} to fetch; "
            f"~{reask_rate:.2f} re-asks expected per new claim",
            flush=True,
        )
    print(f"LLM requests needed now: ~{est['requests']} "
          f"({est['free_requests']} on OpenRouter's free caps)", flush=True)
    if want_quota:
        check_quota(est, gen_ep, judge_ep, fetch=fetch_key_info)  # SystemExit if short

    # The first pass never re-asks inside generate(): the re-ask runs as the cached
    # post-step below, so checkpointed rows stay first-pass rows.
    # The batch retry policy (generator.BATCH_MAX_RETRIES / REASK_MAX_RETRIES), not the
    # API's: no user is waiting, and quick SDK retries absorb transient 429s / 5xx first.
    # votes=1: the eval draws the vote's extra samples itself (cached, resumable), never
    # inside generate().
    generator = LLMGenerator(
        endpoint=gen_ep, reask=False,
        max_retries=BATCH_MAX_RETRIES, reask_max_retries=REASK_MAX_RETRIES,
        votes=1, prompt_variant=variant,
    )
    judge_client = build_client(judge_ep, factory=OpenAI, timeout=60.0)  # Nemotron Ultra: rare 20-27 s calls

    # Spend accounting. `reported` is what OpenRouter billed per its usage.cost; `counted`
    # adds the worst-case bound for any paid generation whose cost went unreported, and
    # is what the ceiling is enforced on — an unknown is never counted as zero. Resumed
    # rows count too: the ceiling is per run, however many sessions it takes.
    worst_q = worst_case_generation_cost(gen_ep)
    worst_reask = (
        cost_upper_bound(MAX_GEN_PROMPT_TOKENS, REASK_MAX_TOKENS, gen_ep.model) if gen_ep.paid else 0.0
    )
    reported = counted = 0.0
    unreported_paid = 0

    def account(cost: float | None, worst: float | None = None) -> None:
        nonlocal reported, counted, unreported_paid
        if cost is not None:
            reported += cost
            counted += cost
        elif gen_ep.paid:
            unreported_paid += 1
            counted += worst_q if worst is None else worst

    for r in done.values():
        account(r.get("generation_cost_usd"))
    for recs in samples.values():  # cached vote samples count toward the per-run ceiling too
        for rec in recs.values():
            account(rec.get("generation_cost_usd"))
            if isinstance(rec.get("reask"), Mapping):
                account(rec["reask"].get("cost_usd"), worst_reask)

    # Requests actually sent this session (successful calls by their own count; a failed
    # call counts 1, a lower bound), for the run block.
    session = {"generation_attempts": 0, "reask_calls": 0, "judge_calls": 0,
               "vote_sample_requests": 0}

    def judge_usd() -> float:
        return prior_judge_usd + getattr(judge_client, "cost_usd", 0.0)

    def spent() -> float:
        # The judge is `:free` by policy, but anything OpenRouter does report counts.
        return counted + judge_usd()

    def checkpoint() -> None:
        save_checkpoint(
            ckpt, sig, sig_fields, done,
            {"judge_calls": judge_calls, "judge_reported_usd": judge_usd()},
        )

    def kept() -> str:
        return (
            f"{len(done)}/{len(qids)} completed rows are checkpointed in {ckpt}"
            if done else "no rows completed yet"
        )

    def with_rate_limit_retries(call, n, qid):
        # An HTTP 200 with no completion (EmptyCompletionError: OpenRouter's upstream
        # failed after accepting the request) is as transient as a per-minute 429,
        # and takes the same path — unless its body names the daily cap.
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return call()
            except (RateLimitError, EmptyCompletionError) as e:
                if _is_daily_cap(e):
                    raise DailyTokenBudgetExhausted(str(e)) from e
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                why = "rate-limited" if isinstance(e, RateLimitError) else "empty completion"
                print(
                    f"  [{n}/{len(qids)}] q{qid:>4} {why}; waiting "
                    f"{RATE_LIMIT_WAIT_S:.0f}s (retry {attempt + 1}/{RATE_LIMIT_RETRIES})",
                    flush=True,
                )
                time.sleep(RATE_LIMIT_WAIT_S)

    def generate_once(q: str, hits: list[SearchHit], counter: str = "generation_attempts"):
        # Checked BEFORE every attempt, retries included: the next attempt's worst
        # case must fit under the ceiling, so the ceiling holds even if this query
        # is the expensive one. A failed attempt's cost is counted (unknown = the
        # worst case on a paid generator) — it may have been billed.
        if spent() + worst_q > ceiling:
            raise SpendCeilingReached(
                f"spent ${spent():.4f} so far and the next query could cost up to "
                f"${worst_q:.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
            )
        try:
            ans = generator.generate(q, hits)
        except EmptyCompletionError as e:
            session[counter] += 1
            account(e.cost_usd)
            raise
        except OpenAIError as e:
            session[counter] += 1
            # generate() attaches what this generation's attempts cost when one of them
            # fails (e.g. the truncation retry after a billed first attempt); an error
            # without it (raised before anything was sent) counts nothing.
            if hasattr(e, "cost_usd"):
                account(e.cost_usd)
            raise
        session[counter] += getattr(ans, "attempts", 1)
        return ans

    def judge_once(q: str, hits: list[SearchHit], text: str) -> dict:
        session["judge_calls"] += 1  # every request sent, retries included
        # Judge must see the SAME context the generator saw (title + full text) — a
        # truncated view would misscore claims grounded in the cut-off part.
        return judge(judge_client, judge_ep.model, q, [hit_passage(h) for h in hits], text)

    # Pass 2 — generate + judge. The judge is called once per query for faithfulness
    # and context relevance (neither has a gold label); its `answered` field is used
    # only where the reply carries no parseable verdict line. SSR_RAG_SKIP_JUDGE: no
    # judge call (every judge field None).
    for qid in todo_ids:
        if qid not in retrieved:
            continue
        n = position[qid]
        q, hits, label = queries[qid], retrieved[qid], labels[qid]
        try:
            # Generation and judging retry SEPARATELY: a judge 429 or empty completion
            # must not re-run (and re-pay for) a generation that already succeeded.
            ans = with_rate_limit_retries(lambda q=q, hits=hits: generate_once(q, hits), n, qid)
            account(getattr(ans, "cost_usd", None))
            if spent() > ceiling:
                raise SpendCeilingReached(
                    f"spent ${spent():.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
                )
            s = None
            if not skip_judge:
                judge_calls += 1
                s = with_rate_limit_retries(
                    lambda q=q, hits=hits, ans=ans: judge_once(q, hits, ans.text), n, qid,
                )
        except DailyTokenBudgetExhausted as e:
            remaining = len(qids) - len(done)
            raise SystemExit(
                f"\nStopped at query {n}/{len(qids)}: the provider's daily cap is exhausted "
                f"(Groq: tokens/day; OpenRouter free tier: "
                f"{OPENROUTER_FREE_REQUESTS_PER_DAY:,} requests/day).\n  {str(e)[:300]}\n"
                f"Nothing was written to {out}; {kept()}.\n"
                f"Resume by re-running the same command after the quota resets: only the "
                f"{remaining} remaining rows run (~{remaining * EST_GEN_TOKENS_PER_QUERY:,} "
                f"generator + ~{remaining * EST_JUDGE_TOKENS_PER_QUERY:,} judge tokens, "
                f"{remaining * free_lo}-{remaining * free_hi} free OpenRouter requests)."
            ) from None
        except SpendCeilingReached as e:
            raise SystemExit(
                f"\nStopped at query {n}/{len(qids)}: spend ceiling reached — {e}.\n"
                f"Nothing was written to {out}; {kept()}."
            ) from None
        except JudgeParseError as e:
            parse_failures += 1  # not checkpointed: a re-run retries it
            print(f"  [{n}/{len(qids)}] q{qid:>4} JUDGE-UNPARSEABLE ({str(e)[:50]})", flush=True)
            time.sleep(throttle)
            continue
        except Exception as e:
            if _is_fatal(e):  # every later query would fail the same way
                raise SystemExit(
                    f"\nStopped at query {n}/{len(qids)}: {type(e).__name__}: {str(e)[:300]}\n"
                    f"Nothing was written to {out}; {kept()}."
                ) from None
            skip(n, qid, e)  # skipped, not checkpointed: a re-run retries it
            time.sleep(throttle)  # a failure is often the rate limit — back off too
            continue
        f = flags[qid]
        done[qid] = build_row(qid, q, label, gold[qid], f, generation_fields(ans, hits), s)
        checkpoint()  # persisted at once: a daily-cap stop loses no finished row
        r = done[qid]
        scores = "judge skipped" if s is None else (
            f"faith={s['faithfulness']:.2f} ctx={s['context_relevance']:.2f}"
        )
        print(
            f"  [{n}/{len(qids)}] q{qid:>4} {label.label:<10} -> {r['predicted_label']:<10} "
            f"{'answered' if r['answered'] else 'abstain '}({r['answered_source'][0]}) "
            f"{'eR' if f['evidence'] else '--'}{'eQ' if f['evidence_qrels'] else '--'} "
            f"{scores}"
            f"{'  TRUNCATED' if r['truncated'] else ''}  {q[:40]}",
            flush=True,
        )
        time.sleep(throttle)

    # Pass 3 — the verdict-only re-ask, over every finished row (resumed ones included)
    # whose claim got no verdict: the reply comes from the re-ask cache when it is there
    # (no LLM call), else from ONE call that is then cached at once. Same retries, spend
    # ceiling and daily-cap stop as pass 2; nothing is written until every row is done.
    final: dict[str, dict] = {}
    reask_stats = {"cached": 0, "fetched": 0, "failed": 0}
    for qid in qids if reask_on else ():
        r = done.get(qid)
        if r is None or not needs_reask(queries[qid], r["verdict"]):
            continue
        msgs, key = reask_request(qid)
        rec = reask_lookup(reask_cache, qid, gen_ep, msgs)
        if rec is not None:
            reask_stats["cached"] += 1
            account(rec.get("cost_usd"), worst_reask)
            final[qid] = apply_reask(r, rec)
            continue
        n = position[qid]

        def reask_once(q=queries[qid], hits=passages(qid)):
            if spent() + worst_reask > ceiling:
                raise SpendCeilingReached(
                    f"spent ${spent():.4f} so far and the next re-ask could cost up to "
                    f"${worst_reask:.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
                )
            session["reask_calls"] += 1
            try:
                return generator.reask_verdict(q, hits)
            except EmptyCompletionError as e:
                account(e.cost_usd, worst_reask)
                raise

        try:
            reply = with_rate_limit_retries(reask_once, n, qid)
        except (DailyTokenBudgetExhausted, SpendCeilingReached) as e:
            raise SystemExit(
                f"\nStopped in the re-ask pass at query {n}/{len(qids)}: {str(e)[:300]}\n"
                f"Nothing was written to {out}; {kept()}; re-ask replies so far are cached "
                f"in {reask_cache_path(dataset)}. Re-run the same command to resume."
            ) from None
        except Exception as e:
            if _is_fatal(e):
                raise SystemExit(
                    f"\nStopped in the re-ask pass at query {n}/{len(qids)}: "
                    f"{type(e).__name__}: {str(e)[:300]}\nNothing was written to {out}; {kept()}."
                ) from None
            # Not cached: a re-run retries it. The row keeps its first-pass verdict.
            reask_stats["failed"] += 1
            print(f"  [{n}/{len(qids)}] q{qid:>4} RE-ASK FAILED ({type(e).__name__})", flush=True)
            final[qid] = apply_reask(r, None, error=type(e).__name__)
            time.sleep(throttle)
            continue
        account(reply.cost_usd, worst_reask)
        rec = {
            "raw": reply.raw, "finish_reason": reply.finish_reason,
            "completion_tokens": reply.completion_tokens, "reasoning_tokens": reply.reasoning_tokens,
            "provider": reply.provider, "cost_usd": reply.cost_usd, "requests": 1,
            "kind": "reask", "query_id": qid, "model": gen_ep.model,
            "max_tokens": REASK_MAX_TOKENS, "source": "rag_eval", "key_version": 2,
            "request": reask_request_fingerprint(gen_ep),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        reask_cache.put(key, rec)  # v2 key; persisted at once, like a checkpointed row
        reask_stats["fetched"] += 1
        final[qid] = apply_reask(r, rec)
        x = final[qid]
        print(f"  [{n}/{len(qids)}] q{qid:>4} re-ask -> {x['reask']['parsed']} "
              f"({reply.finish_reason}, {reply.completion_tokens} tokens)", flush=True)
        time.sleep(throttle)

    # Pass 4 — the self-consistency vote (SSR_LLM_VOTES = k > 1). Sample #1 is each
    # claim's row as passes 2-3 left it. Samples 2..k come from the samples cache, else
    # one generation each with the SAME passages (+ the truncation retry), and — for a
    # claim whose sample has no verdict — that sample's OWN re-ask; every rec is cached at
    # once. Then the vote (combine_sample_rows): when it serves a sample other than #1
    # and the judge is on, that sample's reply is judged once (cached in its rec); a
    # served sample #1 keeps its own judge scores. A sample that failed (not cached) is
    # left out: the claim is voted over the samples it has and listed in
    # missing_vote_samples (a k > 1 run is never canonical, so it still writes). Same
    # retries, spend ceiling, daily-cap and fatal stops as passes 2-3.
    vote_rows: dict[str, dict] = {}
    vote_stats = {"drawn": 0, "cached": 0, "failed": 0, "reasks": 0, "rejudged": 0}
    missing_vote_samples: list[str] = []
    vote_judge_failed: list[str] = []
    for qid in qids if k_votes > 1 else ():
        if qid not in done:
            continue
        n = position[qid]
        q, label, hits = queries[qid], labels[qid], passages(qid)
        if [h.doc_id for h in hits] != done[qid]["retrieved_doc_ids"]:
            raise RuntimeError(f"vote samples for q{qid} would see other passages than sample #1")
        recs = samples.setdefault(qid, {})

        def vote_stop(e: Exception, where: str, n=n) -> SystemExit:
            return SystemExit(
                f"\nStopped in the vote pass ({where}) at query {n}/{len(qids)}: "
                f"{type(e).__name__}: {str(e)[:300]}\nNothing was written to {out}; {kept()}; "
                f"vote samples so far are cached in {spath}. Re-run the same command to "
                f"resume: only the missing samples are drawn."
            )

        def vote_reask_once(q=q, hits=hits):
            if spent() + worst_reask > ceiling:
                raise SpendCeilingReached(
                    f"spent ${spent():.4f} so far and the next re-ask could cost up to "
                    f"${worst_reask:.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
                )
            session["vote_sample_requests"] += 1
            try:
                return generator.reask_verdict(q, hits)
            except EmptyCompletionError as e:
                account(e.cost_usd, worst_reask)
                raise

        for j in range(2, k_votes + 1):
            rec = recs.get(str(j))
            try:
                if rec is None:
                    ans = with_rate_limit_retries(
                        lambda q=q, hits=hits: generate_once(q, hits, "vote_sample_requests"),
                        n, qid,
                    )
                    account(getattr(ans, "cost_usd", None))
                    if spent() > ceiling:
                        raise SpendCeilingReached(
                            f"spent ${spent():.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
                        )
                    rec = {"sample": j, **generation_fields(ans, hits),
                           "requests": getattr(ans, "attempts", 1),
                           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    verdict = build_row(qid, q, label, (), done[qid], rec, None)["verdict"]
                    if not (reask_on and needs_reask(q, verdict)):
                        rec["reask"] = None  # not needed (absent = still to fetch)
                    recs[str(j)] = rec
                    save_samples(spath, sig, samples)  # persisted at once, like a row
                    vote_stats["drawn"] += 1
                    print(f"  [{n}/{len(qids)}] q{qid:>4} vote sample {j}/{k_votes} -> "
                          f"{verdict}", flush=True)
                    time.sleep(vote_throttle)
                else:
                    vote_stats["cached"] += 1
                verdict = build_row(qid, q, label, (), done[qid], rec, None)["verdict"]
                if (reask_on and needs_reask(q, verdict)
                        and not isinstance(rec.get("reask"), Mapping)):
                    reply = with_rate_limit_retries(vote_reask_once, n, qid)
                    account(reply.cost_usd, worst_reask)
                    rec["reask"] = {
                        "raw": reply.raw, "finish_reason": reply.finish_reason,
                        "completion_tokens": reply.completion_tokens,
                        "reasoning_tokens": reply.reasoning_tokens,
                        "cost_usd": reply.cost_usd, "provider": reply.provider,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    rec["requests"] = int(rec.get("requests") or 0) + 1
                    save_samples(spath, sig, samples)
                    vote_stats["reasks"] += 1
                    print(f"  [{n}/{len(qids)}] q{qid:>4} vote sample {j}/{k_votes} re-ask -> "
                          f"{parse_reask(reply.raw)[0]}", flush=True)
                    time.sleep(vote_throttle)
            except (DailyTokenBudgetExhausted, SpendCeilingReached) as e:
                raise vote_stop(e, f"sample {j}") from None
            except Exception as e:
                if _is_fatal(e):
                    raise vote_stop(e, f"sample {j}") from None
                # Not cached (or its re-ask still pending): a re-run retries it.
                vote_stats["failed"] += 1
                print(f"  [{n}/{len(qids)}] q{qid:>4} vote sample {j}/{k_votes} FAILED "
                      f"({type(e).__name__})", flush=True)
                time.sleep(vote_throttle)

        row1 = final.get(qid) or {**done[qid], "reask_attempted": False}
        by_sample = {1: row1}
        for j in range(2, k_votes + 1):
            if (rec := recs.get(str(j))) is not None and (
                r := sample_row(done[qid], q, label, rec, reask_on)
            ) is not None:
                by_sample[j] = r
        if len(by_sample) < k_votes:
            missing_vote_samples.append(qid)
        chosen, block = combine_sample_rows(k_votes, by_sample)
        if chosen == 1:
            vote_rows[qid] = {**row1, "vote": block}
            continue
        rec = recs[str(chosen)]
        scores = None
        if not skip_judge:
            scores = rec.get("judge") if isinstance(rec.get("judge"), Mapping) else None
            if scores is None:
                try:
                    judge_calls += 1
                    scores = with_rate_limit_retries(
                        lambda q=q, hits=hits, text=rec["answer"]: judge_once(q, hits, text), n, qid,
                    )
                except DailyTokenBudgetExhausted as e:
                    raise vote_stop(e, f"judging sample {chosen}") from None
                except Exception as e:
                    if _is_fatal(e):
                        raise vote_stop(e, f"judging sample {chosen}") from None
                    # The verdict stands (the served sample has one); only its judge
                    # scores are missing. Not cached: a re-run retries the judge call.
                    parse_failures += isinstance(e, JudgeParseError)
                    vote_judge_failed.append(qid)
                    print(f"  [{n}/{len(qids)}] q{qid:>4} vote judge FAILED ({type(e).__name__})",
                          flush=True)
                else:
                    rec["judge"] = dict(scores)
                    save_samples(spath, sig, samples)
                    vote_stats["rejudged"] += 1
                time.sleep(vote_throttle)
        vote_rows[qid] = {**sample_row(done[qid], q, label, rec, reask_on, scores), "vote": block}

    # Sample order, resumed rows included; every row says whether the re-ask fired.
    rows = [
        vote_rows.get(q) or final.get(q) or {**done[q], "reask_attempted": False}
        for q in qids if q in done
    ]
    missing_ids = [q for q in qids if q not in done]
    failed_reasks = [r["query_id"] for r in rows if r.get("reask_attempted") and r.get("reask_error")]
    if canonical and (missing_ids or failed_reasks):
        # A thinned canonical artifact would silently shrink the headline's denominator
        # (or score a claim without the re-ask every other claim got): refuse. Completed
        # rows are checkpointed and fetched re-asks cached, so a re-run redoes only these.
        why = []
        if missing_ids:
            why.append(
                f"{len(missing_ids)} of {len(qids)} sampled claims have no row ({skipped} "
                f"skipped, {parse_failures} unparseable judge replies this session): "
                f"{', '.join(missing_ids[:20])}{' ...' if len(missing_ids) > 20 else ''}"
            )
        if failed_reasks:
            why.append(f"{len(failed_reasks)} re-asks failed: {', '.join(failed_reasks[:20])}")
        raise SystemExit(
            f"\nRefusing to write the canonical {OUT / 'rag.json'}: " + "; ".join(why) + ".\n"
            f"Nothing was written; {kept()}; re-ask replies are cached in "
            f"{reask_cache_path(dataset)}.\nRe-run the same command: only the missing claims "
            f"(and failed re-asks) run."
        )
    agg = aggregate(rows, gen_ep.model, judge_ep.model, gen_ep.provider, judge_ep.provider,
                    judge_skipped=skip_judge)
    votes_meta = None
    if k_votes > 1:
        votes_meta = {
            "k": k_votes,
            "rule": VOTE_RULE,
            "samples_cache": str(spath),
            # The vote's own identity: the base signature (which names the checkpoint and
            # the samples cache, shared across k) + k + the rule.
            "vote_signature": rag_signature({**sig_fields, "votes": k_votes, "vote_rule": VOTE_RULE}),
            "samples": vote_stats,
            "judged": "a served sample other than #1 is judged once on its own reply; a "
                      "served sample #1 keeps its judge scores",
        }
    vote_out = {} if k_votes <= 1 else {
        # Claims voted over fewer than k samples (a sample failed this session and is not
        # cached; re-running the same command draws it), and claims whose served sample's
        # judge call failed (verdict kept, judge fields None).
        "missing_vote_samples": missing_vote_samples,
        "vote_judge_failed": vote_judge_failed,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.json").write_text(
        json.dumps(
            {
                **agg,
                "skipped": skipped,
                "skip_reasons": skip_reasons,
                "judge_parse_failures": parse_failures,
                # Sampled claims with no row (never on a canonical run, which refuses to
                # write then): the aggregates above are over n = sampled - missing.
                "missing_query_ids": missing_ids,
                **vote_out,
                "judge_calls": judge_calls,
                "cost": {
                    "reported_usd": round(reported, 6),
                    # reported + worst-case bounds for unreported paid generations: the
                    # figure the ceiling was enforced on.
                    "counted_usd": round(counted, 6),
                    "unreported_paid_generations": unreported_paid,
                    "ceiling_usd": ceiling,
                    "generator_paid": gen_ep.paid,  # False: a $0 run (both roles free)
                    "judge_reported_usd": round(judge_usd(), 6),
                },
                # Provenance + per-query detail go after the aggregates, so the headline
                # fields keep their place at the top of the file.
                "run": {
                    **run_metadata(gen_ep, judge_ep, dataset=dataset, n_requested=n_label,
                                   judge_skipped=skip_judge, votes=votes_meta),
                    "n_sample": len(qids),
                    "eval_limit": limit,
                    "canonical": canonical,
                    "checkpoint_signature": sig,
                    "throttle_s": throttle,
                    "reask_cache": str(reask_cache_path(dataset)) if reask_on else None,
                    "reask_replies": reask_stats,
                    # LLM requests actually sent THIS session (a resumed run's earlier
                    # sessions are not included): first-pass generation attempts, pass-3
                    # re-asks, judge calls (retries included), and the vote's extra samples
                    # (their attempts + their own re-asks). A failed call counts once.
                    "requests_this_session": session,
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    (out / "rag.md").write_text(
        _markdown(agg, skipped, parse_failures, dataset, rows, missing=len(missing_ids))
    )
    print(
        f"\nn={agg['n']}  verdict_accuracy={agg['verdict_accuracy']}  "
        f"evidence={agg['evidence_rate']}  answered={agg['answered_rate']}  "
        f"abstention_precision={agg['abstention_precision']} "
        f"(qrels: {agg['qrels_oracle']['abstention_precision']})  "
        f"faithfulness={agg['faithfulness_answered']} (n={agg['faithfulness_n']}; "
        f"{agg['faithfulness_unjudged_reask']} re-ask placeholders excluded)  "
        f"ctx={agg['context_relevance']}  "
        f"judge_answered_fallbacks={agg['judge_answered_fallbacks']}  "
        f"truncated={agg['truncated_answers']} (retried {agg['retried_answers']})  "
        f"reasked={agg['reasked_answers']} ({agg['reask_verdicts']} verdicts; "
        f"{reask_stats['cached']} cached, {reask_stats['fetched']} fetched, "
        f"{reask_stats['failed']} failed)  "
        f"cost=${reported:.4f} reported (${counted:.4f} counted)"
        + (f"  votes(k={k_votes}): splits={agg['votes']['splits']} "
           f"changed_vs_sample1={agg['votes']['changed_vs_sample1']} "
           f"sample_accuracy={agg['votes']['sample_accuracy']} "
           f"({vote_stats['drawn']} samples drawn, {vote_stats['cached']} cached, "
           f"{vote_stats['failed']} failed, {vote_stats['rejudged']} re-judged)"
           if agg.get("votes") else "")
    )
    if missing_vote_samples:
        print(f"\nINCOMPLETE VOTE: {len(missing_vote_samples)} claims were voted over fewer than "
              f"{k_votes} samples: {', '.join(missing_vote_samples)}. Re-running the same command "
              f"draws only the missing samples.")
    if missing_ids:
        bar = "!" * 88
        print(
            f"\n{bar}\nINCOMPLETE: {len(missing_ids)} of {len(qids)} sampled claims have no row "
            f"(skipped / unparseable), so every number above is over {len(rows)} claims, not "
            f"{len(qids)}.\n  Missing: {', '.join(missing_ids)}\n  Re-running the same command "
            f"retries only those.\n{bar}"
        )
    if replacing:
        print(replacing)
    print(f"Wrote {out/'rag.md'} and {out/'rag.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
