"""RAG answer-quality eval: deterministic verdict scoring + LLM-as-judge.

For a random sample of SciFact claims: retrieve -> generate -> score. Three things are
measured, and each comes from the most deterministic source available:

* **Verdict accuracy** (no judge). The product prompt asks for an optional final
  ``Verdict: SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE`` line when the input is a claim;
  the generator parses it into ``Answer.verdict``. Mapped to SUPPORT / CONTRADICT /
  NEI it is scored against the gold SciFact claim label (3-class accuracy + confusion
  matrix). "Abstained with no verdict line" counts as NEI.
* **Abstention** (no judge where possible). ``answered`` comes from the verdict when
  one was parsed (anything but NOT ENOUGH EVIDENCE is an answer). The judge's
  ``answered`` field is consulted ONLY for replies with no parseable verdict line.
* **Faithfulness and context relevance** (the judge, on every scored query). These have
  no gold label, so they stay LLM-judged. LLM-as-judge agrees with humans ~85-92% of
  the time — treat as signal, not truth.

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
ONLY by a run on the canonical test split with no SSR_EVAL_LIMIT. Every other run — any
train-split run, any limited run — writes to
``data/eval_runs/rag_<dataset-slug>_<n>_<prompt-hash8>/`` (gitignored) and can never touch
eval/results/. A canonical run whose sample size differs from the one recorded in the
committed rag.json (e.g. SSR_RAG_N=all replacing the 50-claim artifact) prints a loud
notice up front and again when it writes: it is replacing the headline with a different N.

Checkpoint + resume: every completed row is persisted at once (atomic temp file +
os.replace) to ``data/eval_cache/rag/<signature>.json``, where the signature covers
everything that changes a row (dataset, sample size, seed, both endpoints, both prompt
hashes, top_k, mode, token budget, reasoning setting, retrieval settings) and nothing
git-specific. A re-run of the same command skips completed rows, so a provider's daily
cap (Groq tokens/day, OpenRouter's free requests/day) just pauses the run: it stops with
the checkpoint saved and the output dir untouched, and resumes after the quota resets.
Skipped rows (pipeline errors, unparseable judge replies) are never checkpointed, so a
re-run retries them. ``SSR_EVAL_REFRESH=1`` ignores the checkpoint; a corrupt or
wrong-shaped one degrades to recompute.

Before any LLM call the run prints the requests still needed (remaining rows x (1
generation + the expected truncation-retry rate + 1 judge)) and the wall-clock estimate.
``--check-quota`` (or ``SSR_RAG_CHECK_QUOTA=1``) also reads the OpenRouter key's remaining
free requests for today (GET /api/v1/key, which is not an LLM call and is not counted
against the quota) and aborts before any LLM call if they are fewer than needed.

Run:
    uv run python -m app.eval.rag_eval
    SSR_RAG_N=all uv run python -m app.eval.rag_eval --check-quota     # full 300-claim test
    SSR_RAG_DATASET=beir/scifact/train SSR_RAG_N=100 uv run python -m app.eval.rag_eval
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
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path

from openai import APIStatusError, OpenAI, RateLimitError

from app.core.config import Settings, settings
from app.core.llm_endpoints import (
    OPENROUTER_BASE_URL,
    PAID_MAX_PRICE_USD_PER_M,
    EmptyCompletionError,
    LLMEndpoint,
    SpendPolicyError,
    build_client,
    completion_choice,
    cost_upper_bound,
    describe_with_ignored,
    model_id,
    resolve_endpoint,
)
from app.core.interfaces import SearchHit, hit_passage
from app.generate.generator import LLMGenerator, resolve_reasoning_effort
from app.generate.prompts import SYSTEM, VERDICTS, build_user_prompt
from app.ingest.corpus import (
    ClaimLabel,
    load_claim_labels,
    load_queries_qrels,
    scifact_source_zip,
)
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


def free_requests_per_query(gen: LLMEndpoint, judge: LLMEndpoint) -> tuple[int, int]:
    """(typical, worst) requests one query makes against OpenRouter's free-tier caps:
    the judge's one call if it is a free OpenRouter model, plus 1 for a free OpenRouter
    generator — 2 in the worst case (the truncation retry)."""
    judge_free = int(judge.provider == "openrouter" and not judge.paid)
    gen_free = int(gen.provider == "openrouter" and not gen.paid)
    return judge_free + gen_free, judge_free + 2 * gen_free


def default_throttle_s(gen: LLMEndpoint | None = None, judge: LLMEndpoint | None = None) -> float:
    """Per-query sleep for this provider mix (see THROTTLE_S)."""
    if THROTTLE_S is not None:
        return THROTTLE_S
    gen = gen or resolve_endpoint("generator", require_key=False)
    judge = judge or resolve_endpoint("judge", require_key=False)
    if "groq" in (gen.provider, judge.provider):
        return GROQ_THROTTLE_S
    _, worst = free_requests_per_query(gen, judge)
    return max(OPENROUTER_MIN_THROTTLE_S, worst * OPENROUTER_S_PER_FREE_REQUEST)


def worst_case_generation_cost(gen: LLMEndpoint) -> float:
    """Most one query's generation can cost: both attempts (the truncation retry runs
    at 2x the budget) at the max_price caps, with a generous prompt. 0 unless paid."""
    if not gen.paid:
        return 0.0
    budget = settings.llm_max_completion_tokens
    return cost_upper_bound(MAX_GEN_PROMPT_TOKENS, budget) + cost_upper_bound(
        MAX_GEN_PROMPT_TOKENS, budget * 2
    )


def typical_generation_cost(gen: LLMEndpoint) -> float:
    if not gen.paid:
        return 0.0
    return cost_upper_bound(EST_GEN_PROMPT_TOKENS, EST_GEN_COMPLETION_TOKENS)


OUT = Path("eval/results")  # canonical run only — the committed artifact
RUNS = Path("data/eval_runs")  # every other run (gitignored)
CACHE = Path("data/eval_cache/rag")  # per-row resume checkpoints (gitignored)

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


def resolve_answered(verdict: str | None, judge_answered: bool) -> tuple[bool, str]:
    """(answered, source). A parsed verdict decides deterministically — anything but
    NOT ENOUGH EVIDENCE is an attempt, a rebuttal included. Only a reply with no
    parseable verdict line falls back to the judge's call."""
    if verdict is not None:
        return verdict != "NOT ENOUGH EVIDENCE", "verdict"
    return judge_answered, "judge"


def predicted_label(verdict: str | None, answered: bool) -> str:
    """Map a verdict onto the gold label space. No verdict + abstained is NEI (it is
    exactly "not enough info"); no verdict + answered is NO_VERDICT, which never
    matches a gold label, so a missing verdict can't be scored as correct."""
    if verdict is not None:
        return VERDICT_TO_LABEL[verdict]
    return NO_VERDICT if answered else "NEI"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def prompt_hash() -> str:
    """sha256 of the generator's system prompt + user-prompt template.

    The template is a function, so it is hashed by *rendering* it on fixed placeholders
    rather than hashing its source: an edit to the wording or layout the model sees
    changes the hash, while a comment or refactor that leaves the prompt identical
    does not. (A change confined to the question sanitizer won't move it — that alters
    only how real queries are cleaned, not the template.)
    """
    ph = SearchHit("{doc_id}", 0.0, "{text}", {"title": "{title}"})
    return _sha256(SYSTEM + "\n\x00\n" + build_user_prompt("{question}", [ph, ph]))


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
) -> dict:
    """Provenance for rag.json: enough to tell whether two runs are comparable. Records
    each role's provider, base URL, model id and the provider fields sent — never a key."""
    gen = gen or resolve_endpoint("generator", require_key=False)
    judge_ep = judge_ep or resolve_endpoint("judge", require_key=False)
    effort = resolve_reasoning_effort(gen.model, settings.llm_reasoning_effort)
    dataset = dataset or settings.eval_dataset
    return {
        "git_sha": _git_sha(),
        "dataset": dataset,
        "prompt_hash": prompt_hash(),
        "judge_prompt_hash": _sha256(JUDGE_SYSTEM),
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


def aggregate(
    rows: list[dict],
    gen_model: str | None = None,
    judge_model: str | None = None,
    gen_provider: str | None = None,
    judge_provider: str | None = None,
) -> dict:
    """Cross the answered/abstained call with whether evidence was actually retrieved,
    so abstention is scored rather than assumed correct — under the rationale oracle
    (headline, top-level keys) and the legacy qrels oracle (``qrels_oracle``)."""
    headline = _abstention(rows, "evidence")
    answered = [r for r in rows if r["answered"]]
    with_verdict = [r for r in rows if r["verdict"] is not None]
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
        "judge_verdict_agreement": _rate(
            sum(r["judge_answered"] == r["answered"] for r in with_verdict), len(with_verdict)
        ),
        "faithfulness_answered": round(statistics.mean(r["faithfulness"] for r in answered), 4)
        if answered
        else None,
        "context_relevance": round(statistics.mean(r["context_relevance"] for r in rows), 4)
        if rows
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


def _markdown(agg: dict, skipped: int, parse_failures: int, dataset: str | None = None) -> str:
    # The canonical header stays verbatim; any other split names itself, so a train-split
    # table can't pass for the headline.
    source = "SciFact claims" if dataset in (None, CANONICAL_DATASET) else f"claims from {dataset}"
    notes = []
    if skipped:
        notes.append(f"{skipped} quer{'y' if skipped == 1 else 'ies'} skipped (pipeline errors)")
    if parse_failures:
        notes.append(
            f"{parse_failures} judge repl{'y' if parse_failures == 1 else 'ies'} unparseable"
        )
    if truncated := agg.get("truncated_answers", 0):
        notes.append(
            f"{truncated} answer{'' if truncated == 1 else 's'} truncated at the token budget "
            f"(scored as written, with no verdict line)"
        )
    note_line = f"\n{'; '.join(notes)}.\n" if notes else ""
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
        f"judge={agg['judge_model']} ({agg['judge_provider']})\n"
        f"{note_line}\n"
        f"## Answer quality (LLM judge)\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| Faithfulness (over answered) | {_fmt(agg['faithfulness_answered'])} |\n"
        f"| Context relevance (all) | {_fmt(agg['context_relevance'])} |\n\n"
        f"## Claim verdicts, scored against the gold label (no judge)\n\n"
        f"The final `Verdict:` line maps SUPPORTED→SUPPORT, REFUTED→CONTRADICT, NOT ENOUGH "
        f"EVIDENCE→NEI. A reply with no verdict line counts as NEI if it abstained and as "
        f"`{NO_VERDICT}` (always wrong) if it answered.\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| 3-class verdict accuracy | {_fmt(agg['verdict_accuracy'])} |\n"
        f"| Verdict line parsed | {_fmt(agg['verdict_parsed_rate'])} |\n"
        f"| Truncated answers (hit the token budget after 1 retry) | "
        f"{agg['truncated_answers']} of {agg['n']} |\n\n"
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
    )


def _estimate_line(n: int, gen: LLMEndpoint, judge_ep: LLMEndpoint, throttle: float) -> str:
    """The up-front spend / request / time estimate, per provider."""
    parts = []
    for role, ep, toks in (
        ("generator", gen, EST_GEN_TOKENS_PER_QUERY),
        ("judge", judge_ep, EST_JUDGE_TOKENS_PER_QUERY),
    ):
        if ep.provider == "groq":
            parts.append(
                f"  {role}: ~{n * toks:,} tokens on {ep.model} (Groq free tier caps tokens "
                f"per day: 200k/day for gpt-oss-120b)"
            )
        elif ep.paid:
            parts.append(
                f"  {role}: {n}-{2 * n} paid requests to {ep.model} (1 + the truncation retry "
                f"when needed); est. ${n * typical_generation_cost(gen):.4f}, worst case "
                f"${n * worst_case_generation_cost(gen):.4f} at the max_price caps "
                f"{PAID_MAX_PRICE_USD_PER_M} USD/1M tokens"
            )
        else:
            count = f"{n}" if role == "judge" else f"{n}-{2 * n}"  # generator may retry once
            parts.append(f"  {role}: {count} free requests to {ep.model} ($0)")
    typical, worst = free_requests_per_query(gen, judge_ep)
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


def output_dir(dataset: str, limit: int, n_sample: int, phash: str) -> tuple[Path, bool]:
    """Where this run's rag.{md,json} go, and whether it is the canonical run.

    Only the canonical test split with no SSR_EVAL_LIMIT may write to eval/results/.
    Anything else — every train-split run, every smoke subset — goes to
    data/eval_runs/rag_<dataset-slug>_<n>_<prompt-hash8>/, so prompt experiments under
    different prompts don't collide and can never overwrite the committed artifact.
    """
    if not limit and dataset == CANONICAL_DATASET:
        return OUT, True
    return RUNS / f"rag_{_slug(dataset)}_{n_sample}_{phash[:8]}", False


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
        f"        README alongside, or use SSR_EVAL_LIMIT / a train-split run to experiment.\n"
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
    dataset: str, n_sample: int, gen: LLMEndpoint, judge_ep: LLMEndpoint
) -> dict:
    """Everything that changes a row's content. Nothing git-specific: a commit that
    leaves every one of these alone must not invalidate hours of rate-limited work.

    The sample is fixed by (dataset, n_sample, seed); each row by the two endpoints
    (provider, base URL, model id, routing fields), both prompts, the generator's token
    budget and reasoning setting, and the retrieval that produced its context.
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


def _stat(stats: Mapping, key: str, default: float = 0) -> float:
    v = stats.get(key, default)
    return v if isinstance(v, int | float) and not isinstance(v, bool) else default


# --- up-front estimate + optional quota check ---------------------------------------------


def observed_retry_rate(rows: Sequence[Mapping]) -> float | None:
    if len(rows) < MIN_ROWS_FOR_OBSERVED_RETRY:
        return None
    return sum((r.get("generation_attempts") or 1) > 1 for r in rows) / len(rows)


def request_estimate(
    remaining: int, gen: LLMEndpoint, judge_ep: LLMEndpoint, throttle: float, retry_rate: float
) -> dict:
    """Requests the remaining rows need: per claim, 1 generation + the expected retry
    rate + 1 judge call (all providers), and the share of those on OpenRouter's free
    caps (expected, and worst case with every generation retrying)."""
    gen_free = int(gen.provider == "openrouter" and not gen.paid)
    judge_free = int(judge_ep.provider == "openrouter" and not judge_ep.paid)
    _, worst = free_requests_per_query(gen, judge_ep)

    def up(x: float) -> int:  # ceil, without float noise turning 672.0000001 into 673
        return math.ceil(round(x, 6))

    return {
        "rows": remaining,
        "retry_rate": retry_rate,
        "requests": up(remaining * (2 + retry_rate)),
        "free_requests": up(remaining * (gen_free * (1 + retry_rate) + judge_free)),
        "free_requests_worst": remaining * worst,
        "seconds": remaining * (throttle + EST_QUERY_LATENCY_S),
    }


def _request_line(est: dict, n_total: int, n_done: int, throttle: float, retry_src: str) -> str:
    days = ""
    if est["free_requests"] > OPENROUTER_FREE_REQUESTS_PER_DAY:
        days = (
            f"\n  that is more than one day's {OPENROUTER_FREE_REQUESTS_PER_DAY:,} free requests: "
            f"expect ~{math.ceil(est['free_requests'] / OPENROUTER_FREE_REQUESTS_PER_DAY)} "
            f"daily-cap stops, each resumed by re-running the same command"
        )
    return (
        f"Remaining work: {est['rows']} of {n_total} claims"
        + (f" ({n_done} resumed from checkpoint)" if n_done else "")
        + f"\n  ~{est['requests']} LLM requests (per claim: 1 generation + "
        f"{est['retry_rate']:.2f} expected truncation retries [{retry_src}] + 1 judge); "
        f"~{est['free_requests']} on OpenRouter's free caps (worst case "
        f"{est['free_requests_worst']})\n"
        f"  est. wall-clock ~{est['seconds'] / 60:.0f} min ({throttle:.1f}s throttle + "
        f"~{EST_QUERY_LATENCY_S:.0f}s assumed call latency per claim){days}"
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
    return p.parse_args(list(argv))


def main(argv: Sequence[str] = ()) -> None:
    args = _parse_args(argv)
    want_quota = args.check_quota or os.environ.get("SSR_RAG_CHECK_QUOTA", "") not in ("", "0")
    dataset = rag_dataset()
    n_req = parse_n(os.environ.get("SSR_RAG_N"))
    limit = int(os.environ.get("SSR_EVAL_LIMIT", "0") or 0)
    if limit < 0:
        raise ValueError("SSR_EVAL_LIMIT must be >= 0")

    # Resolve both endpoints first: a missing key, a key/URL mismatch or a model the
    # spend policy refuses fails here, before minutes of retrieval.
    gen_ep = resolve_endpoint("generator")
    judge_ep = resolve_endpoint("judge")
    print(describe_with_ignored(gen_ep), describe_with_ignored(judge_ep), sep="\n", flush=True)
    throttle = default_throttle_s(gen_ep, judge_ep)
    free_lo, free_hi = free_requests_per_query(gen_ep, judge_ep)
    ceiling = settings.rag_max_spend_usd

    queries, qrels = load_queries_qrels(dataset)
    qids = sample_claims(queries, n_req)
    if os.environ.get("SSR_RAG_N") and n_req is not None and n_req > len(qids):
        print(f"SSR_RAG_N={n_req} exceeds the {len(qids)} claims in {dataset}: using all.")
    if limit:
        qids = qids[:limit]
    labels = load_claim_labels(dataset=dataset, query_ids=set(queries))
    phash = prompt_hash()
    out, canonical = output_dir(dataset, limit, len(qids), phash)
    n_label = "all" if n_req is None else n_req
    print(
        f"Sample: {len(qids)} claims from {dataset} (SSR_RAG_N={n_label}, seed={SEED}"
        + (f", SSR_EVAL_LIMIT={limit}" if limit else "")
        + f") -> {out}"
        + ("" if canonical else f" (non-canonical: never writes {OUT})"),
        flush=True,
    )
    replacing = None
    if canonical and (recorded := committed_sample_size()) is not None and recorded != len(qids):
        replacing = _replace_notice(recorded, len(qids))
        print(replacing, flush=True)

    # Resume: rows completed under the same signature are reused, never re-generated.
    sig_fields = signature_fields(dataset, len(qids), gen_ep, judge_ep)
    sig = rag_signature(sig_fields)
    ckpt = checkpoint_path(sig)
    done, prior = load_checkpoint(ckpt, sig, qids)
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

    print(_estimate_line(len(retrieved), gen_ep, judge_ep, throttle), flush=True)
    observed = observed_retry_rate(list(done.values()))
    retry_rate = EXPECTED_RETRY_RATE if observed is None else observed
    retry_src = "committed-run rate" if observed is None else f"observed on {len(done)} rows"
    est = request_estimate(len(retrieved), gen_ep, judge_ep, throttle, retry_rate)
    print(_request_line(est, len(qids), len(done), throttle, retry_src), flush=True)
    if want_quota:
        check_quota(est, gen_ep, judge_ep, fetch=fetch_key_info)  # SystemExit if short

    generator = LLMGenerator(endpoint=gen_ep)
    judge_client = build_client(judge_ep, factory=OpenAI, timeout=60.0)  # Nemotron Ultra: rare 20-27 s calls

    # Spend accounting. `reported` is what OpenRouter billed per its usage.cost; `counted`
    # adds the worst-case bound for any paid generation whose cost went unreported, and
    # is what the ceiling is enforced on — an unknown is never counted as zero. Resumed
    # rows count too: the ceiling is per run, however many sessions it takes.
    worst_q = worst_case_generation_cost(gen_ep)
    reported = counted = 0.0
    unreported_paid = 0

    def account(cost: float | None) -> None:
        nonlocal reported, counted, unreported_paid
        if cost is not None:
            reported += cost
            counted += cost
        elif gen_ep.paid:
            unreported_paid += 1
            counted += worst_q

    for r in done.values():
        account(r.get("generation_cost_usd"))

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

    # Pass 2 — generate + judge. The judge is called once per query for faithfulness
    # and context relevance (neither has a gold label); its `answered` field is used
    # only where the reply carries no parseable verdict line.
    for qid in todo_ids:
        if qid not in retrieved:
            continue
        n = position[qid]
        q, hits, label = queries[qid], retrieved[qid], labels[qid]

        def with_rate_limit_retries(call, n=n, qid=qid):
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

        def generate_once(q=q, hits=hits):
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
                return generator.generate(q, hits)
            except EmptyCompletionError as e:
                account(e.cost_usd)
                raise

        try:
            # Generation and judging retry SEPARATELY: a judge 429 or empty completion
            # must not re-run (and re-pay for) a generation that already succeeded.
            ans = with_rate_limit_retries(generate_once)
            account(getattr(ans, "cost_usd", None))
            if spent() > ceiling:
                raise SpendCeilingReached(
                    f"spent ${spent():.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
                )
            judge_calls += 1
            # Judge must see the SAME context the generator saw (title + full text) — a
            # truncated view would misscore claims grounded in the cut-off part.
            s = with_rate_limit_retries(
                lambda q=q, hits=hits, ans=ans: judge(
                    judge_client, judge_ep.model, q, [hit_passage(h) for h in hits], ans.text
                )
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
        answered, answered_source = resolve_answered(ans.verdict, s["answered"])
        f = flags[qid]
        done[qid] = {
            "query_id": qid,
            "mode": MODE,
            "gold_label": label.label,
            "rationale_doc_ids": sorted(label.rationale_doc_ids),
            "qrels_doc_ids": sorted(gold[qid]),
            "verdict": ans.verdict,
            "predicted_label": predicted_label(ans.verdict, answered),
            "answered": answered,
            "answered_source": answered_source,
            "judge_answered": s["answered"],
            "faithfulness": s["faithfulness"],
            "context_relevance": s["context_relevance"],
            **f,
            "abstention_class": _abstention_class(answered, f["evidence"]),
            "abstention_class_qrels": _abstention_class(answered, f["evidence_qrels"]),
            "answer": ans.text,
            "cited_doc_ids": ans.citations,
            "retrieved_doc_ids": [h.doc_id for h in hits],
            # How the generation call ended (GeneratedAnswer side channel; a plain
            # Answer from another Generator records None / 1 / False).
            "finish_reason": getattr(ans, "finish_reason", None),
            "truncated": bool(getattr(ans, "truncated", False)),
            "generation_attempts": getattr(ans, "attempts", 1),
            "completion_tokens": getattr(ans, "completion_tokens", None),
            "reasoning_tokens": getattr(ans, "reasoning_tokens", None),
            # OpenRouter-reported cost of this query's generation (all attempts) and
            # the upstream provider that served it; None where not reported.
            "generation_cost_usd": getattr(ans, "cost_usd", None),
            "generation_provider": getattr(ans, "provider", None),
        }
        checkpoint()  # persisted at once: a daily-cap stop loses no finished row
        r = done[qid]
        print(
            f"  [{n}/{len(qids)}] q{qid:>4} {label.label:<10} -> {r['predicted_label']:<10} "
            f"{'answered' if answered else 'abstain '}({answered_source[0]}) "
            f"{'eR' if f['evidence'] else '--'}{'eQ' if f['evidence_qrels'] else '--'} "
            f"faith={s['faithfulness']:.2f} ctx={s['context_relevance']:.2f}"
            f"{'  TRUNCATED' if r['truncated'] else ''}  {q[:40]}",
            flush=True,
        )
        time.sleep(throttle)

    rows = [done[q] for q in qids if q in done]  # sample order, resumed rows included
    agg = aggregate(rows, gen_ep.model, judge_ep.model, gen_ep.provider, judge_ep.provider)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.json").write_text(
        json.dumps(
            {
                **agg,
                "skipped": skipped,
                "skip_reasons": skip_reasons,
                "judge_parse_failures": parse_failures,
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
                    **run_metadata(gen_ep, judge_ep, dataset=dataset, n_requested=n_label),
                    "n_sample": len(qids),
                    "eval_limit": limit,
                    "canonical": canonical,
                    "checkpoint_signature": sig,
                    "throttle_s": throttle,
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    (out / "rag.md").write_text(_markdown(agg, skipped, parse_failures, dataset))
    print(
        f"\nn={agg['n']}  verdict_accuracy={agg['verdict_accuracy']}  "
        f"evidence={agg['evidence_rate']}  answered={agg['answered_rate']}  "
        f"abstention_precision={agg['abstention_precision']} "
        f"(qrels: {agg['qrels_oracle']['abstention_precision']})  "
        f"faithfulness={agg['faithfulness_answered']}  ctx={agg['context_relevance']}  "
        f"judge_answered_fallbacks={agg['judge_answered_fallbacks']}  "
        f"truncated={agg['truncated_answers']} (retried {agg['retried_answers']})  "
        f"cost=${reported:.4f} reported (${counted:.4f} counted)"
    )
    if missing := len(qids) - len(rows):
        print(
            f"{missing} of {len(qids)} claims have no row (skipped / unparseable); re-running "
            f"the same command retries only those."
        )
    if replacing:
        print(replacing)
    print(f"Wrote {out/'rag.md'} and {out/'rag.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
