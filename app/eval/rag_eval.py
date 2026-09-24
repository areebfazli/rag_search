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
(app.core.llm_endpoints; default: paid gpt-oss-120b generator and free Qwen3.8 judge on
OpenRouter, under the spend policy), with a provider-aware throttle so the run stays under
the free-tier limits and a hard per-run spend ceiling (SSR_RAG_MAX_SPEND_USD).

Alongside the aggregates, rag.json keeps every scored query (gold label, rationale
docs, verdict, both evidence flags, answer text, citations, and how generation ended:
finish_reason, token usage, retries) and the run's provenance (prompt hashes, git SHA,
models, generator budget, seed, oracle definition, label source), so a published number
can be traced back to the exact answers and prompts behind it.

Run:
    uv run python -m app.eval.rag_eval
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

from openai import APIStatusError, OpenAI, RateLimitError

from app.core.config import Settings, settings
from app.core.llm_endpoints import (
    PAID_MAX_PRICE_USD_PER_M,
    LLMEndpoint,
    SpendPolicyError,
    build_client,
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
from app.retrieve.service import SearchService

# Generator and judge are different model families, so the judge isn't grading its own
# or a sibling model's output. The ids are what llm_endpoints resolves for the provider
# in use (main() re-resolves them with key + spend-policy checks).
N = 50
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
#   models (paid models are not under that cap): 3.5 s per free request per query stays
#   under it with margin, floored at 5 s.
_THROTTLE_ENV = os.environ.get("SSR_RAG_THROTTLE_S")
THROTTLE_S: float | None = float(_THROTTLE_ENV) if _THROTTLE_ENV else None
GROQ_THROTTLE_S = 30.0
OPENROUTER_S_PER_FREE_REQUEST = 3.5
OPENROUTER_MIN_THROTTLE_S = 5.0
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


def default_throttle_s(gen: LLMEndpoint | None = None, judge: LLMEndpoint | None = None) -> float:
    """Per-query sleep for this provider mix (see THROTTLE_S)."""
    if THROTTLE_S is not None:
        return THROTTLE_S
    gen_provider = gen.provider if gen else GEN_PROVIDER
    judge_provider = judge.provider if judge else JUDGE_PROVIDER
    if "groq" in (gen_provider, judge_provider):
        return GROQ_THROTTLE_S
    free_requests = (1 if judge_provider == "openrouter" else 0) + (
        2 if gen is not None and gen.provider == "openrouter" and not gen.paid else 0
    )  # a free generator can need 2 (the truncation retry)
    return max(OPENROUTER_MIN_THROTTLE_S, free_requests * OPENROUTER_S_PER_FREE_REQUEST)


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


OUT = Path("eval/results")

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
    d = _parse(resp.choices[0].message.content or "")
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


def run_metadata(gen: LLMEndpoint | None = None, judge_ep: LLMEndpoint | None = None) -> dict:
    """Provenance for rag.json: enough to tell whether two runs are comparable. Records
    each role's provider, base URL, model id and the provider fields sent — never a key."""
    gen = gen or resolve_endpoint("generator", require_key=False)
    judge_ep = judge_ep or resolve_endpoint("judge", require_key=False)
    return {
        "git_sha": _git_sha(),
        "prompt_hash": prompt_hash(),
        "judge_prompt_hash": _sha256(JUDGE_SYSTEM),
        "generator_model": gen.model,
        "generator_provider": gen.provider,
        "generator_base_url": gen.base_url,
        "generator_extra_body": gen.extra_body(),
        # The generator's budget and reasoning effort decide whether answers truncate
        # (a truncated reply carries no verdict), so they are part of what a number means.
        "generator_max_completion_tokens": settings.llm_max_completion_tokens,
        "generator_reasoning_effort": resolve_reasoning_effort(
            gen.model, settings.llm_reasoning_effort
        ),
        "generator_reasoning_param": (
            "reasoning.effort" if gen.provider == "openrouter" else "reasoning_effort"
        ),
        "judge_model": judge_ep.model,
        "judge_provider": judge_ep.provider,
        "judge_base_url": judge_ep.base_url,
        "judge_extra_body": judge_ep.extra_body(),
        "mode": MODE,
        "reranker_model": reranker_in_effect(MODE),
        "top_k": TOP_K,
        "n_requested": N,
        "sample_seed": SEED,
        "oracle": ORACLE,
        "oracle_definition": ORACLE_DEFINITIONS[ORACLE],
        "legacy_oracle": "qrels",
        "legacy_oracle_definition": ORACLE_DEFINITIONS["qrels"],
        "label_source": {
            "loader": "app.ingest.corpus.load_claim_labels",
            "file": "scifact/queries.jsonl `metadata` field, BEIR SciFact source.zip",
            "dataset": settings.eval_dataset,
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


def _markdown(agg: dict, skipped: int, parse_failures: int) -> str:
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
        f"{agg['n']} SciFact claims (random sample, seed={agg['sample_seed']}) · "
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
            per_q = 1 if role == "judge" else 2
            parts.append(
                f"  {role}: up to {per_q * n} free requests to {ep.model} ($0; OpenRouter "
                f"free tier: 20/min, {OPENROUTER_FREE_REQUESTS_PER_DAY:,}/day account-wide)"
            )
    return (
        "Estimated LLM usage:\n" + "\n".join(parts) + "\n"
        f"  spend ceiling: ${settings.rag_max_spend_usd:.2f} (SSR_RAG_MAX_SPEND_USD); "
        f"~{n * throttle / 60:.0f} min at a {throttle:.1f}s throttle."
    )


def main() -> None:
    # Resolve both endpoints first: a missing key, a key/URL mismatch or a model the
    # spend policy refuses fails here, before minutes of retrieval.
    gen_ep = resolve_endpoint("generator")
    judge_ep = resolve_endpoint("judge")
    print(describe_with_ignored(gen_ep), describe_with_ignored(judge_ep), sep="\n", flush=True)
    throttle = default_throttle_s(gen_ep, judge_ep)
    ceiling = settings.rag_max_spend_usd

    queries, qrels = load_queries_qrels()
    qids = sorted(queries)
    random.Random(SEED).shuffle(qids)  # fixed random sample, not the first N in id order
    qids = qids[:N]
    labels = load_claim_labels(query_ids=set(queries))

    if (reranker := reranker_in_effect(MODE)) is not None:
        warn = ""
        if reranker == _DEFAULT_RERANKER:
            warn = " — the default MiniLM, which measurably HURTS nDCG@10 on SciFact"
        print(f"MODE={MODE}: reranking with {reranker}{warn}", flush=True)

    service = SearchService()
    skipped = parse_failures = judge_calls = 0
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
    retrieved: dict[str, list[SearchHit]] = {}
    for n, qid in enumerate(qids, start=1):
        try:
            retrieved[qid] = service.retrieve(queries[qid], mode=MODE, top_k=TOP_K)
        except Exception as e:  # skip a query rather than lose the whole run
            skip(n, qid, e)
    gold = {qid: {d for d, rel in qrels.get(qid, {}).items() if rel > 0} for qid in retrieved}
    flags = {
        qid: evidence_flags([h.doc_id for h in hits], labels[qid], gold[qid])
        for qid, hits in retrieved.items()
    }
    n_ev = sum(f["evidence"] for f in flags.values())
    n_evq = sum(f["evidence_qrels"] for f in flags.values())
    mix = ", ".join(f"{g} {sum(labels[q].label == g for q in retrieved)}" for g in LABELS)
    print(
        f"Oracle check (before any LLM call), {len(retrieved)} claims [{mix}]:\n"
        f"  rationale oracle: {n_ev} have a rationale doc in top-{TOP_K} -> "
        f"{len(retrieved) - n_ev} should abstain\n"
        f"  qrels oracle (legacy): {n_evq} have a qrels doc in top-{TOP_K} -> "
        f"{len(retrieved) - n_evq} should abstain",
        flush=True,
    )

    print(_estimate_line(len(retrieved), gen_ep, judge_ep, throttle), flush=True)
    generator = LLMGenerator(endpoint=gen_ep)
    judge_client = build_client(judge_ep, factory=OpenAI, timeout=60.0)  # Nemotron Ultra: rare 20-27 s calls

    # Spend accounting. `reported` is what OpenRouter billed per its usage.cost; `counted`
    # adds the worst-case bound for any paid generation whose cost went unreported, and
    # is what the ceiling is enforced on — an unknown is never counted as zero.
    worst_q = worst_case_generation_cost(gen_ep)
    reported = counted = 0.0
    unreported_paid = 0

    def account(ans) -> None:
        nonlocal reported, counted, unreported_paid
        cost = getattr(ans, "cost_usd", None)
        if cost is not None:
            reported += cost
            counted += cost
        elif gen_ep.paid:
            unreported_paid += 1
            counted += worst_q

    def spent() -> float:
        # The judge is `:free` by policy, but anything OpenRouter does report counts.
        return counted + getattr(judge_client, "cost_usd", 0.0)

    # Pass 2 — generate + judge. The judge is called once per query for faithfulness
    # and context relevance (neither has a gold label); its `answered` field is used
    # only where the reply carries no parseable verdict line.
    rows: list[dict] = []
    for n, qid in enumerate(qids, start=1):
        if qid not in retrieved:
            continue
        q, hits, label = queries[qid], retrieved[qid], labels[qid]

        def with_rate_limit_retries(call, n=n, qid=qid):
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    return call()
                except RateLimitError as e:
                    if _is_daily_cap(e):
                        raise DailyTokenBudgetExhausted(str(e)) from e
                    if attempt == RATE_LIMIT_RETRIES:
                        raise
                    print(
                        f"  [{n}/{len(qids)}] q{qid:>4} rate-limited; waiting "
                        f"{RATE_LIMIT_WAIT_S:.0f}s (retry {attempt + 1}/{RATE_LIMIT_RETRIES})",
                        flush=True,
                    )
                    time.sleep(RATE_LIMIT_WAIT_S)

        try:
            # Checked BEFORE the call: the next query's worst case must fit under the
            # ceiling, so the ceiling holds even if this query is the expensive one.
            if spent() + worst_q > ceiling:
                raise SpendCeilingReached(
                    f"spent ${spent():.4f} so far and the next query could cost up to "
                    f"${worst_q:.4f}, over SSR_RAG_MAX_SPEND_USD=${ceiling:.2f}"
                )
            # Generation and judging retry SEPARATELY: a judge 429 must not re-run (and
            # re-pay for) a generation that already succeeded.
            ans = with_rate_limit_retries(lambda q=q, hits=hits: generator.generate(q, hits))
            account(ans)
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
            raise SystemExit(
                f"\nStopped at query {n}/{len(qids)}: the provider's daily cap is exhausted "
                f"(Groq: tokens/day; OpenRouter free tier: "
                f"{OPENROUTER_FREE_REQUESTS_PER_DAY:,} requests/day).\n  {str(e)[:300]}\n"
                f"Nothing was written; {OUT / 'rag.json'} is unchanged. Re-run once the daily "
                f"budget recovers (~{len(retrieved) * EST_GEN_TOKENS_PER_QUERY:,} generator + "
                f"~{len(retrieved) * EST_JUDGE_TOKENS_PER_QUERY:,} judge tokens, "
                f"~{len(retrieved)} judge requests needed)."
            ) from None
        except SpendCeilingReached as e:
            raise SystemExit(
                f"\nStopped at query {n}/{len(qids)}: spend ceiling reached — {e}.\n"
                f"Nothing was written; {OUT / 'rag.json'} is unchanged."
            ) from None
        except JudgeParseError as e:
            parse_failures += 1
            print(f"  [{n}/{len(qids)}] q{qid:>4} JUDGE-UNPARSEABLE ({str(e)[:50]})", flush=True)
            time.sleep(throttle)
            continue
        except Exception as e:
            if _is_fatal(e):  # every later query would fail the same way
                raise SystemExit(
                    f"\nStopped at query {n}/{len(qids)}: {type(e).__name__}: {str(e)[:300]}\n"
                    f"Nothing was written; {OUT / 'rag.json'} is unchanged."
                ) from None
            skip(n, qid, e)  # skip a query rather than lose the whole run
            time.sleep(throttle)  # a failure is often the rate limit — back off too
            continue
        answered, answered_source = resolve_answered(ans.verdict, s["answered"])
        f = flags[qid]
        rows.append(
            {
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
        )
        r = rows[-1]
        print(
            f"  [{n}/{len(qids)}] q{qid:>4} {label.label:<10} -> {r['predicted_label']:<10} "
            f"{'answered' if answered else 'abstain '}({answered_source[0]}) "
            f"{'eR' if f['evidence'] else '--'}{'eQ' if f['evidence_qrels'] else '--'} "
            f"faith={s['faithfulness']:.2f} ctx={s['context_relevance']:.2f}"
            f"{'  TRUNCATED' if r['truncated'] else ''}  {q[:40]}",
            flush=True,
        )
        time.sleep(throttle)

    agg = aggregate(rows, gen_ep.model, judge_ep.model, gen_ep.provider, judge_ep.provider)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rag.json").write_text(
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
                    "judge_reported_usd": round(getattr(judge_client, "cost_usd", 0.0), 6),
                },
                # Provenance + per-query detail go after the aggregates, so the headline
                # fields keep their place at the top of the file.
                "run": {**run_metadata(gen_ep, judge_ep), "throttle_s": throttle},
                "rows": rows,
            },
            indent=2,
        )
    )
    (OUT / "rag.md").write_text(_markdown(agg, skipped, parse_failures))
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
    print(f"Wrote {OUT/'rag.md'} and {OUT/'rag.json'}")


if __name__ == "__main__":
    main()
