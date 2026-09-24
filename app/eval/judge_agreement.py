"""Judge-consistency eval: re-score the SAME stored RAG answers with the LLM judge.

rag_eval judges every answer once, so its faithfulness / context-relevance numbers (and
the judge-decided ``answered`` fallback) carry no error bar for the judge itself. This
re-judges the answers already stored in eval/results/rag.json — same claim, same
retrieved context, same answer text, same prompt, same model — K times at each of
several temperatures, and measures how much the scores move:

* **T = 0.0** (the production judge temperature): anything that changes is provider
  nondeterminism, so this is the reproducibility of the published numbers.
* **T > 0** (default 0.7): sensitivity — how close the judge's calls sit to a decision
  boundary.

Nothing is regenerated and nothing is retrieved: the context is rebuilt from the corpus
by doc id (no index, no Qdrant) with the same ``hit_passage`` view rag_eval hands the
judge, and each call goes through ``rag_eval.judge`` itself, so the prompt cannot drift
from production. The only thing changed on the wire is the temperature. No ``seed`` is
sent to the provider — production doesn't send one, and sending one would change what
T = 0.0 measures. ``SSR_JUDGE_SEED`` only picks the row subset under a row limit.

Refuses to run if rag.json's judge model or judge-prompt hash doesn't match the current
code: re-scoring a stale file would measure a different judge than the one in use. The
model is compared WITHOUT OpenRouter's ``:free`` suffix, so a Groq-judged rag.json can be
re-judged by the same model's free OpenRouter variant; the report then says which
provider made the original judgement and which made the repeats.

Provider-comparison mode (optional, off by default — it spends Groq budget):
``SSR_JUDGE_COMPARE_PROVIDERS=n`` re-judges the first n rows (``=1``/``yes`` -> 10) once on
Groq and once on OpenRouter at the production temperature and reports per-field
agreement between the two providers. Output goes to data/eval_runs/ only.

Run:
    SSR_JUDGE_DRY_RUN=1 uv run python -m app.eval.judge_agreement   # checks + estimate, no LLM calls
    uv run python -m app.eval.judge_agreement
    SSR_JUDGE_LIMIT=5 SSR_JUDGE_REPEATS=2 uv run python -m app.eval.judge_agreement  # smoke

Env (SSR_ prefix, like the rest of the repo):
    SSR_JUDGE_REPEATS       K re-judgements per row per temperature (default 3, min 2)
    SSR_JUDGE_TEMPERATURES  comma-separated (default "0.0,0.7")
    SSR_JUDGE_LIMIT         judge only this many rows (default 0 = all); output then goes
                            to data/eval_runs/, never to the committed eval/results/
    SSR_JUDGE_SEED          row-subset seed under a limit (default 13)
    SSR_JUDGE_DRY_RUN=1     validate + print the call/time estimate, then exit
    SSR_JUDGE_THROTTLE_S    seconds after each call (default: 15 on Groq, 3.5 on OpenRouter)
    SSR_JUDGE_COMPARE_PROVIDERS  n rows for the Groq-vs-OpenRouter comparison (default off)
    SSR_EVAL_REFRESH=1      ignore the resume checkpoint in data/eval_cache/

The judge provider is SSR_JUDGE_PROVIDER (app.core.llm_endpoints). A provider's daily cap
(Groq tokens/day, OpenRouter's 1,000 free requests/day) stops the run at once with the
completed rows checkpointed, so a re-run resumes; per-minute 429s wait and retry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from openai import RateLimitError

from app.core.config import settings
from app.core.interfaces import SearchHit, hit_passage
from app.core.llm_endpoints import (
    base_model,
    build_client,
    describe_with_ignored,
    model_id,
    resolve_endpoint,
)
from app.eval import rag_eval
from app.eval.rag_eval import JUDGE_SYSTEM, JudgeParseError, judge, resolve_answered

# Seconds slept after every judge call, per provider (SSR_JUDGE_THROTTLE_S overrides):
# * Groq: judge-only calls (~2k tokens each) against the judge model's 8,000
#   tokens/minute bucket — one per 15 s stays under it.
# * OpenRouter free tier: 20 requests/min account-wide — 3.5 s per call (~17/min) is
#   under it with margin. Its 1,000/day cap is handled as a daily cap (see run_repeats).
DEFAULT_THROTTLE_S = {"groq": 15.0, "openrouter": 3.5}


def throttle_s(provider: str, env: Mapping[str, str] | None = None) -> float:
    env = os.environ if env is None else env
    raw = env.get("SSR_JUDGE_THROTTLE_S")
    return float(raw) if raw else DEFAULT_THROTTLE_S[provider]


RATE_LIMIT_RETRIES = rag_eval.RATE_LIMIT_RETRIES
RATE_LIMIT_WAIT_S = rag_eval.RATE_LIMIT_WAIT_S

SOURCE = Path("eval/results/rag.json")
OUT = Path("eval/results")  # canonical (no row limit) run only — the committed artifact
RUNS = Path("data/eval_runs")  # every limited run (gitignored)
CACHE = Path("data/eval_cache")  # resume checkpoint (gitignored)

PRODUCTION_TEMPERATURE = 0.0  # what rag_eval.judge() sends
DEFAULT_K = 3
DEFAULT_TEMPERATURES = (PRODUCTION_TEMPERATURE, 0.7)
DEFAULT_SEED = 13
DEFAULT_COMPARE_N = 10
COMPARE_PROVIDERS = ("groq", "openrouter")
EST_LATENCY_S = 2.0  # rough per-call latency on top of the throttle, for the estimate only
# Three API failures in a row (after the client's own retries) is a spent budget or an
# outage, not a blip: stop and keep the checkpoint rather than burn the rest of the run.
MAX_CONSECUTIVE_API_ERRORS = 3

SCORE_KEYS = ("faithfulness", "context_relevance")
REQUIRED_ROW_KEYS = (
    "query_id",
    "answer",
    "retrieved_doc_ids",
    "verdict",
    "answered",
    "answered_source",
    "judge_answered",
    *SCORE_KEYS,
)


class StaleResultsError(RuntimeError):
    """rag.json doesn't describe the judge (or corpus) the current code would use."""


class AbortRun(RuntimeError):
    """Too many consecutive API failures; completed rows are in the checkpoint."""


class DailyCapReached(AbortRun):
    """A provider's per-day cap was hit; completed rows are in the checkpoint."""


# --- configuration --------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    k: int = DEFAULT_K
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    limit: int = 0
    seed: int = DEFAULT_SEED
    dry_run: bool = False
    compare_n: int = 0  # >0: provider-comparison mode on the first n rows

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env
        k = int(env.get("SSR_JUDGE_REPEATS", DEFAULT_K))
        if k < 2:
            raise ValueError("SSR_JUDGE_REPEATS must be >= 2: agreement needs two repeats")
        raw = env.get("SSR_JUDGE_TEMPERATURES")
        temps = DEFAULT_TEMPERATURES if raw is None else tuple(
            float(t) for t in raw.split(",") if t.strip()
        )
        temps = tuple(dict.fromkeys(temps))  # dedupe, keep order
        if not temps or any(not math.isfinite(t) or not 0.0 <= t <= 2.0 for t in temps):
            raise ValueError(f"SSR_JUDGE_TEMPERATURES must be values in [0, 2], got {raw!r}")
        limit = int(env.get("SSR_JUDGE_LIMIT", 0))
        if limit < 0:
            raise ValueError("SSR_JUDGE_LIMIT must be >= 0")
        raw_cmp = env.get("SSR_JUDGE_COMPARE_PROVIDERS", "").strip().lower()
        if raw_cmp in ("", "0", "no", "false", "off"):
            compare_n = 0
        elif raw_cmp in ("yes", "true", "on", "default"):
            compare_n = DEFAULT_COMPARE_N
        else:
            compare_n = int(raw_cmp)
            if compare_n < 0:
                raise ValueError("SSR_JUDGE_COMPARE_PROVIDERS must be >= 0")
        return cls(
            k=k,
            temperatures=temps,
            limit=limit,
            seed=int(env.get("SSR_JUDGE_SEED", DEFAULT_SEED)),
            dry_run=env.get("SSR_JUDGE_DRY_RUN", "") not in ("", "0"),
            compare_n=compare_n,
        )


def temp_key(t: float) -> str:
    """JSON key for a temperature: "0.0", "0.7" (repr keeps the trailing .0)."""
    return repr(float(t))


def estimate(n_rows: int, cfg: Config, throttle: float) -> tuple[int, float]:
    """(judge calls, estimated seconds). Every call is followed by `throttle`."""
    calls = n_rows * cfg.k * len(cfg.temperatures)
    return calls, calls * (throttle + EST_LATENCY_S)


def estimate_compare(n_rows: int, throttles: Mapping[str, float]) -> tuple[dict[str, int], float]:
    """({provider: calls}, estimated seconds): one T=0 call per row per provider."""
    calls = {p: n_rows for p in throttles}
    return calls, sum(n_rows * (t + EST_LATENCY_S) for t in throttles.values())


# --- input validation -----------------------------------------------------------------


def check_source(blob: object, judge_model: str | None = None) -> None:
    """Refuse a rag.json whose judge isn't the one the current code runs.

    judge_prompt_hash is computed exactly as rag_eval.run_metadata does, so an edit to
    JUDGE_SYSTEM since the run makes the file stale — the stored "original" scores
    came from a different rubric, and comparing against them would mix two judges.
    The judge MODEL is compared without the `:free` suffix (`judge_model` defaults to
    the id the current settings resolve): the same weights on another provider are the
    same judge, and the provider is reported separately (original_judge_provider).
    """
    run = blob.get("run") if isinstance(blob, dict) else None
    if not isinstance(run, dict):
        raise StaleResultsError("rag.json has no `run` metadata; re-run `make eval-rag`")
    current = model_id("judge") if judge_model is None else judge_model
    got = {
        "judge_prompt_hash": run.get("judge_prompt_hash"),
        "judge_model": base_model(run.get("judge_model")),
    }
    expected = {
        "judge_prompt_hash": rag_eval._sha256(JUDGE_SYSTEM),
        "judge_model": base_model(current),
    }
    bad = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    if bad:
        detail = "; ".join(f"{k}: file={got!r} current={want!r}" for k, (got, want) in bad.items())
        raise StaleResultsError(
            f"rag.json was judged by a different judge than the current code ({detail}). "
            "Re-run `make eval-rag` first."
        )
    rows = blob.get("rows")
    if not isinstance(rows, list) or not rows:
        raise StaleResultsError("rag.json has no per-query `rows`")
    for r in rows:
        missing = [k for k in REQUIRED_ROW_KEYS if not isinstance(r, dict) or k not in r]
        if missing:
            raise StaleResultsError(f"rag.json row lacks {missing}; re-run `make eval-rag`")


def original_judge(run: Mapping) -> dict:
    """Who made rag.json's judgement. Files written before provider metadata existed
    were all judged on Groq (the only judge backend then), and are labelled as assumed."""
    provider = run.get("judge_provider")
    return {
        "provider": provider or "groq",
        "provider_source": "recorded" if provider else (
            "assumed: rag.json predates provider metadata, when Groq was the only judge backend"
        ),
        "model": run.get("judge_model"),
        "base_url": run.get("judge_base_url"),
    }


def select_rows(rows: list[dict], limit: int, seed: int) -> tuple[list[dict], list[str]]:
    """(rows to re-judge, query ids skipped for an empty answer).

    An empty answer has no claims to be faithful to, so its scores say nothing about
    judge consistency. A limit takes a seeded random subset of the rest, kept in file
    order.
    """
    skipped = [r["query_id"] for r in rows if not str(r["answer"] or "").strip()]
    keep = [r for r in rows if str(r["answer"] or "").strip()]
    if limit and limit < len(keep):
        chosen = set(random.Random(seed).sample(range(len(keep)), limit))
        keep = [r for i, r in enumerate(keep) if i in chosen]
    return keep, skipped


# --- context reconstruction -------------------------------------------------------------


def rebuild_hits(doc_ids: Sequence[str], docs_by_id: Mapping[str, dict]) -> list[SearchHit]:
    """SearchHits carrying exactly what both indices store per doc: the corpus `text`
    and {"title": title} (see VectorStore.upsert / LexicalIndex.search). The score is
    irrelevant to hit_passage, so it is left at 0."""
    missing = [d for d in doc_ids if d not in docs_by_id]
    if missing:
        raise StaleResultsError(f"retrieved doc ids not in the corpus: {missing[:5]}")
    return [
        SearchHit(d, 0.0, docs_by_id[d]["text"], {"title": docs_by_id[d].get("title", "")})
        for d in doc_ids
    ]


def rebuild_contexts(doc_ids: Sequence[str], docs_by_id: Mapping[str, dict]) -> list[str]:
    """The context strings rag_eval passed to the judge: [hit_passage(h) for h in hits]."""
    return [hit_passage(h) for h in rebuild_hits(doc_ids, docs_by_id)]


# --- judging ----------------------------------------------------------------------------


class TemperatureClient:
    """The ``client.chat.completions.create`` surface judge() uses, with the temperature
    forced. Wrapping the client (rather than copying judge()) keeps the prompt, message
    layout, max_tokens and reply parsing byte-identical to production."""

    def __init__(self, client: object, temperature: float):
        self._client = client
        self.temperature = temperature
        self.requested_temperature: float | None = None  # what judge() itself asked for
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requested_temperature = kwargs.get("temperature")
        kwargs["temperature"] = self.temperature
        return self._client.chat.completions.create(**kwargs)


def _complete(rec: object, cfg: Config) -> bool:
    """A checkpointed row is reusable only if every repeat ran and none hit an API error
    (a parse failure is a real judge outcome and is kept)."""
    if not isinstance(rec, dict) or not isinstance(rec.get("repeats"), dict):
        return False
    for t in cfg.temperatures:
        reps = rec["repeats"].get(temp_key(t))
        if not isinstance(reps, list) or len(reps) != cfg.k:
            return False
        if any(not isinstance(x, dict) or x.get("error_kind") == "api" for x in reps):
            return False
    return True


def _record(row: dict) -> dict:
    return {
        "query_id": row["query_id"],
        "gold_label": row.get("gold_label"),
        "verdict": row["verdict"],
        "answered": row["answered"],
        "answered_source": row["answered_source"],
        "original": {
            "answered": row["judge_answered"],
            **{k: row[k] for k in SCORE_KEYS},
        },
        "repeats": {},
    }


def _judge_once(call: Callable[[], dict], sleep: Callable[[float], None], log: Callable[[str], None]) -> dict:
    """One judge call with rag_eval's rate-limit policy: a daily-cap 429 raises
    DailyCapReached at once (retrying can't clear it); a per-minute 429 waits a full
    window and retries the SAME call, up to RATE_LIMIT_RETRIES, then propagates."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return call()
        except RateLimitError as e:
            if rag_eval._is_daily_cap(e):
                raise DailyCapReached(f"provider daily cap reached: {str(e)[:200]}") from e
            if attempt == RATE_LIMIT_RETRIES:
                raise
            log(f"    rate-limited; waiting {RATE_LIMIT_WAIT_S:.0f}s "
                f"(retry {attempt + 1}/{RATE_LIMIT_RETRIES})")
            sleep(RATE_LIMIT_WAIT_S)
    raise AssertionError("unreachable")


def run_repeats(
    client: object,
    rows: list[dict],
    questions: Mapping[str, str],
    contexts: Mapping[str, list[str]],
    cfg: Config,
    *,
    model: str | None = None,
    throttle: float = 0.0,
    done: Mapping[str, dict] | None = None,
    save: Callable[[dict[str, dict]], None] = lambda _: None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, dict], float | None]:
    """Re-judge every row K times per temperature. Returns ({qid: record}, the
    temperature judge() itself requested — expected to be PRODUCTION_TEMPERATURE).

    `model` is the judge id to send (default: what the settings resolve). `done` rows
    (from the checkpoint) are reused; `save` is called after each new row. A daily-cap
    429 raises DailyCapReached; the rows completed so far have already been saved.
    """
    model = model_id("judge") if model is None else model
    out: dict[str, dict] = {q: r for q, r in (done or {}).items() if _complete(r, cfg)}
    requested: float | None = None
    consecutive_api_errors = 0
    for n, row in enumerate(rows, start=1):
        qid = row["query_id"]
        if qid in out:
            continue
        rec = _record(row)
        for t in cfg.temperatures:
            tc = TemperatureClient(client, t)
            reps: list[dict] = []
            for _ in range(cfg.k):
                try:
                    s = _judge_once(
                        lambda tc=tc: judge(tc, model, questions[qid], contexts[qid], row["answer"]),
                        sleep,
                        log,
                    )
                    consecutive_api_errors = 0
                except DailyCapReached:
                    raise
                except JudgeParseError:
                    s = {"error": "JudgeParseError", "error_kind": "parse"}
                    consecutive_api_errors = 0
                except Exception as e:  # rate limit / network: recorded, row is redone on resume
                    s = {"error": type(e).__name__, "error_kind": "api"}
                    consecutive_api_errors += 1
                requested = tc.requested_temperature if requested is None else requested
                reps.append(s)
                sleep(throttle)
                if consecutive_api_errors >= MAX_CONSECUTIVE_API_ERRORS:
                    raise AbortRun(
                        f"{consecutive_api_errors} consecutive API errors (last: {s['error']}); "
                        f"{len(out)} completed rows are checkpointed — re-run to resume"
                    )
            rec["repeats"][temp_key(t)] = reps
        out[qid] = rec
        if _complete(rec, cfg):
            save(out)
        log(f"  [{n}/{len(rows)}] q{qid:>4} " + "  ".join(
            f"T={k}: " + "".join(
                "?" if "error" in x else ("A" if x["answered"] else "-") for x in reps
            ) + " faith=" + ",".join(
                "?" if "error" in x else f"{x['faithfulness']:.2f}" for x in reps
            )
            for k, reps in rec["repeats"].items()
        ))
    return out, requested


def _compare_complete(rec: object, providers: Sequence[str]) -> bool:
    if not isinstance(rec, dict) or not isinstance(rec.get("by_provider"), dict):
        return False
    return all(
        isinstance(rec["by_provider"].get(p), dict)
        and rec["by_provider"][p].get("error_kind") != "api"
        for p in providers
    )


def run_compare(
    judges: Mapping[str, tuple[object, str]],
    rows: list[dict],
    questions: Mapping[str, str],
    contexts: Mapping[str, list[str]],
    *,
    throttles: Mapping[str, float],
    done: Mapping[str, dict] | None = None,
    save: Callable[[dict[str, dict]], None] = lambda _: None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Judge every row ONCE per provider through the production judge() (T=0.0, the
    production prompt and budget). `judges` maps provider -> (client, model id).

    Same checkpoint/abort contract as run_repeats: a row is saved once every provider
    has a non-API-error result; a daily-cap 429 raises DailyCapReached.
    """
    providers = list(judges)
    out: dict[str, dict] = {
        q: r for q, r in (done or {}).items() if _compare_complete(r, providers)
    }
    consecutive_api_errors = 0
    for n, row in enumerate(rows, start=1):
        qid = row["query_id"]
        if qid in out:
            continue
        rec = {**_record(row), "by_provider": {}}
        del rec["repeats"]
        for p in providers:
            client, model = judges[p]
            try:
                res = _judge_once(
                    lambda c=client, m=model: judge(c, m, questions[qid], contexts[qid], row["answer"]),
                    sleep,
                    log,
                )
                consecutive_api_errors = 0
            except DailyCapReached:
                raise
            except JudgeParseError:
                res = {"error": "JudgeParseError", "error_kind": "parse"}
                consecutive_api_errors = 0
            except Exception as e:  # recorded; the row is redone on resume
                res = {"error": type(e).__name__, "error_kind": "api"}
                consecutive_api_errors += 1
            rec["by_provider"][p] = res
            sleep(throttles[p])
            if consecutive_api_errors >= MAX_CONSECUTIVE_API_ERRORS:
                raise AbortRun(
                    f"{consecutive_api_errors} consecutive API errors (last: {res['error']} "
                    f"on {p}); {len(out)} completed rows are checkpointed — re-run to resume"
                )
        out[qid] = rec
        if _compare_complete(rec, providers):
            save(out)
        log(f"  [{n}/{len(rows)}] q{qid:>4} " + "  ".join(
            f"{p}: " + ("?" if "error" in x else
                        f"{'A' if x['answered'] else '-'} faith={x['faithfulness']:.2f} "
                        f"ctx={x['context_relevance']:.2f}")
            for p, x in rec["by_provider"].items()
        ))
    return out


# --- agreement statistics ---------------------------------------------------------------


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> float | None:
    """Fleiss' kappa for N subjects, each rated by the same number n of raters.

    Fleiss (1971), "Measuring nominal scale agreement among many raters":

        n_ij  = raters assigning subject i to category j
        P_i   = (sum_j n_ij^2 - n) / (n (n - 1))          per-subject agreement
        P_bar = (1/N) sum_i P_i                           observed agreement
        p_j   = (1/(N n)) sum_i n_ij                      category share
        P_e   = sum_j p_j^2                               chance agreement
        kappa = (P_bar - P_e) / (1 - P_e)

    Returns None — never NaN — where kappa is undefined: no subjects, fewer than two
    raters, or every rating in one category (P_e = 1, so 0/0). In that last case the
    raters agree perfectly, but on a single class, which kappa cannot credit; report
    the raw unanimity rate alongside it.
    """
    if not ratings:
        return None
    n = len(ratings[0])
    if any(len(r) != n for r in ratings):
        raise ValueError("Fleiss' kappa needs the same number of ratings per subject")
    if n < 2:
        return None
    big_n = len(ratings)
    totals: Counter = Counter()
    p_bar = 0.0
    for r in ratings:
        counts = Counter(r)
        totals.update(counts)
        p_bar += (sum(c * c for c in counts.values()) - n) / (n * (n - 1))
    p_bar /= big_n
    if len(totals) < 2:  # exact test for P_e == 1, no float comparison
        return None
    p_e = sum((c / (big_n * n)) ** 2 for c in totals.values())
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha_interval(units: Sequence[Sequence[float]]) -> float | None:
    """Krippendorff's alpha with the interval metric delta(c, k) = (c - k)^2.

    Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability" (Annenberg School
    for Communication, Departmental Papers 43). Only pairable values count: units with
    at least two values, n of them in total. With sums over ORDERED pairs of distinct
    positions (i != j):

        D_o   = (1/n) sum_u [ 1/(m_u - 1) sum_{i != j in u} (v_ui - v_uj)^2 ]
        D_e   = 1/(n (n - 1)) sum_{i != j over all n pairable values} (v_i - v_j)^2
        alpha = 1 - D_o / D_e

    Chosen over ICC because it tolerates missing values natively — a repeat lost to an
    unparseable reply just shrinks its unit (m_u) instead of forcing the whole row out —
    and it needs no choice among ICC's ANOVA model variants. Returns None (never NaN)
    when undefined: fewer than two pairable values, or D_e = 0 (every value identical —
    perfect but uninformative agreement). Pairs are summed directly, not via the
    sum-of-squares identity, so identical values give an exact 0 rather than float dust.
    """
    pairable = [[float(v) for v in u] for u in units if len(u) >= 2]
    values = [v for u in pairable for v in u]
    n = len(values)
    if n < 2:
        return None
    # The i == j terms are (v - v)^2 = 0, so summing over all (i, j) equals i != j.
    d_o = sum(sum((a - b) ** 2 for a in u for b in u) / (len(u) - 1) for u in pairable) / n
    d_e = sum((a - b) ** 2 for a in values for b in values) / (n * (n - 1))
    if d_e == 0:
        return None
    return 1.0 - d_o / d_e


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _mean(xs: list[float]) -> float | None:
    return _r(statistics.mean(xs)) if xs else None


def summarize(records: list[dict], t: float, k: int) -> dict:
    """Agreement metrics for one temperature over the re-judged rows."""
    key = temp_key(t)
    per_row = [(rec, [x for x in rec["repeats"][key] if "error" not in x]) for rec in records]
    all_reps = [(rec, x) for rec, valid in per_row for x in valid]
    complete = [valid for _, valid in per_row if len(valid) == k]
    errors = Counter(
        x["error_kind"] for rec in records for x in rec["repeats"][key] if "error" in x
    )

    answered_sets = [[x["answered"] for x in valid] for valid in complete]
    per_repeat_agreement = []
    for i in range(k):
        pairs = [
            (rec["repeats"][key][i], rec["original"]["answered"])
            for rec in records
            if "error" not in rec["repeats"][key][i]
        ]
        per_repeat_agreement.append(_mean([float(x["answered"] == o) for x, o in pairs]))

    def would_flip(rec: dict, x: dict) -> bool:
        return resolve_answered(rec["verdict"], x["answered"])[0] != rec["answered"]

    judge_rows = [(rec, valid) for rec, valid in per_row if rec["answered_source"] == "judge"]
    answered = {
        "rows_all_repeats_valid": len(complete),
        "unanimous_rate": _mean([float(len(set(s)) == 1) for s in answered_sets]),
        "fleiss_kappa": _r(fleiss_kappa(answered_sets)),
        "agreement_with_original": _mean(
            [float(x["answered"] == rec["original"]["answered"]) for rec, x in all_reps]
        ),
        "agreement_with_original_per_repeat": per_repeat_agreement,
        # How often the judge's call disagrees with the published `answered` (which the
        # verdict line decides for most rows) ...
        "disagrees_with_published_answered": _mean(
            [float(x["answered"] != rec["answered"]) for rec, x in all_reps]
        ),
        # ... and how often that would actually flip it: only rows with no parsed
        # verdict let the judge decide (rag_eval.resolve_answered).
        "judge_decided_rows": len(judge_rows),
        "would_flip_published_answered": _mean(
            [float(would_flip(rec, x)) for rec, x in all_reps]
        ),
        "rows_with_any_flip": sum(any(would_flip(rec, x) for x in v) for rec, v in per_row),
    }

    scores = {}
    for s in SCORE_KEYS:
        stds = [statistics.stdev(x[s] for x in v) for _, v in per_row if len(v) >= 2]
        scores[s] = {
            "mean_row_std": _mean(stds),
            "max_row_std": _r(max(stds)) if stds else None,
            "mean_abs_diff_vs_original": _mean(
                [abs(x[s] - rec["original"][s]) for rec, x in all_reps]
            ),
            "alpha_repeats": _r(krippendorff_alpha_interval([[x[s] for x in v] for _, v in per_row])),
            "alpha_with_original": _r(
                krippendorff_alpha_interval(
                    [[rec["original"][s], *(x[s] for x in v)] for rec, v in per_row]
                )
            ),
        }

    triple = ("answered", *SCORE_KEYS)
    return {
        "temperature": t,
        "rows": len(records),
        "calls": len(records) * k,
        "parse_failures": errors.get("parse", 0),
        "api_errors": errors.get("api", 0),
        "identical_to_original_rate": _mean(
            [float(all(x[f] == rec["original"][f] for f in triple)) for rec, x in all_reps]
        ),
        "answered": answered,
        **scores,
    }


def summarize_compare(records: list[dict], a: str, b: str) -> dict:
    """Per-field agreement between provider `a` and provider `b` on the same rows."""
    both = [
        (rec, rec["by_provider"][a], rec["by_provider"][b])
        for rec in records
        if "error" not in rec["by_provider"][a] and "error" not in rec["by_provider"][b]
    ]
    errors = {
        p: Counter(
            rec["by_provider"][p]["error_kind"] for rec in records if "error" in rec["by_provider"][p]
        )
        for p in (a, b)
    }
    fields: dict[str, dict] = {
        "answered": {
            "agreement": _mean([float(x["answered"] == y["answered"]) for _, x, y in both]),
            # Two raters: Fleiss' kappa with n=2 (Scott's pi); None when undefined.
            "kappa": _r(fleiss_kappa([[x["answered"], y["answered"]] for _, x, y in both])),
            f"agreement_with_original_{a}": _mean(
                [float(x["answered"] == rec["original"]["answered"]) for rec, x, _ in both]
            ),
            f"agreement_with_original_{b}": _mean(
                [float(y["answered"] == rec["original"]["answered"]) for rec, _, y in both]
            ),
            f"answered_rate_{a}": _mean([float(x["answered"]) for _, x, _ in both]),
            f"answered_rate_{b}": _mean([float(y["answered"]) for _, _, y in both]),
        }
    }
    for f in SCORE_KEYS:
        fields[f] = {
            "exact_agreement": _mean([float(x[f] == y[f]) for _, x, y in both]),
            "mean_abs_diff": _mean([abs(x[f] - y[f]) for _, x, y in both]),
            f"mean_{a}": _mean([x[f] for _, x, _ in both]),
            f"mean_{b}": _mean([y[f] for _, _, y in both]),
            "alpha": _r(krippendorff_alpha_interval([[x[f], y[f]] for _, x, y in both])),
            f"mean_abs_diff_vs_original_{a}": _mean(
                [abs(x[f] - rec["original"][f]) for rec, x, _ in both]
            ),
            f"mean_abs_diff_vs_original_{b}": _mean(
                [abs(y[f] - rec["original"][f]) for rec, _, y in both]
            ),
        }
    triple = ("answered", *SCORE_KEYS)
    return {
        "providers": [a, b],
        "rows": len(records),
        "rows_both_valid": len(both),
        "calls": {p: len(records) for p in (a, b)},
        "parse_failures": {p: errors[p].get("parse", 0) for p in (a, b)},
        "api_errors": {p: errors[p].get("api", 0) for p in (a, b)},
        "identical_all_fields_rate": _mean(
            [float(all(x[k] == y[k] for k in triple)) for _, x, y in both]
        ),
        **fields,
    }


# --- output -----------------------------------------------------------------------------


def output_dir(limit: int) -> tuple[Path, bool]:
    """Only a full (no row limit) run may write the committed eval/results/ artifact."""
    if not limit:
        return OUT, True
    return RUNS / f"judge_agreement_limit{limit}", False


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _provider_statement(run: Mapping) -> str:
    """Which provider made the repeats vs the original judgement — explicit, because a
    cross-provider "vs original" number is not the same measurement as a same-provider
    one."""
    orig = run.get("original_judge") or {}
    rp, om = run.get("judge_provider"), orig.get("model")
    op = orig.get("provider")
    if not rp or not op:
        return ""
    assumed = "" if orig.get("provider_source") == "recorded" else " (assumed: not recorded in rag.json)"
    line = (
        f"**Repeats** judged on **{rp}** (`{run['judge_model']}`); the **original** judgement "
        f"in rag.json was made on **{op}** (`{om}`){assumed}.\n\n"
    )
    if rp != op:
        line += (
            "The providers differ: every *vs original* row below compares across providers "
            "(same weights, different serving stack), so it mixes provider differences with "
            "judge nondeterminism. Rows computed over the repeats alone (all K agree, kappa, "
            "alpha over repeats) are within one provider.\n\n"
        )
    return line


def compare_markdown(result: dict) -> str:
    run, sm = result["run"], result["summary"]
    a, b = sm["providers"]
    ja_, jb = run["judges"][a], run["judges"][b]

    def row(label: str, va: object, vb: object | None = None, both: bool = False) -> str:
        def fmt(v):
            return _f(v) if v is None or isinstance(v, float) else str(v)
        return f"| {label} | {fmt(va)} |\n" if both else f"| {label} | {fmt(va)} | {fmt(vb)} |\n"

    per = (
        f"| Per provider | {a} | {b} |\n|---|---|---|\n"
        + row("Model id", f"`{ja_['model']}`", f"`{jb['model']}`")
        + row("Judge calls", sm["calls"][a], sm["calls"][b])
        + row("Unparseable replies", sm["parse_failures"][a], sm["parse_failures"][b])
        + row("API errors", sm["api_errors"][a], sm["api_errors"][b])
        + row("`answered` rate", sm["answered"][f"answered_rate_{a}"], sm["answered"][f"answered_rate_{b}"])
        + row("`answered` agrees with rag.json original",
              sm["answered"][f"agreement_with_original_{a}"], sm["answered"][f"agreement_with_original_{b}"])
    )
    for f, label in (("faithfulness", "Faithfulness"), ("context_relevance", "Context relevance")):
        per += row(f"{label}: mean", sm[f][f"mean_{a}"], sm[f][f"mean_{b}"])
        per += row(f"{label}: mean abs diff vs original",
                   sm[f][f"mean_abs_diff_vs_original_{a}"], sm[f][f"mean_abs_diff_vs_original_{b}"])
    agree = (
        f"| {a} vs {b} (rows where both parsed: {sm['rows_both_valid']}) | Value |\n|---|---|\n"
        + row("All 3 fields identical", sm["identical_all_fields_rate"], both=True)
        + row("`answered`: agreement", sm["answered"]["agreement"], both=True)
        + row("`answered`: kappa (2 raters)", sm["answered"]["kappa"], both=True)
    )
    for f, label in (("faithfulness", "Faithfulness"), ("context_relevance", "Context relevance")):
        agree += row(f"{label}: exact agreement", sm[f]["exact_agreement"], both=True)
        agree += row(f"{label}: mean abs diff", sm[f]["mean_abs_diff"], both=True)
        agree += row(f"{label}: Krippendorff alpha", sm[f]["alpha"], both=True)
    orig = run["original_judge"]
    return (
        f"# Judge provider comparison — {a} vs {b}\n\n"
        f"First {result['n_rows']} non-empty answers from rag.json (source run "
        f"{str(run['source']['git_sha'])[:7]}), each judged ONCE per provider at "
        f"T={PRODUCTION_TEMPERATURE} with the production judge prompt "
        f"({run['judge_prompt_hash'][:12]}) and budget. The rag.json original was judged on "
        f"{orig['provider']} (`{orig['model']}`). Non-canonical: written to data/eval_runs/ "
        f"only.\n\n{per}\n{agree}\nKappa / alpha are n/a when undefined.\n"
    )


def to_markdown(result: dict) -> str:
    run, summ = result["run"], result["summary"]
    temps = list(summ)
    head = "| Metric | " + " | ".join(f"T={t}" for t in temps) + " |\n"
    head += "|---|" + "---|" * len(temps) + "\n"

    def line(label: str, get: Callable[[dict], object]) -> str:
        cells = []
        for t in temps:
            v = get(summ[t])
            cells.append(_f(v) if v is None or isinstance(v, float) else str(v))
        return f"| {label} | " + " | ".join(cells) + " |\n"

    body = [
        line("Rows re-judged", lambda s: s["rows"]),
        line("Judge calls", lambda s: s["calls"]),
        line("Unparseable replies", lambda s: s["parse_failures"]),
        line("API errors", lambda s: s["api_errors"]),
        line("Repeat identical to original (all 3 fields)", lambda s: s["identical_to_original_rate"]),
        line("`answered`: all K repeats agree", lambda s: s["answered"]["unanimous_rate"]),
        line("`answered`: Fleiss' kappa", lambda s: s["answered"]["fleiss_kappa"]),
        line("`answered`: agrees with original judgement", lambda s: s["answered"]["agreement_with_original"]),
        line("`answered`: disagrees with published `answered`", lambda s: s["answered"]["disagrees_with_published_answered"]),
        line("`answered`: would flip published `answered`", lambda s: s["answered"]["would_flip_published_answered"]),
    ]
    for s, label in (("faithfulness", "Faithfulness"), ("context_relevance", "Context relevance")):
        body += [
            line(f"{label}: mean per-row std", lambda x, s=s: x[s]["mean_row_std"]),
            line(f"{label}: mean abs diff vs original", lambda x, s=s: x[s]["mean_abs_diff_vs_original"]),
            line(f"{label}: Krippendorff alpha (repeats)", lambda x, s=s: x[s]["alpha_repeats"]),
            line(f"{label}: Krippendorff alpha (+ original)", lambda x, s=s: x[s]["alpha_with_original"]),
        ]
    skipped = result["skipped_empty_answer"]
    judge_rows = next(iter(summ.values()))["answered"]["judge_decided_rows"] if summ else 0
    return (
        f"# Judge consistency — re-judging the stored RAG answers\n\n"
        f"{result['n_rows']} answers from rag.json (source run {str(run['source']['git_sha'])[:7]}) "
        f"· judge={run['judge_model']} · K={run['repeats']} per temperature · "
        f"prompt={run['judge_prompt_hash'][:12]}"
        + ("" if run["canonical"] else f" · SSR_JUDGE_LIMIT={run['row_limit']} (non-canonical)")
        + "\n\n"
        + _provider_statement(run)
        + (f"Skipped (empty answer): {', '.join(skipped)}.\n\n" if skipped else "")
        + "Same claim, context, answer, prompt and model as the original judgement; only the "
        f"temperature differs. T={PRODUCTION_TEMPERATURE} is production, so movement there "
        "is provider nondeterminism. `answered` is published from the verdict line for most "
        f"rows; the judge decides it only for the {judge_rows} with no parsed verdict, so "
        "\"would flip\" counts only those.\n\n"
        + head
        + "".join(body)
        + "\nKappa / alpha are n/a when undefined (all ratings identical: agreement is total "
        "but carries no information beyond that).\n"
    )


# --- checkpoint -------------------------------------------------------------------------


def _signature(source_sha: str, cfg: Config, qids: list[str], judges: Sequence[str]) -> str:
    """`judges`: "provider:model" per judge in use, so a checkpoint never mixes the
    repeats of two providers (or of the comparison mode and the repeats mode)."""
    payload = json.dumps(
        {
            "source_sha256": source_sha,
            "judge_prompt_hash": rag_eval._sha256(JUDGE_SYSTEM),
            "judges": list(judges),
            "mode": "compare" if cfg.compare_n else "repeats",
            "k": cfg.k,
            "temperatures": [temp_key(t) for t in cfg.temperatures],
            "qids": qids,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_checkpoint(path: Path, sig: str) -> dict[str, dict]:
    if os.environ.get("SSR_EVAL_REFRESH") or not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}  # corrupt checkpoint degrades to recompute
    if not isinstance(blob, dict) or blob.get("signature") != sig:
        return {}
    rows = blob.get("rows")
    return rows if isinstance(rows, dict) else {}


# --- entry point ------------------------------------------------------------------------


def _checkpoint_saver(ckpt: Path, sig: str) -> Callable[[dict[str, dict]], None]:
    def save(recs: dict[str, dict]) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signature": sig, "rows": recs}))
        tmp.replace(ckpt)

    return save


def _stop(e: AbortRun) -> SystemExit:
    if isinstance(e, DailyCapReached):
        return SystemExit(
            f"Stopped: {e}.\nCompleted rows are checkpointed in {CACHE}/; re-run the same "
            "command once the provider's daily budget recovers to resume."
        )
    return SystemExit(f"Aborted: {e}")


def main() -> None:
    from openai import OpenAI

    from app.ingest.corpus import load_documents, load_queries_qrels

    cfg = Config.from_env()
    compare = cfg.compare_n > 0
    # Keys are only required once LLM calls will actually be made (not for a dry run).
    need_key = not cfg.dry_run
    if compare:
        endpoints = {
            p: resolve_endpoint("judge", provider=p, require_key=need_key)
            for p in COMPARE_PROVIDERS
        }
    else:
        ep = resolve_endpoint("judge", require_key=need_key)
        endpoints = {ep.provider: ep}
    judge_ep = next(iter(endpoints.values()))
    throttles = {p: throttle_s(p) for p in endpoints}

    raw = SOURCE.read_bytes()
    blob = json.loads(raw)
    for e in endpoints.values():
        check_source(blob, judge_model=e.model)
    if compare:
        rows, skipped = select_rows(blob["rows"], 0, cfg.seed)
        rows = rows[: cfg.compare_n]
    else:
        rows, skipped = select_rows(blob["rows"], cfg.limit, cfg.seed)

    queries, _ = load_queries_qrels()
    missing = [r["query_id"] for r in rows if r["query_id"] not in queries]
    if missing:
        raise StaleResultsError(f"query ids not in {settings.eval_dataset}: {missing[:5]}")
    questions = {r["query_id"]: queries[r["query_id"]] for r in rows}
    docs_by_id = {d["doc_id"]: d for d in load_documents()}
    contexts = {r["query_id"]: rebuild_contexts(r["retrieved_doc_ids"], docs_by_id) for r in rows}

    source_sha = hashlib.sha256(raw).hexdigest()
    qids = [r["query_id"] for r in rows]
    sig = _signature(source_sha, cfg, qids, [f"{p}:{e.model}" for p, e in endpoints.items()])
    ckpt = CACHE / f"judge_agreement-{sig}.json"
    done = {q: r for q, r in _load_checkpoint(ckpt, sig).items() if q in set(qids)}
    src_run = blob["run"]
    orig = original_judge(src_run)
    for e in endpoints.values():
        print(describe_with_ignored(e), flush=True)

    if compare:
        done = {q: r for q, r in done.items() if _compare_complete(r, list(endpoints))}
        out = RUNS / f"judge_provider_compare_n{cfg.compare_n}"
        calls, secs = estimate_compare(len(rows) - len(done), throttles)
        print(
            f"Provider comparison: first {len(rows)} rows ({len(skipped)} skipped for an empty "
            f"answer" + (f", {len(done)} resumed from checkpoint" if done else "") + ") x "
            f"{len(endpoints)} providers, once each at T={PRODUCTION_TEMPERATURE}\n"
            f"  -> {sum(calls.values())} judge calls: "
            + ", ".join(f"{calls[p]} on {p} ({e.model}, {throttles[p]:.1f}s throttle)"
                        for p, e in endpoints.items())
            + f"; est. {secs / 60:.0f} min\n  original judgement: {orig['provider']} "
            f"({orig['model']})\n  output -> {out} (non-canonical)",
            flush=True,
        )
    else:
        done = {q: r for q, r in done.items() if _complete(r, cfg)}
        out, canonical = output_dir(cfg.limit)
        calls_n, secs = estimate(len(rows) - len(done), cfg, throttles[judge_ep.provider])
        print(
            f"Judge consistency: {len(rows)} rows ({len(skipped)} skipped for an empty answer"
            + (f", {len(done)} resumed from checkpoint" if done else "")
            + f") x K={cfg.k} x temperatures {list(cfg.temperatures)}\n"
            f"  judge={judge_ep.provider} {judge_ep.model}  -> {calls_n} judge calls, est. "
            f"{secs / 60:.0f} min ({throttles[judge_ep.provider]:.1f}s throttle + "
            f"~{EST_LATENCY_S:.0f}s latency per call)\n"
            f"  original judgement: {orig['provider']} ({orig['model']})"
            + ("" if orig["provider"] == judge_ep.provider else "  <- different provider")
            + f"\n  output -> {out}{'' if canonical else ' (non-canonical: row limit set)'}",
            flush=True,
        )
    if cfg.dry_run:
        print("SSR_JUDGE_DRY_RUN set: metadata and contexts check out; no LLM calls made.")
        return

    clients = {p: build_client(e, factory=OpenAI) for p, e in endpoints.items()}
    save = _checkpoint_saver(ckpt, sig)
    source_meta = {
        "path": str(SOURCE),
        "sha256": source_sha,
        "git_sha": src_run.get("git_sha"),
        "sample_seed": src_run.get("sample_seed"),
        "generator_model": src_run.get("generator_model"),
    }
    common_run = {
        "git_sha": rag_eval._git_sha(),
        "judge_prompt_hash": rag_eval._sha256(JUDGE_SYSTEM),
        "original_judge": orig,
        "source": source_meta,
    }

    t0 = time.time()
    if compare:
        try:
            records = run_compare(
                {p: (clients[p], e.model) for p, e in endpoints.items()},
                rows, questions, contexts, throttles=throttles, done=done, save=save,
            )
        except AbortRun as e:
            raise _stop(e) from e
        ordered = [records[q] for q in qids]
        a, b = COMPARE_PROVIDERS
        result = {
            "n_rows": len(ordered),
            "skipped_empty_answer": skipped,
            "summary": summarize_compare(ordered, a, b),
            "run": {
                **common_run,
                "mode": "provider_comparison",
                # Per provider: provider, base URL, model id, provider fields sent. No keys.
                "judges": {p: e.metadata() for p, e in endpoints.items()},
                "temperature": PRODUCTION_TEMPERATURE,
                "throttle_s": throttles,
                "wall_time_s": round(time.time() - t0, 1),
            },
            "rows": ordered,
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "judge_provider_compare.json").write_text(json.dumps(result, indent=2))
        (out / "judge_provider_compare.md").write_text(compare_markdown(result))
        print(f"Wrote {out / 'judge_provider_compare.md'} and {out / 'judge_provider_compare.json'}")
        return

    client = clients[judge_ep.provider]
    try:
        records, requested = run_repeats(
            client, rows, questions, contexts, cfg,
            model=judge_ep.model, throttle=throttles[judge_ep.provider], done=done, save=save,
        )
    except AbortRun as e:
        raise _stop(e) from e

    ordered = [records[q] for q in qids]
    result = {
        "n_rows": len(ordered),
        "skipped_empty_answer": skipped,
        "summary": {temp_key(t): summarize(ordered, t, cfg.k) for t in cfg.temperatures},
        "run": {
            **common_run,
            "judge_model": judge_ep.model,
            "judge_provider": judge_ep.provider,
            "judge_base_url": judge_ep.base_url,
            "judge_extra_body": judge_ep.extra_body(),
            "repeats": cfg.k,
            "temperatures": list(cfg.temperatures),
            "production_temperature": PRODUCTION_TEMPERATURE,
            # What judge() itself sent before the override — if this ever differs from
            # production_temperature, the T=0.0 column is no longer "production".
            "judge_requested_temperature": requested,
            "seed": cfg.seed,
            "seed_use": "row subset under SSR_JUDGE_LIMIT only; no seed is sent to the provider",
            "row_limit": cfg.limit,
            "canonical": canonical,
            "throttle_s": throttles[judge_ep.provider],
            "wall_time_s": round(time.time() - t0, 1),
        },
        "rows": ordered,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_agreement.json").write_text(json.dumps(result, indent=2))
    (out / "judge_agreement.md").write_text(to_markdown(result))
    print(f"Wrote {out / 'judge_agreement.md'} and {out / 'judge_agreement.json'}")


if __name__ == "__main__":
    main()
