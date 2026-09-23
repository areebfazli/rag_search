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
code: re-scoring a stale file would measure a different judge than the one in use.

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
    SSR_EVAL_REFRESH=1      ignore the resume checkpoint in data/eval_cache/
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

from app.core.config import settings
from app.core.interfaces import SearchHit, hit_passage
from app.eval import rag_eval
from app.eval.rag_eval import JUDGE_SYSTEM, THROTTLE_S, JudgeParseError, judge, resolve_answered

SOURCE = Path("eval/results/rag.json")
OUT = Path("eval/results")  # canonical (no row limit) run only — the committed artifact
RUNS = Path("data/eval_runs")  # every limited run (gitignored)
CACHE = Path("data/eval_cache")  # resume checkpoint (gitignored)

PRODUCTION_TEMPERATURE = 0.0  # what rag_eval.judge() sends
DEFAULT_K = 3
DEFAULT_TEMPERATURES = (PRODUCTION_TEMPERATURE, 0.7)
DEFAULT_SEED = 13
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


# --- configuration --------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    k: int = DEFAULT_K
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    limit: int = 0
    seed: int = DEFAULT_SEED
    dry_run: bool = False

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
        return cls(
            k=k,
            temperatures=temps,
            limit=limit,
            seed=int(env.get("SSR_JUDGE_SEED", DEFAULT_SEED)),
            dry_run=env.get("SSR_JUDGE_DRY_RUN", "") not in ("", "0"),
        )


def temp_key(t: float) -> str:
    """JSON key for a temperature: "0.0", "0.7" (repr keeps the trailing .0)."""
    return repr(float(t))


def estimate(n_rows: int, cfg: Config) -> tuple[int, float]:
    """(judge calls, estimated seconds). Every call is followed by THROTTLE_S."""
    calls = n_rows * cfg.k * len(cfg.temperatures)
    return calls, calls * (THROTTLE_S + EST_LATENCY_S)


# --- input validation -----------------------------------------------------------------


def check_source(blob: object) -> None:
    """Refuse a rag.json whose judge isn't the one the current code runs.

    judge_prompt_hash is computed exactly as rag_eval.run_metadata does, so an edit to
    JUDGE_SYSTEM since the run makes the file stale — the stored "original" scores
    came from a different rubric, and comparing against them would mix two judges.
    """
    run = blob.get("run") if isinstance(blob, dict) else None
    if not isinstance(run, dict):
        raise StaleResultsError("rag.json has no `run` metadata; re-run `make eval-rag`")
    expected = {
        "judge_prompt_hash": rag_eval._sha256(JUDGE_SYSTEM),
        "judge_model": settings.judge_model,
    }
    bad = {k: (run.get(k), v) for k, v in expected.items() if run.get(k) != v}
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


def run_repeats(
    client: object,
    rows: list[dict],
    questions: Mapping[str, str],
    contexts: Mapping[str, list[str]],
    cfg: Config,
    *,
    done: Mapping[str, dict] | None = None,
    save: Callable[[dict[str, dict]], None] = lambda _: None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, dict], float | None]:
    """Re-judge every row K times per temperature. Returns ({qid: record}, the
    temperature judge() itself requested — expected to be PRODUCTION_TEMPERATURE).

    `done` rows (from the checkpoint) are reused; `save` is called after each new row.
    """
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
                    s = judge(tc, settings.judge_model, questions[qid], contexts[qid], row["answer"])
                    consecutive_api_errors = 0
                except JudgeParseError:
                    s = {"error": "JudgeParseError", "error_kind": "parse"}
                    consecutive_api_errors = 0
                except Exception as e:  # rate limit / network: recorded, row is redone on resume
                    s = {"error": type(e).__name__, "error_kind": "api"}
                    consecutive_api_errors += 1
                requested = tc.requested_temperature if requested is None else requested
                reps.append(s)
                sleep(THROTTLE_S)
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


# --- output -----------------------------------------------------------------------------


def output_dir(limit: int) -> tuple[Path, bool]:
    """Only a full (no row limit) run may write the committed eval/results/ artifact."""
    if not limit:
        return OUT, True
    return RUNS / f"judge_agreement_limit{limit}", False


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


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


def _signature(source_sha: str, cfg: Config, qids: list[str]) -> str:
    payload = json.dumps(
        {
            "source_sha256": source_sha,
            "judge_prompt_hash": rag_eval._sha256(JUDGE_SYSTEM),
            "judge_model": settings.judge_model,
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


def main() -> None:
    from openai import OpenAI

    from app.ingest.corpus import load_documents, load_queries_qrels

    cfg = Config.from_env()
    raw = SOURCE.read_bytes()
    blob = json.loads(raw)
    check_source(blob)
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
    sig = _signature(source_sha, cfg, qids)
    ckpt = CACHE / f"judge_agreement-{sig}.json"
    done = {q: r for q, r in _load_checkpoint(ckpt, sig).items() if q in set(qids)}
    done = {q: r for q, r in done.items() if _complete(r, cfg)}
    out, canonical = output_dir(cfg.limit)

    calls, secs = estimate(len(rows) - len(done), cfg)
    print(
        f"Judge consistency: {len(rows)} rows ({len(skipped)} skipped for an empty answer"
        + (f", {len(done)} resumed from checkpoint" if done else "")
        + f") x K={cfg.k} x temperatures {list(cfg.temperatures)}\n"
        f"  judge={settings.judge_model}  -> {calls} judge calls, est. {secs / 60:.0f} min "
        f"({THROTTLE_S:.0f}s throttle + ~{EST_LATENCY_S:.0f}s latency per call)\n"
        f"  output -> {out}{'' if canonical else ' (non-canonical: row limit set)'}",
        flush=True,
    )
    if cfg.dry_run:
        print("SSR_JUDGE_DRY_RUN set: metadata and contexts check out; no LLM calls made.")
        return
    if not settings.llm_api_key:
        raise SystemExit("SSR_LLM_API_KEY is not set; the judge needs it.")

    client = OpenAI(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=5, timeout=30.0
    )

    def save(recs: dict[str, dict]) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signature": sig, "rows": recs}))
        tmp.replace(ckpt)

    t0 = time.time()
    try:
        records, requested = run_repeats(client, rows, questions, contexts, cfg, done=done, save=save)
    except AbortRun as e:
        raise SystemExit(f"Aborted: {e}") from e

    ordered = [records[q] for q in qids]
    src_run = blob["run"]
    result = {
        "n_rows": len(ordered),
        "skipped_empty_answer": skipped,
        "summary": {temp_key(t): summarize(ordered, t, cfg.k) for t in cfg.temperatures},
        "run": {
            "git_sha": rag_eval._git_sha(),
            "judge_model": settings.judge_model,
            "judge_prompt_hash": rag_eval._sha256(JUDGE_SYSTEM),
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
            "throttle_s": THROTTLE_S,
            "wall_time_s": round(time.time() - t0, 1),
            "source": {
                "path": str(SOURCE),
                "sha256": source_sha,
                "git_sha": src_run.get("git_sha"),
                "sample_seed": src_run.get("sample_seed"),
                "generator_model": src_run.get("generator_model"),
            },
        },
        "rows": ordered,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_agreement.json").write_text(json.dumps(result, indent=2))
    (out / "judge_agreement.md").write_text(to_markdown(result))
    print(f"Wrote {out / 'judge_agreement.md'} and {out / 'judge_agreement.json'}")


if __name__ == "__main__":
    main()
