"""Retrieval latency benchmark — per-query wall-clock for every eval config.

The hybrid-default decision rests partly on cost (reranking roughly ties or loses on
nDCG@10 while costing seconds per query), so the cost side gets a committed artifact
too, not just console output. Every query goes through SearchService.retrieve, the
same call the API and retrieval_eval make, with the same shapes:

* Eval-shaped: top_k = candidate_k = retrieval_eval.DEPTH (100), reranking the
  configured top-settings.rerank_candidates slice. These rows correspond one-to-one
  with the rows of eval/results/retrieval.md.
* API-shaped (top_k=8, candidate_k left to the service default) for bm25 and dense
  only. For hybrid and hybrid_rerank the API shape does the same work — candidate_k
  resolves to max(dense_top_k, top_k) = 100 either way and the rerank slice is fixed —
  so it would only re-measure the eval-shaped row.

First-stage configs run over every test query. The cross-encoder configs run over a
fixed seeded sample of RERANK_SAMPLE queries: at the ~32 s/query previously observed
for bge-reranker-base (and ~7.5 s for MiniLM), 40 queries is ~27 min of reranking,
which keeps the whole benchmark near 30 min on a 4-core laptop CPU while still giving
p90/p95 some support.

Warm-up: WARMUP queries run before each config and are not counted, so model loading,
lazy initialization and first-call allocator/kernel costs stay out of the steady-state
numbers. Those costs are reported separately as cold start (service construction,
reranker load, and each config's first call). Warm-up texts are drawn from queries
outside the rerank sample; the pipeline keeps no per-query result cache, so reusing a
first-stage query later does not make its measured call cheaper.

What is NOT measured: HTTP, JSON serialization, rate limiting and the API's retrieval
lock — this is in-process retrieval time only.

Writes eval/results/latency.{md,json} only for the canonical run (full test split, no
SSR_EVAL_LIMIT); any other run goes to data/eval_runs/ (gitignored), mirroring
retrieval_eval's guard.

Run:
    uv run python -m app.eval.latency
    SSR_EVAL_LIMIT=20 uv run python -m app.eval.latency   # smoke subset
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from app.core.config import settings
from app.eval.retrieval_eval import CANONICAL_DATASET, CONFIGS, DEPTH, OUT, RUNS

API_TOP_K = 8  # the /search endpoint's default top_k
RERANK_SAMPLE = 40  # queries timed per cross-encoder config (see module docstring)
SEED = 13  # fixes which queries form the rerank sample
WARMUP = 3  # untimed queries run before each config

_EVAL = {key: (label, mode, reranker) for key, label, mode, reranker in CONFIGS}


def _config(key: str, top_k: int, sample: str, suffix: str = "") -> dict:
    label, mode, reranker = _EVAL[key]
    return {
        "key": key + suffix,
        "label": label,
        "mode": mode,
        "reranker": reranker,
        "top_k": top_k,
        # retrieval_eval passes candidate_k=DEPTH; the API passes nothing and lets the
        # service default apply. Mirror whichever shape the row claims to be.
        "candidate_k": DEPTH if top_k == DEPTH else None,
        "sample": sample,  # "all" test queries, or the seeded "rerank" sample
    }


LAT_CONFIGS: list[dict] = [
    _config("bm25", DEPTH, "all"),
    _config("dense", DEPTH, "all"),
    _config("hybrid", DEPTH, "all"),
    _config("bm25", API_TOP_K, "all", "_api"),
    _config("dense", API_TOP_K, "all", "_api"),
    _config("rerank_minilm", DEPTH, "rerank"),
    _config("rerank_bge", DEPTH, "rerank"),
]


# --- statistics ------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile (0..100) by linear interpolation between order statistics —
    the same definition as numpy's default ("linear"), hand-rolled to stay small."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


# --- timing ----------------------------------------------------------------------


def time_config(
    service,
    queries: Sequence[tuple[str, str]],
    cfg: dict,
    warmup: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, float], list[float]]:
    """Run `warmup` untimed-for-the-stats, then time each query.

    Returns ({qid: seconds} for the measured queries, [seconds] per warm-up call).
    Warm-up durations are returned only so the first one can be reported as the
    config's cold first call; they never enter the per-query stats.
    """

    def one(text: str) -> float:
        t0 = clock()
        service.retrieve(text, mode=cfg["mode"], top_k=cfg["top_k"], candidate_k=cfg["candidate_k"])
        return clock() - t0

    warm = [one(text) for text in warmup]
    return {qid: one(text) for qid, text in queries}, warm


def pick_queries(
    queries: dict[str, str], n_sample: int = RERANK_SAMPLE, n_warmup: int = WARMUP, seed: int = SEED
) -> tuple[list[tuple[str, str]], list[str]]:
    """(the seeded rerank sample, warm-up texts drawn from outside it when possible).

    Sampled from the load-order list, so the same seed always yields the same queries.
    """
    items = list(queries.items())
    sample = random.Random(seed).sample(items, min(n_sample, len(items)))
    chosen = {qid for qid, _ in sample}
    rest = [text for qid, text in items if qid not in chosen] or [text for _, text in items]
    return sample, rest[:n_warmup]


def output_dir(limit: int, n_queries: int, dataset: str | None = None) -> tuple[Path, bool]:
    """eval/results/ for the canonical full run only; everything else to data/eval_runs/."""
    dataset = dataset or settings.eval_dataset
    if not limit and dataset == CANONICAL_DATASET:
        return OUT, True
    slug = "".join(c if c.isalnum() else "-" for c in dataset).strip("-")
    return RUNS / f"latency_{slug}_{n_queries}", False


# --- environment -----------------------------------------------------------------


def _cpu() -> tuple[str, int | None]:
    """(CPU model name, physical core count) from /proc/cpuinfo, best effort."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor() or "unknown", None
    model, cores, phys, per_socket = "unknown", set(), None, {}
    for line in text.splitlines():
        k, _, v = (s.strip() for s in line.partition(":"))
        if k == "model name" and model == "unknown":
            model = v
        elif k == "physical id":
            phys = v
        elif k == "core id":
            cores.add((phys, v))
        elif k == "cpu cores":
            per_socket[phys] = int(v)
    return model, (len(cores) or sum(per_socket.values()) or None)


def _git() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _loadavg() -> tuple[float, ...] | None:
    try:
        return tuple(round(x, 2) for x in os.getloadavg())
    except OSError:
        return None


def environment(n_rerank: int, loadavg_start: tuple[float, ...] | None) -> dict:
    import torch

    model, physical = _cpu()
    sha, dirty = _git()
    return {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "cpu_model": model,
        "logical_cores": os.cpu_count(),
        "physical_cores": physical,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": sha,
        "git_dirty": dirty,
        "dataset": settings.eval_dataset,
        "candidate_depth": DEPTH,
        "rerank_candidates": settings.rerank_candidates,
        "rerank_batch_size": settings.rerank_batch_size,
        "api_top_k": API_TOP_K,
        # Other processes competing for the CPU inflate every number; the 1/5/15-min
        # load averages at start and end make a contended run visible in the artifact.
        "loadavg_start": loadavg_start,
        "loadavg_end": _loadavg(),
        "rerank_sample": n_rerank,  # as run (SSR_EVAL_LIMIT can cap it)
        "seed": SEED,
        "warmup_queries": WARMUP,
    }


# --- report ----------------------------------------------------------------------


def to_markdown(report: dict, title: str) -> str:
    env = report["environment"]
    lines = [
        f"# Retrieval latency — {title}",
        "",
        "Per-query wall-clock of `SearchService.retrieve` (in-process; excludes HTTP and "
        "the API's retrieval lock), seconds, after "
        f"{env['warmup_queries']} untimed warm-up queries per config.",
        "",
        "| Config | Mode | top_k | n | mean | p50 | p90 | p95 | max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in report["configs"]:
        s = cfg["stats"]
        cells = " | ".join(f"{s[k]:.3f}" for k in ("mean", "p50", "p90", "p95", "max"))
        lines.append(
            f"| {cfg['label']} | {cfg['mode']} | {cfg['top_k']} | {s['n']} | {cells} |"
        )
    cold = report["cold_start"]
    lines += [
        "",
        f"First-stage rows cover every test query; rerank rows a fixed sample of "
        f"{env['rerank_sample']} (seed {env['seed']}). Reranking scores the top "
        f"{env['rerank_candidates']} of {env['candidate_depth']} fused candidates "
        f"(batch {env['rerank_batch_size']}), exactly as in `retrieval.md`. API-shaped "
        f"rows (top_k={env['api_top_k']}) are shown for bm25/dense only: for hybrid and "
        "rerank the API shape does identical work (candidate pool 100, fixed slice).",
        "",
        "## Cold start (not in the table above)",
        "",
        f"- SearchService construction (embedder + Qdrant + BM25 load): "
        f"{cold['service_init_s']:.2f} s",
    ]
    for key, secs in cold["reranker_load_s"].items():
        lines.append(f"- Cross-encoder load ({key}): {secs:.2f} s")
    lines.append("- First call per config (first warm-up query): " + ", ".join(
        f"{c['key']} {c['first_call_s']:.3f} s" for c in report["configs"]
    ))
    dirty = " (dirty tree)" if env["git_dirty"] else ""
    lines += [
        "",
        "## Environment",
        "",
        f"- CPU: {env['cpu_model']} — {env['physical_cores']} physical / "
        f"{env['logical_cores']} logical cores; load average (1/5/15 min) "
        f"{env['loadavg_start']} at start, {env['loadavg_end']} at end",
        f"- torch {env['torch_version']}, {env['torch_num_threads']} threads; "
        f"Python {env['python_version']}; {env['platform']}",
        f"- git {env['git_sha']}{dirty}; run {env['date']}; "
        f"total wall-clock {report['total_runtime_s'] / 60:.1f} min",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from app.ingest.corpus import load_queries_qrels
    from app.retrieve.service import SearchService

    t_start, load_start = time.perf_counter(), _loadavg()
    queries, _ = load_queries_qrels()
    if limit := int(os.environ.get("SSR_EVAL_LIMIT", "0")):  # smoke-test escape hatch
        queries = dict(list(queries.items())[:limit])
    out, canonical = output_dir(limit, len(queries))
    rerank_sample, warmup = pick_queries(queries)
    all_queries = list(queries.items())
    print(
        f"Latency: {settings.eval_dataset}, {len(all_queries)} first-stage queries, "
        f"{len(rerank_sample)} rerank queries, {len(warmup)} warm-up"
    )
    if not canonical:
        print(f"Non-canonical run (limit/dataset) — writing to {out}, not {OUT}")

    t0 = time.perf_counter()
    service = SearchService()
    cold: dict = {"service_init_s": time.perf_counter() - t0, "reranker_load_s": {}}

    results = []
    for cfg in LAT_CONFIGS:
        print(f"  {cfg['key']} (top_k={cfg['top_k']})", flush=True)
        if cfg["reranker"] is not None:
            from app.rerank.cross_encoder import CrossEncoderReranker

            t0 = time.perf_counter()
            service.reranker = CrossEncoderReranker(cfg["reranker"])
            cold["reranker_load_s"][cfg["key"]] = time.perf_counter() - t0
        measured = all_queries if cfg["sample"] == "all" else rerank_sample
        per_query, warm = time_config(service, measured, cfg, warmup)
        if cfg["reranker"] is not None:  # free the cross-encoder before the next one
            service.reranker = None
            gc.collect()
        stats = summarize(list(per_query.values()))
        print(
            f"      n={stats['n']}  mean={stats['mean']:.3f}s  p50={stats['p50']:.3f}s  "
            f"p95={stats['p95']:.3f}s",
            flush=True,
        )
        results.append(
            {**cfg, "stats": stats, "first_call_s": warm[0] if warm else None,
             "warmup_s": warm, "per_query_s": per_query}
        )

    report = {
        "environment": environment(len(rerank_sample), load_start),
        "cold_start": cold,
        "configs": results,
        "total_runtime_s": time.perf_counter() - t_start,
    }
    title = (
        f"BEIR/SciFact ({len(all_queries)} test queries)"
        if canonical
        else f"{settings.eval_dataset} ({len(all_queries)} queries"
        + (f", SSR_EVAL_LIMIT={limit}" if limit else "")
        + ", non-canonical)"
    )
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(report, title)
    (out / "latency.md").write_text(md)
    (out / "latency.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("\n" + md)
    print(f"Wrote {out / 'latency.md'} and {out / 'latency.json'}")


if __name__ == "__main__":
    main()
