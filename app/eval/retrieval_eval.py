"""Retrieval evaluation on BEIR/SciFact gold qrels with ranx.

Runs each config over the test queries and reports nDCG@10, Recall@100, MRR@10 and
MAP@100 with paired significance tests — the headline artifact of the project.
Writes a Markdown table + JSON to eval/results/ — but only for the canonical run (full
test split, no SSR_EVAL_LIMIT). Any other run goes to data/eval_runs/ (gitignored), so
a smoke test can never overwrite the committed table.

Two rerankers are evaluated so "use a domain-appropriate reranker" is a measured
claim rather than an assumed one: the CPU-default MS-MARCO MiniLM (trained on short
web queries) and BAAI/bge-reranker-base (trained on diverse retrieval data including
scientific text).

Per-config runs are cached under data/eval_cache/ (gitignored), keyed by a signature
of everything that affects the result, so an interrupted run resumes cheaply — the
bge reranker alone is ~3h on a laptop CPU.

Run:
    uv run python -m app.eval.retrieval_eval
    SSR_EVAL_REFRESH=1 uv run python -m app.eval.retrieval_eval   # ignore the cache
    SSR_EVAL_LIMIT=20 uv run python -m app.eval.retrieval_eval    # smoke subset
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

from app.core.config import Settings, settings
from app.ingest.build_index import MANIFEST_PATH
from app.ingest.corpus import load_queries_qrels
from app.retrieve.service import SearchService

# (key, table label, retrieval mode, reranker model or None)
CONFIGS: list[tuple[str, str, str, str | None]] = [
    ("bm25", "BM25", "bm25", None),
    ("dense", "Dense (bge-small)", "dense", None),
    ("hybrid", "Hybrid (RRF)", "hybrid", None),
    (
        "rerank_minilm",
        "Hybrid + rerank (MS-MARCO MiniLM)",
        "hybrid_rerank",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
    (
        "rerank_bge",
        "Hybrid + rerank (bge-reranker-base)",
        "hybrid_rerank",
        "BAAI/bge-reranker-base",
    ),
]

METRICS = ["ndcg@10", "recall@100", "mrr@10", "map@100"]
PRETTY = {"ndcg@10": "nDCG@10", "recall@100": "Recall@100", "mrr@10": "MRR@10", "map@100": "MAP@100"}
DEPTH = 100  # first-stage candidate depth = result depth (enough for Recall@100)
MAX_P = 0.05  # significance threshold for the paired tests
CHECKPOINT_EVERY = 20  # persist partial progress this often (cross-encoders are slow)

OUT = Path("eval/results")  # canonical run only — the committed artifact
RUNS = Path("data/eval_runs")  # every other run (gitignored)
CACHE = Path("data/eval_cache")
# Only a full run on this split may write to OUT. Read off the Settings field default
# rather than restated, so the two cannot drift apart.
CANONICAL_DATASET = Settings.model_fields["eval_dataset"].default

# Sentinel for "index built before manifests existed". It is NOT folded into the
# payload: a missing manifest leaves the signature byte-identical to what pre-manifest
# code computed, so the existing data/eval_cache/ entries behind the committed tables
# stay valid instead of being silently invalidated (and hours of cross-encoder runs
# redone). The first `make index` under this code writes a manifest, which then
# changes every signature — correctly, since the index was just rebuilt.
NO_MANIFEST = None


def _index_fingerprint() -> str | None:
    """sha256 of data/index_manifest.json's raw bytes, or NO_MANIFEST if absent.

    Raw bytes, not parsed JSON: a corrupt manifest still hashes to *something* that
    differs from any good one, which fails safe (recompute) rather than matching.
    """
    try:
        return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    except FileNotFoundError:
        return NO_MANIFEST


def _signature(mode: str, reranker: str | None, n_queries: int) -> str:
    """The settings — and the index they run against — that change a run's output. A
    cache entry is only reused when this matches, so changing depth/rerank
    slice/dataset, or reindexing a different corpus/model/BM25 setup, can't silently
    serve stale scores.

    settings.rerank_batch_size is deliberately absent: it is a pure performance knob.
    Measured on MiniLM, batch 8 vs 32 moves scores by ~1e-6 (float reduction noise
    from padding) and leaves the ranking identical, so it cannot change a metric.

    The index is covered through data/index_manifest.json (see _index_fingerprint):
    what it was built FROM, not its bytes. Two caveats remain:
      * A reindex with identical inputs keeps the signature. Nothing in the build is
        random (fixed corpus order, exact BM25, CPU embeddings under the locked torch),
        so that is the same index the cache was scored against.
      * An index built before manifests existed has none; the signature then falls
        back to the settings-only form (NO_MANIFEST), which cannot detect a reindex.
        Use SSR_EVAL_REFRESH=1 if that index may have been rebuilt mid-run.
    """
    fields: dict = {
        "mode": mode,
        "reranker": reranker,
        "depth": DEPTH,
        "rerank_candidates": settings.rerank_candidates,
        "rrf_k": settings.rrf_k,
        "dataset": settings.eval_dataset,
        "embedding_model": settings.embedding_model,
        # Changing the query prefix changes every dense result, so it belongs here.
        "embedding_query_prefix": settings.embedding_query_prefix,
        "n_queries": n_queries,
    }
    if (index := _index_fingerprint()) is not NO_MANIFEST:
        fields["index_manifest"] = index
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _output_dir(limit: int, n_queries: int) -> tuple[Path, bool]:
    """Where this run's tables go, and whether it is the canonical run.

    eval/results/ is the committed headline artifact, so only a full run on the
    canonical split may write there. Anything else — a smoke subset, another split —
    goes to data/eval_runs/<dataset-slug>_<n>/, named so runs don't collide.
    """
    if not limit and settings.eval_dataset == CANONICAL_DATASET:
        return OUT, True
    slug = "".join(c if c.isalnum() else "-" for c in settings.eval_dataset).strip("-")
    return RUNS / f"{slug}_{n_queries}", False


def cached_run(
    service: SearchService, queries: dict[str, str], key: str, mode: str, reranker: str | None
) -> dict[str, dict[str, float]]:
    """Build (or resume) one config's run, checkpointing per query.

    A cross-encoder pass over 300 queries is hours on CPU, so progress is written
    every CHECKPOINT_EVERY queries: an interruption costs at most that many, and a
    partially-finished config still contributes whatever it completed.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    sig = _signature(mode, reranker, len(queries))
    # Signature in the filename, so runs under different settings coexist instead of
    # overwriting each other (a smoke run can't evict the real one).
    path = CACHE / f"{key}-{sig}.json"

    done: dict[str, dict[str, float]] = {}
    if path.exists() and not os.environ.get("SSR_EVAL_REFRESH"):
        # A corrupt checkpoint must degrade to "recompute", never abort the run —
        # otherwise one bad file strands hours of work behind a manual rm. Validate the
        # SHAPE, not just the parse: a file containing `null` or `[]` decodes fine, and
        # `blob["run"]` on it raises TypeError, which an except-clause listing only
        # ValueError/KeyError would let through.
        blob: object = None
        try:
            blob = json.loads(path.read_text())
        except (ValueError, OSError) as e:
            print(f"      (unreadable checkpoint, recomputing: {type(e).__name__})", flush=True)
        if not (isinstance(blob, dict) and isinstance(blob.get("run"), dict)):
            if blob is not None:
                print("      (malformed checkpoint, recomputing)", flush=True)
            blob = {}
        if blob.get("signature") == sig:
            done = blob["run"]
            if blob.get("complete"):
                print(f"      (cached, {len(done)} queries)", flush=True)
                return done
            print(f"      (resuming — {len(done)}/{len(queries)} already done)", flush=True)

    todo = [(q, t) for q, t in queries.items() if q not in done]
    if todo and reranker is not None:
        from app.rerank.cross_encoder import CrossEncoderReranker

        service.reranker = CrossEncoderReranker(reranker)

    def save() -> None:
        # Write-then-rename: os.replace is atomic, so a kill mid-checkpoint leaves the
        # previous good file intact rather than a truncated one. write_text truncates
        # in place, which would corrupt exactly the progress this cache exists to keep.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"signature": sig, "complete": len(done) == len(queries), "run": done})
        )
        os.replace(tmp, path)

    t0 = time.time()
    for i, (qid, qtext) in enumerate(todo, start=1):
        hits = service.retrieve(qtext, mode=mode, top_k=DEPTH, candidate_k=DEPTH)
        # ranx needs strictly-decreasing distinct scores to preserve rank order;
        # rank-derived scores guarantee that regardless of the raw score scale.
        done[qid] = {h.doc_id: float(len(hits) - rank) for rank, h in enumerate(hits)}
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            save()
            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i)
            print(
                f"      {len(done)}/{len(queries)}  ({elapsed / 60:.1f}min elapsed, "
                f"~{eta / 60:.0f}min left)",
                flush=True,
            )
    save()
    if reranker is not None:  # free the cross-encoder before the next one loads
        service.reranker = None
        gc.collect()
    return done


def _json_safe(obj):
    """Replace non-finite floats with null.

    A paired test over two identical score vectors has zero variance, so ranx returns
    NaN — which is exactly what Recall@100 gives between reranked configs, since
    reranking only reorders the returned set and cannot change recall. That NaN is
    meaningful, but `json.dumps` writes it as bare ``NaN``, which RFC 8259 forbids and
    strict parsers reject. Emitting null keeps the committed artifact valid JSON.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def to_markdown(scores: dict[str, dict[str, float]]) -> str:
    best = {m: max(scores[k][m] for k, *_ in CONFIGS) for m in METRICS}
    lines = [
        "| Config | " + " | ".join(PRETTY[m] for m in METRICS) + " |",
        "|" + "---|" * (len(METRICS) + 1),
    ]
    for key, label, _, _ in CONFIGS:
        cells = " | ".join(
            f"**{scores[key][m]:.4f}**" if scores[key][m] == best[m] else f"{scores[key][m]:.4f}"
            for m in METRICS
        )
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    from ranx import Qrels, Run, compare

    queries, qrels_dict = load_queries_qrels()
    if limit := int(os.environ.get("SSR_EVAL_LIMIT", "0")):  # smoke-test escape hatch
        queries = dict(list(queries.items())[:limit])
        qrels_dict = {q: v for q, v in qrels_dict.items() if q in queries}
    out, canonical = _output_dir(limit, len(queries))
    n_judg = sum(len(v) for v in qrels_dict.values())
    print(
        f"Eval set: {settings.eval_dataset}, {len(queries)} queries, "
        f"{n_judg} relevance judgments"
    )
    if not canonical:
        print(f"Non-canonical run (limit/dataset) — writing to {out}, not {OUT}")
    print(f"Rerank slice: top-{settings.rerank_candidates} of {DEPTH} fused candidates\n")

    service = SearchService()

    raw: dict[str, dict[str, dict[str, float]]] = {}
    for key, label, mode, reranker in CONFIGS:
        print(f"  {label}", flush=True)
        t0 = time.time()
        raw[key] = cached_run(service, queries, key, mode, reranker)
        print(f"      done in {time.time() - t0:.0f}s", flush=True)

    # Guard, not a feature: cached_run always finishes its work list, so in normal
    # operation every config covers every query and this intersection is a no-op. It
    # exists so that if that ever stops being true, the table is scored on a common
    # subset and says so, rather than silently comparing mismatched query sets.
    common = sorted(set.intersection(*(set(r) for r in raw.values())))
    if len(common) < len(queries):
        print(
            f"\n  NOTE: only {len(common)}/{len(queries)} queries are complete in every "
            f"config — evaluating all configs on that common subset.",
            flush=True,
        )
    qrels = Qrels({q: v for q, v in qrels_dict.items() if q in set(common)})
    runs = [Run({q: raw[key][q] for q in common}, name=key) for key, *_ in CONFIGS]

    report = compare(qrels, runs, METRICS, max_p=MAX_P)
    blob = report.to_dict()
    scores = {key: blob[key]["scores"] for key, *_ in CONFIGS}

    out.mkdir(parents=True, exist_ok=True)
    table = to_markdown(scores)
    sig_lines = _significance_markdown(blob)
    # The canonical header is kept verbatim so a rerun reproduces the committed file;
    # any other run names its real dataset and size, so it can't pass for the headline.
    title = (
        f"BEIR/SciFact ({len(common)} test queries)"
        if canonical
        else f"{settings.eval_dataset} ({len(common)} queries"
        + (f", SSR_EVAL_LIMIT={limit}" if limit else "")
        + ", non-canonical)"
    )
    (out / "retrieval.md").write_text(
        f"# Retrieval evaluation — {title}\n\n"
        f"{table}\n\n"
        f"Reranked slice: top-{settings.rerank_candidates} of {DEPTH} fused candidates "
        f"(the tail keeps its fused order, so Recall@100 is unchanged by reranking).\n\n"
        f"## Significance\n\n"
        f"Paired two-sided Student's t-test on per-query nDCG@10, p < {MAX_P}.\n\n"
        f"{sig_lines}\n"
    )
    (out / "retrieval.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "dataset": settings.eval_dataset,
                    "n_queries": len(common),
                    "candidate_depth": DEPTH,
                    "rerank_candidates": settings.rerank_candidates,
                    "metrics": scores,
                    "significance": {
                        "stat_test": blob["stat_test"],
                        "max_p": MAX_P,
                        "p_values": {k: blob[k]["comparisons"] for k, *_ in CONFIGS},
                    },
                    "win_tie_loss": {k: blob[k]["win_tie_loss"] for k, *_ in CONFIGS},
                }
            ),
            indent=2,
            allow_nan=False,  # fail loudly if a non-finite value ever slips past _json_safe
        )
    )
    print("\n" + table)
    print("\n" + sig_lines)
    print(f"\nWrote {out/'retrieval.md'} and {out/'retrieval.json'}")


def _significance_markdown(blob: dict) -> str:
    """Pairwise p-values on nDCG@10, plus per-query win/tie/loss, for the key pairs."""
    pairs = [
        ("hybrid", "dense", "Does fusing BM25 into dense retrieval help?"),
        ("hybrid", "bm25", "Does the hybrid beat lexical alone?"),
        ("rerank_minilm", "hybrid", "Does the MS-MARCO cross-encoder help?"),
        ("rerank_bge", "hybrid", "Does a domain-appropriate cross-encoder help?"),
        ("rerank_bge", "rerank_minilm", "Does the reranker choice matter?"),
    ]
    lines = ["| Comparison (nDCG@10) | Δ | p | W/T/L | Significant |", "|---|---|---|---|---|"]
    for a, b, question in pairs:
        delta = blob[a]["scores"]["ndcg@10"] - blob[b]["scores"]["ndcg@10"]
        p = blob[a]["comparisons"][b]["ndcg@10"]
        wtl = blob[a]["win_tie_loss"][b]["ndcg@10"]
        mark = "yes" if p < MAX_P else "no"
        lines.append(
            f"| {a} vs {b} — {question} | {delta:+.4f} | {p:.4f} | "
            f"{wtl['W']}/{wtl['T']}/{wtl['L']} | {mark} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
