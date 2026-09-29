"""NLI verdict eval: a dedicated DeBERTa NLI model as the claim-verdict source, no LLM.

For the same seeded claim sample as rag_eval (SEED=13, shuffle-then-prefix, so claim
sets align and `make rag-compare` pairs them), retrieve the same hybrid top-5, score
every (abstract, claim) pair with app.verify.nli.NLIVerifier, aggregate to a claim
verdict (app.verify.nli.aggregate_claim) and score it against the gold SciFact label.
Rows are rag.json-compatible (query_id, gold_label, predicted_label, answered, evidence,
... plus per-passage probabilities and the chosen doc).

Contamination rule: the 300 "test" claims are SciFact's public dev set. Every choice
(threshold τ, aggregation rule and depth, truncation vs sentence windows, int8) is made
on beir/scifact/train with ``--tune``, which REFUSES any other split; the result is
frozen in app.verify.nli.FROZEN, and a run on any other split uses those frozen values
(or explicit overrides, recorded as such and written to a separate `_custom-` dir).

Output ALWAYS goes to data/eval_runs/verify_<dataset-slug>_<n>_<model-slug>[...]/ —
never eval/results/ (guarded by `assert_safe_output`).

Retrieval: top-5 of mode=hybrid, exactly as rag_eval. For the canonical test split the
cached top-100 hybrid run from retrieval_eval (data/eval_cache/hybrid-<sig>.json) is
reused when its signature matches, so no Qdrant is needed; otherwise SearchService runs.
Either way the top-5 lists are cached under data/eval_cache/nli/.

Per-pair probabilities are cached (app.verify.nli.PairCache) by model, dtype, window
tag, max length, doc id, premise hash and claim hash, appended after every chunk: a
killed run resumes, and re-tuning costs nothing.

Run:
    SSR_RAG_DATASET=beir/scifact/train SSR_RAG_N=all uv run python -m app.eval.verify_eval --tune
    SSR_RAG_N=all uv run python -m app.eval.verify_eval        # frozen settings, test split
    make verify-eval                                           # same, default N (50)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.core.config import settings
from app.eval.rag_eval import (
    LABELS,
    MODE,
    ORACLE,
    ORACLE_DEFINITIONS,
    SEED,
    TOP_K,
    _abstention,
    _abstention_class,
    _file_sha256,
    _git_sha,
    _rate,
    _slug,
    evidence_flags,
    parse_n,
    rag_dataset,
    sample_claims,
    verdict_scores,
)
from app.eval.retrieval_eval import DEPTH, _index_fingerprint, _signature
from app.eval.retrieval_eval import CACHE as RETRIEVAL_CACHE
from app.generate.prompts import VERDICTS
from app.ingest.corpus import load_claim_labels, load_documents, load_queries_qrels, scifact_source_zip
from app.verify import nli
from app.verify.nli import FROZEN, RULES, WINDOWINGS, aggregate_claim

TRAIN_DATASET = "beir/scifact/train"  # the ONLY split --tune accepts
RUNS = Path("data/eval_runs")
FORBIDDEN_OUT = Path("eval/results")  # the committed artifact: never written here
TAU_GRID = tuple(round(i / 100, 2) for i in range(101))
# Aggregations tried by --tune, in preference order: an exact tie goes to the earlier
# entry (the planned default first, then simpler / deeper ones).
TUNE_AGGREGATIONS: tuple[tuple[str, int], ...] = (
    ("max", 5), ("max", 3), ("max", 2), ("max", 1), ("mean", 5), ("mean", 3),
)
LABEL_TO_VERDICT = {lab: v for v, lab in zip(VERDICTS, LABELS, strict=True)}
CHUNK = 32  # pairs per cache append


class UnsafeOutputError(RuntimeError):
    """An output path outside data/eval_runs/, or inside eval/results/."""


# --- output --------------------------------------------------------------------------------


def assert_safe_output(path: Path, runs: Path | None = None, forbidden: Path | None = None) -> Path:
    """`path`, if it is strictly inside `runs` and not inside `forbidden`; else raise.
    Resolved first, so a `..` cannot walk out of data/eval_runs/."""
    runs = RUNS if runs is None else runs
    forbidden = FORBIDDEN_OUT if forbidden is None else forbidden
    p, r, f = path.resolve(), runs.resolve(), forbidden.resolve()
    if p == r or r not in p.parents:
        raise UnsafeOutputError(f"{path} is not inside {runs}")
    if p == f or f in p.parents:
        raise UnsafeOutputError(f"{path} is inside the committed {forbidden}")
    return path


def output_dir(dataset: str, n: int, model: str, dtype: str, custom: Mapping | None = None) -> Path:
    """data/eval_runs/verify_<dataset>_<n>_<model>[_int8][_custom-<hash6>]/ — the frozen
    fp32 settings get the plain name; anything else is suffixed so it cannot pass for it."""
    name = f"verify_{_slug(dataset)}_{n}_{_slug(model)}"
    if dtype != "fp32":
        name += f"_{_slug(dtype)}"
    if custom:
        h = hashlib.sha256(json.dumps(dict(custom), sort_keys=True).encode()).hexdigest()[:6]
        name += f"_custom-{h}"
    return assert_safe_output(RUNS / name)


# --- retrieval -----------------------------------------------------------------------------


def retrieval_cache_path(dataset: str) -> Path:
    fields = {
        "dataset": dataset,
        "mode": MODE,
        "top_k": TOP_K,
        "dense_top_k": settings.dense_top_k,
        "rrf_k": settings.rrf_k,
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "index_manifest": _index_fingerprint(),
    }
    sig = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]
    return nli.CACHE_DIR / f"retrieval_{_slug(dataset)}_{sig}.json"


def _from_retrieval_eval_cache(dataset: str, n_split: int) -> dict[str, list[str]] | None:
    """Top-5 per claim from retrieval_eval's cached hybrid top-100, when that cache was
    built for this split under the current settings/index (its own signature) and
    hybrid's first-stage depth equals the one rag_eval retrieves with (the top-5 of a
    100-deep fusion is then exactly SearchService.retrieve(mode="hybrid", top_k=5))."""
    if dataset != settings.eval_dataset or settings.dense_top_k != DEPTH:
        return None
    path = RETRIEVAL_CACHE / f"hybrid-{_signature('hybrid', None, n_split)}.json"
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not (isinstance(blob, dict) and blob.get("complete") and isinstance(blob.get("run"), dict)):
        return None
    return {
        q: [d for d, _ in sorted(run.items(), key=lambda kv: -kv[1])][:TOP_K]
        for q, run in blob["run"].items()
    }


def load_retrieval(dataset: str, queries: Mapping[str, str], qids: Sequence[str], service_factory=None) -> tuple[dict[str, list[str]], str]:
    """({qid: top-5 doc ids}, where they came from)."""
    path = retrieval_cache_path(dataset)
    cached: dict[str, list[str]] = {}
    try:
        blob = json.loads(path.read_text())
        if isinstance(blob, dict):
            cached = {q: list(v) for q, v in blob.items() if isinstance(v, list)}
    except (OSError, ValueError):
        pass
    source = f"nli retrieval cache ({path})"
    missing = [q for q in qids if q not in cached]
    if missing and (rec := _from_retrieval_eval_cache(dataset, len(queries))) is not None:
        cached.update({q: rec[q] for q in missing if q in rec})
        source = "retrieval_eval hybrid top-100 cache (top-5 slice)"
        missing = [q for q in qids if q not in cached]
    if missing:
        if service_factory is None:
            from app.retrieve.service import SearchService

            service_factory = SearchService
        service = service_factory()
        for q in missing:
            cached[q] = [h.doc_id for h in service.retrieve(queries[q], mode=MODE, top_k=TOP_K)]
        source = f"SearchService.retrieve(mode={MODE!r}, top_k={TOP_K})"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cached))
    os.replace(tmp, path)
    return {q: cached[q] for q in qids}, source


# --- scoring -------------------------------------------------------------------------------


def score_claims(verifier, cache, qids, queries, top, docs, log=None) -> dict[str, list[nli.PairScore]]:
    """{qid: [PairScore per retrieved passage, in rank order]}."""
    log = log or _log
    pairs, owner = [], []
    for q in qids:
        for d in top[q]:
            doc = docs[d]
            pairs.append((d, doc["title"], doc["text"], queries[q]))
            owner.append(q)
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        el = time.time() - t0
        log(f"    {done}/{total} new pairs  ({el / 60:.1f} min, ~{el / done * (total - done) / 60:.0f} min left)")

    scores = nli.score_pairs_cached(verifier, cache, pairs, chunk=CHUNK, progress=progress)
    out: dict[str, list[nli.PairScore]] = {q: [] for q in qids}
    for q, s in zip(owner, scores, strict=True):
        out[q].append(s)
    return out


def build_row(qid, label, qrels_ids, top_ids, scores, tau, rule, k) -> dict:
    probs = [s.probs for s in scores]
    dec = aggregate_claim(probs, tau, rule, k)
    answered = dec.label != "NEI"
    flags = evidence_flags(top_ids, label, qrels_ids)
    chosen = top_ids[dec.index] if dec.index is not None else None
    return {
        "query_id": qid,
        "mode": MODE,
        "gold_label": label.label,
        "rationale_doc_ids": sorted(label.rationale_doc_ids),
        "qrels_doc_ids": sorted(qrels_ids),
        "verdict": LABEL_TO_VERDICT[dec.label],
        "predicted_label": dec.label,
        "answered": answered,
        "answered_source": "nli",
        **flags,
        "abstention_class": _abstention_class(answered, flags["evidence"]),
        "abstention_class_qrels": _abstention_class(answered, flags["evidence_qrels"]),
        # The passage the verdict rests on, when it is not NEI (so rag_compare's
        # "has a citation" outcome means "pointed at a passage").
        "cited_doc_ids": [chosen] if answered and chosen else [],
        "retrieved_doc_ids": list(top_ids),
        "chosen_doc_id": chosen,
        "chosen_rank": None if dec.index is None else dec.index + 1,
        "candidate_label": dec.candidate,
        "confidence": round(dec.confidence, 6),
        "passage_probs": [
            {
                "doc_id": d,
                "rank": i + 1,
                **{lab: round(s.probs[lab], 6) for lab in LABELS},
                "n_windows": s.n_windows,
                "window": s.window,
            }
            for i, (d, s) in enumerate(zip(top_ids, scores, strict=True))
        ],
    }


# --- metrics -------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 4), round(c + h, 4)


def class_scores(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """Accuracy, per-label accuracy (= recall), per-label P/R/F1 and macro-F1."""
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    per = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        ng = sum(g == lab for g in gold)
        npred = sum(p == lab for p in pred)
        prec = tp / npred if npred else 0.0
        rec = tp / ng if ng else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"n": ng, "predicted": npred, "precision": round(prec, 4),
                    "recall": round(rec, 4), "f1": round(f1, 4)}
    return {
        "accuracy": round(correct / n, 4) if n else None,
        "correct": correct,
        "macro_f1": round(sum(per[lab]["f1"] for lab in LABELS) / len(LABELS), 4),
        "per_label": per,
    }


def summarize(rows: Sequence[Mapping]) -> dict:
    cs = class_scores([r["gold_label"] for r in rows], [r["predicted_label"] for r in rows])
    head = _abstention(list(rows), "evidence")
    answered = [r for r in rows if r["answered"]]
    return {
        "n": len(rows),
        "verdict_accuracy": cs["accuracy"],
        "verdict_accuracy_ci95": wilson(cs["correct"], len(rows)),
        "per_label_accuracy": {lab: cs["per_label"][lab]["recall"] for lab in LABELS},
        "macro_f1": cs["macro_f1"],
        "per_label": cs["per_label"],
        **verdict_scores(list(rows)),
        "evidence_rate": head.pop("evidence_rate"),
        "answered_rate": _rate(len(answered), len(rows)),
        **{k: v for k, v in head.items() if k != "quadrants"},
        "oracle": ORACLE,
        "quadrants": head["quadrants"],
        "qrels_oracle": _abstention(list(rows), "evidence_qrels"),
        "by_label": {
            g: {"n": sum(r["gold_label"] == g for r in rows),
                "answered": sum(r["gold_label"] == g and r["answered"] for r in rows)}
            for g in LABELS
        },
        "top_k": TOP_K,
        "sample_seed": SEED,
    }


# --- tuning (train only) ---------------------------------------------------------------


def curve(probs: Mapping[str, Sequence[Mapping[str, float]]], gold: Mapping[str, str], rule: str, k: int, grid: Sequence[float] = TAU_GRID) -> list[dict]:
    """Accuracy and macro-F1 at every τ in `grid` for one aggregation."""
    qids = list(probs)
    gl = [gold[q] for q in qids]
    decisions = [aggregate_claim(probs[q], 0.0, rule, k) for q in qids]
    rows = []
    for tau in grid:
        pred = [d.candidate if d.confidence >= tau else "NEI" for d in decisions]
        cs = class_scores(gl, pred)
        rows.append({"tau": tau, "accuracy": cs["accuracy"], "macro_f1": cs["macro_f1"],
                     "answered_rate": round(sum(p != "NEI" for p in pred) / len(pred), 4)})
    return rows


def best_tau(points: Sequence[Mapping]) -> dict:
    """The τ maximising (accuracy, macro-F1); among exact ties, the median tied τ (the
    middle of the plateau, not its edge)."""
    top = max((p["accuracy"], p["macro_f1"]) for p in points)
    tied = [p for p in points if (p["accuracy"], p["macro_f1"]) == top]
    return dict(tied[(len(tied) - 1) // 2], n_tied=len(tied))


def tune(probs_by_windowing: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]], gold: Mapping[str, str]) -> dict:
    """Pick windowing x aggregation x τ by 3-class accuracy (macro-F1 breaks ties, then
    the order of WINDOWINGS x TUNE_AGGREGATIONS)."""
    table, curves, best = [], {}, None
    for w in WINDOWINGS:
        if w not in probs_by_windowing:
            continue
        for rule, k in TUNE_AGGREGATIONS:
            pts = curve(probs_by_windowing[w], gold, rule, k)
            curves[f"{w}/{rule}@{k}"] = pts
            b = best_tau(pts)
            entry = {"windowing": w, "rule": rule, "k": k, **b}
            table.append(entry)
            if best is None or (b["accuracy"], b["macro_f1"]) > (best["accuracy"], best["macro_f1"]):
                best = entry
    return {"best": best, "table": table, "curves": curves}


def two_tau_diagnostic(probs, gold, rule, k, grid=TAU_GRID[::5]) -> dict:
    """Diagnostic only (never adopted): separate thresholds for SUPPORT and CONTRADICT."""
    qids = list(probs)
    gl = [gold[q] for q in qids]
    decs = [aggregate_claim(probs[q], 0.0, rule, k) for q in qids]
    best = None
    for ts in grid:
        for tc in grid:
            pred = [d.candidate if d.confidence >= (ts if d.candidate == "SUPPORT" else tc) else "NEI" for d in decs]
            cs = class_scores(gl, pred)
            cand = {"tau_support": ts, "tau_contradict": tc, "accuracy": cs["accuracy"], "macro_f1": cs["macro_f1"]}
            if best is None or (cand["accuracy"], cand["macro_f1"]) > (best["accuracy"], best["macro_f1"]):
                best = cand
    return best


# --- markdown ------------------------------------------------------------------------------


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def run_markdown(agg: Mapping, run: Mapping) -> str:
    v = run["verifier"]
    lines = [
        f"# NLI verdict eval — {run['dataset']} ({agg['n']} claims, non-canonical)",
        "",
        f"Verifier `{v['model']}` ({v['dtype']}, {v['windowing']}), aggregation "
        f"`{v['rule']}@{v['k']}`, τ = {v['tau']} — {run['settings_source']}.",
        f"Retrieval: {run['mode']} top-{run['top_k']} ({run['retrieval_source']}).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Verdict accuracy (3-class) | {_f(agg['verdict_accuracy'])} (95% CI {agg['verdict_accuracy_ci95']}) |",
        f"| Macro-F1 | {_f(agg['macro_f1'])} |",
        *[f"| Accuracy on gold {lab} (n={agg['per_label'][lab]['n']}) | {_f(agg['per_label_accuracy'][lab])} |" for lab in LABELS],
        f"| Answered rate | {_f(agg['answered_rate'])} |",
        f"| Evidence (rationale doc in top-{agg['top_k']}) | {_f(agg['evidence_rate'])} |",
        f"| Abstention precision / recall | {_f(agg['abstention_precision'])} / {_f(agg['abstention_recall'])} |",
        f"| False abstention rate | {_f(agg['false_abstention_rate'])} |",
        f"| Answered without evidence | {_f(agg['answered_without_evidence_rate'])} |",
        "",
        "Confusion (rows gold, columns predicted):",
        "",
        "| gold \\ pred | " + " | ".join(LABELS) + " |",
        "|---|---|---|---|",
        *[f"| {g} | " + " | ".join(str(agg["confusion"][g][p]) for p in LABELS) + " |" for g in LABELS],
        "",
        f"Timing: {run['timing']}",
        "",
    ]
    return "\n".join(lines)


def tuning_markdown(t: Mapping, n: int, dataset: str) -> str:
    b = t["best"]
    lines = [
        f"# NLI tuning on {dataset} ({n} claims)",
        "",
        f"Chosen: windowing `{b['windowing']}`, `{b['rule']}@{b['k']}`, τ = {b['tau']} "
        f"(accuracy {b['accuracy']:.4f}, macro-F1 {b['macro_f1']:.4f}; {b['n_tied']} τ tied).",
        "",
        "| Windowing | Aggregation | best τ | Accuracy | Macro-F1 | τ tied |",
        "|---|---|---|---|---|---|",
        *[f"| {e['windowing']} | {e['rule']}@{e['k']} | {e['tau']} | {e['accuracy']:.4f} | "
          f"{e['macro_f1']:.4f} | {e['n_tied']} |" for e in t["table"]],
        "",
        f"τ curve ({b['windowing']}/{b['rule']}@{b['k']}):",
        "",
        "| τ | Accuracy | Macro-F1 | Answered |",
        "|---|---|---|---|",
        *[f"| {p['tau']:.2f} | {p['accuracy']:.4f} | {p['macro_f1']:.4f} | {p['answered_rate']:.4f} |"
          for p in t["curves"][f"{b['windowing']}/{b['rule']}@{b['k']}"] if round(p["tau"] * 100) % 5 == 0],
        "",
        f"Two-τ diagnostic (not adopted): {t.get('two_tau')}",
        "",
    ]
    return "\n".join(lines)


# --- entry point ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m app.eval.verify_eval", description=__doc__.split("\n")[0])
    p.add_argument("--tune", action="store_true", help=f"tune windowing/aggregation/τ (only on {TRAIN_DATASET})")
    p.add_argument("--tau", type=float, help="override the frozen τ (recorded; custom output dir)")
    p.add_argument("--rule", choices=RULES, help="override the frozen aggregation rule")
    p.add_argument("--k", type=int, help="override the frozen aggregation depth")
    p.add_argument("--windowing", choices=WINDOWINGS, help="override the frozen windowing")
    return p.parse_args(list(argv))


def resolve_settings(args: argparse.Namespace, dataset: str, int8: bool) -> tuple[dict, dict, str]:
    """(settings, overrides, source). --tune only on train; a non-tune run needs FROZEN to
    have been tuned, or explicit overrides for every frozen field it relies on."""
    if args.tune and dataset != TRAIN_DATASET:
        raise SystemExit(
            f"--tune is only allowed on {TRAIN_DATASET}, not {dataset}: the test claims are "
            "SciFact's public dev set and must be touched once, with frozen settings."
        )
    chosen = {k: FROZEN[k] for k in ("windowing", "rule", "k", "tau")}
    overrides = {k: v for k, v in (("windowing", args.windowing), ("rule", args.rule),
                                   ("k", args.k), ("tau", args.tau)) if v is not None}
    if int8 != FROZEN["int8"]:
        overrides["int8"] = int8
    chosen.update({k: v for k, v in overrides.items() if k != "int8"})
    if args.tune:
        return chosen, overrides, f"tuned in-sample on {dataset} (--tune)"
    if FROZEN["tuned_on"] is None and not {"tau", "rule", "k", "windowing"} <= overrides.keys():
        raise SystemExit(
            "app.verify.nli.FROZEN has not been tuned yet (tuned_on is None): run --tune on "
            f"{TRAIN_DATASET} first, or pass every setting explicitly."
        )
    source = f"frozen, tuned on {FROZEN['tuned_on']}"
    if overrides:
        source += f" — OVERRIDDEN: {overrides}"
    return chosen, overrides, source


def _log(*a) -> None:
    print(*a, flush=True)  # flushed: long runs go to a log file under nohup


def main(argv: Sequence[str] = (), verifier_factory=None, service_factory=None, log=_log) -> Path:
    args = _parse_args(argv)
    t_start = time.time()
    dataset = rag_dataset()
    n_req = parse_n(os.environ.get("SSR_RAG_N"))
    limit = int(os.environ.get("SSR_EVAL_LIMIT", "0") or 0)
    int8 = settings.nli_int8
    chosen, overrides, source = resolve_settings(args, dataset, int8)

    queries, qrels = load_queries_qrels(dataset)
    qids = sample_claims(queries, n_req)
    if limit:
        qids = qids[:limit]
    labels = load_claim_labels(dataset=dataset, query_ids=set(queries))

    factory = verifier_factory or (lambda w: nli.NLIVerifier(windowing=w, int8=int8))
    verifier = factory(chosen["windowing"])
    out = output_dir(dataset, len(qids), verifier.model_name, verifier.dtype,
                     {k: v for k, v in overrides.items() if k != "int8"} or None)
    log(f"Sample: {len(qids)} claims from {dataset} (seed={SEED}) -> {out}  [{source}]")

    top, retrieval_source = load_retrieval(dataset, queries, qids, service_factory)
    docs = {d["doc_id"]: d for d in load_documents()}
    gold_qrels = {q: {d for d, rel in qrels.get(q, {}).items() if rel > 0} for q in qids}
    cache = nli.PairCache(nli.cache_path_for(verifier.model_name, verifier.dtype))
    log(f"Verifier {verifier.model_name} ({verifier.dtype}); pair cache {cache.path} "
        f"({len(cache.entries)} entries)")

    windowings = WINDOWINGS if args.tune else (chosen["windowing"],)
    scored: dict[str, dict[str, list[nli.PairScore]]] = {}
    for w in windowings:
        verifier.windowing = w
        log(f"  scoring {len(qids) * TOP_K} pairs, windowing={w}")
        scored[w] = score_claims(verifier, cache, qids, queries, top, docs, log)

    tuning = None
    if args.tune:
        gold = {q: labels[q].label for q in qids}
        tuning = tune({w: {q: [s.probs for s in scored[w][q]] for q in qids} for w in windowings}, gold)
        b = tuning["best"]
        tuning["two_tau"] = two_tau_diagnostic(
            {q: [s.probs for s in scored[b["windowing"]][q]] for q in qids}, gold, b["rule"], b["k"]
        )
        chosen = {"windowing": b["windowing"], "rule": b["rule"], "k": b["k"], "tau": b["tau"]}
        long_pairs = sum(s.n_windows > 1 for q in qids for s in scored["window"][q])
        tuning["long_pairs_windowed"] = long_pairs
        log(f"  tuned: {chosen}  acc={b['accuracy']} macro_f1={b['macro_f1']}")

    final = scored[chosen["windowing"]]
    rows = [
        build_row(q, labels[q], gold_qrels[q], top[q], final[q], chosen["tau"], chosen["rule"], chosen["k"])
        for q in qids
    ]
    agg = summarize(rows)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    fwd_pairs = getattr(verifier, "pairs_scored", 0)
    fwd_s = getattr(verifier, "forward_seconds", 0.0)
    timing = {
        "total_s": round(time.time() - t_start, 1),
        "pairs_forwarded": fwd_pairs,
        "windows_forwarded": getattr(verifier, "windows_scored", 0),
        "forward_s": round(fwd_s, 1),
        "ms_per_pair": round(1000 * fwd_s / fwd_pairs, 1) if fwd_pairs else None,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "peak_rss_mb": round(peak_mb),
    }
    try:
        import torch
        import transformers

        versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                    "threads": torch.get_num_threads()}
    except ImportError:  # pragma: no cover
        versions = {}
    run = {
        "git_sha": _git_sha(),
        "dataset": dataset,
        # rag_compare's settings table reads these rag_eval keys.
        "generator_model": f"nli:{verifier.model_name}",
        "generator_provider": "local-nli",
        "judge_model": None,
        "judge_provider": None,
        "prompt_hash": None,
        "judge_prompt_hash": None,
        "mode": MODE,
        "top_k": TOP_K,
        "reranker_model": None,
        "n_requested": "all" if n_req is None else n_req,
        "sample_seed": SEED,
        "n_sample": len(qids),
        "eval_limit": limit,
        "canonical": False,
        "verifier": {
            "model": verifier.model_name,
            "dtype": verifier.dtype,
            **chosen,
            "max_length": getattr(verifier, "max_length", nli.MAX_LENGTH),
            "window_overlap_sentences": nli.OVERLAP_SENTENCES,
            "batch_size": getattr(verifier, "batch_size", None),
            "premise": "title + '. ' + abstract (premise first, claim as hypothesis)",
            **versions,
        },
        "settings_source": source,
        "frozen": FROZEN,
        "overrides": overrides,
        "retrieval_source": retrieval_source,
        "oracle": ORACLE,
        "oracle_definition": ORACLE_DEFINITIONS[ORACLE],
        "label_source": {
            "loader": "app.ingest.corpus.load_claim_labels",
            "dataset": dataset,
            "source_zip_sha256": _file_sha256(scifact_source_zip()),
        },
        "pair_cache": str(cache.path),
        "timing": timing,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.json").write_text(json.dumps({**agg, "run": run, "rows": rows}, indent=2))
    (out / "rag.md").write_text(run_markdown(agg, run))
    if tuning is not None:
        (out / "tuning.json").write_text(json.dumps(tuning, indent=2))
        (out / "tuning.md").write_text(tuning_markdown(tuning, len(qids), dataset))
    log(
        f"\nn={agg['n']}  verdict_accuracy={agg['verdict_accuracy']} {agg['verdict_accuracy_ci95']}  "
        f"macro_f1={agg['macro_f1']}  per_label={agg['per_label_accuracy']}  "
        f"answered={agg['answered_rate']}\ntiming={timing}\nWrote {out}/rag.json"
    )
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
