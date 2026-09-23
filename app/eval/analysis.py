"""Where does fusion's recall gain come from? A label-stratified look at the eval runs.

SciFact claims come in three kinds (see app.ingest.corpus.ClaimLabel): SUPPORT and
CONTRADICT claims cite abstracts with rationale sentences; NEI claims have none, yet
BEIR's qrels still mark a doc relevant for every one of them. The headline table in
eval/results/retrieval.md averages over all 300 claims, so this module splits the same
runs by label and asks whether hybrid-vs-dense behaves the same on each stratum. It also
replays RRF over the cached dense + BM25 lists at other k values and dense weights, to show
how sensitive the committed hybrid numbers are to those two settings.

It re-scores the cached runs retrieval_eval already produced — no retrieval is re-run
and the embedded Qdrant store is never opened (only the BM25 index is loaded, to rank
missed gold docs past depth 100). Missing or incomplete caches fail loudly.

Run (after `make eval`):
    uv run python -m app.eval.analysis
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence

from app.core.config import Settings, settings
from app.eval.retrieval_eval import CACHE, CONFIGS, DEPTH, MAX_P, OUT, _json_safe, _signature
from app.ingest.corpus import ClaimLabel, load_claim_labels, load_queries_qrels
from app.retrieve.fusion import reciprocal_rank_fusion

KEYS = ("bm25", "dense", "hybrid")  # first-stage configs; reranking cannot move R@100
LABELS = {key: label for key, label, *_ in CONFIGS}
METRICS = ["ndcg@10", "recall@100"]
PRETTY = {"ndcg@10": "nDCG@10", "recall@100": "Recall@100"}
PAIRS = [("hybrid", "dense"), ("hybrid", "bm25")]
LABEL_ORDER = ("SUPPORT", "CONTRADICT", "NEI")
# RRF sensitivity grid: k swept at equal weight, dense weight swept at k = BASE_K.
BASE_K = 60  # the production default (settings.rrf_k), and the sweep's baseline
K_SWEEP = (1, 2, 5, 10, 20, 60, 100)
DENSE_WEIGHTS = (0.3, 0.5, 0.7)

Run_ = dict[str, dict[str, float]]


def load_cached_runs(n_queries: int) -> dict[str, Run_]:
    """The bm25/dense/hybrid runs retrieval_eval cached, located by its own signature.

    Refuses anything short of a complete run over exactly `n_queries` queries: this
    analysis re-scores committed numbers, so a partial or stale cache must not pass.
    """
    by_key = {key: (mode, reranker) for key, _, mode, reranker in CONFIGS}
    runs: dict[str, Run_] = {}
    for key in KEYS:
        mode, reranker = by_key[key]
        sig = _signature(mode, reranker, n_queries)
        path = CACHE / f"{key}-{sig}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached '{key}' run at {path} (signature {sig}, {n_queries} queries). "
                "Run `make eval` first — this analysis only re-scores its cached runs, and "
                "a signature mismatch means the settings or index differ from that run."
            )
        blob = json.loads(path.read_text())
        run = blob.get("run") if isinstance(blob, dict) else None
        if not isinstance(run, dict) or blob.get("signature") != sig or not blob.get("complete"):
            raise ValueError(f"{path} is not a complete run with signature {sig}; rerun `make eval`.")
        if len(run) != n_queries:
            raise ValueError(f"{path} covers {len(run)} queries, expected {n_queries}.")
        runs[key] = run
    return runs


def ranked(run: Run_, qid: str) -> list[str]:
    """Doc ids best-first. Cached scores are rank-derived (len - rank), so distinct."""
    return sorted(run[qid], key=lambda d: -run[qid][d])


def check_hybrid_replay(runs: dict[str, Run_]) -> None:
    """Re-fuse the cached dense + BM25 lists with the production RRF and require the
    cached hybrid run back, query for query. Same argument order as SearchService
    (dense first), so ties break identically. A mismatch means the three caches were
    not produced by the same code/settings, and any stratified comparison between them
    would be meaningless."""
    bad = [
        q
        for q in runs["hybrid"]
        if [
            d
            for d, _ in reciprocal_rank_fusion(
                [ranked(runs["dense"], q), ranked(runs["bm25"], q)], k=settings.rrf_k
            )[:DEPTH]
        ]
        != ranked(runs["hybrid"], q)
    ]
    if bad:
        raise ValueError(
            f"Cached hybrid run does not match RRF over the cached dense+BM25 runs on "
            f"{len(bad)} queries (e.g. {bad[:5]}); the caches are inconsistent."
        )


def score_and_compare(
    runs: dict[str, Run_],
    qids: list[str],
    relevant: dict[str, set[str]],
    keys: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> dict:
    """ranx scores + paired t-tests (retrieval_eval's method) of `keys` on one query
    subset, with Δ / p / W-T-L for each (a, b) in `pairs`."""
    from ranx import Qrels, Run, compare

    qrels = Qrels({q: {d: 1 for d in relevant[q]} for q in qids})
    ranx_runs = [Run({q: runs[k][q] for q in qids}, name=k) for k in keys]
    blob = compare(qrels, ranx_runs, METRICS, max_p=MAX_P).to_dict()
    return {
        "n": len(qids),
        "scores": {k: blob[k]["scores"] for k in keys},
        "comparisons": {
            f"{a}_vs_{b}": {
                m: {
                    "delta": blob[a]["scores"][m] - blob[b]["scores"][m],
                    "p": float(blob[a]["comparisons"][b][m]),
                    "win_tie_loss": blob[a]["win_tie_loss"][b][m],
                }
                for m in METRICS
            }
            for a, b in pairs
        },
    }


def stratified(
    runs: dict[str, Run_], qids: list[str], relevant: dict[str, set[str]]
) -> dict:
    """bm25/dense/hybrid scored and compared on one query subset."""
    return score_and_compare(runs, qids, relevant, KEYS, PAIRS)


def weighted_rrf(
    rankings: Sequence[Sequence[str]], weights: Sequence[float], k: int
) -> list[tuple[str, float]]:
    """RRF with a per-list weight: score(d) = sum_i w_i * 1/(k + rank_i(d)), 1-based.

    Implemented here, not in app.retrieve.fusion: production `reciprocal_rank_fusion`
    takes no weights (every list counts 1), and widening the serving path's signature
    for an offline sweep is not warranted. It mirrors that function otherwise — a doc
    absent from a list gets nothing from it, lists accumulate in argument order, and
    the sort is stable — so equal weights of 0.5 give exactly half of every production
    score (scaling by a power of two is exact in floating point) and hence the
    identical ranking, ties included. `rrf_sensitivity` checks that on the real runs.
    """
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _as_run(ranking: Sequence[str]) -> dict[str, float]:
    """A best-first list as a run, scored like retrieval_eval's cache (len - rank)."""
    return {d: float(len(ranking) - i) for i, d in enumerate(ranking)}


def _refuse(runs: dict[str, Run_], qids: Iterable[str], fuse) -> Run_:
    """Apply `fuse` to each query's cached [dense, BM25] lists (dense first, as in
    SearchService) and cut the result to DEPTH, as the harness does."""
    return {
        q: _as_run([d for d, _ in fuse([ranked(runs["dense"], q), ranked(runs["bm25"], q)])[:DEPTH]])
        for q in qids
    }


def fused_run(
    runs: dict[str, Run_], qids: Iterable[str], k: int, dense_weight: float = 0.5
) -> Run_:
    """Dense + BM25 re-fused at (k, dense weight). Equal weight goes through the
    production `reciprocal_rank_fusion` itself; any other weight through `weighted_rrf`."""
    if dense_weight == 0.5:
        return _refuse(runs, qids, lambda lists: reciprocal_rank_fusion(lists, k=k))
    weights = (dense_weight, 1.0 - dense_weight)
    return _refuse(runs, qids, lambda lists: weighted_rrf(lists, weights, k))


def _cfg(k: int, w: float) -> str:
    return f"k={k},w={w}"


def rrf_sensitivity(
    runs: dict[str, Run_],
    qids: list[str],
    relevant: dict[str, set[str]],
    committed: dict[str, float],
) -> dict:
    """Sweep RRF's k (equal weight) and the dense weight (k = BASE_K), each scored
    against the BASE_K / equal-weight baseline with retrieval_eval's paired tests.

    The baseline is the production config, so it must *be* the committed hybrid run:
    query for query against the cached run, and metric for metric against
    `committed` (retrieval.json's hybrid row). Either mismatch raises — a sensitivity
    table whose baseline isn't the published number would be measuring something else.
    """
    base = _cfg(BASE_K, 0.5)
    grid = [(k, 0.5) for k in K_SWEEP] + [(BASE_K, w) for w in DENSE_WEIGHTS if w != 0.5]
    fused = {_cfg(k, w): fused_run(runs, qids, k, w) for k, w in grid}

    bad = [q for q in qids if fused[base][q] != runs["hybrid"][q]]
    if bad:
        raise ValueError(
            f"RRF k={BASE_K} equal-weight replay differs from the cached hybrid run on "
            f"{len(bad)}/{len(qids)} queries (e.g. {bad[:5]}); the sensitivity baseline "
            "would not be the committed configuration."
        )
    if _refuse(runs, qids, lambda lists: weighted_rrf(lists, (0.5, 0.5), BASE_K)) != fused[base]:
        raise ValueError("weighted_rrf at w=0.5 disagrees with production RRF; fix weighted_rrf.")

    blob = score_and_compare(
        fused, qids, relevant, list(fused), [(c, base) for c in fused if c != base]
    )
    got = {m: blob["scores"][base][m] for m in METRICS}
    want = {m: committed[m] for m in METRICS}
    if got != want:
        raise ValueError(
            f"Sensitivity baseline scores {got} do not reproduce the committed hybrid row {want}."
        )
    rows = []
    for k, w in grid:
        c = _cfg(k, w)
        rows.append(
            {
                "k": k,
                "dense_weight": w,
                "baseline": c == base,
                "scores": blob["scores"][c],
                "vs_baseline": None if c == base else blob["comparisons"][f"{c}_vs_{base}"],
            }
        )
    return {
        "n": blob["n"],
        "baseline": {"k": BASE_K, "dense_weight": 0.5},
        "union_recall@100_ceiling": sum(
            len((set(runs["dense"][q]) | set(runs["bm25"][q])) & relevant[q]) / len(relevant[q])
            for q in qids
        )
        / len(qids),
        "rows": rows,
    }


def miss_table(
    runs: dict[str, Run_],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    labels: dict[str, ClaimLabel],
) -> list[dict]:
    """Every (query, gold doc) pair that hybrid fails to return in its top-100.

    BM25 is re-ranked over the whole corpus so the depth at which it *would* have found
    the doc is exact; dense ranks are only known to depth 100 (the cache), since the
    dense index can't be queried here without taking the embedded Qdrant lock.
    """
    from app.index.lexical import LexicalIndex

    lex = LexicalIndex().load()
    rows = []
    for q in sorted(qrels, key=int):
        hybrid = set(runs["hybrid"][q])
        missed = [d for d in sorted(qrels[q]) if d not in hybrid]
        if not missed:
            continue
        bm25_full = [h.doc_id for h in lex.search(queries[q], len(lex.docs))]
        if bm25_full[:DEPTH] != ranked(runs["bm25"], q):
            raise ValueError(f"Live BM25 top-{DEPTH} for query {q} differs from the cache.")
        dense = ranked(runs["dense"], q)
        for d in missed:
            bm25_rank = bm25_full.index(d) + 1 if d in bm25_full else None
            dense_rank = dense.index(d) + 1 if d in dense else None
            rows.append(
                {
                    "qid": q,
                    "label": labels[q].label,
                    "doc_id": d,
                    "rationale_doc": d in labels[q].rationale_doc_ids,
                    "hybrid_recall@100": len(hybrid & set(qrels[q])) / len(qrels[q]),
                    "bm25_rank": bm25_rank,
                    "dense_rank": dense_rank,
                    # Hybrid fuses both retrievers' top-DEPTH, so a doc in either pool is
                    # one fusion could have kept; one in neither, no fusion rule can.
                    "reachable": (bm25_rank is not None and bm25_rank <= DEPTH)
                    or dense_rank is not None,
                }
            )
    return rows


def _p(p: float) -> str:
    return "n/a (identical per query)" if not math.isfinite(p) else f"{p:.4f}"


def to_markdown(report: dict) -> str:
    dist, strata, misses = report["label_distribution"], report["strata"], report["misses"]
    n = report["n_queries"]

    lines = [
        f"# Label-stratified retrieval analysis — BEIR/SciFact ({n} test claims)",
        "",
        "Re-scores the cached `make eval` runs (no retrieval re-run). Claim labels come from "
        "the `metadata` field of BEIR SciFact's `queries.jsonl`, which ir_datasets drops "
        "(`app.ingest.corpus.load_claim_labels`). Generated by "
        "`uv run python -m app.eval.analysis`.",
        "",
        "## 1. Label distribution",
        "",
        "| Claim label | Claims | Share | qrels ≠ rationale docs |",
        "|---|---|---|---|",
    ]
    for lab in LABEL_ORDER:
        d = dist[lab]
        diff = "—" if lab == "NEI" else str(d["qrels_differs_from_rationale"])
        lines.append(f"| {lab} | {d['n']} | {d['n'] / n:.1%} | {diff} |")
    ndiff = len(report["qrels_differs_from_rationale"])
    how = (
        "is a strict superset of the rationale docs (it adds cited docs without rationale)"
        if report["qrels_superset_of_rationale"]
        else "differs from the rationale docs"
    )
    lines += [
        "",
        "Every claim has at least one qrels-relevant doc, NEI claims included: for those, "
        "BEIR marks the cited abstract relevant even though the annotators found no "
        f"rationale in it. On {ndiff} evidence-bearing claims the qrels set {how}, so "
        "evidence-bearing claims are scored below under **both** relevant sets. NEI claims "
        "have no rationale docs, so they can only be scored against qrels.",
        "",
        "## 2. Stratified metrics",
        "",
        "| Subset | Relevant set | n | "
        + " | ".join(f"{LABELS[k]} {PRETTY[m]}" for m in METRICS for k in KEYS)
        + " |",
        "|---|---|---|" + "---|" * (len(KEYS) * len(METRICS)),
    ]
    for s in strata:
        cells = " | ".join(f"{s['scores'][k][m]:.4f}" for m in METRICS for k in KEYS)
        lines.append(f"| {s['subset']} | {s['relevant_set']} | {s['n']} | {cells} |")
    lines += [
        "",
        f"Paired two-sided Student's t-test via ranx (as in `retrieval_eval`), p < {MAX_P}, "
        "uncorrected for multiple comparisons:",
        "",
        "| Subset | Relevant set | Comparison | Metric | Δ | p | W/T/L |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in strata:
        for a, b in PAIRS:
            for m in METRICS:
                c = s["comparisons"][f"{a}_vs_{b}"][m]
                wtl = c["win_tie_loss"]
                sig = " **(sig.)**" if math.isfinite(c["p"]) and c["p"] < MAX_P else ""
                lines.append(
                    f"| {s['subset']} | {s['relevant_set']} | {a} vs {b} | {PRETTY[m]} | "
                    f"{c['delta']:+.4f} | {_p(c['p'])}{sig} | {wtl['W']}/{wtl['T']}/{wtl['L']} |"
                )

    full = sum(1 for r in misses if r["hybrid_recall@100"] == 0)
    lines += [
        "",
        f"## 3. Hybrid misses at depth {DEPTH}",
        "",
        f"Every qrels-relevant doc hybrid fails to return in its top-{DEPTH} "
        f"({len(misses)} docs over {len({r['qid'] for r in misses})} claims; {full} of those "
        f"claims lose every gold doc). BM25 rank is exact (full-corpus BM25); dense rank is "
        f"known only to depth {DEPTH}. *Reachable* = in either retriever's top-{DEPTH} "
        "candidate pool, i.e. a different fusion could have kept it.",
        "",
        "| qid | Label | Gold doc | Rationale doc? | Hybrid R@100 | BM25 rank | Dense rank "
        "| Reachable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in misses:
        rat = "—" if r["label"] == "NEI" else ("yes" if r["rationale_doc"] else "no")
        bm = str(r["bm25_rank"]) if r["bm25_rank"] is not None else "not ranked"
        dn = str(r["dense_rank"]) if r["dense_rank"] is not None else f"not in top-{DEPTH}"
        lines.append(
            f"| {r['qid']} | {r['label']} | {r['doc_id']} | {rat} | "
            f"{r['hybrid_recall@100']:.2f} | {bm} | {dn} | {'yes' if r['reachable'] else 'no'} |"
        )
    lines += sensitivity_markdown(report["rrf_sensitivity"], misses)
    lines += ["", "## 5. Interpretation", "", report["interpretation"], ""]
    return "\n".join(lines)


def sensitivity_markdown(sens: dict, misses: list[dict]) -> list[str]:
    """Section 4: the RRF k / dense-weight sweep, with prose worded from its numbers."""
    rows = sens["rows"]
    base = next(r for r in rows if r["baseline"])
    lines = [
        "",
        "## 4. RRF sensitivity (offline replay)",
        "",
        f"The cached dense and BM25 top-{DEPTH} lists re-fused at other settings, each fused "
        f"list cut to depth {DEPTH} and scored like the rows above. Equal-weight rows use the "
        "production `reciprocal_rank_fusion`; weighted rows use "
        "score(d) = w·1/(k + r_dense) + (1 − w)·1/(k + r_bm25), 1-based ranks, a doc absent "
        f"from a list getting nothing from it. Baseline: k = {sens['baseline']['k']}, "
        f"w = {sens['baseline']['dense_weight']} (the production config), verified to "
        f"reproduce the cached hybrid run on {sens['n']}/{sens['n']} queries and the committed "
        "hybrid row of `retrieval.md` exactly. Δ, p (paired two-sided t-test, uncorrected) and "
        "W/T/L are against that baseline.",
        "",
        "| k | Dense weight w | nDCG@10 | Δ | p | W/T/L | Recall@100 | Δ | p | W/T/L |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = []
        for m in METRICS:
            cells.append(f"{r['scores'][m]:.4f}")
            if r["baseline"]:
                cells += ["baseline", "—", "—"]
                continue
            c = r["vs_baseline"][m]
            wtl = c["win_tie_loss"]
            sig = " **(sig.)**" if math.isfinite(c["p"]) and c["p"] < MAX_P else ""
            cells += [f"{c['delta']:+.4f}", f"{_p(c['p'])}{sig}", f"{wtl['W']}/{wtl['T']}/{wtl['L']}"]
        lines.append(f"| {r['k']} | {r['dense_weight']} | " + " | ".join(cells) + " |")

    k_rows = [r for r in rows if not r["baseline"] and r["dense_weight"] == 0.5]
    w_rows = [r for r in rows if not r["baseline"] and r["dense_weight"] != 0.5]

    def c(r: dict, m: str) -> dict:
        return r["vs_baseline"][m]

    def ps(rs: list[dict], m: str) -> list[float]:
        return [c(r, m)["p"] for r in rs if math.isfinite(c(r, m)["p"])]

    n_k = ps(k_rows, "ndcg@10")
    k_recall_tied = all(
        c(r, "recall@100")["win_tie_loss"]["T"] == sens["n"] for r in k_rows
    )
    fell = [r for r in rows if not r["baseline"] and c(r, "recall@100")["delta"] < 0]
    reachable = sum(1 for r in misses if r["reachable"])
    base_r = base["scores"]["recall@100"]
    ks = ", ".join(str(r["k"]) for r in k_rows)
    if k_recall_tied:
        k_recall = f"Recall@100 is identical to the baseline's on every query ({base_r:.4f})"
    else:
        k_recall = "Recall@100 ranges over " + ", ".join(
            f"{r['scores']['recall@100']:.4f} (k = {r['k']})" for r in k_rows
        ) + f" against the baseline's {base_r:.4f}"
    lines += [
        "",
        f"For k ∈ {{{ks}}} at equal weight, nDCG@10 moves by at most "
        f"{max(abs(c(r, 'ndcg@10')['delta']) for r in k_rows):.4f} from the baseline "
        + (f"(all p ≥ {min(n_k):.4f})" if n_k else "(identical on every query)")
        + f", and {k_recall}. "
        + " ".join(
            f"Dense weight {r['dense_weight']}: nDCG@10 {c(r, 'ndcg@10')['delta']:+.4f} "
            f"(p = {_p(c(r, 'ndcg@10')['p'])}), Recall@100 {r['scores']['recall@100']:.4f} "
            f"({c(r, 'recall@100')['delta']:+.4f}, p = {_p(c(r, 'recall@100')['p'])})."
            for r in w_rows
        ),
        "",
        f"Recall@100 has little room to rise under any fusion of these two pools: only "
        f"{reachable} of the {len(misses)} gold docs hybrid misses are in either retriever's "
        f"top-{DEPTH} ([§3](#3-hybrid-misses-at-depth-{DEPTH})), so no k or weighting can "
        f"exceed the union ceiling of {sens['union_recall@100_ceiling']:.4f}."
        + (
            " It can fall, though: the two top-"
            f"{DEPTH} lists can hold up to {2 * DEPTH} docs, and a setting that leans "
            "on one retriever cuts gold docs only the other ranked highly ("
            + "; ".join(
                f"k = {r['k']}, w = {r['dense_weight']}: "
                f"{c(r, 'recall@100')['delta']:+.4f}"
                for r in fell
            )
            + ")."
            if fell
            else ""
        ),
    ]
    return lines


def interpret(strata: list[dict], misses: list[dict]) -> str:
    """Prose read of the numbers — worded from them, so it can't drift from the table."""
    get = {(s["subset"], s["relevant_set"]): s for s in strata}
    allq, nei = get[("all", "qrels")], get[("NEI", "qrels")]
    ev_q, ev_r = get[("evidence-bearing", "qrels")], get[("evidence-bearing", "rationale")]

    def hd(s: dict, m: str) -> dict:
        return s["comparisons"]["hybrid_vs_dense"][m]

    def tied(c: dict) -> bool:
        return c["win_tie_loss"]["W"] == c["win_tie_loss"]["L"] == 0

    def wtl(c: dict) -> str:
        w = c["win_tie_loss"]
        return f"{w['W']}/{w['T']}/{w['L']}"

    r_all, r_nei = hd(allq, "recall@100"), hd(nei, "recall@100")
    r_evq, r_evr, n_evr = hd(ev_q, "recall@100"), hd(ev_r, "recall@100"), hd(ev_r, "ndcg@10")
    discordant = r_all["win_tie_loss"]["W"] + r_all["win_tie_loss"]["L"]
    nei_discordant = r_nei["win_tie_loss"]["W"] + r_nei["win_tie_loss"]["L"]
    unreachable = sum(1 for r in misses if not r["reachable"])
    nei_misses = sum(1 for r in misses if r["label"] == "NEI")

    if tied(r_evq) and tied(r_evr):
        ev = (
            f"On the {ev_r['n']} evidence-bearing claims, hybrid and dense are **identical at "
            f"Recall@100** — every query tied, against either relevant set "
            f"({ev_r['scores']['hybrid']['recall@100']:.4f} on rationale docs, "
            f"{ev_q['scores']['hybrid']['recall@100']:.4f} on qrels) — and their nDCG@10 on "
            f"rationale docs differs by {n_evr['delta']:+.4f} (p = {_p(n_evr['p'])})."
        )
    else:
        ev = (
            f"On the {ev_r['n']} evidence-bearing claims, hybrid vs dense Recall@100 differs "
            f"by {r_evr['delta']:+.4f} on rationale docs (p = {_p(r_evr['p'])}, W/T/L "
            f"{wtl(r_evr)}) and {r_evq['delta']:+.4f} on qrels (p = {_p(r_evq['p'])}); "
            f"nDCG@10 by {n_evr['delta']:+.4f} (p = {_p(n_evr['p'])})."
        )
    return (
        f"Fusion's significant Recall@100 gain over dense on the full set "
        f"({r_all['delta']:+.4f}, p = {_p(r_all['p'])}, W/T/L {wtl(r_all)}) comes from the "
        f"**NEI** claims: {nei_discordant} of the {discordant} queries where the two differ are "
        f"NEI, and on that stratum hybrid gains {r_nei['delta']:+.4f} (p = {_p(r_nei['p'])}, "
        f"W/T/L {wtl(r_nei)}). {ev} For an NEI claim the qrels doc is a cited abstract in "
        f"which annotators found no rationale, so the recall fusion adds is recall of "
        f"documents that cannot ground a SUPPORT or CONTRADICT verdict. It still widens the "
        f"candidate pool, which is why hybrid stays the default, but it is not evidence that "
        f"fusion finds more supporting or refuting abstracts than dense alone; on this corpus "
        f"it does not. Of the {len(misses)} gold docs hybrid misses, {nei_misses} belong to NEI "
        f"claims and {unreachable} were in neither retriever's top-{DEPTH}, so no fusion rule "
        f"over these two candidate pools could have recovered them. All p-values are "
        f"uncorrected; p = {_p(r_nei['p'])} on {nei_discordant} discordant queries is "
        f"suggestive, not settled."
    )


def main() -> None:
    canonical = Settings.model_fields["eval_dataset"].default
    if settings.eval_dataset != canonical:
        raise SystemExit(
            f"analysis writes the committed eval/results/analysis.* and is defined for "
            f"{canonical}; SSR_EVAL_DATASET={settings.eval_dataset} is not supported."
        )
    queries, qrels = load_queries_qrels()
    labels = load_claim_labels(query_ids=set(queries))
    if not (set(queries) == set(qrels) == set(labels)):
        raise ValueError(
            f"query/qrels/label sets disagree: {len(queries)} queries, {len(qrels)} with "
            f"qrels, {len(labels)} labelled"
        )
    runs = load_cached_runs(len(queries))
    for key, run in runs.items():
        if set(run) != set(queries):
            raise ValueError(f"cached '{key}' run covers a different query set than the split")
    check_hybrid_replay(runs)

    all_q = sorted(queries, key=int)
    ev = [q for q in all_q if labels[q].label != "NEI"]
    nei = [q for q in all_q if labels[q].label == "NEI"]
    qrels_sets = {q: set(qrels[q]) for q in all_q}
    rationale = {q: labels[q].rationale_doc_ids for q in ev}
    differs = [q for q in ev if rationale[q] != qrels_sets[q]]

    strata = []
    for subset, qids, rel_name, rel in [
        ("all", all_q, "qrels", qrels_sets),
        ("evidence-bearing", ev, "qrels", qrels_sets),
        ("evidence-bearing", ev, "rationale", rationale),
        ("NEI", nei, "qrels", qrels_sets),
    ]:
        strata.append({"subset": subset, "relevant_set": rel_name, **stratified(runs, qids, rel)})

    counts = Counter(labels[q].label for q in all_q)
    diff_by_label = Counter(labels[q].label for q in differs)
    misses = miss_table(runs, queries, qrels, labels)
    report = {
        "dataset": settings.eval_dataset,
        "n_queries": len(all_q),
        "candidate_depth": DEPTH,
        "stat_test": "student (paired, two-sided, via ranx)",
        "max_p": MAX_P,
        "label_distribution": {
            lab: {"n": counts[lab], "qrels_differs_from_rationale": diff_by_label[lab]}
            for lab in LABEL_ORDER
        },
        "qrels_differs_from_rationale": differs,
        "qrels_superset_of_rationale": all(rationale[q] < qrels_sets[q] for q in differs),
        "strata": strata,
        "misses": misses,
    }
    committed = json.loads((OUT / "retrieval.json").read_text())["metrics"]["hybrid"]
    report["rrf_sensitivity"] = rrf_sensitivity(runs, all_q, qrels_sets, committed)
    report["interpretation"] = interpret(strata, misses)

    OUT.mkdir(parents=True, exist_ok=True)
    md = to_markdown(report)
    (OUT / "analysis.md").write_text(md)
    (OUT / "analysis.json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    print(md)
    print(f"\nWrote {OUT / 'analysis.md'} and {OUT / 'analysis.json'}")


if __name__ == "__main__":
    main()
