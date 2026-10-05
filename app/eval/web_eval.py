"""Web retrieval eval: can Semantic Scholar's open search find SciFact's gold papers?

Runs S2 paper relevance search (the `web` mode, through SearchService like every other
eval) for each SciFact test claim and scores it against the SAME gold qrels as the local
pipeline, next to the committed local `hybrid` run (read from retrieval_eval's cache —
the Qdrant index is never opened), with retrieval_eval's paired t-test via ranx.

What this measures, and what it doesn't:
  * S2 searches its whole graph (~200M papers); our index holds 5,183. So this is "can
    open-web retrieval surface the gold paper among everything", not a like-for-like
    comparison of rankers — the local hybrid only has to beat 5,182 distractors.
  * S2's index and ranking change over time. Every response is cached on disk with its
    fetch time (data/s2_cache/), the report records the fetch-date range, and a re-run
    re-scores the cached snapshot rather than re-querying.
  * Scoring relies on SciFact doc ids being S2 corpus ids (allenai/scifact doc/data.md:
    doc_id is "The document's S2ORC ID"). That is CHECKED first, with one paper/batch
    request over the gold doc ids (titles compared to the local corpus); below
    MAPPING_MIN_MATCH the run stops before any search request is spent.
  * `web` is the retriever as served (papers without an abstract are dropped, since
    they carry no passage to ground an answer on); `web_any` re-parses the same cached
    responses keeping them, i.e. "did S2 find the paper at all".

Cost: one request per claim (limit=100 is S2's max page size, so Recall@100 comes free)
plus one batch request, spaced by the process-wide limiter (settings.s2_rate_per_s,
default 1/s = the documented per-key rate). The count and minimum time print up front.
Interrupted (e.g. persistent 429s)? Re-run the same command: cached responses are reused.

Run:
    SSR_S2_API_KEY=... uv run --locked python -m app.eval.web_eval     # or: make web-eval
    SSR_EVAL_LIMIT=2 uv run --locked python -m app.eval.web_eval       # 3-request smoke run
    SSR_EVAL_REFRESH=1 ...   # re-fetch everything, ignoring (and overwriting) the cache
Outputs eval/results/web_retrieval.{md,json} for the full test split only (pooled rows
on, id mapping enforced); any other run — sampled, SSR_EVAL_LIMIT, another split,
SSR_WEB_EVAL_POOL=0 or SSR_WEB_EVAL_ALLOW_UNMAPPED=1 — goes to
data/eval_runs/web_<dataset>_<n>[_nopool][_unmapped]/.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings, settings
from app.core.interfaces import SearchHit
from app.core.paths import RESULTS
from app.retrieve.web_sources import SourceError
from app.retrieve.semantic_scholar import (
    S2_BATCH_MAX_IDS,
    S2_MAX_LIMIT,
    SEARCH_PATH,
    ResponseCache,
    S2Error,
    SemanticScholarRetriever,
    search_params,
)

DEPTH = S2_MAX_LIMIT  # one page of S2's max size; the local hybrid run is also depth 100
METRICS = ["recall@5", "recall@10", "recall@100", "ndcg@10"]
PRETTY = {
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "recall@100": "Recall@100",
    "ndcg@10": "nDCG@10",
}
MAX_P = 0.05
# Share of gold doc ids whose S2 title must match the local SciFact title for the id
# mapping to count as confirmed. Titles differ slightly across sources (punctuation,
# trailing period, case), hence the normalised fuzzy match below.
MAPPING_MIN_MATCH = 0.9
TITLE_MATCH_RATIO = 0.9
BATCH_FIELDS = "title,corpusId"
EST_LATENCY_S = 1.0  # rough per-request S2 latency for the up-front time estimate

OUT = RESULTS  # canonical run only — the committed artifact (repo-anchored, not cwd)
RUNS = Path("data/eval_runs")
CANONICAL_DATASET = Settings.model_fields["eval_dataset"].default
SEED = 13  # sample seed for SSR_WEB_EVAL_N (rag_eval's shuffle-then-prefix sampling)

# The web pipeline variants (app/retrieve/web_search.WebSearch): key -> (rewrite, rerank).
# Each also has an `<key>_any` twin over the same responses, keeping papers that have no
# abstract ("did S2 find the paper at all").
VARIANTS = {
    "web": (False, False),
    "web_rewrite": (True, False),
    "web_rerank": (False, True),
    "web_rewrite_rerank": (True, True),
}
LABELS = {
    "web": "S2, raw claim as query (web mode as before)",
    "web_rewrite": "S2, keyword rewrite",
    "web_rerank": "S2, raw claim + local rerank",
    "web_rewrite_rerank": "S2, keyword rewrite + local rerank",
}
# Multi-source candidate pools (WebSearch.search_pooled), rewrite + rerank on. Chosen on
# the seeded 100-claim train sample with app/eval/web_pool_eval.py, then frozen:
#   web_pool_live    — the latency-bounded choice: S2 rewrite + S2 snippet search (claim)
#                      + PubMed Best Match (rewrite and claim), dense rerank of the top 50
#                      (train: R@5 .285 / R@100 .453 vs .208 / .218 for rewrite+rerank)
#   web_pool_offline — the recall-first choice: + 3 extra S2 keyword queries + snippet
#                      search on the rewrite too, dense rerank of the top 100
#                      (train: R@5 .285 / R@100 .473; best train R@100 of the served pools)
# Served view only (no `_any` twin): each extra pooled view is another ~100 embeddings per
# claim; train's _any numbers are in data/eval_runs/web_pool_*.
POOL_VARIANTS = {
    "web_pool_live": {
        "pubmed_queries": ("rewrite", "claim"), "snippet_queries": ("claim",),
        "multi_query": 0, "dense_cap": 50,
    },
    "web_pool_offline": {
        "pubmed_queries": ("rewrite", "claim"), "snippet_queries": ("claim", "rewrite"),
        "multi_query": 3, "dense_cap": 100,
    },
}
POOL_LABELS = {
    "web_pool_live": "Pooled: S2 rewrite + S2 snippets + PubMed, dense rerank of top 50 (live)",
    "web_pool_offline": "Pooled: + 3 extra S2 queries + rewrite snippets, dense rerank of top 100 (offline)",
}
# (key, label) of the rows in the report; "hybrid" is the committed local run.
ROWS = [("hybrid", "Local hybrid (RRF, 5,183 docs)")]
for _key, _label in LABELS.items():
    ROWS += [(_key, _label), (f"{_key}_any", f"{_label}, incl. no-abstract papers")]
ROWS += list(POOL_LABELS.items())
# Every web row vs the local hybrid; every new variant vs raw S2 in the same view; every
# pooled variant vs the previous best (rewrite + rerank) in the same view.
PAIRS = [(k, "hybrid") for k, _ in ROWS[1:]] + [
    (f"{k}{view}", f"web{view}") for k in VARIANTS if k != "web" for view in ("", "_any")
] + [
    (k, "web_rewrite_rerank") for k in POOL_VARIANTS
] + [("web_pool_offline", "web_pool_live")]


def pool_enabled() -> bool:
    """SSR_WEB_EVAL_POOL=0 skips the pooled rows (no PubMed / snippet requests)."""
    return os.environ.get("SSR_WEB_EVAL_POOL", "1").strip() not in ("0", "", "false")

Run_ = dict[str, dict[str, float]]


class _NoLocalIndex:
    """Stands in for the dense/BM25 retrievers: web mode must never touch them (and
    this eval must never open the single-process embedded Qdrant store)."""

    def search(self, query, top_k):  # pragma: no cover - reaching it is the bug
        raise AssertionError("web_eval touched the local index")


class _WriteOnlyCache(ResponseCache):
    """SSR_EVAL_REFRESH: never read the cache, but still record fresh responses."""

    def get(self, endpoint, params):
        return None


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def titles_match(a: str, b: str) -> bool:
    ka, kb = _title_key(a), _title_key(b)
    if not ka or not kb:
        return False
    return ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= TITLE_MATCH_RATIO


def check_mapping(
    retriever: SemanticScholarRetriever, local_titles: dict[str, str], doc_ids: list[str]
) -> dict:
    """Look up each gold doc id as `CorpusId:<id>` (paper/batch, <= 500 per request) and
    compare S2's title with the local one. Returns counts + a few mismatch examples."""
    ids = sorted(set(doc_ids), key=int)
    found = matched = 0
    mismatches: list[dict] = []
    for i in range(0, len(ids), S2_BATCH_MAX_IDS):
        chunk = ids[i : i + S2_BATCH_MAX_IDS]
        papers = retriever.batch([f"CorpusId:{d}" for d in chunk], BATCH_FIELDS)
        for doc_id, paper in zip(chunk, papers, strict=False):
            if not isinstance(paper, dict):
                continue
            found += 1
            s2_title = paper.get("title") if isinstance(paper.get("title"), str) else ""
            if str(paper.get("corpusId")) == doc_id and titles_match(s2_title, local_titles.get(doc_id, "")):
                matched += 1
            elif len(mismatches) < 5:
                mismatches.append(
                    {"doc_id": doc_id, "local": local_titles.get(doc_id, ""), "s2": s2_title[:200]}
                )
    n = len(ids)
    return {
        "n_gold_docs": n,
        "found_in_s2": found,
        "title_matches": matched,
        "match_rate": matched / n if n else 0.0,
        "threshold": MAPPING_MIN_MATCH,
        "confirmed": bool(n) and matched / n >= MAPPING_MIN_MATCH,
        "mismatch_examples": mismatches,
    }


def load_hybrid_run(n_total: int) -> Run_:
    """The committed local hybrid run, from retrieval_eval's cache (same signature
    logic, so a settings/index change can't pass off a stale run)."""
    from app.eval.retrieval_eval import CACHE, _signature

    sig = _signature("hybrid", None, n_total)
    path = CACHE / f"hybrid-{sig}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached hybrid run at {path} ({n_total} queries). Run `make eval` first; "
            "a signature mismatch means the settings or index differ from that run."
        )
    blob = json.loads(path.read_text())
    run = blob.get("run") if isinstance(blob, dict) else None
    if not isinstance(run, dict) or blob.get("signature") != sig or not blob.get("complete"):
        raise ValueError(f"{path} is not a complete hybrid run; rerun `make eval`.")
    return run


def to_run(hits: list[SearchHit]) -> dict[str, float]:
    # Rank-derived scores, as retrieval_eval does, so ranx preserves the order exactly.
    return {h.doc_id: float(len(hits) - rank) for rank, h in enumerate(hits)}


def score(qrels_dict: dict[str, dict[str, int]], runs: dict[str, Run_]) -> dict:
    """ranx scores + paired two-sided t-tests (retrieval_eval's method) for ROWS/PAIRS."""
    from ranx import Qrels, Run, compare

    qids = sorted(qrels_dict)
    qrels = Qrels({q: qrels_dict[q] for q in qids})
    keys = [k for k, _ in ROWS if k in runs]
    ranx_runs = [Run({q: runs[k].get(q, {}) for q in qids}, name=k) for k in keys]
    blob = compare(qrels, ranx_runs, METRICS, max_p=MAX_P).to_dict()
    comparisons = {}
    for a, b in PAIRS:
        if a in runs and b in runs:
            comparisons[f"{a}_vs_{b}"] = {
                m: {
                    "delta": blob[a]["scores"][m] - blob[b]["scores"][m],
                    "p": float(blob[a]["comparisons"][b][m]),
                    "wtl": blob[a]["win_tie_loss"][b][m],
                }
                for m in METRICS
            }
    return {
        "n_queries": len(qids),
        "scores": {k: {m: float(v) for m, v in blob[k]["scores"].items()} for k in keys},
        "comparisons": comparisons,
        "stat_test": blob.get("stat_test", "student"),
    }


def to_markdown(result: dict, meta: dict) -> str:
    scores = result["scores"]
    labels = dict(ROWS)
    lines = [
        f"# Web retrieval (Semantic Scholar) — {meta['title']}",
        "",
        f"S2 responses fetched {meta['fetched_from']} → {meta['fetched_to']} (UTC). "
        "S2's index and ranking change over time; re-running re-scores this cached snapshot.",
        "",
        "S2 searches its whole graph (~200M papers); the local index holds 5,183. This "
        "measures whether open-web retrieval surfaces the gold paper among everything, not "
        "a like-for-like ranker comparison.",
        "",
        "Variants (app/retrieve/web_search.py; no LLM anywhere): *raw* sends the claim "
        "sentence as the S2 query; *rewrite* sends a keyword query built from it "
        f"(app/retrieve/query_rewrite.py: ≤{meta['rewrite']['max_terms']} content terms, "
        f"plus a {meta['rewrite']['fallback_terms']}-term fallback query whose results are "
        "pooled with the first's, unless the first alone fills the page); *rerank* re-orders "
        "the S2 candidates (≤100 per query) locally by RRF of bge-small similarity to the "
        "claim and BM25 over the candidates. Rewrite rules were chosen on a seeded 100-claim "
        "beir/scifact/train sample and frozen before the test run.",
        "",
        "*Pooled* rows (app/retrieve/web_search.py `search_pooled`) widen CANDIDATE "
        "GENERATION: the rewrite's S2 results are pooled with S2 snippet search "
        "(app/retrieve/s2_extra.py; papers quoting the claim verbatim or mentioning SciFact "
        "are excluded) and PubMed Best Match (app/retrieve/pubmed.py), external ids "
        "resolved to S2 corpus ids with S2 paper/batch, deduped, and re-ranked by the same "
        "local RRF — with only the top N of a BM25 pre-rank embedded (latency bound). "
        "Sources and settings chosen on the train sample with app/eval/web_pool_eval.py "
        "and frozen before this run.",
        "",
        "| Config | " + " | ".join(PRETTY[m] for m in METRICS) + " |",
        "|" + "---|" * (len(METRICS) + 1),
    ]
    for key, _ in ROWS:
        if key in scores:
            cells = " | ".join(f"{scores[key][m]:.4f}" for m in METRICS)
            lines.append(f"| {labels[key]} | {cells} |")
    lines += [
        "",
        f"## Significance (paired two-sided Student's t-test, p < {MAX_P})",
        "",
        "| Comparison | Metric | Δ | p | W/T/L | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for name, per_metric in result["comparisons"].items():
        for m, c in per_metric.items():
            wtl = c["wtl"]
            lines.append(
                f"| {name.replace('_vs_', ' vs ')} | {PRETTY[m]} | {c['delta']:+.4f} | "
                f"{c['p']:.4f} | {wtl['W']}/{wtl['T']}/{wtl['L']} | "
                f"{'yes' if c['p'] < MAX_P else 'no'} |"
            )
    mp = meta["mapping"]
    rw = meta["rewrite"]
    lines += [
        "",
        "## Id mapping check",
        "",
        f"SciFact doc ids looked up as S2 `CorpusId:<id>`: {mp['found_in_s2']}/{mp['n_gold_docs']} "
        f"gold docs found, {mp['title_matches']} with a matching title "
        f"(match rate {mp['match_rate']:.3f}, threshold {mp['threshold']}).",
        "",
        f"Queries: {meta['n_queries']} · S2 requests this run: {meta['requests_sent']} · "
        f"authenticated: {meta['authenticated']} · raw-query web hits dropped for having no "
        f"abstract: {meta['dropped_no_abstract']} · rewrite fallback query sent for "
        f"{rw['fallback_used']}/{meta['n_queries']} claims · rewritten claims with no content "
        f"terms (sent raw): {rw['no_terms']} · candidate embeddings cached/computed this run: "
        f"{meta['embeddings']['cached']}/{meta['embeddings']['computed']} · PubMed requests "
        f"this run: {meta.get('pubmed_requests_sent', 0)}.",
        "",
    ]
    return "\n".join(lines)


def allow_unmapped() -> bool:
    return bool(os.environ.get("SSR_WEB_EVAL_ALLOW_UNMAPPED", "").strip())


def output_dir(limit: int, n_queries: int, sampled: bool = False) -> tuple[Path, bool]:
    """(directory, canonical). Only the full test split, with the pooled rows and the id
    mapping check enforced, may write the committed eval/results/: SSR_WEB_EVAL_POOL=0
    (pooled rows missing) or SSR_WEB_EVAL_ALLOW_UNMAPPED=1 (possibly unconfirmed ids)
    goes to data/eval_runs/ like any other non-canonical run."""
    flags = ("" if pool_enabled() else "_nopool") + ("_unmapped" if allow_unmapped() else "")
    if not limit and not sampled and not flags and settings.eval_dataset == CANONICAL_DATASET:
        return OUT, True
    slug = "".join(c if c.isalnum() else "-" for c in settings.eval_dataset).strip("-")
    return RUNS / f"web_{slug}_{n_queries}{flags}", False


def make_embedder():
    """bge-small for the rerank (a seam for tests). Loads the model only — never Qdrant."""
    from app.index.embedder import Embedder

    return Embedder()


def load_rarity(docs: list[dict]):
    from app.ingest.corpus import document_passage
    from app.retrieve.query_rewrite import TermRarity

    return TermRarity.from_texts(document_passage(d) for d in docs)


def local_hybrid_run(queries: dict[str, str], n_total: int) -> Run_:
    """The local hybrid run for these claims. On the canonical split it is the committed
    run from retrieval_eval's cache (Qdrant is never opened); on any other split (the
    train dev sample) it is computed once through SearchService — which opens the local
    index — and cached by retrieval_eval.cached_run under its own key."""
    if settings.eval_dataset == CANONICAL_DATASET:
        run = load_hybrid_run(n_total)
    else:
        from app.eval.retrieval_eval import cached_run
        from app.retrieve.service import SearchService

        print("Non-canonical split: computing the local hybrid run (opens the local index).")
        run = cached_run(SearchService(), queries, f"hybrid_web-eval-s{SEED}", "hybrid", None)
    missing = [q for q in queries if q not in run]
    if missing:
        raise ValueError(f"local hybrid run lacks {len(missing)} of these claims (e.g. {missing[:3]})")
    return {q: run[q] for q in queries}


def pipelines(retriever, embedder_factory, rarity, emb_cache, extras=None) -> dict:
    """One WebSearch per row key, sharing the retriever, embedder, rarity and caches.
    strict: a failed fallback request stops the run instead of scoring a thinner pool."""
    from app.retrieve.web_search import WebSearch

    embedder: list = []

    def shared_embedder():
        if not embedder:
            embedder.append(embedder_factory())
        return embedder[0]

    out = {}
    for key, (rewrite, rerank) in VARIANTS.items():
        for view, keep in (("", False), ("_any", True)):
            out[key + view] = WebSearch(
                retriever,
                rewrite=rewrite,
                rerank=rerank,
                embedder=shared_embedder,
                rarity=rarity,
                emb_cache=emb_cache,
                keep_no_abstract=keep,
                strict=True,
            )
    if extras is not None:  # {"pubmed": PubMedSource, "resolver": S2IdResolver}
        for key, kw in POOL_VARIANTS.items():
            for view, keep in (("", False),):
                out[key + view] = WebSearch(
                    retriever,
                    rewrite=True,
                    rerank=True,
                    embedder=shared_embedder,
                    rarity=rarity,
                    emb_cache=emb_cache,
                    keep_no_abstract=keep,
                    strict=True,
                    snippets=True,
                    pubmed=extras["pubmed"],
                    resolver=extras["resolver"],
                    **kw,
                )
    return out


def make_extras(retriever):
    """PubMed source + S2 id resolver for the pooled rows: cached on disk (data/), long
    backoff like the S2 retriever here, keyless."""
    from app.retrieve.pubmed import PubMedSource
    from app.retrieve.s2_extra import S2IdResolver

    return {
        "pubmed": PubMedSource(
            cache=ResponseCache(Path(settings.s2_cache_dir).parent / "web_cache"),
            rate=settings.pubmed_rate_per_s,
            max_retries=max(3, settings.s2_max_retries),
            max_backoff_s=180.0,
        ),
        "resolver": S2IdResolver(retriever, Path(settings.s2_cache_dir) / "ids"),
    }


def estimate_requests(web: dict, queries: dict[str, str], reader: ResponseCache, refresh: bool):
    """(min, max) uncached S2 searches: raw claims + rewrite primaries, plus fallbacks —
    known exactly when the primary is cached, else counted as possible."""
    from app.retrieve.semantic_scholar import parse_search

    def uncached(q: str) -> bool:
        return refresh or reader.get(SEARCH_PATH, search_params(q, DEPTH)) is None

    rewrite = web["web_rewrite"]
    lo = hi = 0
    for text in queries.values():
        raw = int(uncached(text))
        lo += raw
        hi += raw
        qs = rewrite.queries(text)
        first = uncached(qs[0])
        lo += first
        hi += first
        if len(qs) > 1:
            if first:
                hi += uncached(qs[1])
            else:
                entry = reader.get(SEARCH_PATH, search_params(qs[0], DEPTH)) or {}
                need = len(parse_search(entry.get("response"), DEPTH)) < DEPTH
                lo += need and uncached(qs[1])
                hi += need and uncached(qs[1])
    return lo, hi


def main() -> int:
    from app.eval.rag_eval import sample_claims
    from app.eval.retrieval_eval import _json_safe
    from app.ingest.corpus import load_documents, load_queries_qrels
    from app.retrieve import query_rewrite
    from app.retrieve.service import SearchService
    from app.retrieve.web_search import EmbeddingCache

    queries, qrels_dict = load_queries_qrels()
    n_total = len(queries)
    limit = int(os.environ.get("SSR_EVAL_LIMIT", "0"))
    n_env = os.environ.get("SSR_WEB_EVAL_N", "all").strip().lower()
    sampled = n_env not in ("", "all")
    if sampled:
        queries = {q: queries[q] for q in sample_claims(queries, int(n_env), SEED)}
    if limit:
        queries = dict(list(queries.items())[:limit])
    qrels_dict = {q: v for q, v in qrels_dict.items() if q in queries}
    out, canonical = output_dir(limit, len(queries), sampled)

    refresh = bool(os.environ.get("SSR_EVAL_REFRESH"))
    cache = (_WriteOnlyCache if refresh else ResponseCache)(settings.s2_cache_dir)
    # S2 has answered 429 with Retry-After: 64 s even with a key at <=1 req/s (2026-09-29), so
    # the offline eval waits up to 3 min per 429 instead of stopping; the API path keeps
    # its short max_wait_s budget.
    retriever = SemanticScholarRetriever(cache=cache, max_wait_s=None, max_backoff_s=180.0)
    docs = load_documents()
    local_titles = {d["doc_id"]: d["title"] for d in docs}
    emb_cache = EmbeddingCache(Path(settings.s2_cache_dir) / "emb")
    extras = make_extras(retriever) if pool_enabled() else None
    web = pipelines(retriever, make_embedder, load_rarity(docs), emb_cache, extras)

    # Up-front cost: uncached searches + the mapping batch (if uncached).
    gold_ids = sorted({d for v in qrels_dict.values() for d in v}, key=int)
    reader = ResponseCache(settings.s2_cache_dir)  # always reads, even under refresh
    lo, hi = estimate_requests(web, queries, reader, refresh)
    n_batches = -(-len(gold_ids) // S2_BATCH_MAX_IDS)
    print(
        f"Eval set: {settings.eval_dataset}, {len(queries)} claims"
        + (f" (seeded sample, seed {SEED})" if sampled else "")
        + f", {len(gold_ids)} gold docs. "
        f"S2 key: {'set' if retriever.authenticated else 'NOT set (shared public pool; expect 429s)'}."
    )
    print(
        f"Expected S2 requests: ~{lo + n_batches}–{hi + n_batches} (≤{n_batches} batch + "
        f"{lo}–{hi} uncached searches; retries extra) → ≥{(lo + n_batches) / settings.s2_rate_per_s / 60:.1f}–"
        f"{(hi + n_batches) / settings.s2_rate_per_s / 60:.1f} min at {settings.s2_rate_per_s:g} req/s, "
        f"~{(hi + n_batches) * (1 / settings.s2_rate_per_s + EST_LATENCY_S) / 60:.1f} min worst case "
        "with latency."
    )
    if not canonical:
        print(f"Non-canonical run — writing to {out}, not {OUT}")

    hybrid = local_hybrid_run(queries, n_total)

    try:
        mapping = check_mapping(retriever, local_titles, gold_ids)
    except S2Error as e:
        print(f"\nS2 id-mapping check failed ({e}). Nothing scored; re-run to retry.")
        return 1
    print(
        f"Id mapping: {mapping['title_matches']}/{mapping['n_gold_docs']} gold docs match "
        f"by title (rate {mapping['match_rate']:.3f}, need ≥{MAPPING_MIN_MATCH})."
    )
    if not mapping["confirmed"] and not allow_unmapped():
        print(
            "SciFact ids do not map to S2 corpus ids well enough to score against gold "
            f"qrels (examples: {mapping['mismatch_examples'][:3]}). Stopping before any "
            "search request. Set SSR_WEB_EVAL_ALLOW_UNMAPPED=1 to score anyway."
        )
        return 1

    runs: dict[str, Run_] = {"hybrid": hybrid, **{k: {} for k in web}}
    fetched: list[str] = []
    dropped = fallback_used = no_terms = 0
    t0 = time.time()
    for i, (qid, text) in enumerate(queries.items(), start=1):
        for key, pipeline in web.items():
            # Through SearchService, as the API runs it (web mode, this pipeline).
            service = SearchService(dense=_NoLocalIndex(), lexical=_NoLocalIndex(), web=pipeline)
            try:
                res = service.retrieve_web(text, mode="web", top_k=DEPTH)
            except (S2Error, SourceError) as e:
                print(
                    f"\nStopped at claim {i}/{len(queries)} ({key}): {e}. Responses so far are "
                    "cached — re-run the same command to resume."
                )
                return 1
            runs[key][qid] = to_run(res.hits)
            fetched += [e["fetched_at"] for e in res.entries if e.get("fetched_at")]
            if key == "web_rewrite":
                fallback_used += len(res.entries) > 1
                no_terms += not query_rewrite.claim_terms(text)
        dropped += len(runs["web_any"][qid]) - len(runs["web"][qid])
        if i % 20 == 0 or i == len(queries):
            print(f"  {i}/{len(queries)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    result = score(qrels_dict, runs)
    today = datetime.now(timezone.utc).date().isoformat()
    title = (
        f"BEIR/SciFact ({result['n_queries']} test claims)"
        if canonical
        else f"{settings.eval_dataset} ({result['n_queries']} claims"
        + (f", seeded sample (seed {SEED})" if sampled else "")
        + (f", SSR_EVAL_LIMIT={limit}" if limit else "")
        + ("" if extras else ", no pooled rows")
        + (", SSR_WEB_EVAL_ALLOW_UNMAPPED" if allow_unmapped() else "")
        + ", non-canonical)"
    )
    meta = {
        "title": title,
        "dataset": settings.eval_dataset,
        "n_queries": result["n_queries"],
        "sample_seed": SEED if sampled else None,
        "depth": DEPTH,
        "fetched_from": min(fetched, default=today),
        "fetched_to": max(fetched, default=today),
        "scored_on": today,
        "requests_sent": retriever.requests_sent,
        "pubmed_requests_sent": extras["pubmed"].requests_sent if extras else 0,
        "pool": {k: {**v, "pubmed": True, "snippets": True} for k, v in POOL_VARIANTS.items()} if extras else None,
        "authenticated": retriever.authenticated,
        "dropped_no_abstract": dropped,
        "rewrite": {
            "max_terms": query_rewrite.MAX_TERMS,
            "fallback_terms": query_rewrite.FALLBACK_TERMS,
            "term_priority": "local-corpus IDF",
            "fallback_used": fallback_used,
            "no_terms": no_terms,
        },
        "rerank": {
            "embedding_model": settings.embedding_model,
            "fusion": f"RRF(k={settings.rrf_k}) of dense + BM25 over the S2 candidates",
            "candidates_per_query": DEPTH,
        },
        "embeddings": {"cached": emb_cache.hits, "computed": emb_cache.misses},
        "runtime_s": round(time.time() - t0, 1),
        "mapping": mapping,
    }
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(result, meta)
    (out / "web_retrieval.md").write_text(md)
    (out / "web_retrieval.json").write_text(
        json.dumps(_json_safe({**meta, **result, "max_p": MAX_P}), indent=2, allow_nan=False)
    )
    print("\n" + md)
    print(f"Wrote {out / 'web_retrieval.md'} and {out / 'web_retrieval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
