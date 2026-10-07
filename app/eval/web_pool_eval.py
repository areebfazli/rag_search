"""Dev harness for the web pipeline's CANDIDATE GENERATION: which open-web sources, alone
and pooled, put SciFact's gold papers in front of the local re-ranker?

Sources (each cached on disk, rate-limited per host, bounded retries; no LLM):
  s2        S2 paper/search with the frozen keyword rewrite (primary + fallback, the
            fallback sent only when the live pipeline sends it) — the current `web` pool
            (WebSearch.candidates, app/retrieve/web_search.py)
  s2_multi  extra deterministic S2 keyword queries (query_rewrite.multi_queries)
  snip_claim / snip_rw   S2 snippet search with the claim / the primary rewrite
  pm_claim / pm_rw       PubMed E-utilities Best Match with the claim / the rewrite
  oa_sem / oa_rw         OpenAlex semantic search with the claim / works search with the rewrite
  cites     references + citations of the top seeds of the base pool (S2 graph)

External ids (PMID/DOI) are resolved to S2 corpus ids with one paper/batch call per
claim (bulk-prefetched here; S2IdResolver caches per id), because SciFact doc ids are S2
corpus ids. Every combination is pooled (web_search.merge_pools), re-ranked with the same
local RRF(bge-small, BM25) as `web` (web_search.rerank_pool), and scored on the gold qrels.

Develop on the seeded 100-claim train sample only:
    SSR_EVAL_DATASET=beir/scifact/train SSR_WEB_EVAL_N=100 \
        uv run --locked python -m app.eval.web_pool_eval
SSR_POOL_SOURCES=s2,snip_claim,pm_rw,pm_claim fetches only the live pool's sources.
Writes data/eval_runs/web_pool_<dataset>_<n>/web_pool.{md,json} (+ runs.json, the
per-claim ranked lists, for paired comparisons between runs). Never the canonical
eval/results/ (the frozen choice is run on test through app.eval.web_eval).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.retrieve.pubmed import quiet_http_logs

METRICS = ["recall@5", "recall@10", "recall@100", "ndcg@10"]
SEED = 13
RUNS = Path("data/eval_runs")
SOURCE_NAMES = ["s2", "s2_multi", "s2_deep", "snip_claim", "snip_rw", "pm_claim", "pm_rw", "oa_sem", "oa_rw", "cites"]
SNIPPET_LIMIT = int(os.environ.get("SSR_POOL_SNIPPET_LIMIT", "100"))
OA_CLAIMS = int(os.environ.get("SSR_POOL_OA_CLAIMS", "45"))  # first N claims only
N_SEEDS = int(os.environ.get("SSR_POOL_SEEDS", "3"))  # citation seeds: top-N of the pre-ranked seed pool
SCREEN = bool(os.environ.get("SSR_POOL_SCREEN"))  # cheap BM25+source pre-rank, no embedding
CITE_CAP = int(os.environ.get("SSR_POOL_CITE_CAP", "30"))
SEED_POOL = os.environ.get("SSR_POOL_SEED_POOL", "s2").split("+")


def fetch_sources() -> set[str]:
    """SSR_POOL_SOURCES (comma list of SOURCE_NAMES): fetch only these (s2 is always
    fetched — it is every row's baseline). Unset = all. Lets a rewrite change be measured
    on the live pool (s2,snip_claim,pm_rw,pm_claim) without paying for every other source."""
    raw = os.environ.get("SSR_POOL_SOURCES", "")
    if not raw.strip():
        return set(SOURCE_NAMES)
    chosen = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = chosen - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"SSR_POOL_SOURCES: unknown source(s) {sorted(unknown)}; expected {SOURCE_NAMES}")
    return chosen | {"s2"}


def _ids(hits):
    return [h.doc_id for h in hits]


class Sources:
    """Builds every source's candidate list for one claim (SearchHits, keep_no_abstract),
    counting logical requests (cached or not) per source — the live per-query cost."""

    def __init__(self, env: dict):
        from app.retrieve.openalex import OpenAlexSource
        from app.retrieve.pubmed import PubMedSource
        from app.retrieve.s2_extra import S2IdResolver
        from app.retrieve.semantic_scholar import RateLimiter, ResponseCache, SemanticScholarRetriever

        cache = ResponseCache(settings.s2_cache_dir)
        self.cache = cache
        self.s2 = SemanticScholarRetriever(
            cache=cache, limiter=RateLimiter(settings.s2_rate_per_s), max_wait_s=None, max_backoff_s=180.0
        )
        ext = ResponseCache(Path(settings.s2_cache_dir).parent / "web_cache")
        self.pubmed = PubMedSource(cache=ext, max_backoff_s=180.0)
        self.openalex = OpenAlexSource(cache=ext, max_backoff_s=180.0)
        self.resolver = S2IdResolver(self.s2, Path(settings.s2_cache_dir) / "ids")
        self.rarity = env["rarity"]
        self.logical: dict[str, int] = {n: 0 for n in SOURCE_NAMES + ["resolve"]}

    def s2_base(self, claim: str) -> list[SearchHit]:
        """The live `web` base pool: WebSearch.candidates with the frozen rewrite, so the
        fallback query is sent exactly when the live pipeline sends it (not when the
        primary alone filled the page with abstract-bearing papers). Title-only papers
        are kept (another source may lend the abstract; each view filters at merge)."""
        from app.retrieve.web_search import WebSearch

        entries: list[dict] = []
        ws = WebSearch(self.s2, rewrite=True, rarity=self.rarity, strict=True)
        hits = ws.candidates(claim, 100, keep_no_abstract=True, entries=entries)
        self.logical["s2"] += len(entries)
        for h in hits:
            h.metadata["source"] = "s2"
        return hits

    def s2_queries(self, queries: list[str], name: str) -> list[SearchHit]:
        from app.retrieve.semantic_scholar import parse_search

        out: list[SearchHit] = []
        seen: set[str] = set()
        for q in queries:
            self.logical[name] += 1
            entry = self.s2.fetch(q, 100)
            for h in parse_search(entry["response"], 100, keep_no_abstract=True):
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    h.metadata["source"] = name
                    out.append(h)
        return out

    def s2_deep(self, query: str) -> list[SearchHit]:
        """Pages 2-3 (offsets 100, 200) of the primary rewrite query."""
        from app.retrieve.semantic_scholar import (
            SEARCH_FIELDS,
            SEARCH_PATH,
            S2Error,
            normalize_query,
            parse_search,
            payload_error,
            valid_search_payload,
        )

        out: list[SearchHit] = []
        for off in (100, 200):
            p = {"query": normalize_query(query), "limit": 100, "fields": SEARCH_FIELDS, "offset": off}
            self.logical["s2_deep"] += 1
            hit = self.cache.get(SEARCH_PATH, p)
            if hit is not None and not valid_search_payload(hit.get("response")):
                hit = None  # an error body cached by older code is a miss
            if hit is None:
                try:
                    payload = self.s2._request("GET", SEARCH_PATH, p)
                except S2Error as e:
                    if "HTTP 400" in str(e):  # offset past the matches
                        break
                    raise
                if not valid_search_payload(payload):  # a 200 with an error body: never cached
                    raise S2Error(payload_error(payload))
                payload.setdefault("data", [])
                hit = self.cache.put(SEARCH_PATH, p, payload)
            got = parse_search(hit["response"], 100, keep_no_abstract=True)
            for h in got:
                h.metadata["source"] = "s2_deep"
            out += got
            if len(hit["response"].get("data") or []) < 100:
                break
        return out

    def external(self, claim: str, idx: int, only: str = "", sources: set[str] | None = None) -> dict[str, list]:
        """ExternalPaper lists per external source (before id resolution). `only` =
        "ext" fetches the non-S2 hosts only (PubMed/OpenAlex), "s2" the S2 ones only."""
        from app.retrieve.query_rewrite import rewrite_queries
        from app.retrieve.s2_extra import snippet_search

        rw = (rewrite_queries(claim, self.rarity) or [claim])[0]
        plan = [
            ("snip_claim", "s2", 1, lambda: snippet_search(self.s2, claim, SNIPPET_LIMIT, self.cache)),
            ("snip_rw", "s2", 1, lambda: snippet_search(self.s2, rw, SNIPPET_LIMIT, self.cache)),
            ("pm_claim", "ext", 2, lambda: self.pubmed.search(claim, 100)),
            ("pm_rw", "ext", 2, lambda: self.pubmed.search(rw, 100)),
        ]
        if idx < OA_CLAIMS:  # keyless OpenAlex budget: ~100 searches/day (see openalex.py)
            plan += [
                ("oa_sem", "ext", 1, lambda: self.openalex.search(claim, 50, mode="semantic")),
                ("oa_rw", "ext", 1, lambda: self.openalex.search(rw, 100)),
            ]
        out = {}
        for name, host, cost, fn in plan:
            if (only and host != only) or (sources is not None and name not in sources):
                continue
            self.logical[name] = self.logical.get(name, 0) + cost
            out[name] = fn()
        return out


def baseline_of(name: str) -> str:
    return "s2" + ("_any" if name.endswith("_any") else "")


def score_on_subsets(
    qrels: dict[str, dict[str, int]], runs: dict[str, dict], subsets: dict[str, list[str]]
) -> dict[str, dict]:
    """ranx scores per run on ITS OWN claim subset, plus its baseline (s2 in the same
    view) re-scored on that subset and the paired t-test p-value against it.

    Runs sharing a subset are scored together (one ranx.compare per subset). Returns
    {name: {"scores", "baseline_scores", "vs_s2_p"}} ("vs_s2_p" absent for the baseline
    itself); a run with an empty subset gets nothing."""
    from ranx import Qrels, Run, compare

    groups: dict[tuple[str, ...], list[str]] = {}
    for name, subset in subsets.items():
        if subset:
            groups.setdefault(tuple(subset), []).append(name)
    out: dict[str, dict] = {}
    for subset, names in groups.items():
        bases = [b for b in dict.fromkeys(baseline_of(n) for n in names) if b in runs]
        members = list(dict.fromkeys(names + bases))
        q = Qrels({k: qrels[k] for k in subset})
        blob = compare(q, [Run({k: runs[nm].get(k, {}) for k in subset}, name=nm) for nm in members],
                       METRICS, max_p=0.05).to_dict()
        for nm in names:
            base = baseline_of(nm)
            entry = {"scores": {m: float(blob[nm]["scores"][m]) for m in METRICS}}
            if base in blob:
                entry["baseline_scores"] = {m: float(blob[base]["scores"][m]) for m in METRICS}
                if nm != base:
                    entry["vs_s2_p"] = {m: float(blob[nm]["comparisons"][base][m]) for m in METRICS}
            out[nm] = entry
    return out


_TAG_OK = re.compile(r"[A-Za-z0-9_-]{1,64}")


def pool_tag() -> str:
    """SSR_POOL_TAG (an output-dir suffix), or "" if unset. Only [A-Za-z0-9_-] (1-64
    chars) is accepted; anything else ("../x", "a/b", spaces, dots) is refused with
    ValueError before any work starts, so the tag can never move the output dir out
    of data/eval_runs/."""
    tag = os.environ.get("SSR_POOL_TAG", "")
    if tag and not _TAG_OK.fullmatch(tag):
        raise ValueError(f"SSR_POOL_TAG must match [A-Za-z0-9_-]{{1,64}}, got {tag!r}")
    return tag


def main() -> int:
    from app.eval.rag_eval import sample_claims
    from app.eval.web_eval import load_rarity, make_embedder, to_run
    from app.ingest.corpus import load_documents, load_queries_qrels
    from app.retrieve.query_rewrite import multi_queries, rewrite_queries
    from app.retrieve.s2_extra import expand_citations
    from app.retrieve.web_search import EmbeddingCache, merge_pools, prerank, rerank_pool

    tag = pool_tag()  # refuse a bad tag up front, not after hours of fetching
    wanted = fetch_sources()
    quiet_http_logs()
    queries, qrels = load_queries_qrels()
    n = int(os.environ.get("SSR_WEB_EVAL_N", "100"))
    ids = sample_claims(queries, n, SEED)
    limit = int(os.environ.get("SSR_EVAL_LIMIT", "0"))
    if limit:
        ids = ids[:limit]
    docs = load_documents()
    env = {"rarity": load_rarity(docs)}
    src = Sources(env)
    emb_cache = EmbeddingCache(Path(settings.s2_cache_dir) / "emb")
    embedder = make_embedder()
    dense_cap = int(os.environ.get("SSR_POOL_DENSE_CAP", "0")) or None

    # 1. Fetch every source for every claim (cached; resumable by re-running).
    t0 = time.time()
    per: dict[str, dict[str, list]] = {}
    ext_all: dict[str, dict[str, list]] = {}
    only = os.environ.get("SSR_POOL_ONLY", "")
    for i, qid in enumerate(ids, 1):
        claim = queries[qid]
        rws = rewrite_queries(claim, env["rarity"]) or [claim]
        ext_all[qid] = src.external(claim, i - 1, only, wanted)
        if only == "ext":
            continue
        per[qid] = {"s2": src.s2_base(claim)}
        if "s2_multi" in wanted:
            per[qid]["s2_multi"] = src.s2_queries(multi_queries(claim, env["rarity"]), "s2_multi")
        if "s2_deep" in wanted and not os.environ.get("SSR_POOL_NO_DEEP"):
            per[qid]["s2_deep"] = src.s2_deep(rws[0])
        if i % 10 == 0:
            print(f"  fetched {i}/{len(ids)} ({(time.time() - t0) / 60:.1f} min, S2 requests {src.s2.requests_sent})", flush=True)

    if only == "ext":
        print(f"External sources fetched: PubMed {src.pubmed.requests_sent}, OpenAlex {src.openalex.requests_sent} requests")
        return 0

    # 2. Resolve external ids in bulk, then per claim (cache hits only).
    every = [p for q in ext_all.values() for lst in q.values() for p in lst]
    src.resolver.prefetch(every)
    for qid in ids:
        for name in SOURCE_NAMES:
            if name in ("s2", "s2_multi", "s2_deep", "cites"):
                continue
            papers = ext_all[qid].get(name)
            if papers is None:  # not fetched for this claim (OpenAlex subset)
                continue
            per[qid][name] = [
                SearchHit(h.doc_id, h.score, h.text, {**h.metadata, "source": name})
                for h in src.resolver.to_hits(papers, keep_no_abstract=True)
            ]
        src.logical["resolve"] += 1

    # 3. Citation expansion from (SSR_POOL_NO_CITES=1 skips it) the top seeds of the reranked base (s2) pool.
    for qid in (ids if "cites" in wanted and not os.environ.get("SSR_POOL_NO_CITES") else []):
        claim = queries[qid]
        base = prerank(claim, merge_pools([per[qid][s] for s in SEED_POOL]))
        seeds = [h.doc_id for h in base[:N_SEEDS] if h.doc_id.isdigit()]
        papers = expand_citations(src.s2, seeds, max_refs_per_seed=CITE_CAP, max_cites_per_seed=CITE_CAP, cache=src.cache) if seeds else []
        src.logical["cites"] += 1 if seeds else 0
        per[qid]["cites"] = [
            SearchHit(h.doc_id, h.score, h.text, {**h.metadata, "source": "cites"})
            for h in src.resolver.to_hits(papers, keep_no_abstract=True)
        ]
    print(f"Fetched all sources in {(time.time() - t0) / 60:.1f} min; S2 requests {src.s2.requests_sent}, "
          f"PubMed {src.pubmed.requests_sent}, OpenAlex {src.openalex.requests_sent}", flush=True)

    if os.environ.get("SSR_POOL_FETCH_ONLY"):
        return 0

    # 4. Score combinations. A combination is scored ONLY on the claims where every one
    # of its sources was actually queried (OpenAlex ran on the first OA_CLAIMS claims
    # only), and its baseline (s2, same view) is re-scored on exactly those claims — a
    # source that was never asked must not count as a miss.
    combos_env = os.environ.get("SSR_POOL_COMBOS")
    combos = [tuple(c.split("+")) for c in combos_env.split(",")] if combos_env else (
        [(s,) for s in SOURCE_NAMES]
        + [("s2", s) for s in SOURCE_NAMES if s != "s2"]
    )
    if ("s2",) not in combos:
        combos = [("s2",)] + list(combos)  # the baseline every row is compared with
    results = {}
    runs = {}
    subsets: dict[str, list[str]] = {}
    for combo in combos:
        for view, keep in (("", False), ("_any", True)):
            name = "+".join(combo) + view
            run, pool_hit, pool_sizes = {}, 0, []
            subset = [qid for qid in ids if all(s in per[qid] for s in combo)]
            for qid in subset:
                pool = merge_pools([per[qid][s] for s in combo], keep_no_abstract=keep)
                pool_sizes.append(len(pool))
                pool_hit += any(h.doc_id in qrels[qid] for h in pool)
                ranked = (prerank(queries[qid], pool) if SCREEN else
                          rerank_pool(queries[qid], pool, embedder, emb_cache, dense_cap))[:100]
                run[qid] = to_run(ranked)
            runs[name] = run
            subsets[name] = subset
            results[name] = {"n": len(subset), "pool_any_gold": pool_hit / max(1, len(subset)),
                             "pool_mean": sum(pool_sizes) / max(1, len(subset)),
                             "subset": len(subset) < len(ids)}
        print(f"  scored {'+'.join(combo)} (n={len(subsets['+'.join(combo)])})", flush=True)
    for nm, scored in score_on_subsets(qrels, runs, subsets).items():
        results[nm].update(scored)
    names = list(runs)
    slug = "".join(c if c.isalnum() else "-" for c in settings.eval_dataset).strip("-")
    out = RUNS / (f"web_pool_{slug}_{len(ids)}" + (tag and "_" + tag))
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Web candidate pool — {settings.eval_dataset}, {len(ids)} claims (seed {SEED})", "",
             f"dense_cap={dense_cap}; screen={SCREEN}; logical requests per claim: "
             + ", ".join(f"{k}={v / len(ids):.2f}" for k, v in src.logical.items()), "",
             "Each row is scored only on the n claims where all of its sources were queried; "
             "`s2 R@100 (same n)` and the p-value compare it with the s2 baseline (same view) "
             f"on exactly those claims. OpenAlex (oa_*) ran on the first {OA_CLAIMS} claims only.", "",
             "| Pool | n | gold in pool | mean pool | R@5 | R@10 | R@100 | nDCG@10 | s2 R@100 (same n) | p(R@100 vs s2) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for nm in names:
        r = results[nm]
        if "scores" not in r:
            lines.append(f"| {nm} | 0 | | | | | | | | |")
            continue
        s = r["scores"]
        p = r.get("vs_s2_p", {}).get("recall@100")
        base = r.get("baseline_scores", {}).get("recall@100")
        label = nm + (f" (OpenAlex subset, n={r['n']})" if "oa_" in nm else (f" (subset, n={r['n']})" if r["subset"] else ""))
        lines.append(f"| {label} | {r['n']} | {r['pool_any_gold']:.2f} | {r['pool_mean']:.0f} | {s['recall@5']:.4f} | "
                     f"{s['recall@10']:.4f} | {s['recall@100']:.4f} | {s['ndcg@10']:.4f} | "
                     f"{'' if base is None else f'{base:.4f}'} | {'' if p is None else f'{p:.4f}'} |")
    md = "\n".join(lines) + "\n"
    (out / "web_pool.md").write_text(md)
    (out / "web_pool.json").write_text(json.dumps({"results": results, "logical": src.logical, "n": len(ids),
                                                   "dense_cap": dense_cap}, indent=2))
    # Per-claim ranked lists, for paired comparisons between two harness runs.
    (out / "runs.json").write_text(json.dumps(runs))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
