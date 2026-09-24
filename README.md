# Semantic Search with RAG Engine

[![CI](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml/badge.svg)](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml)

Hybrid retrieval (**BM25 + dense embeddings**, fused with Reciprocal Rank Fusion) and grounded
**RAG** answers, with an evaluation harness that measures every stage on **gold relevance
labels** and paired significance tests, including two cross-encoder rerankers that, measured,
did not earn their place in the pipeline.

![Search UI: hybrid (RRF) results for "Vitamin D deficiency is associated with increased risk of multiple sclerosis" over 5,183 SciFact abstracts](docs/ui.png)

## Results at a glance

BEIR/SciFact, 300 test queries, gold qrels. Bold is best per column.

| Config | nDCG@10 | Recall@100 |
|---|---|---|
| BM25 | 0.6863 | 0.9127 |
| Dense (bge-small) | 0.7127 | 0.9417 |
| **Hybrid (RRF)**, default | 0.7241 | **0.9650** |
| Hybrid + rerank (MS-MARCO MiniLM) | 0.6975 | **0.9650** |
| Hybrid + rerank (bge-reranker-base) | **0.7242** | **0.9650** |

MRR@10, MAP@100 and the per-comparison significance tests are in
[`eval/results/retrieval.md`](eval/results/retrieval.md); what the numbers mean is in
[Findings](#evaluation-headline-artifact) below.

## Pipeline

```
query ─┬─► BM25 (bm25s)      top-100 ─┐
       └─► dense (Qdrant)    top-100 ─┴─► RRF fusion ─► top-8 ──► search results
                                              │
                        context + grounded prompt ─► LLM ─► answer + [n] citations

optional 2nd stage: cross-encoder rerank over the top-32 fused candidates.
Measured below, and on this corpus it does not improve ranking, so it is off by default.
```

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Lexical | `bm25s` |
| Fusion | Reciprocal Rank Fusion (hand-rolled) |
| Vector store | Qdrant (embedded local mode) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base` (both evaluated; swappable via `SSR_RERANKER_MODEL`) |
| API | FastAPI + slowapi rate limiting |
| Retrieval eval | `ranx` on BEIR/SciFact gold qrels, with paired significance tests |
| RAG eval | Verdicts + abstention scored against SciFact rationale labels; LLM judge for faithfulness (`app/eval/rag_eval.py`) |
| LLM | Groq `openai/gpt-oss-120b` generator, `qwen/qwen3.8-27b` judge (any OpenAI-compatible endpoint via `SSR_LLM_BASE_URL`) |

## Evaluation (headline artifact)

Measured on BEIR/SciFact: 300 test queries, gold relevance judgments (`make eval`; the table is
[above](#results-at-a-glance)). Point estimates invite over-reading, so every claim below is
backed by a **paired two-sided t-test on per-query nDCG@10** (300 pairs), reported as Δ, p, and
per-query win/tie/loss. The full per-comparison table is committed in
[`eval/results/retrieval.md`](eval/results/retrieval.md#significance), and the label-stratified
re-score behind Finding 1 in [`eval/results/analysis.md`](eval/results/analysis.md).

Six tests are reported here (the five in that significance table plus Recall@100 in Finding 1),
and none are corrected for multiple comparisons. Bonferroni at α = 0.05 would set the bar at
p ≈ 0.008, which only hybrid-vs-BM25 clears, so treat the two results at p ≈ 0.035–0.038 as
suggestive. The load-bearing conclusions below are the *null* ones, which correction only
strengthens; p = 0.996 is not a near-miss.

**Findings.**

**1. Fusion's win is recall, not ranking.** Hybrid RRF beats BM25 by +0.038 nDCG@10
(p = 0.0007), but its +0.011 edge over *dense alone* is not significant (p = 0.26, W/T/L
53/213/34). Where fusion does separate from dense is **Recall@100: 0.942 → 0.965 (p = 0.035,
W/T/L 9/289/2)**, and that, not top-10 ordering, is the reason to keep it: a fuller candidate
pool for the stages downstream. On 11 discordant queries out of 300, that is the right call on
this evidence rather than a settled one.[^rrf]

Stratified by SciFact's claim labels ([`eval/results/analysis.md`](eval/results/analysis.md)),
that recall gain is **entirely on NEI claims**, the 112 of 300 where annotators found no rationale
in any abstract but BEIR's qrels still mark the cited one relevant. All 11 discordant queries are
NEI: hybrid gains +0.0625 Recall@100 there (p = 0.034, W/T/L 9/101/2). On the 188
evidence-bearing (SUPPORT/CONTRADICT) claims, hybrid and dense are identical at Recall@100 on
every query (0.9947 against rationale docs), and nDCG@10 differs by +0.0003 (p = 0.977). So the
fuller pool is fuller in abstracts that carry no evidence: fusion does not retrieve more
supporting or refuting evidence than dense alone on this corpus.

**2. Neither cross-encoder reranker paid off.** The CPU-default MS-MARCO MiniLM, trained on
short web queries, *costs* 0.027 nDCG@10 against plain hybrid (p = 0.056, W/T/L 45/192/63). The
domain-appropriate `bge-reranker-base` repairs exactly that damage, beating MiniLM by the same
0.027 (p = 0.038), and then lands on top of doing nothing at all: **Δ = +0.0001, p = 0.996**.

**3. So the reranker *choice* matters and reranking itself doesn't.** A mismatched cross-encoder
degrades ranking; the right one returns you to where you started, expensively. On a 4-core
laptop CPU, reranking a 32-candidate slice costs **3.94 s/query (MiniLM)** and **23.4 s/query
(bge)** against 0.124 s/query for hybrid alone: ≈188× the latency for a statistical tie (MiniLM
≈32×). Those are per-query means of `SearchService.retrieve` after 3 warm-ups on an Intel
i5-10210U with torch pinned to 4 threads, hybrid over all 300 queries and rerank over a seeded
40-query sample ([`eval/results/latency.md`](eval/results/latency.md)); earlier figures here were
console readings from the eval sweep and ran higher. **Hybrid RRF is the default**; reranking
stays available behind `mode=hybrid_rerank`.

This replaces an earlier claim in this README, that reranking "only pays off with a
domain-appropriate model such as bge-reranker", which was never run. Measured, it is wrong: the
domain-appropriate model doesn't pay off either, it just stops the mismatched one from hurting.

Bounding the reranked slice to 32 (a latency guard, see [Operational notes](#operational-notes))
also caps the damage a bad reranker can do: MiniLM scored 0.6715 reordering all 100 fused
candidates versus 0.6975 over 32, having fewer chances to promote a bad document into the top
10.[^depth]

[^rrf]: RRF is insensitive to its k on this data: for k ∈ {1, 2, 5, 10, 20, 100}, nDCG@10 moves
by at most 0.0046 against the default k = 60 (all p ≥ 0.19), and Recall@100 stays at 0.965 on
every query. No fusion of these two pools could raise it much: only 3 of the 11 misses are in
either retriever's top-100, a ceiling of 0.975. Weighting dense at 0.3 or 0.7 instead of equally
gains no significant nDCG@10 (-0.0096, p = 0.18; +0.0034, p = 0.50) and costs recall: 0.928 at
0.3 dense (p = 0.002), 0.952 at 0.7 (p = 0.16). The numbers are an offline replay of the cached
top-100 lists, verified to reproduce the committed hybrid run exactly, in §4 of
[`eval/results/analysis.md`](eval/results/analysis.md#4-rrf-sensitivity-offline-replay).

[^depth]: A point estimate from the previous committed run at full depth (same model, corpus and
fusion, with the BM25/dense/hybrid rows bit-identical), not a row in the current table. Reproduce
with `SSR_RERANK_CANDIDATES=100 uv run python -m app.eval.retrieval_eval`.

## Grounded answers (RAG)

`GET /answer?q=...` runs the hybrid retrieval, then generates a grounded answer with an OpenAI-compatible LLM (Groq `openai/gpt-oss-120b` by default; switch providers via `SSR_LLM_BASE_URL`). Answers cite sources as `[n]` mapped back to document ids, and the model is instructed to **abstain when the retrieved context lacks the evidence** rather than hallucinate.

For example, `/answer?q=Can aspirin reduce the risk of colorectal cancer?`:
> "Aspirin has been shown to reduce the risk of colorectal cancer [1][2][3] … a pooled analysis of four randomized trials showed a 34% reduction in 20-year colorectal cancer mortality [3]."

**Answer-quality eval** (`app/eval/rag_eval.py`) scores each answer's final `Verdict:` line
(supported / refuted / not enough evidence) and its answer/abstain decision **against SciFact's
labels rather than assuming them correct**; an LLM judge scores only faithfulness and context
relevance. Generator = Groq `openai/gpt-oss-120b`, judge = `qwen/qwen3.8-27b`, a different model
family, so it isn't grading its own output. Random 50-claim sample (seed 13), all 50 scored,
top-5 context; full tables in [`eval/results/rag.md`](eval/results/rag.md). Groq retired the
Llama models the previous run used, so these numbers are not directly comparable to that run's.

Abstention is scored against a **rationale oracle**: the context has evidence when a document the
annotators cited *with rationale sentences* is in the top-5. NEI claims have no rationale
document, so abstaining on them is the correct action.

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.98 |
| Context relevance (all) | 0.70 |
| **3-class verdict accuracy** (vs gold label, no judge) | **0.70** |
| Verdict line parsed | 0.86 |
| Evidence retrieved (rationale doc in top-5) | 0.58 |
| Answered (model attempted an answer) | 0.60 |
| **Abstention precision** (abstained & no evidence) | **0.80** |
| Abstention recall (no evidence & abstained) | 0.76 |
| False abstention (had evidence, abstained anyway) | 0.14 |
| Answered without evidence, as a share of answers given (hallucination risk) | 0.17 |

|  | evidence retrieved (29) | no evidence (21) |
|---|---|---|
| **answered** (30) | 25 answered with evidence | 5 answered with nothing to go on |
| **abstained** (20) | 4 declined despite having the evidence | 16 correct |

Verdicts against the gold label (`NONE` = answered with no parseable verdict line, always wrong):

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** (19) | 12 | 1 | 2 | 4 |
| **CONTRADICT** (13) | 0 | 9 | 4 | 0 |
| **NEI** (18) | 4 | 0 | 14 | 0 |

**Finding: the oracle decides whether abstention looks broken.** Scored the legacy way, against
BEIR qrels (which mark a cited abstract relevant for NEI claims too), the *same answers* give
abstention precision 0.35 and 13 "false" abstentions (false-abstention rate 0.32). Under the
rationale oracle, 9 of those 13 are NEI claims where refusing was the right call, and precision
is 0.80. What remains is smaller and real: 4 SUPPORT claims answered without a parseable verdict
line (scored `NONE`), 4 CONTRADICT claims abstained on, and 4 NEI claims answered SUPPORT, which
are 4 of the 5 answers given without evidence. Faithfulness over the answers it does give is 0.98.

(An earlier version of this eval reported "answered 0.50" from the first 10 query ids in dataset
order: a head slice, not a sample. The harness now takes a seeded random sample and re-runs at any size via `make eval-rag`.)

## Limitations

- **300 queries bound what can be detected.** For hybrid vs dense, the minimum detectable effect
  at 80% power (α = 0.05, paired) is ≈ 0.028 nDCG@10 and ≈ 0.031 Recall@100. The nulls above
  rule out large effects, not small ones.
- **Recall@100 has a ceiling fusion cannot move.** 8 of the 11 gold docs hybrid misses are in
  neither retriever's top-100, so no fusion rule over these two candidate pools could recover
  them ([`eval/results/analysis.md`](eval/results/analysis.md), §3).
- **NEI labelling.** For the 112 NEI claims, BEIR's qrels mark the cited abstract relevant even
  though annotators found no rationale in it, and the whole hybrid-vs-dense recall gain sits on
  that stratum (Finding 1).
- **The RAG eval is small.** N = 50 claims, so every rate in that table rests on a few dozen
  answers, and 10 of the 12 confusion-matrix cells are single digits.
- **The LLM judge is single-sample.** Each answer is judged once, so judge variance on
  faithfulness and context relevance is unmeasured. Verdict accuracy and abstention need no judge,
  except for the 7 of 50 answer/abstain calls it settles when no verdict line parsed.
- **No before/after on the RAG eval.** The model change (retired Groq Llama models →
  `openai/gpt-oss-120b` / `qwen/qwen3.8-27b`) breaks comparability with the earlier published run,
  so no RAG-quality change can be attributed across it.

## What I'd do next

- **Fix verdict-line compliance**: 14% of answers end without a parseable `Verdict:` line, which
  alone costs 4 SUPPORT claims (scored `NONE`). Enforce the format (structured output or a
  retry) so verdict accuracy measures judgment, not formatting.
- **Measure judge agreement by re-sampling**: re-run the judge several times per answer to put a
  variance on faithfulness and context relevance, which currently rest on one call each.
- **Stratified reporting as the default**: report every retrieval comparison by claim label, not
  only as a follow-up analysis.

## Quickstart

```bash
uv sync                                   # Python 3.12 env + deps          (make install)
uv run python -m app.ingest.build_index   # build dense (Qdrant) + BM25     (make index, one-time)
uv run uvicorn app.api.main:app --reload  # UI + API at localhost:8000      (make api)
uv run python -m app.eval.retrieval_eval  # reproduce the metrics table     (make eval)
uv run python -m app.eval.analysis        # label-stratified re-score       (make analysis)
uv run python -m app.eval.latency         # per-query retrieval latency     (make latency)
uv run pytest                             # tests                           (make test)

cp .env.example .env                      # only for /answer + `make eval-rag`: add a Groq key
```

First run downloads the SciFact corpus (~9 MB via `ir_datasets`) and the embedding model
(~130 MB from HuggingFace); indexing takes a few minutes. Search and `make eval` need no API
key. Only the **grounded-answer** endpoint does, and without one `/answer` returns a clear
503 rather than failing obscurely. `make analysis` re-scores the cached `make eval` runs, so run
`make eval` first. `make lint` runs ruff, and `make ci` is what CI runs: a locked install, then
lint + test.

Open <http://localhost:8000> for the search UI, or query the API directly:
`GET /search?q=...&mode=hybrid&top_k=8` with modes `bm25`, `dense`, `hybrid`, `hybrid_rerank`.

## Operational notes

Things that are deliberate rather than accidental, and the reasoning behind them:

- **Reranking is bounded to the top-32 fused candidates** (`SSR_RERANK_CANDIDATES`); the tail
  keeps its fused order, so recall past the slice is untouched. A cross-encoder is a full
  forward pass *per candidate*. Reranking all 100 takes ~20s on CPU, and because retrieval is
  serialized behind a lock, an unbounded slice lets a single client at the rate limit hold the
  service for minutes. It is a denial-of-service guard first and a latency knob second.
- **The API is unauthenticated**, so per-IP rate limiting (30/min search, 10/min answer) is the
  only guard. `SSR_TRUST_PROXY=true` keys on the X-Forwarded-For entry `SSR_TRUSTED_PROXY_HOPS`
  in from the *right* (default 1), after joining repeated header lines. Set the hop count to
  exactly how many proxies you run, because the count *is* the trust boundary: only the rightmost
  `hops` entries were written by your own infrastructure. Too **low** stops short and keys on one
  of your proxies, putting every user behind it in one bucket. Too **high** indexes past your
  proxies into the part of the list the *client* supplied, letting a client choose its own key
  and rotate it per request, bypassing the limit entirely. The default of 1 is the safe
  value and cannot be over-indexed.
- **Embedded Qdrant locks to a single process.** Don't run the API and `make eval`/`make index`
  at the same time; the second one will fail to acquire the storage lock.
- **`make eval` caches per config** under `data/eval_cache/`, keyed by a signature of everything
  that affects the numbers, and checkpoints every 20 queries, because a cross-encoder sweep over 300
  queries is hours on a laptop CPU. `SSR_EVAL_REFRESH=1` forces recomputation.
- **Retrieval is serialized** by a process-wide lock (embedded Qdrant and the shared
  sentence-transformers models are not thread-safe), so this serves one search at a time per
  worker. Running Qdrant in server mode is the fix if that ceiling ever matters.

## License

[MIT](LICENSE)
