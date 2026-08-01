# Semantic Search + RAG Engine

Hybrid retrieval (**BM25 + dense embeddings**, fused with Reciprocal Rank Fusion) and grounded
**RAG** answers, with an evaluation harness that measures every stage on **gold relevance
labels** and paired significance tests, including two cross-encoder rerankers that, measured,
did not earn their place in the pipeline.

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
| RAG eval | LLM-as-judge with abstention scored against gold qrels (`app/eval/rag_eval.py`) |
| LLM | Groq `llama-3.3-70b-versatile` (any OpenAI-compatible endpoint via `SSR_LLM_BASE_URL`) |

## Evaluation (headline artifact)

Measured on BEIR/SciFact: 300 test queries, gold relevance judgments (`uv run python -m app.eval.retrieval_eval`):

| Config | nDCG@10 | Recall@100 | MRR@10 | MAP@100 |
|---|---|---|---|---|
| BM25 | 0.6863 | 0.9127 | 0.6492 | 0.6439 |
| Dense (bge-small) | 0.7127 | 0.9417 | 0.6822 | 0.6736 |
| **Hybrid (RRF)**, default | 0.7241 | **0.9650** | 0.6886 | 0.6816 |
| Hybrid + rerank (MS-MARCO MiniLM) | 0.6975 | **0.9650** | 0.6632 | 0.6558 |
| Hybrid + rerank (bge-reranker-base) | **0.7242** | **0.9650** | **0.6901** | **0.6834** |

Point estimates invite over-reading, so every claim below is backed by a **paired two-sided
t-test on per-query nDCG@10** (300 pairs), reported as Δ, p, and per-query win/tie/loss:

| Comparison | Δ nDCG@10 | p | W/T/L | Significant (p<0.05) |
|---|---|---|---|---|
| Hybrid vs BM25 | +0.0378 | 0.0007 | 72/202/26 | **yes** |
| Hybrid vs Dense | +0.0114 | 0.2578 | 53/213/34 | no |
| Rerank (MiniLM) vs Hybrid | −0.0266 | 0.0555 | 45/192/63 | no |
| Rerank (bge) vs Hybrid | +0.0001 | 0.9964 | 48/196/56 | no |
| Rerank (bge) vs Rerank (MiniLM) | +0.0266 | 0.0376 | 65/190/45 | **yes** |

Six tests are reported here (the five above plus Recall@100 in Finding 1), and none are
corrected for multiple comparisons. Bonferroni at α = 0.05 would set the bar at p ≈ 0.008, which
only hybrid-vs-BM25 clears, so treat the two results at p ≈ 0.035–0.038 as suggestive. The
load-bearing conclusions below are the *null* ones, which correction only strengthens; p = 0.996
is not a near-miss.

**Findings.**

**1. Fusion's win is recall, not ranking.** Hybrid RRF beats BM25 by +0.038 nDCG@10
(p = 0.0007), but its +0.011 edge over *dense alone* is not significant (p = 0.26, W/T/L
53/213/34). Where fusion does separate from dense is **Recall@100: 0.942 → 0.965 (p = 0.035,
W/T/L 9/289/2)**, and that, not top-10 ordering, is the reason to keep it: a fuller candidate
pool for the stages downstream. On 11 discordant queries out of 300, that is the right call on
this evidence rather than a settled one.

**2. Neither cross-encoder reranker paid off.** The CPU-default MS-MARCO MiniLM, trained on
short web queries, *costs* 0.027 nDCG@10 against plain hybrid (p = 0.056, W/T/L 45/192/63). The
domain-appropriate `bge-reranker-base` repairs exactly that damage, beating MiniLM by the same
0.027 (p = 0.038), and then lands on top of doing nothing at all: **Δ = +0.0001, p = 0.996**.

**3. So the reranker *choice* matters and reranking itself doesn't.** A mismatched cross-encoder
degrades ranking; the right one returns you to where you started, expensively. Over the same 300
queries on a 4-core laptop CPU, reranking a 32-candidate slice costs **7.5 s/query (MiniLM)** and
**32.3 s/query (bge)** against ~0.2 s/query for hybrid alone: ~160× the latency for a
statistical tie. **Hybrid RRF is the default**; reranking stays available behind
`mode=hybrid_rerank`.

This replaces an earlier claim in this README, that reranking "only pays off with a
domain-appropriate model such as bge-reranker", which was never run. Measured, it is wrong: the
domain-appropriate model doesn't pay off either, it just stops the mismatched one from hurting.

Bounding the reranked slice to 32 (a latency guard, see [Operational notes](#operational-notes))
also caps the damage a bad reranker can do: MiniLM scored 0.6715 reordering all 100 fused
candidates versus 0.6975 over 32, having fewer chances to promote a bad document into the top
10.[^depth]

[^depth]: A point estimate from the previous committed run at full depth (same model, corpus and
fusion, with the BM25/dense/hybrid rows bit-identical), not a row in the current table. Reproduce
with `SSR_RERANK_CANDIDATES=100 uv run python -m app.eval.retrieval_eval`.

## Grounded answers (RAG)

`GET /answer?q=...` runs the hybrid retrieval, then generates a grounded answer with an OpenAI-compatible LLM (Groq `llama-3.3-70b-versatile` by default; switch providers via `SSR_LLM_BASE_URL`). Answers cite sources as `[n]` mapped back to document ids, and the model is instructed to **abstain when the retrieved context lacks the evidence** rather than hallucinate.

For example, `/answer?q=Can aspirin reduce the risk of colorectal cancer?`:
> "Aspirin has been shown to reduce the risk of colorectal cancer [1][2][3] … a pooled analysis of four randomized trials showed a 34% reduction in 20-year colorectal cancer mortality [3]."

**Answer-quality eval** (`app/eval/rag_eval.py`) is an LLM-as-judge whose abstention decisions
are **scored against the gold qrels rather than assumed correct**. Generator = Groq
`llama-3.3-70b-versatile`, judge = `llama-3.1-8b-instant`, a separate, lighter model, so it
isn't grading its own output. Random 50-claim sample (seed 13); 49 scored, 1 skipped on a
pipeline error, counted and reported rather than silently dropped (`eval/results/rag.md`).

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.83 |
| Context relevance (all) | 0.72 |
| Evidence retrieved (gold doc in top-5) | 0.82 |
| Answered (model attempted an answer) | 0.73 |
| **Abstention precision** (abstained & no evidence) | **0.38** |
| Abstention recall (no evidence & abstained) | 0.56 |
| False abstention (had evidence, abstained anyway) | 0.20 |
| Answered without evidence, as a share of answers given (hallucination risk) | 0.11 |

Crossing the judge's answer/abstain call with whether a gold document was actually retrieved
gives a 2×2 over the 49 claims:

|  | evidence retrieved (40) | no evidence (9) |
|---|---|---|
| **answered** (36) | 32 correct | 4 answered with nothing to go on |
| **abstained** (13) | 8 declined despite having the evidence | 5 correct |

**Finding: abstention is the weakest part of this system, and scoring it against qrels is what
exposed it.** An earlier version of this eval reported "answered 0.50" and concluded the system
"correctly abstains". But an abstention rate alone cannot tell a calibrated refusal from an
over-cautious one. Against the labels: retrieval surfaces the gold document 82% of the time, yet
**only 5 of 13 abstentions were justified** (in 8 cases the evidence was in the context and the
model declined to use it), while it answered 4 of the 9 genuinely unanswerable claims. The
generator is both too conservative where it has evidence and not conservative enough where it
doesn't, so the prompt's abstention instruction is the thing to work on next. Faithfulness over
the answers it does give is solid at 0.83.

(That old 0.50 also came from the first 10 query ids in dataset order: a head slice, not a
sample. The harness now takes a seeded random sample and re-runs at any size via `make eval-rag`.)

## Quickstart

```bash
uv sync                                   # Python 3.12 env + deps
uv run python -m app.ingest.build_index   # build dense (Qdrant) + BM25 indices (one-time)
uv run uvicorn app.api.main:app --reload  # serve UI + API at http://localhost:8000
uv run python -m app.eval.retrieval_eval  # reproduce the metrics table above
uv run pytest                             # tests

cp .env.example .env                      # only for /answer + `make eval-rag`: add a Groq key
```

First run downloads the SciFact corpus (~9 MB via `ir_datasets`) and the embedding model
(~130 MB from HuggingFace); indexing takes a few minutes. Search and `make eval` need no API
key. Only the **grounded-answer** endpoint does, and without one `/answer` returns a clear
503 rather than failing obscurely.

Open <http://localhost:8000> for the search UI, or query the API directly:
`GET /search?q=...&mode=hybrid&top_k=8` with modes `bm25`, `dense`, `hybrid`, `hybrid_rerank`.
(`make install|index|api|eval|test` wrap these.)

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
