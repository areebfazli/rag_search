# Project details

The long-form companion to the [README](../README.md): methodology, significance tests,
per-label breakdowns, what was tried and not adopted, limitations, and operational notes. Every
number comes from the committed files in [`eval/results/`](../eval/results/).

Contents: [Retrieval results](#retrieval-results-at-a-glance) ·
[Pipeline](#pipeline) · [Stack](#stack) · [Code layout](#code-layout) ·
[Retrieval evaluation](#retrieval-evaluation) · [Grounded answers](#grounded-answers-rag) ·
[Label audit](#label-audit) · [Tried and not adopted](#tried-and-measured-not-adopted) ·
[Judge reliability](#judge-reliability) · [Web search](#web-search-semantic-scholar) ·
[Limitations](#limitations) · [What I'd do next](#what-id-do-next) ·
[Commands and API](#commands-and-api) · [Operational notes](#operational-notes)

## Retrieval results at a glance

BEIR/SciFact, 300 test queries, gold qrels. Bold is best per column.

| Config | nDCG@10 | Recall@100 |
|---|---|---|
| BM25 | 0.6863 | 0.9127 |
| Dense (bge-small) | 0.7127 | 0.9417 |
| **Hybrid (RRF)**, default | 0.7241 | **0.9650** |
| Hybrid + rerank (MS-MARCO MiniLM) | 0.6975 | **0.9650** |
| Hybrid + rerank (bge-reranker-base) | **0.7242** | **0.9650** |

RAG answers (Ling 3.0 Flash Sante, free, 300 test claims): **verdict accuracy 0.80** (95% CI
0.751–0.841), faithfulness 0.97, abstention precision 0.83; see [Grounded answers](#grounded-answers-rag).

MRR@10, MAP@100 and the per-comparison significance tests are in
[`eval/results/retrieval.md`](../eval/results/retrieval.md); what the numbers mean is in
[Findings](#retrieval-evaluation) below.

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
| RAG eval | Verdicts (with a one-call verdict re-ask) + abstention scored against SciFact rationale labels, plus a label-audit sensitivity check; free Nemotron 3 Ultra judge for faithfulness, with a judge-consistency harness (`app/eval/rag_eval.py`, `app/eval/judge_agreement.py`) |
| LLM | OpenRouter free models by default ($0): `inclusionai/ling-3.0-flash-sante:free` (Ling 3.0 Flash Sante) generator and `nvidia/nemotron-3-ultra-550b-a55b:free` (Nemotron 3 Ultra) judge, from different model families so the generator never grades its own output. Paid `openai/gpt-oss-120b` is an optional generator override via `SSR_OPENROUTER_LLM_MODEL` (pinned to a bf16 provider, no fallbacks, price-capped; [`app/core/llm_endpoints.py`](../app/core/llm_endpoints.py)). Groq, Ollama or any OpenAI-compatible endpoint via `SSR_LLM_PROVIDER=groq` + `SSR_LLM_BASE_URL` |

## Code layout

```
app/core       config (pydantic-settings, SSR_ env prefix) · interfaces (Retriever/Reranker/Generator Protocols) · llm_endpoints (provider -> URL/key/model + OpenRouter spend policy)
app/ingest     corpus (BEIR/SciFact via ir_datasets, + claim labels) · build_index
app/index      embedder (bge-small) · vector_store (Qdrant embedded) · lexical (bm25s)
app/retrieve   dense · fusion (RRF) · service (SearchService, the single retrieval entry point) · semantic_scholar · query_rewrite · web_search
app/rerank     cross_encoder (MS-MARCO MiniLM, bge-reranker-base)
app/generate   generator (OpenAI-compatible client) · prompts (grounded prompt + injection sanitizer)
app/verify     NLI and fine-tuned verifiers + Kaggle training kit (measured, not adopted)
app/eval       retrieval_eval · rag_eval · rag_compare · label_audit · web_eval · latency · analysis · judge_agreement · verify_eval/verify_combine
app/api        main (FastAPI: /search, /answer, rate-limited)
frontend/      index.html (vanilla JS UI, served at /)
```

`SearchService.retrieve(query, mode, top_k)` is shared by the API and every eval harness, so
both exercise identical logic. There is no docker-compose: Qdrant runs embedded (on disk under
`data/qdrant`), and the `Dockerfile` packages the API only.

## Retrieval evaluation

Measured on BEIR/SciFact: 300 test queries, gold relevance judgments (`make eval`; the table is
[above](#retrieval-results-at-a-glance)). Point estimates invite over-reading, so every claim below is
backed by a **paired two-sided t-test on per-query nDCG@10** (300 pairs), reported as Δ, p, and
per-query win/tie/loss. The full per-comparison table is committed in
[`eval/results/retrieval.md`](../eval/results/retrieval.md#significance), and the label-stratified
re-score behind Finding 1 in [`eval/results/analysis.md`](../eval/results/analysis.md).

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

Stratified by SciFact's claim labels ([`eval/results/analysis.md`](../eval/results/analysis.md)),
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
40-query sample ([`eval/results/latency.md`](../eval/results/latency.md)); earlier figures here were
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
[`eval/results/analysis.md`](../eval/results/analysis.md#4-rrf-sensitivity-offline-replay).

[^depth]: A point estimate from the previous committed run at full depth (same model, corpus and
fusion, with the BM25/dense/hybrid rows bit-identical), not a row in the current table. Reproduce
with `SSR_RERANK_CANDIDATES=100 uv run python -m app.eval.retrieval_eval`.

## Grounded answers (RAG)

`GET /answer?q=...` runs the hybrid retrieval, then generates a grounded answer with an OpenAI-compatible LLM (the free `inclusionai/ling-3.0-flash-sante:free` on OpenRouter by default; paid `openai/gpt-oss-120b` via `SSR_OPENROUTER_LLM_MODEL`, or Groq, Ollama or another endpoint via `SSR_LLM_PROVIDER=groq` and `SSR_LLM_BASE_URL`). Answers cite sources as `[n]` mapped back to document ids, and the model is instructed to **abstain when the retrieved context lacks the evidence** rather than hallucinate. When the input is a claim and the reply has no parseable verdict (truncated, or prose without a `Verdict:` line), the generator makes one verdict-only follow-up call (`SSR_LLM_REASK`, on by default; the response's `verdict_source` says where the verdict came from).

For example, `/answer?q=Can aspirin reduce the risk of colorectal cancer?`:
> "Aspirin has been shown to reduce the risk of colorectal cancer [1][2][3] … a pooled analysis of four randomized trials showed a 34% reduction in 20-year colorectal cancer mortality [3]."

**Answer-quality eval** (`app/eval/rag_eval.py`) scores each answer's final `Verdict:` line
(supported / refuted / not enough evidence) and its answer/abstain decision **against SciFact's
labels rather than assuming them correct**; an LLM judge scores only faithfulness and context
relevance. Generator = Ling 3.0 Flash Sante (`inclusionai/ling-3.0-flash-sante:free`), a free
health/medicine-tuned model, with a 2048-token budget and one retry at 2x on
`finish_reason=length`. Judge = Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`), a
different model family, so the generator isn't grading its own output (picked from a bench of 10
free models; the previous judge's free variant served 0 of 4 calls). **All 300 SciFact test
claims** (seed 13, so the first 50 are the earlier 50-claim sample), 0 skipped, top-5 context;
full tables in [`eval/results/rag.md`](../eval/results/rag.md). **A full run costs $0 on the default
free models.** The first-pass answers span two days (OpenRouter's 1,000 requests/day free cap hit
mid-run, resumed from the per-row checkpoint).

**Verdict accuracy is 0.80 (240 of 300), 95% Wilson CI 0.751–0.841.**

| Gold label | n | Verdict accuracy | Answered (not abstained) |
|---|---|---|---|
| SUPPORT | 124 | 0.815 (101) | 107 |
| CONTRADICT | 64 | 0.891 (57) | 58 |
| NEI | 112 | 0.732 (82) | 30 |

**What changed.** Two post-processing steps on the same Ling answers, each developed on SciFact
*train* claims and then measured once on the 300 test claims; paired exact McNemar via
`make rag-compare` (b = earlier run right and this run wrong, c = the reverse):

| Step | Verdict accuracy | b | c | p |
|---|---|---|---|---|
| Strict `Verdict:` line only | 0.73 | | | |
| + verdict-recovery parser (verdict restated in prose, or a stance rule; commits ec465ef, b3de17b) | 0.7767 | 0 | 14 | 0.0001 |
| **+ verdict re-ask** (one verdict-only call for a claim with no verdict; commits e367318, 3dcc1a3) | **0.8000** | **0** | **7** | **0.016** |

The re-ask fired on 15 of 300 claims (10 answers truncated at the token budget, 5 prose replies
without a verdict line), produced a verdict for 14 (12 correct), and broke nothing. Abstention
(+0.01, p = 0.25) and citation rate (0.94, unchanged) don't move significantly. Faithfulness
moved 0.98 → 0.97 only because three answers that were truncated to nothing now count as
answered. Verdict sources over the 300 rows: `line` 257, `inline` 10, `stance` 18, `reask` 14,
none 1 (still truncated after the re-ask, scored as an abstention).

Abstention is scored against a **rationale oracle**: the context has evidence when a document the
annotators cited *with rationale sentences* is in the top-5. NEI claims have no rationale
document, so abstaining on them is the correct action.

| Metric | Score |
|---|---|
| Faithfulness (over answered, LLM judge) | 0.97 |
| Context relevance (all, LLM judge) | 0.72 |
| **3-class verdict accuracy** (vs gold label, no judge) | **0.80** (240/300) |
| Verdict parsed (incl. recovered and re-asked) | 299 of 300 |
| Truncated answers (hit the token budget after 1 retry) | 11 of 300 |
| Answers that needed the length retry | 63 of 300 |
| Replies citing at least one passage | 282 of 300 |
| Evidence retrieved (rationale doc in top-5) | 0.58 |
| Answered (model attempted an answer) | 0.65 |
| **Abstention precision** (abstained & no evidence) | **0.83** |
| Abstention recall (no evidence & abstained) | 0.70 |
| False abstention (had evidence, abstained anyway) | 0.10 |
| Answered without evidence, as a share of answers given (hallucination risk) | 0.19 |

|  | evidence retrieved (175) | no evidence (125) |
|---|---|---|
| **answered** (195) | 157 answered with evidence | 38 answered with nothing to go on |
| **abstained** (105) | 18 declined despite having the evidence | 87 correct |

Verdicts against the gold label (`NONE` = answered with no verdict, always wrong):

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** (124) | 101 | 6 | 17 | 0 |
| **CONTRADICT** (64) | 1 | 57 | 6 | 0 |
| **NEI** (112) | 17 | 13 | 82 | 0 |

**Remaining failure modes.** 60 of 300 verdicts are wrong, and almost all are an evidence
*judgement* error, not a format one:

- **NEI claims answered anyway: 30 of 112** (17 SUPPORTED, 13 REFUTED), the largest group. The
  model is over-eager: it infers a verdict from weak or tangential context. They account for most
  of the 38 answers given without evidence.
- **SUPPORT/CONTRADICT claims abstained on: 23** (17 SUPPORT, 6 CONTRADICT). For 14 of the 17
  SUPPORT claims the rationale doc is in the context: the model reads the passage too literally.
- Smaller: 6 SUPPORT claims answered REFUTED and 1 CONTRADICT answered SUPPORTED.

**The oracle decides whether abstention looks broken.** Scored the legacy way, against BEIR qrels
(which mark a cited abstract relevant for NEI claims too), the *same answers* give abstention
precision 0.41 and 62 "false" abstentions (rate 0.26); under the rationale oracle only 18 are
false and precision is 0.83.

**Paired comparison with the paid gpt-oss-120b run** (50 shared claims; both predate the
verdict-recovery parser, so they use the old strict parser and are not comparable to the 0.80
headline). Accuracy is indistinguishable and citations are the one significant difference:

| On the 50 shared claims | gpt-oss-120b (paid) | Ling | b | c | p |
|---|---|---|---|---|---|
| Verdict correct | 0.78 | 0.76 | 2 | 1 | 1.00 |
| Abstention correct (rationale oracle) | 0.84 | 0.78 | 3 | 0 | 0.25 |
| **Cites at least one passage** | **0.42** | **0.94** | **1** | **27** | **< 0.0001** |

On those claims 28 gpt-oss replies were a bare `Verdict:` line against 0 for Ling, and the run
cost $0.0049 against $0. Same-model Ling-vs-Ling repeats flip 7 of 50 verdicts (p = 1.00), the
noise floor for 50 claims.

### Label audit

SciFact's labels contain errors. Sylvestre, *Gold Label Errors in the SciFact Benchmark: An
LLM-Assisted Annotation Audit* ([BioNLP 2026](https://aclanthology.org/2026.bionlp-1.9/); data
[Kefez/scifact-audit-bionlp2026](https://github.com/Kefez/scifact-audit-bionlp2026) at pinned
commit `5711f3f`, CC BY 4.0) corrected 11 of our 300 claims' labels and marked 8 more debatable.
Re-scoring the same stored answers offline ([`eval/results/rag_label_audit.md`](../eval/results/rag_label_audit.md)):

| Labels | n | Verdict accuracy |
|---|---|---|
| Original SciFact labels (headline) | 300 | 0.8000 |
| Corrected (the 11 confirmed errors) | 300 | 0.8367 |
| Corrected, 8 debatable claims excluded | 292 | 0.8425 |

**The headline stays on the original labels.** Caveats: the corrections come from a single
annotator, 8 of the 11 were first flagged by an LLM (so an LLM generator agreeing with them is
partly expected), and only the 188 evidence-bearing claims were audited, never the NEI ones. The
predictions are identical across rows, so this measures the labels, not the pipeline. To compare
two runs under the corrected key: `make rag-compare A=... B=... ARGS=--labels=audit`.

### Tried and measured, not adopted

All on the 300 test claims unless noted; paired exact McNemar.

| Experiment | Verdict accuracy | Result |
|---|---|---|
| Paid GPT-6 Luna as generator ($0.0985) | 0.66 vs 0.80 | worse (57 broken, 14 fixed, p < 0.0001); cites on every reply, but 27 replies have no parseable verdict and 56 SUPPORT/CONTRADICT claims get NEI |
| Off-the-shelf DeBERTa-v3 NLI verifier (tuned on 809 train claims) | 0.64 vs 0.80 | worse (70 broken, 21 fixed, p < 0.0001); weakest on CONTRADICT (0.56) |
| Fine-tuned PubMedBERT verifier (Kaggle; HealthVer + PubMedQA, then SciFact train) | alone 0.66; combined with Ling via rule R3 0.787 vs 0.777 (before the re-ask) | no gain (11 fixed, 8 broken, p = 0.65); R3 won on its 99 tuning claims (0.859 vs 0.828) and didn't transfer |
| Evidence-first prompt (quote the finding sentence first) | 0.847 vs 0.837 on 100 *train* claims | no gain (p = 1.0); only 64 of 98 replies followed the format |
| Disagreement-triggered second look (Ling vs the verifier) | 0.8033 vs 0.7767 (before the re-ask) | not significant alone (p = 0.057), adds nothing on top of the re-ask (0.8067 vs 0.80, p = 0.75), and costs citations (0.94 → 0.92) |

Verdict accuracy and abstention are judge-free and comparable across all runs, including the
oldest Groq/Qwen-judged gpt-oss run (0.70 → 0.78 on 50 claims after a larger token budget and a
`Verdict: REFUTED[1]` parser fix); its faithfulness and context-relevance numbers are **not**
comparable to the Nemotron-judged runs.

### Judge reliability

[`eval/results/judge_agreement.md`](../eval/results/judge_agreement.md) re-judges 50 stored
answers 3 times at each of two temperatures, with the same claim, context, answer, prompt and
model as the published judgement. It was measured on the **previous gpt-oss-120b run's answers**
(source run 133e9c7, 50 claims), **not** on the 300-claim Ling run above. It characterizes the
judge, and the judge (model and prompt) is unchanged, but those answers included the 28
bare-verdict replies, so the study has not yet been repeated on the longer, cited Ling
explanations.

| Agreement across 3 repeats | T = 0.0 (production) | T = 0.7 |
|---|---|---|
| `answered`: Fleiss' kappa | 1.00 | 0.84 |
| Faithfulness: Krippendorff's alpha | 0.89 | 0.51 |
| Context relevance: Krippendorff's alpha | 0.99 | 0.93 |
| Repeat identical to the original (all 3 fields) | 0.89 | 0.59 |
| Published `answered` values flipped | 0 | 0 |

At the production temperature of 0.0 the judge is close to deterministic: its `answered` call
never changed (kappa 1.00), and faithfulness (alpha 0.89) and context relevance (alpha 0.99)
barely moved. At T = 0.7 the `answered` call holds up (kappa 0.84) but faithfulness drops to
alpha 0.51, so faithfulness is the least stable score the judge produces.

(An earlier version of this eval reported "answered 0.50" from the first 10 query ids in dataset
order: a head slice, not a sample. The harness now takes a seeded random sample and re-runs at any size via `make eval-rag`.)

## Web search (Semantic Scholar)

An optional, networked retrieval source, **off the default path**: `hybrid` over the local
index stays the default and the measured headline. Two extra modes on `/search`, `/answer` and
the UI's mode selector (`GET /search?q=...&mode=web` or `mode=hybrid_web`):

- `web`: [Semantic Scholar](https://www.semanticscholar.org/product/api) paper search alone
  (`GET /graph/v1/paper/search`, ≤ 100 results per request).
- `hybrid_web`: the local hybrid list and the S2 list fused with the same hand-rolled RRF.
  SciFact doc ids are S2 corpus ids, so a paper in both appears once (local text, S2 URL and
  year). If S2 fails, `hybrid_web` returns local results plus a `warnings` entry; `web` has
  nothing to fall back to and returns 503.

Both modes now run a **deterministic, LLM-free pipeline** by default
([`app/retrieve/web_search.py`](../app/retrieve/web_search.py)): the claim is rewritten into a
≤ 6-term keyword query (plus a 3-term fallback query, results pooled) because S2 search handles a
whole sentence badly, then the pooled S2 candidates are re-ranked locally by RRF of bge-small
similarity to the original claim and BM25 over the candidates. Switch either step off with
`SSR_S2_QUERY_REWRITE=false` / `SSR_S2_RERANK=false`.

**Measured** (`make web-eval`, 300 test claims, gold-paper retrieval; rewrite rules chosen on 100
train claims and frozen before the test run;
[`eval/results/web_retrieval.md`](../eval/results/web_retrieval.md)):

| Config | Recall@5 | Recall@100 | nDCG@10 |
|---|---|---|---|
| S2, raw claim as query (the earlier `web`) | 0.040 | 0.056 | 0.035 |
| S2, raw claim + local rerank | 0.040 | 0.056 | 0.040 |
| S2, keyword rewrite | 0.102 | 0.236 | 0.089 |
| **S2, keyword rewrite + local rerank (now the default)** | **0.185** | **0.244** | **0.159** |
| Local hybrid (RRF, 5,183 docs), for scale | 0.766 | 0.965 | 0.724 |

Rewrite + rerank vs the raw claim: 45 claims better, 1 worse at Recall@5 (p < 0.0001), and every
rewrite comparison is significant. Reranking alone changes nothing (Recall@5 p = 1.0): the
rewrite is what gives it candidates worth reordering. The rerank stays within S2's own page, so
it can lift Recall@5/10 and nDCG but barely Recall@100. Web search still trails local hybrid on
every metric (p < 0.0001, e.g. Recall@5 0.185 vs 0.766), which is expected: S2 searches ~200M
papers against our 5,183, so this asks whether open-web retrieval can find the gold paper at all,
not which ranker is better. The id mapping holds (256 of 283 gold titles match S2 corpus ids; the
eval stops if under 90%). S2's index also changes, so responses are cached with their fetch date.

Web hits carry `source: "semantic_scholar"`, `url` (http/https only, checked server- and
client-side) and `year`. Titles and abstracts are untrusted third-party text, flattened to one
line with quote runs and control characters removed so they cannot forge the grounded prompt's
structure.

**Key and limits.** No key is required, but unauthenticated requests share S2's public pool and
often get 429. A free key ([request form](https://www.semanticscholar.org/product/api#api-key-form))
goes in `.env` as `SSR_S2_API_KEY` (sent as `x-api-key` only when set; 1 request/s). A
process-wide limiter spaces every request, retries included, to `SSR_S2_RATE_PER_S` (default 1);
API requests wait at most `SSR_S2_MAX_WAIT_S` (5 s), then degrade. Raw responses cache under
`data/s2_cache/`: always for the eval, and for the API only with `SSR_S2_API_CACHE=true`.
`make web-eval` makes ~300 requests (≈ 5 min at 1 req/s; `SSR_EVAL_LIMIT=2` is a 3-request
smoke run).

## Limitations

- **300 queries bound what can be detected.** For hybrid vs dense, the minimum detectable effect
  at 80% power (α = 0.05, paired) is ≈ 0.028 nDCG@10 and ≈ 0.031 Recall@100. The nulls above
  rule out large effects, not small ones.
- **Recall@100 has a ceiling fusion cannot move.** 8 of the 11 gold docs hybrid misses are in
  neither retriever's top-100, so no fusion rule over these two candidate pools could recover
  them ([`eval/results/analysis.md`](../eval/results/analysis.md), §3).
- **NEI labelling.** For the 112 NEI claims, BEIR's qrels mark the cited abstract relevant even
  though annotators found no rationale in it, and the whole hybrid-vs-dense recall gain sits on
  that stratum (Finding 1).
- **300 claims still leave a ~5-point interval on RAG accuracy.** The eval covers every SciFact
  test claim, and the 95% Wilson interval on verdict accuracy is 0.751–0.841 (0.80 ± 0.045), so a
  gap of a few points between two runs' headline rates means nothing on its own; compare runs
  claim by claim with `make rag-compare`. The re-ask gain (+7 claims, p = 0.016) is real but
  small, and rests on 15 re-asked claims.
- **Paired comparisons with the gpt-oss run cover only 50 claims**, scored with the old strict
  parser, so only large effects (the citation gap) can reach significance.
- **The label audit is a sensitivity check, not a new key.** One annotator, mostly LLM-flagged, no
  NEI claims audited; the 0.8367 corrected figure is an upper-side reading.
- **Remaining errors are evidence-judgement errors.** 30 of 112 NEI claims are answered anyway
  (over-eager) and 17 of 124 SUPPORT claims are abstained on (too literal); format and parsing
  failures are now ≤ 1 of 300.
- **Web search is far below local retrieval on SciFact.** Even with rewrite + rerank, Recall@5 is
  0.185 against 0.766, and the S2 results move with S2's index; it is an optional source, not a
  replacement.
- **One judge model.** Faithfulness and context relevance come from a single judge (Nemotron 3
  Ultra), with no second model or human labels to check it against. It is from a different model
  family than the generator to avoid self-evaluation bias, but a single LLM judge still carries
  its own biases. Its repeat consistency is measured ([above](#judge-reliability)), not its
  correctness.
- **The judge change breaks comparability with the oldest run.** Faithfulness and context
  relevance from the Qwen3.8-judged Groq run can't be compared with the Nemotron-judged runs.
  The gpt-oss and Ling runs share the Nemotron judge and are comparable. Verdict accuracy and
  abstention don't depend on the judge.
- **Free models can disappear.** Both default models are OpenRouter `:free` variants, served by
  whichever upstream providers carry them, and availability changes over time: the previous
  judge's free variant (`qwen/qwen3.8-27b:free`) became unavailable (0 of 4 calls), which is why
  the judge is now Nemotron. Nemotron or Ling can go the same way, forcing another model change and
  a re-run (the paid gpt-oss generator remains as a fallback).

## What I'd do next

- **Attack the two remaining error modes directly.** Over-eager answers on NEI claims (30 of 112)
  and too-literal abstentions on SUPPORT claims (17 of 124) are what is left. A stricter prompt
  (evidence-first), an off-the-shelf NLI verifier, a fine-tuned verifier and a second-look check
  have all been measured and not adopted ([above](#tried-and-measured-not-adopted)), so the next
  lever is better supervision, such as SciFact-train hard negatives for NEI, or a stronger
  generator that keeps Ling's citation behavior (paid GPT-6 Luna did not).
- **Close the web-search gap** (Recall@5 0.185 vs 0.766 local): the rewrite is rule-based and was
  tuned on 100 claims; a learned or LLM query rewrite, or scoring candidates against more of S2's
  graph, is untested.
- **Re-run the judge-consistency study on the 300 Ling answers**, and human spot-check faithfulness
  on the rows where judge repeats disagree (faithfulness alpha 0.51 at T = 0.7).
- **Stratified reporting as the default**: report every retrieval comparison by claim label, not
  only as a follow-up analysis.

## Commands and API

```bash
uv sync                                   # Python 3.12 env + deps          (make install)
uv run python -m app.ingest.build_index   # build dense (Qdrant) + BM25     (make index, one-time)
uv run uvicorn app.api.main:app --reload  # UI + API at localhost:8000      (make api)
uv run python -m app.eval.retrieval_eval  # reproduce the metrics table     (make eval)
uv run python -m app.eval.analysis        # label-stratified re-score       (make analysis)
uv run python -m app.eval.latency         # per-query retrieval latency     (make latency)
uv run pytest                             # tests                           (make test)

cp .env.example .env                      # only for /answer + `make eval-rag`: add an OpenRouter key
```

First run downloads the SciFact corpus (~9 MB via `ir_datasets`) and the embedding model
(~130 MB from HuggingFace); indexing takes a few minutes. Search and `make eval` need no API
key. Only the **grounded-answer** endpoint does, and without one `/answer` returns a clear
503 rather than failing obscurely. `make analysis` re-scores the cached `make eval` runs, so run
`make eval` first. `make lint` runs ruff, and `make ci` is what CI runs: a locked install, then
lint + test.

Open <http://localhost:8000> for the search UI, or query the API directly:
`GET /search?q=...&mode=hybrid&top_k=8` with modes `bm25`, `dense`, `hybrid`, `hybrid_rerank`,
plus the optional networked `web` and `hybrid_web` ([Web search](#web-search-semantic-scholar)).
`GET /answer?q=...&top_k=5` returns the grounded answer, its verdict and citations.

The RAG and web targets (`make eval-rag`, `make rag-compare`, `make rag-audit`, `make web-eval`)
and the verifier experiments (`make verify-eval`, `verify-export`, `verify-train`,
`verify-combine`) are listed with one-line descriptions in the [`Makefile`](../Makefile).

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
