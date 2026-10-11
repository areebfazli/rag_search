# Project details

The technical companion to the [README](../README.md): how retrieval and answering work, how
they are measured, where the remaining errors come from, what was tried and not shipped,
operations and limitations. Every number comes from a committed file in
[`eval/results/`](../eval/results/) unless it is marked as coming from an archived experiment.

Experiments that were measured and removed from `main` are preserved in the git tag
[`experiments-archive`][archive] (code, `make` targets and their result files).

Contents: [Retrieval](#retrieval) · [Answers](#answers) ·
[Evaluation methodology](#evaluation-methodology) ·
[Diagnosis](#diagnosis-where-the-remaining-errors-come-from) ·
[Tried and not shipped](#tried-and-not-shipped) · [Operations](#operations) ·
[Limitations](#limitations) · [What I'd do next](#what-id-do-next) · [Code layout](#code-layout)

[archive]: https://github.com/areebfazli/rag_search/tree/experiments-archive

## Retrieval

### Method

```
query ─┬─► BM25 (bm25s)      top-100 ─┐
       └─► dense (Qdrant)    top-100 ─┴─► RRF fusion ─► top-k ──► search results / answer context

optional 2nd stage (mode=hybrid_rerank): cross-encoder rerank of the top-32 fused candidates.
Measured below; on this corpus it does not improve ranking, so it is off by default.
```

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5`; its query prefix goes on queries only, never on documents |
| Lexical | `bm25s` |
| Fusion | Reciprocal Rank Fusion, k = 60, hand-rolled (no LangChain / LlamaIndex) |
| Vector store | Qdrant, embedded local mode (on disk under `data/qdrant`) |
| Reranker (not default) | `cross-encoder/ms-marco-MiniLM-L-6-v2`; `BAAI/bge-reranker-base` also evaluated (`SSR_RERANKER_MODEL`) |
| Retrieval eval | `ranx` on BEIR/SciFact gold qrels, paired significance tests |

`SearchService.retrieve(query, mode, top_k)` is the single entry point, shared by the API and
every eval harness, so both run identical logic. Modes: `bm25`, `dense`, `hybrid` (default) and
`hybrid_rerank`. The UI offers the first three.

BM25 returns only documents that share a term with the query: bm25s pads its top-k with score-0
documents in index order, and those are dropped, so a query with no known token (e.g. "5-HT2A",
or only stopwords) gets no BM25 hits instead of k arbitrary ones that fusion would then rank.
Local retrieval metrics are unchanged on all 300 test claims.

### Results

BEIR/SciFact, 300 test queries, gold qrels ([`retrieval.md`](../eval/results/retrieval.md)).
Bold is best per column.

| Config | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | MAP@100 |
|---|---|---|---|---|---|
| BM25 | 0.6863 | 0.8187 | 0.9127 | 0.6492 | 0.6439 |
| Dense (bge-small) | 0.7127 | 0.8362 | 0.9417 | 0.6822 | 0.6736 |
| **Hybrid (RRF)**, default | 0.7241 | **0.8554** | **0.9650** | 0.6886 | 0.6816 |
| Hybrid + rerank (MS-MARCO MiniLM) | 0.6975 | 0.8322 | **0.9650** | 0.6632 | 0.6558 |
| Hybrid + rerank (bge-reranker-base) | **0.7242** | 0.8494 | **0.9650** | **0.6901** | **0.6834** |

The reranked slice is the top 32 of 100 fused candidates; the tail keeps its fused order, so
Recall@100 is unchanged by reranking.

Every comparison below is a **paired two-sided t-test on per-query nDCG@10** (300 pairs),
reported as Δ, p and per-query win/tie/loss; the table is in
[`retrieval.md`](../eval/results/retrieval.md#significance) and the label-stratified re-score in
[`analysis.md`](../eval/results/analysis.md). Six tests are reported (the five in that table plus
Recall@100 in Finding 1), none corrected for multiple comparisons. Bonferroni at α = 0.05 would
set the bar at p ≈ 0.008, which only hybrid vs BM25 clears (its nDCG@10 gain is spread over 98
differing queries, 72 better and 26 worse; an exact sign test agrees, p < 0.0001), so treat the
two results at p ≈ 0.035–0.038 as suggestive. The load-bearing conclusions are the *null* ones,
which correction only strengthens; p = 0.996 is not a near-miss.

### Finding 1: fusion's case is recall, not ranking, and it is only suggestive

Hybrid RRF beats BM25 by +0.038 nDCG@10 (p = 0.0007), but its +0.011 edge over *dense alone* is
not significant (p = 0.26, W/T/L 53/213/34). Where fusion separates from dense is **Recall@100:
0.942 → 0.965 (W/T/L 9/289/2)**. The paired t-test gives p = 0.035, but per-query recall is
almost binary and only 11 queries differ, so an exact sign test (9 vs 2) is the fairer test, and
it gives **p = 0.065: suggestive, not significant**. That possible gain in the candidate pool,
not top-10 ordering, is the reason to keep fusion; it is a judgement call on this evidence, not a
settled result.[^rrf]

Stratified by SciFact's claim labels ([`analysis.md`](../eval/results/analysis.md)), that recall
gain is **entirely on NEI claims**, the 112 of 300 where annotators found no rationale in any
abstract but BEIR's qrels still mark the cited one relevant. All 11 discordant queries are NEI:
hybrid gains +0.0625 Recall@100 there (t-test p = 0.034, sign test p = 0.065, W/T/L 9/101/2;
`analysis.md` reports both tests and marks Recall@100 significance by the sign test). On the 188
evidence-bearing (SUPPORT/CONTRADICT) claims, hybrid and dense are identical at Recall@100 on
every query (0.9947 against rationale docs), and nDCG@10 differs by +0.0003 (p = 0.977). So the
fuller pool is fuller in abstracts that carry no evidence: fusion may widen the candidate pool,
but it does not retrieve more supporting or refuting evidence than dense alone on this corpus.
Against BM25, hybrid's Recall@100 gain is significant over all claims (18/281/1, sign test
p = 0.0001) but not on the evidence-bearing ones alone (5/183/0, p = 0.0625).

[^rrf]: RRF is insensitive to its k on this data: for k ∈ {1, 2, 5, 10, 20, 100}, nDCG@10 moves
by at most 0.0046 against the default k = 60 (all p ≥ 0.19), and Recall@100 stays at 0.965 on
every query. No fusion of these two pools could raise it much: only 3 of the 11 misses are in
either retriever's top-100, a ceiling of 0.975. Weighting dense at 0.3 or 0.7 instead of equally
gains no significant nDCG@10 (-0.0096, p = 0.18; +0.0034, p = 0.50) and costs recall: 0.928 at
0.3 dense (1 better / 13 worse, sign test p = 0.002), 0.952 at 0.7 (2 better / 6 worse, sign
test p = 0.29). The numbers are an offline replay of the cached top-100 lists, verified to
reproduce the committed hybrid run exactly, in §4 of
[`analysis.md`](../eval/results/analysis.md#4-rrf-sensitivity-offline-replay).

### Finding 2: reranking does not pay off here

The CPU-default MS-MARCO MiniLM, trained on short web queries, *costs* 0.027 nDCG@10 against
plain hybrid (p = 0.056, W/T/L 45/192/63). The domain-appropriate `bge-reranker-base` repairs
exactly that damage, beating MiniLM by the same 0.027 (p = 0.038), and then lands on top of doing
nothing at all: **Δ = +0.0001, p = 0.996**. So the reranker *choice* matters and reranking itself
doesn't: a mismatched cross-encoder degrades ranking, and the right one returns you to where you
started, expensively. An earlier version of this README claimed reranking "only pays off with a
domain-appropriate model such as bge-reranker" before that was run; measured, it is wrong.

Bounding the reranked slice to 32 (a latency and DoS guard, see [Operations](#operations)) also
caps the damage a bad reranker can do: MiniLM scored 0.6715 reordering all 100 fused candidates
versus 0.6975 over 32.[^depth]

[^depth]: A point estimate from the previous committed run at full depth (same model, corpus and
fusion, with the BM25/dense/hybrid rows bit-identical), not a row in the current table. Reproduce
with `SSR_RERANK_CANDIDATES=100 uv run python -m app.eval.retrieval_eval`.

### Latency

Per-query mean of `SearchService.retrieve` after 3 warm-ups, Intel i5-10210U laptop CPU, torch
pinned to 4 threads ([`latency.md`](../eval/results/latency.md)): BM25 0.001 s, dense 0.124 s,
**hybrid 0.124 s** (all 300 queries); reranking a 32-candidate slice costs **3.94 s (MiniLM)**
and **23.4 s (bge)** over a seeded 40-query sample. That is ≈188× hybrid's latency for a
statistical tie (MiniLM ≈32×). Earlier figures here were console readings from the eval sweep and
ran higher. Hybrid RRF is the default; reranking stays available as `mode=hybrid_rerank`.

## Answers

### How an answer is made

`GET /answer?q=...` runs hybrid retrieval and passes the top 5 abstracts to an OpenAI-compatible
LLM with a grounded prompt (retrieved text is sanitized so it cannot forge the prompt's
structure). Answers cite sources as `[n]`, mapped back to document ids, and the model is told to
**abstain when the context lacks the evidence** rather than guess. For a claim it ends with
`Verdict: SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE`.

- **Generator:** Mistral's `ministral-14b-latest` on Mistral's free tier (provider `mistral`,
  key `SSR_MISTRAL_API_KEY`, 30 requests/min), a different family from the Nvidia judge. If a
  call fails with an availability error (429 after retries, 5xx, 403/404, timeout, empty reply),
  the same request is tried once on `ministral-8b-latest`; the answer records which model replied
  (`model_used`, also returned by `/answer`). It is not a reasoning model. The completion budget
  is 2,048 tokens with **one retry at 2×** (4,096) on `finish_reason=length`; in the canonical
  run no answer needed the retry.
- **Earlier generator (history).** Until October 2026 the generator was Ling 3.0 Flash Sante
  (`inclusionai/ling-3.0-flash-sante:free`) on OpenRouter, a free health/medicine-tuned
  reasoning model picked on a 2026-09-24 bench of 10 free models. It was withdrawn upstream
  (requests return 404), so its canonical run (0.797, git `348c919`) can no longer be
  regenerated; see [the generator change](#the-generator-change-ling-to-mistral). The 2,048-token
  budget and its retry were sized for Ling's hidden reasoning: at 1,024 it needed the retry on 36
  of 50 eval claims and still truncated 7; at 2,048, 12 of 50 retried and 1 truncated.
- **Verdict parser** (built on the Ling run). The final `Verdict:` line is parsed first; if it
  is missing, the parser recovers a verdict restated in prose (`inline`) or, for a complete
  reply, from the first sentence's stance (`stance`). A truncated reply never gets a stance
  verdict, since it was cut off mid-reply.
- **Verdict re-ask** (`SSR_LLM_REASK`, default on; adopted on the Ling run). When a claim's
  reply still has no parseable verdict (truncated, or prose without a verdict line), the
  generator makes one verdict-only call (8,192-token budget) and takes its verdict. The
  response's `verdict_source` says where the verdict came from (`line`, `inline`, `stance`,
  `reask`). The re-ask goes to the model that answered. A failed re-ask never fails `/answer`;
  it becomes a `warnings` entry.
- **Claim gate.** Only inputs that read as a claim get the re-ask or a stance verdict
  (`looks_like_claim`): a question, a keyword search ("BRCA1 breast cancer risk") or an
  instruction gets neither, though an explicit `Verdict:` line the model writes is still parsed.
  The heuristic accepts all 300 test and 809 train claims; a short claim with no final period can
  be missed and is then treated like a question.
- **Retries.** `/answer` gives each generation call one SDK retry (30 s timeout) and the re-ask
  none (120 s), so a stuck provider holds a worker for minutes at most; the eval harness, with no
  user waiting, allows 5 retries (re-ask 1). A failed retry keeps the earlier attempt's cost.

Example (from the earlier Ling generator), `/answer?q=Can aspirin reduce the risk of colorectal cancer?`:
> "Aspirin has been shown to reduce the risk of colorectal cancer [1][2][3] … a pooled analysis of four randomized trials showed a 34% reduction in 20-year colorectal cancer mortality [3]."

### Scoring

`app/eval/rag_eval.py` scores each answer's verdict and its answer/abstain decision **against
SciFact's labels** (no judge), and uses an LLM judge only for faithfulness and context relevance.

- **Verdicts** map SUPPORTED → SUPPORT, REFUTED → CONTRADICT, NOT ENOUGH EVIDENCE → NEI and are
  compared with the gold claim label. **No-verdict rule (commit 3e360cc):** a reply with no
  verdict counts as NEI only if it is complete and abstained; one cut off at the token budget, or
  empty, scores `NONE` (always wrong).
- **Abstention** is scored against a **rationale oracle**: the context "has evidence" when a
  document the annotators cited *with rationale sentences* is in the top 5. NEI claims have no
  rationale document, so abstaining on them is correct. `answered` comes from the verdict; the
  judge decides it only when no verdict was parsed (0 of 300 in the Mistral run, 1 on Ling).
- **Judge:** Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`), a different model family
  from the generator so it never grades its own output. On the same 10-model bench it was the only
  one with 22/22 calls served and 100% parseable verdicts; the previous judge's free variant
  (`qwen/qwen3.8-27b:free`) served 0 of 4. Faithfulness is averaged over answered rows the judge
  saw text for.

### Results

All 300 SciFact test claims (seed 13, so the first 50 are the earlier 50-claim sample), 0
skipped, top-5 context, `ministral-14b-latest` generator (all 300 answers from the primary
model, none from the fallback) and the free Nemotron judge: **a full run costs $0** (about 360
Mistral requests and 305 free OpenRouter requests). Full tables in
[`rag.md`](../eval/results/rag.md); the confidence interval is from
[`rag_label_audit.md`](../eval/results/rag_label_audit.md).

**Verdict accuracy is 0.7500 (225 of 300), 95% Wilson CI 0.698–0.796.**

| Gold label | n | Verdict accuracy | Answered (not abstained) |
|---|---|---|---|
| SUPPORT | 124 | 0.718 (89) | 94 |
| CONTRADICT | 64 | 0.813 (52) | 55 |
| NEI | 112 | 0.750 (84) | 28 |

| Metric | Score |
|---|---|
| Faithfulness (over answered, LLM judge; n = 177) | 0.99 (0.986) |
| Context relevance (all, LLM judge) | 0.69 |
| **3-class verdict accuracy** (vs gold label, no judge) | **0.75** (225/300) |
| Verdict parsed (incl. recovered and re-asked) | 300 of 300 |
| Truncated answers (hit the token budget after 1 retry) | 0 of 300 |
| Answers that needed the length retry | 0 of 300 |
| Replies citing at least one passage | 281 of 300 |
| Evidence retrieved (rationale doc in top 5) | 0.58 |
| Answered (model attempted an answer) | 0.59 |
| **Abstention precision** (abstained & no evidence) | **0.76** (0.7561) |
| Abstention recall (no evidence & abstained) | 0.74 |
| False abstention (had evidence, abstained anyway) | 0.17 |
| Answered without evidence, as a share of answers given (hallucination risk) | 0.18 |

|  | evidence retrieved (175) | no evidence (125) |
|---|---|---|
| **answered** (177) | 145 answered with evidence | 32 answered with nothing to go on |
| **abstained** (123) | 30 declined despite having the evidence | 93 correct |

Verdicts against the gold label (`NONE` = no verdict on a reply that answered, was cut off at the
token budget or was empty; always wrong):

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** (124) | 89 | 5 | 30 | 0 |
| **CONTRADICT** (64) | 3 | 52 | 9 | 0 |
| **NEI** (112) | 14 | 14 | 84 | 0 |

Verdict sources: `line` 175, `inline` 5, `stance` 59, `reask` 61. **The re-ask fired on 61 of
300 claims** (Ling: 16), every one returned a verdict, and 42 of the 61 are correct. None was a
truncation: ministral-14b often answers in prose and leaves out the final `Verdict:` line, which
the stance rule recovers for 59 claims and the re-ask for the rest. That costs one extra request
per affected claim and leans on the two recovery steps; getting the line on the first reply is
listed under [What I'd do next](#what-id-do-next), not fixed.

**The oracle decides whether abstention looks broken.** Scored the legacy way, against BEIR qrels
(which mark a cited abstract relevant for NEI claims too), the *same answers* give abstention
precision 0.41 and 73 "false" abstentions (rate 0.31); under the rationale oracle only 30 are
false and precision is 0.76. `rag.md` reports both.

### The generator change: Ling to Mistral

The free Ling 3.0 Flash Sante generator was withdrawn from OpenRouter in October 2026, so the
default generator moved to Mistral's free `ministral-14b-latest` (with `ministral-8b-latest` as
a fallback) and the canonical run was repeated with the pipeline otherwise unchanged: same
retrieval, prompt, parser, re-ask, judge and claims. The Ling run stays in git at `348c919`
(`eval/results/rag.json` there); it can no longer be regenerated. Paired exact McNemar on the
same 300 claims (`make rag-compare`; b = Ling right and Mistral wrong, c = the reverse):

| On the 300 claims | Ling (348c919) | Mistral ministral-14b | b | c | p |
|---|---|---|---|---|---|
| Verdict correct | 0.7967 | 0.7500 | 31 | 17 | 0.0595 |
| Abstention correct (rationale oracle) | 0.8100 | 0.7933 | | | 0.58 |
| Cites at least one passage | 0.9400 | 0.9367 | | | 1.0 |

The 4.7-point drop is not significant at 0.05. Where it moves: Mistral declines more claims
that have a stance (SUPPORT → NEI 30 against Ling's 16) and answers fewer NEI claims (28 against
31). Faithfulness (0.99 vs 0.98) and context relevance (0.69 vs 0.72) come from the same judge
and prompt, but over different answers. Mistral's run-to-run noise has not been measured.

### How the parser and re-ask were adopted (Ling run)

Everything in this subsection was measured on the earlier Ling generator. Two post-processing
steps on the same Ling answers, scored on the 300 test claims with paired exact McNemar via
`make rag-compare` (b = earlier run right and this one wrong, c = the reverse). Neither was
developed on train alone; see [Test-set reuse](#test-set-reuse). Both were carried over to
Mistral unchanged.

| Step | Verdict accuracy | b | c | p |
|---|---|---|---|---|
| Strict `Verdict:` line only | 0.73 | | | |
| + verdict-recovery parser (verdict restated in prose, or a stance rule; commits ec465ef, b3de17b) | 0.7767 | 0 | 14 | 0.0001 |
| **+ verdict re-ask** (one verdict-only call for a claim with no verdict; commits e367318, 3dcc1a3) | **0.8000** | **0** | **7** | **0.016** |
| Current scoring rules (commit 3e360cc) | **0.7967** | 1 | 0 | 1.00 |

The first three rows were measured as committed at the time. Commit 3e360cc then tightened the
two rules above (a truncated or empty reply with no verdict scores `NONE`, and a truncated reply
gets no first-sentence verdict). On the same stored answers that moved two claims, both now
wrong: claim 343 (gold SUPPORT, truncated, no verdict even after the re-ask) went from NEI to
`NONE`, and claim 514 (gold NEI, truncated) lost its first-sentence NEI verdict, was re-asked (the
only new LLM call when the run was regenerated) and answered SUPPORTED. Under the current rules
the answers without the re-ask score 0.76 (228/300), and the re-ask fixes 11 and breaks 0
(p = 0.001); the extra 4 are NEI claims whose replies were cut off before any text and used to
count as correct abstentions, so the scoring change, not the re-ask, accounts for the difference,
and **7 fixed / 0 broken (p = 0.016) is the figure the re-ask was adopted on**.

On the Ling run the re-ask fired on 16 of 300 claims (11 truncated, 5 prose replies without a
verdict line) and produced a verdict for 15 (12 correct). Abstention (+0.007, p = 0.63) and
citation rate (0.94, unchanged) didn't move significantly. Faithfulness stayed 0.98: the re-ask is
not re-judged, and the 3 claims answered only through the re-ask, after a first reply that hit the
token budget before writing any answer text, are left out because the judge saw no answer text.
Claim 514 is counted: its truncated first reply had text, which the judge scored 0.0. Verdict
sources over the 300 rows: `line` 257, `inline` 10, `stance` 17, `reask` 15, none 1 (claim 343:
`NONE` for the verdict, an abstention in that run's tables).

### Remaining failure modes

75 of 300 verdicts are wrong in the Mistral run, and all are an evidence *judgement* error, not a
format one (every claim got a verdict):

- **SUPPORT/CONTRADICT claims abstained on: 39** (30 SUPPORT, 9 CONTRADICT), the largest group.
  For 26 of the 30 SUPPORT claims (and 4 of the 9 CONTRADICT) the rationale doc is in the
  context, so the model had the passage and judged it not enough.
- **NEI claims answered anyway: 28 of 112** (14 SUPPORTED, 14 REFUTED). They account for most of
  the 32 answers given without evidence.
- Smaller: 5 SUPPORT claims answered REFUTED and 3 CONTRADICT answered SUPPORTED.

On the Ling run the two big groups were the other way round (31 NEI claims answered, 22
SUPPORT/CONTRADICT claims abstained on, out of 61 errors). The
[diagnosis](#diagnosis-where-the-remaining-errors-come-from) of whether these are really model
errors was done on Ling's 61.

## Evaluation methodology

### Retrieval harness

`make eval` (`app/eval/retrieval_eval.py`) scores every config on the 300 test queries with
`ranx` and runs the paired tests above. Each config's run is cached under `data/eval_cache/`
(gitignored), keyed by a signature of everything that affects the result, and checkpointed every
20 queries, because a cross-encoder sweep is hours on a laptop CPU. `SSR_EVAL_REFRESH=1` ignores
the cache, `SSR_EVAL_LIMIT=n` runs a smoke subset, and a corrupt checkpoint degrades to
recompute. `make analysis` re-scores the cached runs by claim label (loading BM25 only, never
Qdrant) and replays RRF at other settings; `make latency` times every config.

### Answer harness and the canonical artifact

`make eval-rag` samples claims with a seeded shuffle (`SSR_RAG_N`, default 50, or `all`), so
samples nest: the first 50 of the 300-claim run are exactly the 50-claim sample.
`SSR_RAG_DATASET=beir/scifact/train` (809 claims, disjoint from test) is for prompt development.

- **Only one kind of run writes the committed artifact.** `eval/results/rag.{md,json}` is
  written only by a run over the whole test split (`SSR_RAG_N=all`), with no `SSR_EVAL_LIMIT`,
  both roles on the code-default models, the re-ask on and no fallback answers. Every other run
  (the default 50 claims, any train or limited run, another model, re-ask off) writes to
  `data/eval_runs/rag_<dataset>_<n>_<prompt-hash8>[_<model-slug>][_noreask]/`. A canonical run
  that ends with a claim missing or a failed re-ask refuses to write (finished rows stay
  checkpointed, so a re-run redoes only those); non-canonical runs write and name the missing
  claims.
- **Checkpoint and resume.** Every completed row is saved at once under
  `data/eval_cache/rag/<signature>.json` (signature = dataset, n, seed, both endpoints, both
  prompt hashes, top-k, mode, token budget, reasoning and retrieval settings). A daily-cap stop is
  resumed by re-running the same command; skipped rows are retried.
- **Re-ask reply cache.** Re-ask replies are cached in `data/eval_cache/secondlook/`
  (`app/eval/reply_cache.py`), keyed by the exact request including provider, base URL and model,
  so a reply is never reused across endpoints. The older v1 keys, which only served the
  withdrawn Ling endpoint, are no longer read.
- **Fallback answers.** Each row records `model_used`. A row answered by the fallback model is
  never checkpointed, and a canonical run refuses to write while any remain, so the committed
  artifact is all primary-model answers.
- **Free-tier budget.** With the default providers a 300-claim run is about 360 Mistral requests
  (300 answers plus the re-asks; free tier 30/min) and about 305 free OpenRouter judge requests,
  at a 5 s per-claim throttle (`SSR_RAG_THROTTLE_S` overrides). The judge is the binding limit:
  OpenRouter's free tier allows 20 requests/min and 1,000/day only on accounts that have ever
  bought $10 of credits (50/day otherwise). The run prints its request counts up front, and
  `SSR_RAG_CHECK_QUOTA=1` (or `--check-quota`) reads the OpenRouter key's remaining free
  requests (not an LLM call) and aborts if they are short.
- **Provenance.** `rag.json` keeps every row (gold label, rationale docs, verdict and its source,
  answer text, citations, finish reason, token usage, cost) and the run's prompt hashes, git SHA,
  models and seed, so each published number traces back to the exact answers.

### Comparing runs

`make rag-compare A=… B=…` runs paired exact McNemar tests on verdict, abstention and citation
between two `rag.json` files and lists which settings differ (`ARGS=--labels=audit` compares them
under the audited labels). `python -m app.eval.rag_rescore IN OUT` re-reads a finished `rag.json`
with the current verdict parser, with no LLM calls, and refuses to write into `eval/results/`.

### Test-set reuse

The 300 test claims (SciFact's public dev split) are no longer a clean held-out set. The
verdict-recovery parser's patterns came from an error analysis of *test* failures and were then
refined and checked on train. The re-ask was adopted after three candidate fixes were scored on
test (on train it fixed 0 and broke 1). In all, 8 variants have been scored on the same 300 claims
(paid GPT-6 Luna, the NLI verifier, the parser, the fine-tuned verifier alone and combined, the
second look, the re-ask, and second look + re-ask), so raw p-values are optimistic. The parser's
gain (14 fixed, 0 broken, p = 1.2e-4) survives any correction. The re-ask's (7 fixed, 0 broken,
raw p = 0.016) becomes borderline: Holm-adjusted p = 0.047 over the 3 fixes tried together, 0.063
over the 4 or 5 post-parser candidates, and 0.078 over all 8. All of this was on the Ling
generator.

The Mistral run is a single run of that existing pipeline: the parser, re-ask and prompt were
carried over unchanged, and nothing was tuned on its answers. The test set was still used to
build the pipeline it runs, so 0.750 is not a clean held-out number either.

The generator is also not deterministic: same-model Ling-vs-Ling repeats changed 9 of 50
predicted labels, flipping correctness on 7 (3 one way, 4 the other; p = 1.00). That was the
noise floor for 50 claims on Ling; Mistral's has not been measured, and the Wilson interval on
one run does not capture it. A fresh held-out set would be needed to confirm either score.

## Diagnosis: where the remaining errors come from

Four checks ask whether the test errors are the model's, the retrieval's, or the benchmark's.
**Checks 2–4 and the conclusion were measured on the earlier Ling run's 61 errors** (git
`348c919`) and were not repeated for Mistral; check 1 is re-scored on the current run. All are
**secondary**: **the headline stays 0.750 (225/300) on the original SciFact labels.** The two
blind re-labellings and the cited-abstract diagnostic used tooling that has since been removed
from `main` (archived in [`experiments-archive`][archive]); their labels and runs are local
(`data/` is gitignored), not committed.

### 1. Published label audit

SciFact's labels contain errors. Sylvestre, *Gold Label Errors in the SciFact Benchmark: An
LLM-Assisted Annotation Audit* ([BioNLP 2026](https://aclanthology.org/2026.bionlp-1.9/); data
[Kefez/scifact-audit-bionlp2026](https://github.com/Kefez/scifact-audit-bionlp2026) at pinned
commit `5711f3f`, CC BY 4.0) corrected 11 of our 300 claims' labels and marked 8 more debatable.
Re-scoring the same stored answers offline (`make rag-audit`,
[`rag_label_audit.md`](../eval/results/rag_label_audit.md) for the Mistral run; the Ling column
is the same re-score of the Ling run as committed at `348c919`):

| Labels | n | Mistral (current) | Ling (348c919) |
|---|---|---|---|
| Original SciFact labels (headline) | 300 | 0.7500 | 0.7967 |
| Corrected (the 11 confirmed errors) | 300 | 0.7667 | 0.8300 |
| Corrected, 8 debatable claims excluded | 292 | 0.7740 | 0.8356 |

Read the corrected rows as a **one-sided sensitivity check**, not an upper bound or a second
headline. Only the 188 evidence-bearing claims were audited, never the 112 NEI ones, so label
errors there, which could move the score either way, were never looked for. On the Mistral run,
of the 11 corrected claims 6 go wrong → right, 1 right → wrong (claim 770, SUPPORT → NEI, which
Mistral called SUPPORTED) and 4 stay wrong; on Ling it was 10 wrong → right and none right →
wrong. The corrections come from a single annotator, and 8 of the 11 were first flagged by an
LLM, so an LLM generator agreeing with them is partly expected. The predictions are identical
across rows, so this measures the labels, not the pipeline.

### 2. Blind re-labelling of the NEI disagreements (LLM annotators, Ling run)

The audit never looked at NEI claims, which is where most of Ling's errors were: 31 NEI-gold
claims got SUPPORT (18) or CONTRADICT (13). SciFact's NEI means the one *cited* abstract has no
rationale, while the system searches all 5,183 abstracts, so another passage may decide the claim.
To test that, the 31 disagreements were labelled blind against exactly the passages the generator
saw.

- **Packet.** 81 items from `eval/results/rag.json` (git_sha `aa98231`): the 31 disagreements plus
  50 controls the model got right (30 NEI, 10 SUPPORT, 10 CONTRADICT), shuffled with a fixed seed.
- **Annotators.** Three independent LLM agents (2 Claude Opus, 1 Claude Sonnet), each shown only
  the claim and the 5 passages, with no model verdict or answer, gold label or claim id, and the
  same written guidelines. Majority vote. Unanimous on 71 of 81 items, no 3-way splits, Fleiss
  κ = 0.87.
- **Disagreements.** The majority sides with the model on 25 of 31 (15 SUPPORT, 10 CONTRADICT) and
  with gold NEI on 6 (claims 1175, 1213, 1344, 350, 384, 410). Per annotator: 28, 23 and 23 of 31.
  Of the 25, the SciFact-cited abstract was among the passages in 14 and not in 11.
- **Controls.** The majority matches SciFact gold on 49 of 50 (0.98; NEI controls 29/30,
  SUPPORT/CONTRADICT 20/20); per annotator 48, 49 and 48 of 50.

| Labels | n | Verdict accuracy (95% Wilson CI) |
|---|---|---|
| Original SciFact labels (Ling run) | 300 | 0.7967 (239/300) |
| Annotator labels on the 31 disagreements only (one-sided, can only rise) | 300 | 0.8800 (264/300; 0.838–0.912) |
| Annotator labels on every sampled item | 300 | 0.8767 (263/300; 0.835–0.909) |

The annotators are LLMs, not a human, so this is a weaker check, and they may share the
generator's reading of the evidence. The controls show they are not simply agreeing with the
model: they match gold wherever model and gold agree. The question asked is also not SciFact's:
"do these passages decide it?" (open corpus) against "does the cited abstract decide it?" (closed
corpus), the mismatch SciFact-Open (Wadden et al. 2022) documents. It does not show the verdicts
are correct, and it is not a corrected score. How much is the open-corpus mismatch was tested next
and is smaller than this check alone suggests.

### 3. Cited-abstract diagnostic (train, Ling)

An eval-only context mode gave the generator only SciFact's cited abstract(s) instead of the
retrieved top 5. Same seeded 100 `beir/scifact/train` claims, free Ling, judge skipped.

| Gold | Retrieved top 5 | Cited abstract only | Fixed / broken |
|---|---|---|---|
| All 100 | 0.81 | 0.84 | 7 / 4 (exact McNemar p = 0.55) |
| SUPPORT | 0.864 | 0.864 | 2 / 2 |
| CONTRADICT | 0.864 | 0.955 | 2 / 0 |
| NEI | 0.706 | 0.735 | 3 / 2 |

The +3 points is within run-to-run noise (Ling repeats flip about 7 of 50 verdicts), so the
retrieval ceiling here is roughly +3 points at most. The cited doc was already in the retrieved top
5 for 88 of 100 claims. Of the baseline's 19 errors, 3 had no cited doc in the top 5 (all NEI-gold,
called SUPPORT from other papers) and the cited abstract fixed all 3; of the 16 with it in the
top 5, 4 were fixed and 12 stayed wrong. Seven NEI-gold claims are called CONTRADICT even when
shown only the cited abstract.

### 4. Second blind re-labelling (test, errors on a stance claim, Ling run)

A second packet (`a08a3152bfcba3d5`) of 70 items: the 30 SUPPORT/CONTRADICT-gold errors plus 40
controls (20 NEI-gold predicted NEI, 10 SUPPORT and 10 CONTRADICT correct), excluding the first
packet's claims. Same three blind LLM annotators, majority vote: unanimous on 56 of 70, no 3-way
splits, Fleiss κ = 0.79.

| Error type (n) | Majority sides with gold (model judgement error) | With the model | Third label |
|---|---|---|---|
| SUPPORT → NEI (16) | 8 | 8 | 0 |
| CONTRADICT → NEI (6) | 3 | 3 | 0 |
| SUPPORT → CONTRADICT (6) | 0 | 6 | 0 |
| CONTRADICT → SUPPORT (1) | 0 | 1 | 0 |
| SUPPORT → no verdict (1) | 0 | 0 | 1 |
| **All 30** | **11** | **18** | **1** |

Per annotator, 11, 13 and 5 of the 30 side with gold. SciFact's rationale abstract was among the
passages for 25 of the 30; among the 18 sided with the model, 4 had it missing. Controls: the
majority matches gold on 36 of 40 (SUPPORT/CONTRADICT 20/20, NEI 16/20); the annotators called 4
NEI-gold controls CONTRADICT, so they lean toward finding a stance and the 18 may be high.
Secondary accuracy: 0.8567 (257/300) relabelling these errors only, 0.8433 (253/300) over all
sampled items.

### Conclusion (Ling run)

On the Ling run, the remaining errors were mostly not retrieval (a ceiling of about +3 points on
train; the cited doc in the top 5 for 88%), not code or parsing, and not mainly the open-corpus
mismatch: that is smaller than the earlier unblinded estimate (about 25 of 61) suggested. On train
only 3 errors came from it, and in the first blind check the cited abstract was shown in 14 of the
25 cases sided with the model. The main source is **where the line is drawn between NEI and a
stance**. The model, and the LLM annotators, read the same abstract more liberally than SciFact's
explicit-rationale-sentence convention on NEI claims (over-reading, mostly toward CONTRADICT), and
more cautiously on some implied SUPPORT (about 11 genuine too-cautious errors). Combined over
Ling's 61 test errors: about 43 where blind LLM annotators side with the model (25 + 18), about 17
model judgement errors (6 + 11), and 1 with no verdict.

Caveats: this is an estimate from LLM annotators, not a human; the diagnostic is on train and the
re-labellings on test, which are not the same claims; and the train deltas are within noise.

**The Mistral run, as an observation only.** Its 75 errors sit on the same NEI line: 67 are a
claim with a stance called NEI (39) or an NEI claim given a stance (28), and only 8 swap SUPPORT
and CONTRADICT. The biggest error cell is SUPPORT → NEI (30), so Mistral errs more on the
cautious side than Ling did. No re-labelling or diagnostic was run on these errors, so how many
are the model's and how many the benchmark's is not known.

**Implication.** Further gains on SciFact mean calibrating to its annotation convention (for
example few-shot examples of its NEI threshold). That is benchmark calibration, not a smarter
system. A few-shot prompt was prepared but never run (see below).

## Tried and not shipped

Everything here was measured and then removed from `main` (or, for reranking, kept only as a
non-default mode). **Code: tag [`experiments-archive`][archive]**, which also holds the removed
`make` targets and result files. All generator experiments here were run with the earlier free
Ling generator, not Mistral. Unless noted, results are on the 300 test claims with paired exact
McNemar. These were scored before commit 3e360cc's scoring rules and not re-run: "0.80" there is
the Ling canonical run as committed at dc1816b (240/300), which differs from the final Ling run
(348c919) only on claims 343 and 514, and "0.7767" is the same answers before the re-ask under the
old rules.

| Experiment | Verdict accuracy | Result |
|---|---|---|
| Cross-encoder reranking | n/a (retrieval) | No gain: bge ties hybrid (p = 0.996) at ≈188× the latency, MiniLM scores lower ([Finding 2](#finding-2-reranking-does-not-pay-off-here)). Kept as `mode=hybrid_rerank` in the API and eval, not in the UI |
| Paid GPT-6 Luna as generator ($0.0985), re-scored with the current parser | 0.71 vs 0.80 | worse (42 broken, 15 fixed, p = 0.0005; like for like, without the re-ask, 0.71 vs 0.7767: 39 broken, 19 fixed, p = 0.012); cites on every reply, but 60 SUPPORT/CONTRADICT claims get NEI |
| Paid gpt-oss-120b as generator, 50 claims ($0.0049) | 0.78 vs 0.76 | indistinguishable (p = 1.00; 3 discordant claims, so a difference is ruled neither in nor out); cites far less (table below) |
| Off-the-shelf DeBERTa-v3 NLI verifier (tuned on 809 train claims) | 0.64 vs 0.80 | worse (70 broken, 21 fixed, p < 0.0001); weakest on CONTRADICT (0.56) |
| Fine-tuned PubMedBERT verifier (Kaggle; HealthVer + PubMedQA, then SciFact train) | alone 0.66; combined with Ling via rule R3 0.787 vs 0.777 (before the re-ask) | no gain (11 fixed, 8 broken, p = 0.65); R3 won on its 99 tuning claims (0.859 vs 0.828) and didn't transfer |
| Disagreement-triggered second look (Ling vs the verifier) | 0.8033 vs 0.7767 (before the re-ask) | not significant alone (p = 0.057), adds nothing on top of the re-ask (0.8067 vs 0.80, p = 0.75), and costs citations (0.94 → 0.92) |
| Evidence-first prompt (quote the finding sentence first), 100 *train* claims | 0.847 vs 0.837 | no gain (p = 1.0); only 64 of 98 replies followed the format |
| Majority vote over k = 3 samples, 100 *train* claims | 0.82 vs 0.81 | no gain (2 fixed, 1 broken, p = 1.00); fails the pre-registered gate; 3.0× generation calls |
| "Finding" prompt variant, 100 *train* claims, judge skipped | 0.83 vs 0.81 | not adopted (5 fixed, 3 broken, p = 0.73); passes the gate exactly at the threshold, with the disclosures below |
| Few-shot NEI-threshold prompt | n/a | prepared, never run |
| Web search (Semantic Scholar + PubMed) | n/a (retrieval) | far below local retrieval ([below](#web-search-semantic-scholar-and-pubmed)) |

### Paid gpt-oss-120b, paired on 50 claims

Both runs predate the verdict-recovery parser, so they use the old strict parser and are not
comparable to the Ling run's 0.797. Accuracy is indistinguishable; citations are the one
significant difference:

| On the 50 shared claims | gpt-oss-120b (paid) | Ling | b | c | p |
|---|---|---|---|---|---|
| Verdict correct | 0.78 | 0.76 | 2 | 1 | 1.00 |
| Abstention correct (rationale oracle) | 0.84 | 0.78 | 3 | 0 | 0.25 |
| **Cites at least one passage** | **0.42** | **0.94** | **1** | **27** | **< 0.0001** |

On those claims 28 gpt-oss replies were a bare `Verdict:` line against 0 for Ling, and the run
cost $0.0049 against $0. The paid path ran under a per-model spend policy (pinned bf16 provider,
no fallbacks, price caps, a per-run spend ceiling); `main` now sends only free models to
OpenRouter. Verdict accuracy and abstention are judge-free and comparable across all runs,
including the oldest Groq/Qwen-judged gpt-oss run (0.70 → 0.78 on 50 claims after a larger token
budget and a `Verdict: REFUTED[1]` parser fix); its faithfulness and context-relevance numbers are
**not** comparable to the Nemotron-judged runs. The Groq-specific provider was generalised into
the `openai_compat` provider, which still works with Groq.

### Voting and the "finding" prompt (train, pre-registered)

Both ran on the seeded 100-claim `beir/scifact/train` sample (seed 13) with the free Ling
generator, and both were pre-registered before any LLM call: k = 3 samples at temperature 0.1,
each with its own verdict re-ask; plurality vote, ties going to the lowest-numbered sample. The
gate: accuracy must not drop, fixed minus broken must be at least 2, and for the prompt NEI net
must be at least -1. Paired exact McNemar against the single-sample baseline, which scored **0.81
(81/100)**. The test split was never run.

- **Majority vote, k = 3: 0.82, 2 fixed / 1 broken, p = 1.00, fails the gate.** 90 of 100 claims
  were unanimous, 10 split 2-1 and none split three ways; the individual samples scored
  0.81 / 0.80 / 0.81. Cost: 3.0× the generation calls with all samples drawn, about 2.2× with an
  early stop.
- **"Finding" prompt** (judge the claim by its main finding, not its exact wording, with a guard
  against over-reaching on NEI), single sample, judge skipped (a reply with no verdict counts as
  wrong): **0.83, 5 fixed / 3 broken, p = 0.73.** That passes the gate exactly at the threshold,
  and it is not adopted. Disclosures: the first scoring was 0.82 (net +1, a fail); a rate-limited
  re-ask (claim 1054) was then retried by the eval's normal resume, after that score had been
  seen, which tipped it to 0.83; 2 of the 5 fixes (claims 1114 and 9) were read while writing the
  wording, so the net is 0 without them; against same-day single samples it is only +1 to +2. The
  error category it targets came from the *test*-set analysis.

### Web search (Semantic Scholar and PubMed)

An optional networked retrieval source (`web` and `hybrid_web` modes) ran a deterministic,
LLM-free pipeline: a ≤ 6-term keyword rewrite of the claim sent to Semantic Scholar (S2), pooled
with S2 snippet search (dropping papers that reprint SciFact claims) and PubMed Best Match, then
re-ranked locally by RRF of bge-small similarity and BM25 over the candidates. Measured on the 300
test claims, rules chosen on 100 train claims and frozen
([`web_retrieval.md` at the tag](https://github.com/areebfazli/rag_search/blob/experiments-archive/eval/results/web_retrieval.md)):

| Config | Recall@5 | Recall@100 | nDCG@10 |
|---|---|---|---|
| S2, raw claim as query | 0.040 | 0.056 | 0.035 |
| S2, keyword rewrite + local rerank | 0.185 | 0.244 | 0.159 |
| **Pooled: S2 rewrite + S2 snippets + PubMed** | **0.228** | **0.384** | **0.203** |
| Local hybrid (5,183 docs), for scale | 0.766 | 0.965 | 0.724 |

Rewrite + rerank beat the raw claim (45 better, 1 worse at Recall@5, p < 0.0001); reranking alone
changed nothing (p = 1.0); pooling raised Recall@100 on 46 claims and lowered it on none
(p < 0.0001), Recall@5 0.185 → 0.228 (p = 0.012). Web search still trailed local hybrid on every
metric (p < 0.0001), which is expected: S2 searches ~200M papers against our 5,183. Scoring by
SciFact corpus id caps web Recall@100 at ≈ 0.90 (256 of 283 gold docs title-match). Caveats: the
rows predate a later rewrite fix (commit aa98231) that changes the S2 queries for 56 of the 300
claims and was never re-measured on test (on train: Recall@5 0.285 → 0.305, not significant); the
rows used the eval's settings, not the API's; a query took about 17 s with caches off (measured
informally); OpenAlex helped on 45 train claims (Recall@100 0.219 → 0.319, p = 0.027) but its
keyless quota was too small for a test run. Removed because it is far below local retrieval on
this benchmark, slow, and adds third-party dependencies and rate limits to an unauthenticated API.

### Judge consistency study

A harness re-judged 50 stored answers 3 times at each of two temperatures, with the same claim,
context, answer, prompt and model as the published judgement
([`judge_agreement.md` at the tag](https://github.com/areebfazli/rag_search/blob/experiments-archive/eval/results/judge_agreement.md)).
It was measured on the **earlier gpt-oss-120b run's answers** (source run 133e9c7, 50 claims),
not on the 300-claim Ling run; the judge model and prompt are unchanged, but those answers
included the 28 bare-verdict replies.

| Agreement across 3 repeats | T = 0.0 (production) | T = 0.7 |
|---|---|---|
| `answered`: Fleiss' kappa | 1.00 | 0.84 |
| Faithfulness: Krippendorff's alpha | 0.89 | 0.51 |
| Context relevance: Krippendorff's alpha | 0.99 | 0.93 |
| Repeat identical to the original (all 3 fields) | 0.89 | 0.59 |
| Published `answered` values flipped | 0 | 0 |

At the production temperature the judge is close to deterministic; at T = 0.7 faithfulness drops
to alpha 0.51, so faithfulness is the least stable score it produces. This measures repeat
consistency, not correctness.

## Operations

### API

- `GET /search?q=...&mode=hybrid&top_k=8`: modes `bm25`, `dense`, `hybrid`, `hybrid_rerank`.
- `GET /answer?q=...&top_k=5`: the grounded answer, its verdict, `verdict_source`, citations,
  hits, `model_used` (the model that answered: the primary or its fallback) and `warnings`.
  Without a key (`SSR_MISTRAL_API_KEY` by default) it returns a clear 503 naming the missing
  variable (never its value); a provider failure is a 502, not a 500.
- `GET /health`, and the UI at `/`.

First run downloads the SciFact corpus (~9 MB via `ir_datasets`) and the embedding model (~130 MB
from Hugging Face); indexing takes a few minutes. Search and `make eval` need no API key; `/answer` needs
`SSR_MISTRAL_API_KEY`, and `make eval-rag` also needs `SSR_OPENROUTER_API_KEY` for the judge.

### Guards and deliberate limits

- **The API is unauthenticated**, so per-IP rate limiting (slowapi; 30/min search, 10/min answer)
  is the only guard. `/answer` is free with the default models; with an `openai_compat` endpoint
  you pay whatever that provider charges, so set a limit on that key too.
- **Proxy trust.** `SSR_TRUST_PROXY=true` keys on the X-Forwarded-For entry
  `SSR_TRUSTED_PROXY_HOPS` in from the *right* (default 1), after joining repeated header lines.
  Set the hop count to exactly how many proxies you run, because the count *is* the trust
  boundary: only the rightmost `hops` entries were written by your own infrastructure. Too **low**
  keys on one of your proxies, putting every user behind it in one bucket. Too **high** indexes
  past your proxies into the part of the list the *client* supplied, letting a client choose and
  rotate its own key and bypass the limit. The default of 1 is safe and cannot be over-indexed.
- **Reranking is bounded to the top 32 fused candidates** (`SSR_RERANK_CANDIDATES`); the tail
  keeps its fused order. A cross-encoder is a full forward pass per candidate: reranking all 100
  takes ~20 s on CPU, and because retrieval is serialized behind a lock, an unbounded slice would
  let one client at the rate limit hold the service for minutes. It is a DoS guard first and a
  latency knob second.
- **Retrieval is serialized** by a process-wide lock (embedded Qdrant and the shared
  sentence-transformers models are not thread-safe), so each worker serves one search at a time.
  The LLM call runs outside the lock. Qdrant in server mode is the fix if that ever matters.
- **Embedded Qdrant locks to a single process.** Don't run the API and `make eval`/`make index` at
  the same time; the second fails to acquire the storage lock.
- **OpenRouter is free-only.** OpenRouter serves the judge by default (and the generator only if
  `SSR_LLM_PROVIDER=openrouter`). Every request to openrouter.ai passes a guard before it is
  sent: only `:free` model ids, `allow_fallbacks: false` and a $0 `max_price`, and no `models`,
  `route` or `plugins` fields (`app/core/llm_endpoints.py`).
- **Each key goes only to its own provider.** The provider setting decides URL, key and model:
  `mistral` always uses `https://api.mistral.ai/v1` with `SSR_MISTRAL_API_KEY`, `openrouter`
  always uses openrouter.ai with `SSR_OPENROUTER_API_KEY`, and `openai_compat` refuses an
  openrouter.ai or api.mistral.ai URL. API keys are `SecretStr` settings, masked in reprs and
  logs.
- **bm25s `load()` deserializes on-disk arrays**: only point it at an index this repo built.

### Settings

All settings are `SSR_`-prefixed environment variables (or `.env`), defined in
[`app/core/config.py`](../app/core/config.py); [`.env.example`](../.env.example) shows the common
ones.

| Setting | Default | Purpose |
|---|---|---|
| `SSR_MISTRAL_API_KEY` | empty | Key for the default generator (answers and the RAG eval) |
| `SSR_MISTRAL_LLM_MODEL` / `SSR_MISTRAL_LLM_FALLBACK_MODEL` | `ministral-14b-latest` / `ministral-8b-latest` | Generator and its one-shot fallback (`""` turns the fallback off) |
| `SSR_OPENROUTER_API_KEY` | empty | Key for the default judge (RAG eval only), or the generator with `SSR_LLM_PROVIDER=openrouter` |
| `SSR_LLM_PROVIDER` / `SSR_JUDGE_PROVIDER` | `mistral` / `openrouter` | Generator: `mistral`, `openrouter` or `openai_compat`; judge: `openrouter` or `openai_compat` |
| `SSR_LLM_BASE_URL` / `SSR_LLM_API_KEY` / `SSR_LLM_MODEL` / `SSR_JUDGE_MODEL` | Groq URL, empty, model ids | Used only by `openai_compat` |
| `SSR_LLM_MAX_COMPLETION_TOKENS` | 2048 | Generation budget (one retry at 2×) |
| `SSR_LLM_REASONING_EFFORT` | `auto` | `auto` sends nothing; any other value is passed through |
| `SSR_LLM_REASK` | `true` | Verdict-only re-ask; a run with it off is never canonical |
| `SSR_RERANKER_MODEL`, `SSR_RERANK_CANDIDATES`, `SSR_RERANK_BATCH_SIZE` | MiniLM, 32, 8 | `hybrid_rerank` only |
| `SSR_DENSE_TOP_K`, `SSR_RERANK_TOP_K`, `SSR_RRF_K` | 100, 8, 60 | Candidate depth, default results returned, RRF k |
| `SSR_TRUST_PROXY`, `SSR_TRUSTED_PROXY_HOPS` | `false`, 1 | Rate-limit keying behind a proxy (above) |
| `SSR_EVAL_REFRESH`, `SSR_EVAL_LIMIT` | unset | Eval: ignore caches / smoke subset |
| `SSR_RAG_N`, `SSR_RAG_DATASET`, `SSR_RAG_CHECK_QUOTA`, `SSR_RAG_THROTTLE_S` | 50, test split, off, provider-aware | Answer eval sample, split, quota check, pacing |

The model ids (`mistral_llm_model`, `openrouter_judge_model`) are code defaults; a run that
overrides either is never canonical. `openrouter_llm_model` still holds the withdrawn Ling id as
the record of the earlier run; set `SSR_OPENROUTER_LLM_MODEL` to a live `:free` model to use
OpenRouter as the generator.

### Docker

The `Dockerfile` packages the API only (no docker-compose: Qdrant runs embedded). Mount the
built index at runtime (`-v "$PWD/data:/app/data"`) or build it inside the container, and pass
`SSR_MISTRAL_API_KEY` with `-e` at run time (plus `SSR_OPENROUTER_API_KEY` only for the eval's
judge); never bake a key into the image. The base image is
digest-pinned and the install uses the exact `uv.lock`.

## Limitations

- **300 queries bound what can be detected.** For hybrid vs dense, the minimum detectable effect
  at 80% power (α = 0.05, paired) is ≈ 0.028 nDCG@10 and ≈ 0.031 Recall@100. The nulls above rule
  out large effects, not small ones.
- **Recall@100 has a ceiling fusion cannot move.** 8 of the 11 gold docs hybrid misses are in
  neither retriever's top-100 ([`analysis.md`](../eval/results/analysis.md), §3).
- **NEI labelling.** For the 112 NEI claims, BEIR's qrels mark the cited abstract relevant even
  though annotators found no rationale in it, and the whole hybrid-vs-dense recall gain (itself
  only suggestive: sign test p = 0.065) sits on that stratum.
- **300 claims still leave a ~5-point interval on RAG accuracy** (0.750, 95% CI 0.698–0.796),
  so a gap of a few points between two runs means nothing on its own; compare runs claim by claim
  with `make rag-compare`. The Ling → Mistral drop of 4.7 points is exactly that case (p = 0.06).
  The re-ask gain on Ling (+7 claims, raw p = 0.016) rests on 15 re-asked claims (16 under the
  current rules) and is borderline once the other variants are counted (Holm 0.047–0.078). The
  interval also ignores generator nondeterminism.
- **The test set has been reused** ([details](#test-set-reuse)): the parser and re-ask were
  chosen on these claims (on Ling), so 0.750 is not a clean held-out number.
- **Mistral leans on the re-ask.** ministral-14b leaves out the `Verdict:` line often enough that
  the re-ask fires on 61 of 300 claims (Ling: 16). It recovers every one, but that is an extra
  request per claim and a dependence on the recovery steps, not a fix.
- **The label checks are secondary.** The audit is one annotator, mostly LLM-flagged, with no NEI
  claims audited; the re-labellings use LLM annotators, not humans, and were run on Ling's
  errors, not Mistral's. Neither replaces the headline.
- **One judge model.** Faithfulness and context relevance come from a single judge with no human
  labels to check it against; its repeat consistency was measured, not its correctness.
  Faithfulness from the oldest Qwen-judged run is not comparable with the Nemotron-judged runs.
- **Free models can disappear.** It has already happened twice: the previous judge's free variant
  became unavailable (0 of 4 calls), and the Ling generator was withdrawn in October 2026, which
  forced the move to Mistral and a new canonical run. The judge is still an OpenRouter `:free`
  model and the generator depends on Mistral's free tier; if either goes, the model must change
  and the eval be re-run (any `openai_compat` endpoint can stand in).

## What I'd do next

- **Calibrate to the benchmark, and say so.** Most remaining errors sit on SciFact's NEI line; a
  few-shot prompt showing its threshold, screened on unused train claims first, is the obvious
  next step. It would be benchmark calibration, not a smarter system.
- **Get the verdict line on the first reply.** ministral-14b needs the re-ask on 61 of 300
  claims; a prompt or format change that fixes that should be screened on train claims first.
- **Confirm the score on fresh claims** never used for development, and repeat the run a few
  times to measure Mistral's run-to-run noise.
- **Re-run the error diagnosis on Mistral's errors**; the blind re-labellings were done on
  Ling's.
- **Human spot-checks**: of the LLM re-labels, and of faithfulness on rows where judge repeats
  disagreed.
- **Stratified reporting by default**: report every retrieval comparison by claim label, not
  only as a follow-up analysis.

## Code layout

```
app/core       config (pydantic-settings, SSR_ prefix) · interfaces (Retriever/Reranker/Generator Protocols) · llm_endpoints (provider mistral/openrouter/openai_compat -> URL/key/model + OpenRouter free-only guard) · paths
app/ingest     corpus (BEIR/SciFact via ir_datasets, + claim labels from the source zip) · build_index
app/index      embedder (bge-small) · vector_store (Qdrant embedded) · lexical (bm25s)
app/retrieve   dense · fusion (RRF) · service (SearchService, the single retrieval entry point)
app/rerank     cross_encoder (MS-MARCO MiniLM default; bge-reranker-base evaluated)
app/generate   generator (OpenAI-compatible client, ministral-8b fallback, verdict parser, re-ask) · prompts (grounded prompt + injection sanitizer)
app/eval       retrieval_eval · analysis · latency · rag_eval · reply_cache · rag_compare · rag_rescore · label_audit
app/schemas    api (Pydantic request/response models)
app/api        main (FastAPI: /search, /answer, rate-limited)
frontend/      index.html (vanilla JS UI, served at /)
```
