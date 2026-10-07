# Project details

The long-form companion to the [README](../README.md): methodology, significance tests,
per-label breakdowns, what was tried and not adopted, limitations, and operational notes. Every
number comes from the committed files in [`eval/results/`](../eval/results/).

Contents: [Retrieval results](#retrieval-results-at-a-glance) ·
[Pipeline](#pipeline) · [Stack](#stack) · [Code layout](#code-layout) ·
[Retrieval evaluation](#retrieval-evaluation) · [Grounded answers](#grounded-answers-rag) ·
[Label audit](#label-audit) · [NEI re-labelling](#blind-re-labelling-of-the-nei-disagreements-llm-annotators-secondary) ·
[Where the remaining errors come from](#where-the-remaining-verdict-errors-come-from-diagnosis) · [Tried and not adopted](#tried-and-measured-not-adopted) ·
[Test-set reuse](#how-much-to-trust-the-test-set) ·
[Judge reliability](#judge-reliability) · [Web search](#web-search-semantic-scholar-and-pubmed) ·
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

RAG answers (Ling 3.0 Flash Sante, free, 300 test claims): **verdict accuracy 0.80** (239/300 =
0.797, 95% CI 0.748–0.838), faithfulness 0.98 (LLM judge), abstention precision 0.83; see [Grounded answers](#grounded-answers-rag).

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
app/retrieve   dense · fusion (RRF) · service (SearchService, the single retrieval entry point) · semantic_scholar · query_rewrite · web_search · s2_extra (S2 snippets, id resolver) · pubmed · openalex (not wired in)
app/rerank     cross_encoder (MS-MARCO MiniLM, bge-reranker-base)
app/generate   generator (OpenAI-compatible client) · prompts (grounded prompt + injection sanitizer)
app/verify     NLI and fine-tuned verifiers + Kaggle training kit (measured, not adopted)
app/eval       retrieval_eval · rag_eval · rag_compare · label_audit · nei_relabel (blind re-labelling packet + scorer, offline) · web_eval · web_pool_eval · latency · analysis · judge_agreement · verify_eval/verify_combine
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
p ≈ 0.008, which only hybrid-vs-BM25 clears (and its nDCG@10 gain is spread over 98 differing
queries, 72 better and 26 worse; an exact sign test agrees, p < 0.0001), so treat the two results
at p ≈ 0.035–0.038 as suggestive. The load-bearing conclusions below are the *null* ones, which correction only
strengthens; p = 0.996 is not a near-miss.

**Findings.**

**1. Fusion's case is recall, not ranking, and it is only suggestive.** Hybrid RRF beats BM25 by
+0.038 nDCG@10 (p = 0.0007), but its +0.011 edge over *dense alone* is not significant (p = 0.26,
W/T/L 53/213/34). Where fusion separates from dense is **Recall@100: 0.942 → 0.965 (W/T/L
9/289/2)**. The paired t-test gives p = 0.035, but per-query recall here is almost binary and only
11 queries differ, so an exact sign test (9 vs 2) is the fairer test, and it gives **p = 0.065:
suggestive, not significant**. That possible gain in the candidate pool, not top-10 ordering, is
the reason to keep fusion; it is a judgement call on this evidence, not a settled result.[^rrf]

Stratified by SciFact's claim labels ([`eval/results/analysis.md`](../eval/results/analysis.md)),
that recall gain is **entirely on NEI claims**, the 112 of 300 where annotators found no rationale
in any abstract but BEIR's qrels still mark the cited one relevant. All 11 discordant queries are
NEI: hybrid gains +0.0625 Recall@100 there (t-test p = 0.034, sign test p = 0.065, W/T/L
9/101/2; `analysis.md` reports both tests and marks Recall@100 significance by the sign test).
On the 188 evidence-bearing (SUPPORT/CONTRADICT) claims, hybrid and dense are identical at
Recall@100 on every query (0.9947 against rationale docs), and nDCG@10 differs by +0.0003 (p = 0.977). So the
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
0.3 dense (1 better / 13 worse, sign test p = 0.002), 0.952 at 0.7 (2 better / 6 worse, sign
test p = 0.29). The numbers are an offline replay of the cached top-100 lists, verified to
reproduce the committed hybrid run exactly, in §4 of
[`eval/results/analysis.md`](../eval/results/analysis.md#4-rrf-sensitivity-offline-replay).

[^depth]: A point estimate from the previous committed run at full depth (same model, corpus and
fusion, with the BM25/dense/hybrid rows bit-identical), not a row in the current table. Reproduce
with `SSR_RERANK_CANDIDATES=100 uv run python -m app.eval.retrieval_eval`.

## Grounded answers (RAG)

`GET /answer?q=...` runs the hybrid retrieval, then generates a grounded answer with an OpenAI-compatible LLM (the free `inclusionai/ling-3.0-flash-sante:free` on OpenRouter by default; paid `openai/gpt-oss-120b` via `SSR_OPENROUTER_LLM_MODEL`, or Groq, Ollama or another endpoint via `SSR_LLM_PROVIDER=groq` and `SSR_LLM_BASE_URL`). Answers cite sources as `[n]` mapped back to document ids, and the model is instructed to **abstain when the retrieved context lacks the evidence** rather than hallucinate. When the input is a claim and the reply has no parseable verdict (truncated, or prose without a `Verdict:` line), the generator makes one verdict-only follow-up call (`SSR_LLM_REASK`, on by default; the response's `verdict_source` says where the verdict came from). A truncated reply never gets a verdict inferred from its first sentence (it was cut off mid-reply); it goes to the re-ask instead. Only inputs that read as a claim get that re-ask or a verdict inferred from the reply's first sentence: a question, a keyword search ("BRCA1 breast cancer risk") or an instruction gets neither (a simple heuristic in `looks_like_claim`, checked to accept all 300 test and 809 train claims; a short claim with no final period can be missed and is then treated like a question). If a web source or the re-ask fails, the answer is still served and the failure is listed in the response's `warnings`. `/answer` gives each generation call one SDK retry and the re-ask none (30 s and 120 s timeouts), so a stuck provider holds a worker for minutes at most; the eval harness, with no user waiting, allows 5 retries (re-ask 1).

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

**Verdict accuracy is 0.80 (239 of 300 = 0.797), 95% Wilson CI 0.748–0.838.**

| Gold label | n | Verdict accuracy | Answered (not abstained) |
|---|---|---|---|
| SUPPORT | 124 | 0.815 (101) | 107 |
| CONTRADICT | 64 | 0.891 (57) | 58 |
| NEI | 112 | 0.723 (81) | 31 |

**What changed.** Two post-processing steps on the same Ling answers, scored on the 300 test
claims with paired exact McNemar via `make rag-compare` (b = earlier run right and this run
wrong, c = the reverse). Neither was developed on train alone; see
[How much to trust the test set](#how-much-to-trust-the-test-set):

| Step | Verdict accuracy | b | c | p |
|---|---|---|---|---|
| Strict `Verdict:` line only | 0.73 | | | |
| + verdict-recovery parser (verdict restated in prose, or a stance rule; commits ec465ef, b3de17b) | 0.7767 | 0 | 14 | 0.0001 |
| **+ verdict re-ask** (one verdict-only call for a claim with no verdict; commits e367318, 3dcc1a3) | **0.8000** | **0** | **7** | **0.016** |
| Current scoring rules (commit 3e360cc; see below) | **0.7967** | 1 | 0 | 1.00 |

The first three rows were measured as committed at the time. Commit 3e360cc then tightened two
scoring rules: a reply cut off at the token budget (or empty) with no verdict now scores `NONE`
(always wrong) instead of NEI, since it abstained from nothing, and a truncated reply no longer
gets a verdict from its first sentence (it is re-asked instead). On the same stored answers that
moved two claims, both now wrong: claim 343 (gold SUPPORT, truncated, no verdict even after the
re-ask) went from NEI to `NONE`, and claim 514 (gold NEI, truncated) lost its first-sentence NEI
verdict, was re-asked (the only new LLM call when the run was regenerated) and answered
SUPPORTED. Under the current rules the answers without the re-ask score 0.76 (228/300), and the
re-ask fixes 11 and breaks 0 (p = 0.001); the extra 4 are NEI claims whose replies were cut off
before any text and used to count as correct abstentions, so the scoring change, not the re-ask,
accounts for the difference from the 7 above, and the 7 fixed / 0 broken (p = 0.016) is still the
figure the re-ask was adopted on.

The re-ask now fires on 16 of 300 claims (11 answers truncated at the token budget, 5 prose
replies without a verdict line) and produces a verdict for 15 (12 correct). Abstention (+0.007,
p = 0.63) and citation rate (0.94, unchanged) don't move significantly. Faithfulness stays 0.98
(0.979 over 193 judged answers): the re-ask is not re-judged, and the 3 claims that count as
answered only through the re-ask, after a first reply that hit the token budget (4,096 tokens
after its retry) before writing any answer text, are left out because the judge saw no answer
text. Claim 514 is counted: its truncated first reply had text, which the judge scored
(faithfulness 0.0). Verdict sources over the 300 rows: `line` 257, `inline` 10, `stance` 17,
`reask` 15, none 1 (claim 343, still truncated after the re-ask: `NONE` for the verdict, an
abstention in the tables below).

Abstention is scored against a **rationale oracle**: the context has evidence when a document the
annotators cited *with rationale sentences* is in the top-5. NEI claims have no rationale
document, so abstaining on them is the correct action.

| Metric | Score |
|---|---|
| Faithfulness (over answered, LLM judge; n = 193, 3 re-ask-only answers excluded) | 0.98 |
| Context relevance (all, LLM judge) | 0.72 |
| **3-class verdict accuracy** (vs gold label, no judge) | **0.80** (239/300) |
| Verdict parsed (incl. recovered and re-asked) | 299 of 300 |
| Truncated answers (hit the token budget after 1 retry) | 11 of 300 |
| Answers that needed the length retry | 63 of 300 |
| Replies citing at least one passage | 282 of 300 |
| Evidence retrieved (rationale doc in top-5) | 0.58 |
| Answered (model attempted an answer) | 0.65 |
| **Abstention precision** (abstained & no evidence) | **0.83** |
| Abstention recall (no evidence & abstained) | 0.69 |
| False abstention (had evidence, abstained anyway) | 0.10 |
| Answered without evidence, as a share of answers given (hallucination risk) | 0.20 |

|  | evidence retrieved (175) | no evidence (125) |
|---|---|---|
| **answered** (196) | 157 answered with evidence | 39 answered with nothing to go on |
| **abstained** (104) | 18 declined despite having the evidence | 86 correct |

Verdicts against the gold label (`NONE` = no verdict on a reply that answered, was cut off at the
token budget or was empty; always wrong):

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** (124) | 101 | 6 | 16 | 1 |
| **CONTRADICT** (64) | 1 | 57 | 6 | 0 |
| **NEI** (112) | 18 | 13 | 81 | 0 |

**Remaining failure modes.** 61 of 300 verdicts are wrong, and almost all are an evidence
*judgement* error, not a format one:

- **NEI claims answered anyway: 31 of 112** (18 SUPPORTED, 13 REFUTED), the largest group. The
  model is over-eager: it infers a verdict from weak or tangential context. They account for most
  of the 39 answers given without evidence.
- **SUPPORT/CONTRADICT claims abstained on: 22** (16 SUPPORT, 6 CONTRADICT). For 13 of the 16
  SUPPORT claims the rationale doc is in the context: the model reads the passage too literally.
- Smaller: 6 SUPPORT claims answered REFUTED, 1 CONTRADICT answered SUPPORTED, and 1 SUPPORT
  claim (343) truncated with no verdict even after the re-ask.

**The oracle decides whether abstention looks broken.** Scored the legacy way, against BEIR qrels
(which mark a cited abstract relevant for NEI claims too), the *same answers* give abstention
precision 0.41 and 61 "false" abstentions (rate 0.26); under the rationale oracle only 18 are
false and precision is 0.83.

**Paired comparison with the paid gpt-oss-120b run** (50 shared claims; both predate the
verdict-recovery parser, so they use the old strict parser and are not comparable to the 0.797
headline). Accuracy is indistinguishable on these 50 claims (p = 1.00), which with 3
discordant claims rules a difference neither in nor out; citations are the one significant
difference:

| On the 50 shared claims | gpt-oss-120b (paid) | Ling | b | c | p |
|---|---|---|---|---|---|
| Verdict correct | 0.78 | 0.76 | 2 | 1 | 1.00 |
| Abstention correct (rationale oracle) | 0.84 | 0.78 | 3 | 0 | 0.25 |
| **Cites at least one passage** | **0.42** | **0.94** | **1** | **27** | **< 0.0001** |

On those claims 28 gpt-oss replies were a bare `Verdict:` line against 0 for Ling, and the run
cost $0.0049 against $0. Same-model Ling-vs-Ling repeats change 9 of 50 predicted labels, flipping correctness on 7 (3 one way, 4 the
other; p = 1.00), the noise floor for 50 claims: the generator is not deterministic, which the
Wilson interval on a single run does not capture.

### Label audit

SciFact's labels contain errors. Sylvestre, *Gold Label Errors in the SciFact Benchmark: An
LLM-Assisted Annotation Audit* ([BioNLP 2026](https://aclanthology.org/2026.bionlp-1.9/); data
[Kefez/scifact-audit-bionlp2026](https://github.com/Kefez/scifact-audit-bionlp2026) at pinned
commit `5711f3f`, CC BY 4.0) corrected 11 of our 300 claims' labels and marked 8 more debatable.
Re-scoring the same stored answers offline ([`eval/results/rag_label_audit.md`](../eval/results/rag_label_audit.md)):

| Labels | n | Verdict accuracy |
|---|---|---|
| Original SciFact labels (headline) | 300 | 0.7967 |
| Corrected (the 11 confirmed errors) | 300 | 0.8300 |
| Corrected, 8 debatable claims excluded | 292 | 0.8356 |

**The headline stays on the original labels.** Read 0.830 as a one-sided sensitivity check, not
as an upper bound or a second headline. Only the 188 evidence-bearing claims were audited, never
the 112 NEI ones, so label errors there, which could move the score either way, were never
looked for; of the 11 corrected claims, 10 go wrong → right and none right → wrong. The eleventh,
claim 343 (SUPPORT → NEI), stays wrong: its answer was truncated with no verdict, which now scores
`NONE` under either label (it was credited through the judge's "not answered" call before commit
3e360cc). The corrections come from a single annotator, and 8 of the 11 were first flagged
by an LLM, so an LLM generator agreeing with them is partly expected. The predictions are
identical across rows, so this measures the labels, not the pipeline. To compare
two runs under the corrected key: `make rag-compare A=... B=... ARGS=--labels=audit`.

### Blind re-labelling of the NEI disagreements (LLM annotators, secondary)

The audit above never looked at NEI claims, which is where most remaining errors are: 31 NEI-gold
claims got SUPPORT (18) or CONTRADICT (13). SciFact's NEI means the one *cited* abstract has no
rationale, while the system searches all 5,183 abstracts, so another passage may decide the claim.
To test that, the 31 disagreements were labelled blind against exactly the passages the generator saw.

- **Packet.** 81 items from `eval/results/rag.json` (git_sha `aa98231`): the 31 disagreements plus
  50 controls the model got right (30 NEI, 10 SUPPORT, 10 CONTRADICT), shuffled with a fixed seed.
  Built by `make rag-relabel` (`app/eval/nei_relabel.py`).
- **Annotators.** Three independent LLM agents (2 Claude Opus, 1 Claude Sonnet), each shown only the
  claim and the 5 passages, with no model verdict or answer, gold label or claim id, and the same
  written guidelines as the HTML page. Majority vote. Unanimous on 71 of 81 items, no 3-way
  splits, Fleiss κ = 0.87.
- **Disagreements.** The majority sides with the model on 25 of 31 (15 SUPPORT, 10 CONTRADICT) and
  with gold NEI on 6 (claims 1175, 1213, 1344, 350, 384, 410). Per annotator: 28, 23 and 23 of 31.
  Of the 25, the SciFact-cited abstract was among the passages in 14 and not in 11.
- **Controls.** The majority matches SciFact gold on 49 of 50 (0.98; NEI controls 29/30,
  SUPPORT/CONTRADICT 20/20); per annotator 48, 49 and 48 of 50.

| Labels | n | Verdict accuracy (95% Wilson CI) |
|---|---|---|
| Original SciFact labels (headline) | 300 | 0.7967 (239/300) |
| Annotator labels on the 31 disagreements only (one-sided, can only rise) | 300 | 0.8800 (264/300; 0.838–0.912) |
| Annotator labels on every sampled item | 300 | 0.8767 (263/300; 0.835–0.909) |

**Secondary; the headline stays on the original SciFact labels.** The annotators are LLMs, not a
human, so this is a weaker check, and they may share the generator's reading of the evidence (the
generator is also an LLM). The controls show they are not simply agreeing with the model: they match
gold wherever model and gold agree. The question asked is also not SciFact's: "do these passages
decide it?" (open corpus) against "does the cited abstract decide it?" (closed corpus), the mismatch
SciFact-Open (Wadden et al. 2022) documents. Annotator spread on the disagreements is 23–28 of 31.
The labels are not committed (`data/` is gitignored). Read together, this suggests most remaining
NEI "errors" reflect where the benchmark draws the NEI line (and, to a lesser extent, the
open-corpus mismatch) rather than the model, retrieval or code; it does not show the verdicts are
correct, and it is not a corrected score. How much of this is the open-corpus mismatch was tested
directly afterwards and is smaller than this check alone suggests; see
[the diagnosis](#where-the-remaining-verdict-errors-come-from-diagnosis).

### Where the remaining verdict errors come from (diagnosis)

Two follow-up measurements ask where the 61 test errors (31 NEI-gold answered, 30
SUPPORT/CONTRADICT-gold wrong) come from: is it retrieval, the open-corpus mismatch, or the
evidence judgement itself? Both are diagnostics, not product changes, and neither is canonical.

**1. Cited-abstract diagnostic (train).** `SSR_RAG_CONTEXT=oracle_cited` (eval-only, never
canonical) gives the generator only SciFact's cited abstract(s) instead of the retrieved top 5.
Same seeded 100 `beir/scifact/train` claims, free Ling, judge skipped
(`data/eval_runs/rag_beir-scifact-train_100_d0921f4e_nojudge_oracle-cited/`, local, gitignored).

| Gold | Retrieved top 5 | Cited abstract only | Fixed / broken |
|---|---|---|---|
| All 100 | 0.81 | 0.84 | 7 / 4 (exact McNemar p = 0.55) |
| SUPPORT | 0.864 | 0.864 | 2 / 2 |
| CONTRADICT | 0.864 | 0.955 | 2 / 0 |
| NEI | 0.706 | 0.735 | 3 / 2 |

The +3 points is within run-to-run noise (Ling repeats flip about 7 of 50 verdicts), so the
retrieval ceiling here is roughly +3 points at most. The cited doc was already in the retrieved
top 5 for 88 of 100 claims. Of the baseline's 19 errors, 3 had no cited doc in the top 5 (all
NEI-gold, called SUPPORT from other papers) and the cited abstract fixed all 3; of the 16 with it
in the top 5, 4 were fixed and 12 stayed wrong. Seven NEI-gold claims are called CONTRADICT even
when shown only the cited abstract.

**2. Second blind re-labelling (test, errors on a stance claim).** `nei_relabel build --set
evidence_errors` (packet `a08a3152bfcba3d5`): 70 items, the 30 SUPPORT/CONTRADICT-gold errors plus
40 controls (20 NEI-gold predicted NEI, 10 SUPPORT and 10 CONTRADICT correct), excluding the first
packet's claims. Same three blind LLM annotators (2 Claude Opus, 1 Claude Sonnet), majority vote:
unanimous on 56 of 70, no 3-way splits, Fleiss κ = 0.79.

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
sampled items; the headline stays 0.7967.

**Conclusion (measured, not triumphant).** The remaining errors are mostly not retrieval (oracle
ceiling about +3 points on train; cited doc in the top 5 for 88%), not code or parsing, and not
mainly the open-corpus mismatch: that is smaller than the earlier unblinded estimate (about 25 of
61) suggested. On train only 3 errors came from it, and in the first blind check the cited
abstract was shown in 14 of the 25 cases sided with the model. The main source is where the line
is drawn between NEI and a stance. The model, and the LLM annotators, read the same abstract more
liberally than SciFact's explicit-rationale-sentence convention on NEI claims (over-reading, mostly
toward CONTRADICT), and more cautiously on some implied SUPPORT (about 11 genuine too-cautious
errors). Combined over the 61 test errors: about 43 where blind LLM annotators side with the model
(25 + 18), about 17 model judgement errors (6 + 11), and 1 with no verdict. Caveats: this is an
estimate from LLM annotators, not a human; the diagnostic is on train and the re-labelling on
test, and they are not the same claims; and the train deltas are within noise.

**Implication.** Further gains on SciFact mean calibrating to its annotation convention (for
example few-shot examples of its NEI threshold). That is benchmark calibration, not a smarter
system, and should be described that way. Next step: few-shot calibration screened on train (in
progress, nothing adopted).

### Tried and measured, not adopted

All on the 300 test claims unless noted; paired exact McNemar. These were scored before commit
3e360cc's scoring rules and not re-run: "0.80" is the canonical run as committed at dc1816b
(240/300), which differs from the current one only on claims 343 and 514
([above](#grounded-answers-rag)), and "0.7767" is the same answers before the re-ask under the old
rules.

| Experiment | Verdict accuracy | Result |
|---|---|---|
| Paid GPT-6 Luna as generator ($0.0985), re-scored with the current verdict parser | 0.71 vs 0.80 | worse (42 broken, 15 fixed, p = 0.0005; like for like, without the re-ask, 0.71 vs 0.7767: 39 broken, 19 fixed, p = 0.012); cites on every reply, but 60 SUPPORT/CONTRADICT claims get NEI |
| Off-the-shelf DeBERTa-v3 NLI verifier (tuned on 809 train claims) | 0.64 vs 0.80 | worse (70 broken, 21 fixed, p < 0.0001); weakest on CONTRADICT (0.56) |
| Fine-tuned PubMedBERT verifier (Kaggle; HealthVer + PubMedQA, then SciFact train) | alone 0.66; combined with Ling via rule R3 0.787 vs 0.777 (before the re-ask) | no gain (11 fixed, 8 broken, p = 0.65); R3 won on its 99 tuning claims (0.859 vs 0.828) and didn't transfer |
| Evidence-first prompt (quote the finding sentence first) | 0.847 vs 0.837 on 100 *train* claims | no gain (p = 1.0); only 64 of 98 replies followed the format |
| Disagreement-triggered second look (Ling vs the verifier) | 0.8033 vs 0.7767 (before the re-ask) | not significant alone (p = 0.057), adds nothing on top of the re-ask (0.8067 vs 0.80, p = 0.75), and costs citations (0.94 → 0.92) |
| Majority vote over k=3 samples (`SSR_LLM_VOTES=3`), 100 *train* claims | 0.82 vs 0.81 | no gain (2 fixed, 1 broken, p = 1.00); fails the gate; 3.0x generation calls |
| "Finding" prompt variant (`SSR_LLM_PROMPT_VARIANT=finding`), 100 *train* claims, judge skipped | 0.83 vs 0.81 | not adopted (5 fixed, 3 broken, p = 0.73); passes the gate exactly at the threshold, with the disclosures below |

Verdict accuracy and abstention are judge-free and comparable across all runs, including the
oldest Groq/Qwen-judged gpt-oss run (0.70 → 0.78 on 50 claims after a larger token budget and a
`Verdict: REFUTED[1]` parser fix); its faithfulness and context-relevance numbers are **not**
comparable to the Nemotron-judged runs.

**Majority voting and the "finding" prompt (train, pre-registered).** Both were run on the seeded
100-claim `beir/scifact/train` sample (seed 13) with the free Ling generator, and both were
pre-registered before any LLM call: k=3 samples at temperature 0.1, each with its own verdict
re-ask; plurality vote, ties going to the lowest-numbered sample; the answer text taken from the
lowest-numbered majority sample; at most one judge call per claim. The gate: accuracy must not
drop, fixed minus broken must be at least 2, and for the prompt NEI net must be at least -1.
Paired exact McNemar against the single-sample baseline, which scored **0.81 (81/100)**. The test
split was never run.

- **Majority vote, k=3: 0.82, 2 fixed / 1 broken, p = 1.00, fails the gate.** 90 of 100 claims
  were unanimous, 10 split 2-1 and none split three ways; the individual samples scored
  0.81 / 0.80 / 0.81. Cost: 3.0x the generation calls with all samples drawn (as the eval does),
  about 2.2x with `/answer`'s early stop; latency scales the same way. Kept as an opt-in setting
  (`SSR_LLM_VOTES`), off by default.
- **"Finding" prompt variant** (judge the claim by its main finding, not its exact wording, with a
  guard against over-reaching on NEI), single sample, judge skipped (a reply with no verdict
  counts as wrong): **0.83, 5 fixed / 3 broken, p = 0.73.** That passes the gate exactly at the
  threshold, and it is not adopted. Disclosures: the first scoring was 0.82 (net +1, a fail); a
  rate-limited re-ask (claim 1054) was then retried by the eval's normal resume, after that score
  had been seen, which tipped it to 0.83; 2 of the 5 fixes (claims 1114 and 9) were read while
  writing the wording, so the net is 0 without them; against same-day single samples it is only
  +1 to +2. The error category it targets came from the *test*-set analysis.
- **Next step:** replicate on unused train claims (101-300 of the seeded shuffle) before any test
  run. A test run would be variant #9 or later on the same 300 claims
  ([below](#how-much-to-trust-the-test-set)).

### How much to trust the test set

The 300 test claims (SciFact's public dev split) are no longer a clean held-out set. The
verdict-recovery parser's patterns came from an error analysis of *test* failures and were then
refined and checked on train. The re-ask was adopted after three candidate fixes were scored on
test (on train it fixed 0 and broke 1). In all, 8 variants have now been scored on the same 300
claims (paid Luna, the NLI verifier, the parser, the fine-tuned verifier alone and combined, the
second look, the re-ask, and second look + re-ask), so raw p-values are optimistic. The parser's
gain (14 fixed, 0 broken, p = 1.2e-4) survives any correction. The re-ask's (7 fixed, 0 broken,
raw p = 0.016) becomes borderline: Holm-adjusted p = 0.047 over the 3 fixes tried together, 0.063
over the 4 or 5 post-parser candidates, and 0.078 over all 8. The generator is also not
deterministic (repeat runs flip about 7 of 50 verdicts, [above](#grounded-answers-rag)), so the
Wilson interval on one run understates the real uncertainty. A fresh held-out set would be needed
to confirm 0.80.

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

## Web search (Semantic Scholar and PubMed)

An optional, networked retrieval source, **off the default path**: `hybrid` over the local
index stays the default and the measured headline. Two extra modes on `/search`, `/answer` and
the UI's mode selector (`GET /search?q=...&mode=web` or `mode=hybrid_web`):

- `web`: the web pipeline below on its own.
- `hybrid_web`: the local hybrid list and the web list fused with the same hand-rolled RRF.
  SciFact doc ids are S2 corpus ids, so a paper in both appears once (local text, S2 URL and
  year). If the base S2 search fails, `hybrid_web` returns local results plus a `warnings`
  entry; `web` has nothing to fall back to and returns 503.

Both modes run a **deterministic, LLM-free pipeline**
([`app/retrieve/web_search.py`](../app/retrieve/web_search.py)), pooled from three sources by
default:

1. **S2 keyword search.** The claim is rewritten into a ≤ 6-term keyword query (plus a 3-term
   fallback query, results pooled), because S2 search handles a whole sentence badly.
2. **S2 snippet search** on the claim ([`app/retrieve/s2_extra.py`](../app/retrieve/s2_extra.py)).
   Papers that quote the claim verbatim or mention SciFact are dropped: NLP papers reprint
   SciFact claims together with their labels.
3. **PubMed Best Match** on the rewrite and the claim
   ([`app/retrieve/pubmed.py`](../app/retrieve/pubmed.py)), with PMIDs/DOIs resolved to S2
   corpus ids in one S2 `paper/batch` call.

The pool is deduplicated and re-ranked locally by RRF of bge-small similarity to the original
claim and BM25 over the candidates, embedding only the top 50 of a BM25 pre-rank
(`SSR_WEB_DENSE_CAP`). If snippets or PubMed fail, the search continues without them and the
failure is reported in `warnings`. `SSR_S2_QUERY_REWRITE=false` / `SSR_S2_RERANK=false` switch
the rewrite and the re-rank off.

**Measured** (`make web-eval`, 300 test claims, gold-paper retrieval; rewrite rules, sources and
settings chosen on 100 train claims and frozen before the test run;
[`eval/results/web_retrieval.md`](../eval/results/web_retrieval.md)):

| Config | Recall@5 | Recall@100 | nDCG@10 |
|---|---|---|---|
| S2, raw claim as query | 0.040 | 0.056 | 0.035 |
| S2, raw claim + local rerank | 0.040 | 0.056 | 0.040 |
| S2, keyword rewrite | 0.102 | 0.236 | 0.089 |
| S2, keyword rewrite + local rerank | 0.185 | 0.244 | 0.159 |
| **Pooled: S2 rewrite + S2 snippets + PubMed, rerank of top 50 (the default)** | **0.228** | **0.384** | **0.203** |
| Local hybrid (RRF, 5,183 docs), for scale | 0.766 | 0.965 | 0.724 |

Rewrite + rerank vs the raw claim: 45 claims better, 1 worse at Recall@5 (p < 0.0001), and every
rewrite comparison is significant; reranking alone changes nothing (Recall@5 p = 1.0). Pooling
then adds candidates the S2 keyword search never returns: vs rewrite + rerank, Recall@100 rises
on 46 claims and falls on none (0.244 → 0.384, p < 0.0001), Recall@5 0.185 → 0.228 (p = 0.012)
and nDCG@10 0.159 → 0.203 (p = 0.002). Extra S2 queries and wider re-ranking (the offline row in
the results file) add nothing significant. Web search still trails local hybrid on every metric
(p < 0.0001), which is expected: S2 searches ~200M papers against our 5,183, so this asks whether
open-web retrieval can find the gold paper at all, not which ranker is better.

**These rows predate commit aa98231**, which fixed how the keyword rewrite handles numbered names
(interleukin-2, TDP-43), unseen words, currency and comparator numbers and compound duplicates.
The fix changes the S2 queries for 56 of the 300 test claims (41 of them in the primary query).
On the 100 train claims it moved Recall@5 0.285 → 0.305 and Recall@100 0.453 → 0.463 (2 claims
better, 0 worse; not significant), but `make web-eval` has not been re-run on test, so the table
is for the pre-fix rewrite until it is.

These rows were measured with the eval's web settings, not the API's: the pooled row embeds the
top 50 candidates (`SSR_WEB_DENSE_CAP`), has no per-request deadline and does not apply the dataset-dump snippet filter.
The API instead embeds at most 30 (`SSR_WEB_API_DENSE_CAP`), abandons extra sources still
pending after 20 s (`SSR_WEB_DEADLINE_S`) and drops dataset-dump snippets that overlap a SciFact
claim (`SSR_WEB_SNIPPET_DATASET_FILTER`, on), so its web results can differ from the table; those
API settings are not what the committed numbers measure.

The scoring itself caps web recall below 1: of the 283 gold docs, S2 returns 264 under their
SciFact corpus id and 256 with a matching title, so a perfect web search would score Recall@100
≈ 0.90 here (the eval stops if the title match rate drops under 90%). S2's index also changes,
so responses are cached with their fetch date.

OpenAlex ([`app/retrieve/openalex.py`](../app/retrieve/openalex.py)) is implemented but off. On
the 45 train claims where it was queried, adding its semantic search on the claim to S2 raised
Recall@100 from 0.219 to 0.319 (`s2+oa_sem`, p = 0.027), and adding its works search on the
keyword rewrite instead to 0.330 (`s2+oa_rw`, p = 0.024), in the train screening run
`data/eval_runs/web_pool_beir-scifact-train_100_screen_c5fix/` (gitignored). Keyless use allows
about 100 searches a day, too few for the API or a 300-claim eval, so it was never measured on
test.

Web hits carry `source`, `url` (http/https only, checked server- and client-side) and `year`.
Titles and abstracts are untrusted third-party text, flattened to one line with quote runs and
control characters removed so they cannot forge the grounded prompt's structure.

**Keys, limits and speed.** No key is required, but unauthenticated S2 requests share a public
pool and often get 429. A free key ([request form](https://www.semanticscholar.org/product/api#api-key-form))
goes in `.env` as `SSR_S2_API_KEY` (sent as `x-api-key` only when set; 1 request/s). A
process-wide limiter spaces every S2 request, retries included, to `SSR_S2_RATE_PER_S`
(default 1); API requests wait at most `SSR_S2_MAX_WAIT_S` (5 s), then degrade. PubMed needs no
key and has its own limiter (`SSR_PUBMED_RATE_PER_S`, default 2/s); NCBI asks API users to
identify themselves, so set `SSR_NCBI_EMAIL` (and optionally `SSR_NCBI_API_KEY`). Both are empty
by default and sent only to NCBI. In the API, a pooled web search has a 20 s budget
(`SSR_WEB_DEADLINE_S`; sources still pending are dropped and listed in `warnings`) and embeds at
most 30 candidates (`SSR_WEB_API_DENSE_CAP`; the eval uses 50). One web search makes several S2
and PubMed calls; measured informally with caches off it takes about 17 s per query on the
laptop CPU, so web mode is slow. Raw responses cache under `data/s2_cache/` and `data/web_cache/`:
always for the eval, and for the API only with `SSR_S2_API_CACHE=true`. A cached `make web-eval`
re-scores with no requests; an uncached full run makes a few thousand S2 and PubMed requests at
about 1 per second (`SSR_EVAL_LIMIT=2` is a smoke run).

## Limitations

- **300 queries bound what can be detected.** For hybrid vs dense, the minimum detectable effect
  at 80% power (α = 0.05, paired) is ≈ 0.028 nDCG@10 and ≈ 0.031 Recall@100. The nulls above
  rule out large effects, not small ones.
- **Recall@100 has a ceiling fusion cannot move.** 8 of the 11 gold docs hybrid misses are in
  neither retriever's top-100, so no fusion rule over these two candidate pools could recover
  them ([`eval/results/analysis.md`](../eval/results/analysis.md), §3).
- **NEI labelling.** For the 112 NEI claims, BEIR's qrels mark the cited abstract relevant even
  though annotators found no rationale in it, and the whole hybrid-vs-dense recall gain (itself
  only suggestive: sign test p = 0.065) sits on that stratum (Finding 1).
- **300 claims still leave a ~5-point interval on RAG accuracy.** The eval covers every SciFact
  test claim, and the 95% Wilson interval on verdict accuracy is 0.748–0.838 (0.797 ± 0.045), so a
  gap of a few points between two runs' headline rates means nothing on its own; compare runs
  claim by claim with `make rag-compare`. The re-ask gain (+7 claims, raw p = 0.016) is small,
  rests on 15 re-asked claims (16 under the current scoring rules), and is borderline once the other variants scored on the same
  claims are counted (Holm 0.047–0.078). The interval also ignores generator nondeterminism.
- **The test set has been reused.** The parser and the re-ask were chosen with the 300 test
  claims in view, and 8 variants have been scored on them, so 0.80 is no longer a clean held-out
  number ([details](#how-much-to-trust-the-test-set)).
- **Paired comparisons with the gpt-oss run cover only 50 claims**, scored with the old strict
  parser, so only large effects (the citation gap) can reach significance.
- **The label audit is a sensitivity check, not a new key.** One annotator, mostly LLM-flagged, no
  NEI claims audited; the 0.830 corrected figure is a one-sided check, not an upper bound.
- **Remaining errors are evidence-judgement errors.** 31 of 112 NEI claims are answered anyway
  (over-eager) and 16 of 124 SUPPORT claims are abstained on (too literal); format and parsing
  failures are now ≤ 1 of 300. A secondary blind re-label by LLM annotators sided with the model
  on 25 of the 31 NEI cases, so some of the over-eager count is where the benchmark draws the NEI
  line, not the model ([details](#blind-re-labelling-of-the-nei-disagreements-llm-annotators-secondary));
  the open-corpus mismatch explains less than that check alone suggested, and about 11 of the 30
  stance-claim errors look like genuine too-cautious judgements
  ([diagnosis](#where-the-remaining-verdict-errors-come-from-diagnosis)).
- **Web search is far below local retrieval on SciFact, and slow.** Even pooled, Recall@5 is
  0.228 against 0.766 (Recall@100 0.384 against 0.965), a query takes many seconds, and the
  results move with S2's and PubMed's indexes; it is an optional source, not a replacement.
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

- **Attack the two remaining error modes directly.** Over-eager answers on NEI claims (31 of 112)
  and too-literal abstentions on SUPPORT claims (16 of 124) are what is left. A stricter prompt
  (evidence-first), an off-the-shelf NLI verifier, a fine-tuned verifier and a second-look check
  have all been measured and not adopted ([above](#tried-and-measured-not-adopted)), so the next
  lever is better supervision, such as SciFact-train hard negatives for NEI, or a stronger
  generator that keeps Ling's citation behavior (paid GPT-6 Luna did not).
- **Close the web-search gap** (Recall@5 0.228 vs 0.766 local): the rewrite is rule-based and was
  tuned on 100 claims; a learned or LLM query rewrite is untested, and OpenAlex helped on train
  but needs a quota that allows a full test run.
- **Confirm 0.80 on fresh claims** never used for development, since the test set has been
  reused, and repeat the run a few times to measure generator noise.
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
plus the optional networked `web` and `hybrid_web` ([Web search](#web-search-semantic-scholar-and-pubmed)).
`GET /answer?q=...&top_k=5` returns the grounded answer, its verdict and citations.

The RAG and web targets (`make eval-rag`, `make rag-compare`, `make rag-audit`,
`make rag-relabel`, `make web-eval`) and the verifier experiments (`make verify-eval`,
`verify-export`, `verify-train`, `verify-combine`) are listed with one-line descriptions in the [`Makefile`](../Makefile).

## Operational notes

Things that are deliberate rather than accidental, and the reasoning behind them:

- **Reranking is bounded to the top-32 fused candidates** (`SSR_RERANK_CANDIDATES`); the tail
  keeps its fused order, so recall past the slice is untouched. A cross-encoder is a full
  forward pass *per candidate*. Reranking all 100 takes ~20s on CPU, and because retrieval is
  serialized behind a lock, an unbounded slice lets a single client at the rate limit hold the
  service for minutes. It is a denial-of-service guard first and a latency knob second.
- **The API is unauthenticated**, so per-IP rate limiting (30/min search, 10/min answer) is the
  only guard. Web modes add their own guards, because one web search fans out to several S2 and
  PubMed calls: a stricter per-IP limit shared by `/search` and `/answer` (`SSR_WEB_RATE_LIMIT`,
  default `6/minute`, 429 when exceeded) and a per-process cap on concurrent web searches
  (`SSR_WEB_MAX_CONCURRENT`, default 2; when every slot is busy the request gets 503 with
  `Retry-After` at once instead of queueing). Local modes never touch either. `SSR_TRUST_PROXY=true` keys on the X-Forwarded-For entry `SSR_TRUSTED_PROXY_HOPS`
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
- **BM25 returns only documents that share a term with the query.** bm25s fills its top-k with
  score-0 documents in index order; those are dropped, so a query with no known token (e.g.
  "5-HT2A", or only stopwords) gets no BM25 hits instead of k arbitrary ones that fusion would
  rank. The web re-rank keeps them, since it orders a whole candidate pool. Local retrieval
  metrics are unchanged on all 300 test claims.
- **Opt-in generation experiments (all off by default, none in the committed numbers).**
  `SSR_LLM_VOTES` (default 1, up to 5) draws k samples for a claim and takes a plurality verdict,
  so `/answer` can cost up to k times the calls. `SSR_LLM_PROMPT_VARIANT` (`default` | `finding`)
  picks the product system prompt; `default` is the one every committed result used.
  `SSR_RAG_SKIP_JUDGE=1` is eval-only: no judge calls, so no faithfulness or context relevance, and
  a reply with no verdict scores as wrong. A rag_eval run with any of them is **never canonical**
  and writes to `data/eval_runs/`: a skipped-judge run gets a `_nojudge` suffix and a voting run
  `_votes<k>`. Commented examples are in `.env.example`.
- **Only one RAG run writes the committed artifact.** `make eval-rag` writes
  `eval/results/rag.{md,json}` only for a full test-split run (`SSR_RAG_N=all`, no
  `SSR_EVAL_LIMIT`) with both models on the code defaults and the re-ask on. Every other run,
  including the default 50-claim sample, writes to `data/eval_runs/`. A canonical run that ends
  with a claim missing or a failed re-ask refuses to write (finished rows stay checkpointed, so a
  re-run redoes only those); non-canonical runs write and name the missing claims.
- **Retrieval is serialized** by a process-wide lock (embedded Qdrant and the shared
  sentence-transformers models are not thread-safe), so this serves one search at a time per
  worker. Running Qdrant in server mode is the fix if that ceiling ever matters.
