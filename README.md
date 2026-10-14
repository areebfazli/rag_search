# Scientific Claim Checker

[![CI](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml/badge.svg)](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml)

You type a scientific or health claim, such as *"Vitamin D deficiency increases the risk of
multiple sclerosis"*. The tool searches 5,183 biomedical paper abstracts (the SciFact benchmark)
for the most relevant papers, then has a language model read the top five and say whether they
**support** the claim, **contradict** it, or **don't give enough evidence**, citing the exact
papers it used. Retrieval is hand-rolled (keyword search + embeddings, fused), the answers come
from free models, so running it costs $0, and every headline number is measured against the
benchmark's expert labels.

In the UI you type a claim (or click one of three examples) and press **Check claim**: one click
returns a colour-coded verdict card (the verdict, a cited explanation whose [n] links jump to the
paper, and the model that answered) above the ranked list of matching papers.

![The UI after checking the example claim "Bariatric surgery has a positive impact on mental health.": the claim box with its Check claim button and search settings, three example claims, a green Supported verdict card answered by ministral-14b-latest with clickable numbered citations, and the first of the papers found, marked cited](docs/ui.png)

## Results

All numbers are on the 300 SciFact test claims with default settings, and each one comes from a
committed file in [`eval/results/`](eval/results/). Answers come from Mistral's free
`ministral-14b-latest` (every one of the 300 in the canonical run; `ministral-8b-latest` is the
fallback if a call fails), and a free Nemotron 3 Ultra model on OpenRouter is the judge.

| What is measured | Result | Source |
|---|---|---|
| Right paper in the top 10 results (Recall@10) | 0.855 | [`retrieval.md`](eval/results/retrieval.md) |
| Right paper in the top 100 results (Recall@100) | 0.965 | [`retrieval.md`](eval/results/retrieval.md) |
| Ranking quality (nDCG@10) | 0.724 | [`retrieval.md`](eval/results/retrieval.md) |
| Correct verdict (support / contradict / not enough evidence) | **0.757** (227 of 300; 95% CI 0.705–0.802) | [`rag.md`](eval/results/rag.md), [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| Correct verdict under a published label audit (one-sided check, not the headline) | 0.780 | [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| When it abstains, the papers really lacked evidence (abstention precision) | 0.73 | [`rag.md`](eval/results/rag.md) |
| Answers citing at least one paper | 291 of 300 | [`rag.json`](eval/results/rag.json) |
| Answers that stick to the papers (faithfulness, LLM judge, 169 answers) | 0.98 | [`rag.md`](eval/results/rag.md) |
| Search time per query (laptop CPU) | 0.124 s | [`latency.md`](eval/results/latency.md) |
| Cost of the full 300-claim answer evaluation | $0 | [`rag.json`](eval/results/rag.json) (`cost`) |

## What I found

- **Retrieval is strong, and it is not the main bottleneck.** The right paper is in the top 10 for
  86% of claims. Combining keyword and embedding search clearly beats keyword search alone
  (nDCG@10 p = 0.0007), but its edge over embeddings alone is not significant for ranking
  (p = 0.26). It *may* widen the pool of papers found (top-100 recall 0.942 → 0.965), but only 11
  claims differ (9 better, 2 worse; sign test p = 0.065, suggestive, not significant), and all of
  them are claims the dataset says have no evidence, so fusion does not find more evidence.
  [Details](docs/DETAILS.md#retrieval).
- **Verdicts are right 75.7% of the time (227 of 300).** Of the 73 wrong verdicts, 43 are
  supported/contradicted claims the model declined to call (29 of them supported claims, the
  largest error cell) and 24 are "not enough evidence" (NEI) claims it answered anyway.
- **The answer model changed, and the score difference is not significant.** The project was
  built on the free Ling 3.0 Flash Sante model on OpenRouter, which scored 0.797 (239 of 300,
  git `348c919`) and was withdrawn upstream in October 2026. It now uses Mistral's free
  `ministral-14b-latest`. On the same 300 claims, Ling was right on 27 claims Mistral got wrong
  and Mistral on 15 that Ling got wrong: McNemar p = 0.09, not significant. Abstention
  (p = 0.38) and citation rate (p = 0.11) don't move significantly either.
  [Details](docs/DETAILS.md#the-generator-change-ling-to-mistral).
- **On the Ling run, most errors were about where SciFact draws the NEI line.** Secondary checks
  on that run pointed away from retrieval and code: on 100 train claims, the paper SciFact cites
  was already in the top 5 for 88, and giving the model only that paper moved accuracy
  0.81 → 0.84 (within noise, p = 0.55). Two blind re-labellings by three LLM annotators (not a
  human) sided with the model on about 43 of Ling's 61 test errors and with the benchmark on
  about 17. These checks were not repeated on Mistral, whose errors sit in the same place (the
  biggest cell is supported claims called NEI). Further gains on SciFact would mostly be
  calibrating to its labelling convention, not a smarter system.
  [Diagnosis](docs/DETAILS.md#diagnosis-where-the-remaining-errors-come-from).
- **The headline stays on the original labels.** A published 2026 audit corrects 11 of the 300
  test labels; scored against it, accuracy is 0.780 (7 of the changed claims go wrong → right,
  none right → wrong), but the audit skipped NEI claims, so that is a one-sided sensitivity check.
  The LLM re-labellings are weaker evidence still: the annotators may share the generator's
  reading of the papers.
- **The test set has been reused.** The verdict parser and the re-ask step were chosen on the
  Ling run with these same 300 claims in view, and several variants were scored on them, so they
  are no longer a clean held-out set; the re-ask's gain on Ling (7 fixed, 0 broken, raw
  p = 0.016) is borderline after correction. The Mistral run is a single run. The one change
  made for Mistral, a firmer verdict-line instruction in the prompt, was chosen on train claims,
  not on these. [Details](docs/DETAILS.md#test-set-reuse).
- **The verdict line now comes on the first reply.** Under the original prompt, ministral-14b
  often answered in prose without the final `Verdict:` line (or added a caveat after it), so the
  re-ask step (one verdict-only follow-up call) fired on 61 of 300 claims in the first Mistral run
  (0.750). The prompt now says the line is required for every claim and that caveats belong in
  the prose above it. On 100 train claims that cut the re-ask from 23 to 2 with accuracy within
  noise. On test it fired on 2 of 300, and accuracy moved 0.750 → 0.757 (20 broken, 22 fixed,
  McNemar p = 0.88): no change. [Details](docs/DETAILS.md#the-verdict-line-prompt-change).

## Tried, measured, not shipped

Each was measured on the same harness, on the earlier Ling run, and left out of the product.
The code is preserved in the
git tag [`experiments-archive`](https://github.com/areebfazli/rag_search/tree/experiments-archive);
full results are in [DETAILS](docs/DETAILS.md#tried-and-not-shipped).

| Idea | Result |
|---|---|
| Cross-encoder reranking | `bge-reranker-base` ties plain hybrid (nDCG@10 +0.0001, p = 0.996) at 23.4 s/query vs 0.124 s; MS-MARCO MiniLM scores lower (−0.027, p = 0.056). Still callable as `mode=hybrid_rerank` in the API and eval, off by default |
| Paid generators | GPT-6 Luna: 0.71 vs 0.7767 for the free Ling generator then in use (both without the re-ask; p = 0.012). gpt-oss-120b: indistinguishable on 50 claims (0.78 vs 0.76, p = 1.00) but cited a paper far less often (0.42 vs 0.94, p < 0.0001) |
| Web search (Semantic Scholar + PubMed) | Right paper in top 5 / top 100: 0.228 / 0.384, vs 0.766 / 0.965 for local search |
| Verifier models | Off-the-shelf DeBERTa NLI: 0.64. Fine-tuned PubMedBERT combined with the generator: 0.787 vs 0.777 (p = 0.65) |
| Second look, voting, prompt variants | Second look adds nothing on top of the re-ask (p = 0.75); 3-sample voting 0.82 vs 0.81 on train (p = 1.00, 3x calls); evidence-first and "finding" prompts 0.847 vs 0.837 (p = 1.0) and 0.83 vs 0.81 (p = 0.73) on train |

## Quick start

You need `git`, `make`, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if missing)
and about 2 GB of disk. Search needs no key. Verdicts need a free
[Mistral](https://console.mistral.ai/) API key (`SSR_MISTRAL_API_KEY`); the answer evaluation
also needs a free [OpenRouter](https://openrouter.ai/) key (`SSR_OPENROUTER_API_KEY`) for the
judge.

```bash
git clone https://github.com/areebfazli/rag_search.git
cd rag_search
make install          # Python 3.12 env + dependencies (uv sync)
cp .env.example .env  # then set SSR_MISTRAL_API_KEY (and SSR_OPENROUTER_API_KEY for make eval-rag)
make index            # one-time: download SciFact + the embedding model, build the indices (~5 min)
make api              # open http://localhost:8000
```

The vector store runs embedded and allows one process, so don't run `make api` at the same time
as `make eval` or `make index`.

## Commands

| Command | What it does |
|---|---|
| `make install` / `make index` / `make api` | Set up, build the indices, serve the UI + API |
| `make eval` | Retrieval metrics on the 300 test claims → `eval/results/retrieval.md` (no key needed) |
| `make analysis` | Re-score the cached retrieval runs by claim label → `eval/results/analysis.md` |
| `make latency` | Per-query search time for every configuration → `eval/results/latency.md` |
| `make eval-rag` | Answer-quality eval (needs both keys): 50 claims by default → `data/eval_runs/`. `SSR_RAG_N=all make eval-rag` runs all 300 (the canonical run made about 300 Mistral requests and 305 free OpenRouter judge requests) and is the only run that writes `eval/results/rag.md`; add `SSR_RAG_CHECK_QUOTA=1` to stop up front if today's free OpenRouter quota is short |
| `make rag-compare A=a/rag.json B=b/rag.json` | Claim-by-claim significance test (exact McNemar) between two answer runs |
| `make rag-audit` | Re-score the answers under the published label audit (offline) → `eval/results/rag_label_audit.md` |
| `make test` / `make lint` / `make ci` | Unit tests / ruff / both, as CI runs them |

The judge's limit is the one to watch in the evaluation: OpenRouter's free tier allows 1,000
requests a day only on accounts that have bought at least $10 of credits at some point (50
otherwise), and a 300-claim run needs about 305 judge requests. Mistral's free tier allows 30
requests a minute, which the run's throttle stays under. A run that hits a limit stops, and
re-running the same command resumes it. Any OpenAI-compatible endpoint (Groq, Ollama, OpenAI)
can replace either role; see [`.env.example`](.env.example).

## Architecture

```
claim ─┬─► BM25 (bm25s)            top-100 ─┐
       └─► bge-small + Qdrant      top-100 ─┴─► RRF fusion ─► top-5 ─► grounded prompt
                                                                         │
     free Mistral ministral-14b (8b fallback) ◄──────────────────────────┘
                 │
                 └─► answer with [n] citations + Verdict: SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE
                     (verdict parser; one verdict-only re-ask if a claim gets no verdict)
```

- `app/retrieve`: BM25, dense and hand-written RRF fusion behind one `SearchService`, shared by
  the API and the eval harness so both run identical code.
- `app/generate`: OpenAI-compatible client, grounded prompt with an injection sanitizer, verdict
  parser, re-ask and a one-shot fallback to `ministral-8b-latest`. Each key only goes to its own
  provider, and OpenRouter requests pass a free-only guard (`app/core/llm_endpoints.py`).
- `app/eval`: retrieval eval (ranx, paired significance tests), answer eval (verdicts scored
  against SciFact labels; a free Nemotron 3 Ultra judge on OpenRouter, from a different model
  family (Nvidia) than the Mistral generator, for faithfulness), McNemar comparisons, label
  audit, latency.
- `app/api` + `frontend/`: FastAPI (`/search`, `/answer`, per-IP rate limits) and a vanilla JS UI
  where one **Check claim** button runs both.

## More detail

[`docs/DETAILS.md`](docs/DETAILS.md) covers the method, every significance test, per-label
tables, scoring rules, the error diagnosis, everything tried and not shipped, operations and
limitations.

## License

[MIT](LICENSE)
