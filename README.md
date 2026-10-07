# Scientific Claim Checker

[![CI](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml/badge.svg)](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml)

You type a scientific or health claim, such as *"Vitamin D deficiency increases the risk of
multiple sclerosis"*. The tool searches 5,183 biomedical paper abstracts (the SciFact benchmark)
for the most relevant papers, then has a language model read the top five and say whether they
**support** the claim, **contradict** it, or **don't give enough evidence**, citing the exact
papers it used. Retrieval is hand-rolled (keyword search + embeddings, fused), the answers come
from free models, so running it costs $0, and every headline number is measured against the
benchmark's expert labels.

![The search UI after a hybrid search for "Vitamin D deficiency increases the risk of multiple sclerosis", showing ranked SciFact abstracts with scores](docs/ui.png)

## Results

All numbers are on the 300 SciFact test claims with default settings, and each one comes from a
committed file in [`eval/results/`](eval/results/).

| What is measured | Result | Source |
|---|---|---|
| Right paper in the top 10 results (Recall@10) | 0.855 | [`retrieval.md`](eval/results/retrieval.md) |
| Right paper in the top 100 results (Recall@100) | 0.965 | [`retrieval.md`](eval/results/retrieval.md) |
| Ranking quality (nDCG@10) | 0.724 | [`retrieval.md`](eval/results/retrieval.md) |
| Correct verdict (support / contradict / not enough evidence) | **0.797** (239 of 300; 95% CI 0.748–0.838) | [`rag.md`](eval/results/rag.md), [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| Correct verdict under a published label audit (one-sided check, not the headline) | 0.830 | [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| When it abstains, the papers really lacked evidence (abstention precision) | 0.83 | [`rag.md`](eval/results/rag.md) |
| Answers citing at least one paper | 282 of 300 | [`rag.json`](eval/results/rag.json) |
| Answers that stick to the papers (faithfulness, LLM judge, 193 answers) | 0.98 | [`rag.md`](eval/results/rag.md) |
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
- **Verdicts are right 79.7% of the time, and most remaining errors are about where SciFact draws
  the "not enough evidence" (NEI) line.** Of the 61 wrong verdicts, 31 are NEI claims the model
  answered anyway and 22 are supported/contradicted claims it declined to call. Secondary checks
  point away from retrieval and code: on 100 train claims, the paper SciFact cites was already in
  the top 5 for 88, and giving the model only that paper moved accuracy 0.81 → 0.84 (within
  noise, p = 0.55). Two blind re-labellings by three LLM annotators (not a human) sided with the
  model on about 43 of the 61 test errors and with the benchmark on about 17. Further gains on
  SciFact would mostly be calibrating to its labelling convention, not a smarter system.
  [Diagnosis](docs/DETAILS.md#diagnosis-where-the-remaining-errors-come-from).
- **The headline stays on the original labels.** A published 2026 audit corrects 11 of the 300
  test labels; scored against it, accuracy is 0.830, but the audit skipped NEI claims and 10 of its
  11 changes favour the model, so that is a one-sided sensitivity check. The LLM re-labellings are
  weaker evidence still: the annotators may share the generator's reading of the papers.
- **The test set has been reused.** The verdict parser and the re-ask step behind 0.797 were
  chosen with these same 300 claims in view, and several variants were scored on them, so they
  are no longer a clean held-out set; the re-ask's gain (7 fixed, 0 broken, raw p = 0.016) is
  borderline after correction. [Details](docs/DETAILS.md#test-set-reuse).

## Tried, measured, not shipped

Each was measured on the same harness and left out of the product. The code is preserved in the
git tag [`experiments-archive`](https://github.com/areebfazli/rag_search/tree/experiments-archive);
full results are in [DETAILS](docs/DETAILS.md#tried-and-not-shipped).

| Idea | Result |
|---|---|
| Cross-encoder reranking | `bge-reranker-base` ties plain hybrid (nDCG@10 +0.0001, p = 0.996) at 23.4 s/query vs 0.124 s; MS-MARCO MiniLM scores lower (−0.027, p = 0.056). Still callable as `mode=hybrid_rerank` in the API and eval, off by default |
| Paid generators | GPT-6 Luna: 0.71 vs 0.7767 for free Ling (both without the re-ask; p = 0.012). gpt-oss-120b: indistinguishable on 50 claims (0.78 vs 0.76, p = 1.00) but cited a paper far less often (0.42 vs 0.94, p < 0.0001) |
| Web search (Semantic Scholar + PubMed) | Right paper in top 5 / top 100: 0.228 / 0.384, vs 0.766 / 0.965 for local search |
| Verifier models | Off-the-shelf DeBERTa NLI: 0.64. Fine-tuned PubMedBERT combined with the generator: 0.787 vs 0.777 (p = 0.65) |
| Second look, voting, prompt variants | Second look adds nothing on top of the re-ask (p = 0.75); 3-sample voting 0.82 vs 0.81 on train (p = 1.00, 3x calls); evidence-first and "finding" prompts 0.847 vs 0.837 (p = 1.0) and 0.83 vs 0.81 (p = 0.73) on train |

## Quick start

You need `git`, `make`, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if missing),
about 2 GB of disk, and, for answers only, a free [OpenRouter](https://openrouter.ai/) API key.
Search works without a key.

```bash
git clone https://github.com/areebfazli/rag_search.git
cd rag_search
make install          # Python 3.12 env + dependencies (uv sync)
cp .env.example .env  # then set SSR_OPENROUTER_API_KEY=sk-or-...
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
| `make eval-rag` | Answer-quality eval: 50 claims by default → `data/eval_runs/`. `SSR_RAG_N=all make eval-rag` runs all 300 (about 670 free requests, ~1.5 h) and is the only run that writes `eval/results/rag.md`; add `SSR_RAG_CHECK_QUOTA=1` to stop up front if today's free quota is short |
| `make rag-compare A=a/rag.json B=b/rag.json` | Claim-by-claim significance test (exact McNemar) between two answer runs |
| `make rag-audit` | Re-score the answers under the published label audit (offline) → `eval/results/rag_label_audit.md` |
| `make test` / `make lint` / `make ci` | Unit tests / ruff / both, as CI runs them |

OpenRouter's free tier allows 1,000 requests a day only on accounts that have bought at least $10
of credits at some point. A run that hits the limit stops, and re-running the same command
resumes it. Any OpenAI-compatible endpoint (Groq, Ollama, OpenAI) can replace the defaults; see
[`.env.example`](.env.example).

## Architecture

```
claim ─┬─► BM25 (bm25s)            top-100 ─┐
       └─► bge-small + Qdrant      top-100 ─┴─► RRF fusion ─► top-5 ─► grounded prompt
                                                                         │
           free Ling 3.0 Flash Sante (OpenRouter) ◄──────────────────────┘
                 │
                 └─► answer with [n] citations + Verdict: SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE
                     (verdict parser; one verdict-only re-ask if a claim gets no verdict)
```

- `app/retrieve`: BM25, dense and hand-written RRF fusion behind one `SearchService`, shared by
  the API and the eval harness so both run identical code.
- `app/generate`: OpenAI-compatible client, grounded prompt with an injection sanitizer, verdict
  parser and re-ask. OpenRouter requests pass a free-only guard (`app/core/llm_endpoints.py`).
- `app/eval`: retrieval eval (ranx, paired significance tests), answer eval (verdicts scored
  against SciFact labels; a free Nemotron 3 Ultra judge, from a different model family, for
  faithfulness), McNemar comparisons, label audit, latency.
- `app/api` + `frontend/`: FastAPI (`/search`, `/answer`, per-IP rate limits) and a vanilla JS UI.

## More detail

[`docs/DETAILS.md`](docs/DETAILS.md) covers the method, every significance test, per-label
tables, scoring rules, the error diagnosis, everything tried and not shipped, operations and
limitations.

## License

[MIT](LICENSE)
