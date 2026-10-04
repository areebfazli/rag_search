# Scientific Claim Checker (Semantic Search with RAG Engine)

[![CI](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml/badge.svg)](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml)

You type a scientific or health claim, such as *"Vitamin D deficiency increases the risk of
multiple sclerosis"*. The tool finds the research papers most relevant to it, reads them, and
says whether they **support** the claim, **refute** it, or **don't contain enough evidence**,
citing the exact papers it relied on. It searches a built-in collection of 5,183 biomedical
paper abstracts (the SciFact benchmark) and can optionally search the web through Semantic
Scholar and PubMed. It runs on free AI models, so it costs nothing to use. Every number below is measured
against answers written by expert annotators, not estimated.

![Search UI: results for "Vitamin D deficiency is associated with increased risk of multiple sclerosis" over 5,183 SciFact abstracts](docs/ui.png)

## How it works

1. **Find papers.** Keyword search and meaning-based search run side by side and their rankings
   are combined (BM25 + `bge-small` dense embeddings, fused with Reciprocal Rank Fusion).
2. **Read them.** The top 5 abstracts go to a language model that is told to use only them
   (free Ling 3.0 Flash Sante on OpenRouter, behind a prompt-injection sanitizer).
3. **Decide and cite.** It answers SUPPORTED, REFUTED or NOT ENOUGH EVIDENCE, citing passages as
   `[n]`, and holds back when the papers lack the evidence (grounded RAG with abstention).
4. **Measure everything.** Search and verdicts are scored against SciFact's expert labels, and a
   separate model checks that answers stick to the papers (gold qrels via `ranx`, paired
   significance tests, Nemotron 3 Ultra as LLM judge).

```
claim ─┬─► keyword search (BM25)  ─┐
       └─► meaning search (dense) ─┴─► combine (RRF) ─► top 5 papers ─► LLM ─► verdict + [n] citations
```

## Results

All on the 300 SciFact test claims, default settings.

| What is measured | Result | Source |
|---|---|---|
| Right paper among the top 10 results (Recall@10) | 0.855 | [`web_retrieval.md`](eval/results/web_retrieval.md) (local hybrid row) |
| Right paper among the top 100 results (Recall@100) | 0.965 | [`retrieval.md`](eval/results/retrieval.md) |
| Ranking quality (nDCG@10) | 0.724 | [`retrieval.md`](eval/results/retrieval.md) |
| Correct verdict (support / refute / not enough evidence) | **0.80** (240 of 300; 95% CI 0.751–0.841) | [`rag.md`](eval/results/rag.md), [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| Correct verdict, with the published label corrections applied | 0.837 | [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| Answers that cite at least one paper | 282 of 300 (94%) | [`rag.json`](eval/results/rag.json) |
| Answers that stick to the papers (faithfulness, LLM judge) | 0.97 | [`rag.md`](eval/results/rag.md) |
| Search time per query (laptop CPU) | 0.124 s | [`latency.md`](eval/results/latency.md) |
| Cost of a full 300-claim evaluation run | $0 | [`rag.json`](eval/results/rag.json) (`cost`) |
| Web search (Semantic Scholar + PubMed): right paper in top 5 / top 100 | 0.228 / 0.384 (local search: 0.766 / 0.965) | [`web_retrieval.md`](eval/results/web_retrieval.md) |

**Key findings**

- **Combining keyword and meaning search widens the pool of papers found** (top-100 recall 0.942
  → 0.965, p = 0.035). The extra papers are all on claims the dataset says have no evidence, so
  it does not find more evidence; it does beat keyword search alone on ranking (p = 0.0007).
- **A heavier re-ranking model didn't help.** The best one (`bge-reranker-base`) ties plain
  search (nDCG@10 +0.0001, p = 0.996) at 23.4 s per query, about 190× slower. It stays off.
- **Bigger, paid models were not better.** Paid GPT-6 Luna got fewer verdicts right, and paid
  gpt-oss-120b matched the free model's accuracy but cited papers far less often.
- **Some of the dataset's own answers are wrong.** A published 2026 audit corrects 11 of the 300
  test labels (about 4%) and calls 8 more debatable; scored against the corrections, accuracy is
  0.84. The headline keeps the original labels.
- **Tried, measured, not adopted:** an evidence-first prompt, two verifier models and a
  "second look" step; none beat the current pipeline
  ([details](docs/DETAILS.md#tried-and-measured-not-adopted)).

## Run it yourself

**You need:** `git` and `make`, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if
missing), ~2 GB of disk for the environment, models and index, and a free
[OpenRouter](https://openrouter.ai/) API key (only for answers; search works without one).
Optional: a free [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)
for web search (requests need an institutional email; without a key, web search uses a shared
pool that is often rate-limited).

```bash
git clone https://github.com/areebfazli/rag_search.git
cd rag_search
make install                  # Python 3.12 env + dependencies (uv sync)
cp .env.example .env          # then edit .env:
                              #   SSR_OPENROUTER_API_KEY=sk-or-...
                              #   SSR_S2_API_KEY=...        (optional, web search)
make index                    # one-time: download SciFact + embedding model, build the indices (~5 min)
make api                      # open http://localhost:8000
```

Don't run `make api` and `make eval`/`make index` at the same time: the embedded vector store
allows one process.

**Reproduce the numbers**

| Command | What it does |
|---|---|
| `make eval` | Search quality on the 300 test claims → `eval/results/retrieval.md` (no API key) |
| `SSR_RAG_N=all make eval-rag` | Full answer-quality run on 300 claims → `eval/results/rag.md`; ~670 free requests, about 1.5 hours. Add `SSR_RAG_CHECK_QUOTA=1` to stop up front if today's free quota is short |
| `make rag-compare A=a/rag.json B=b/rag.json` | Claim-by-claim significance test between two answer runs |
| `make rag-audit` | Re-score answers under the published label corrections (offline, no LLM) |
| `make web-eval` | Web search (Semantic Scholar + PubMed) vs local search (a few thousand requests at 1 per second; set the S2 key) |
| `make latency` | Per-query search speed for every configuration |
| `make test` / `make lint` | Unit tests / ruff |

OpenRouter's free tier allows 1,000 requests a day (on accounts that have bought at least $10 of
credits at some point; far fewer otherwise). If a run hits the limit it stops, and re-running
the same command resumes where it left off.

Groq, Ollama, any OpenAI-compatible endpoint, or a price-capped paid OpenRouter model can be used
instead of the free defaults; see [`.env.example`](.env.example).

## More detail

- [`docs/DETAILS.md`](docs/DETAILS.md): methodology, significance tests, per-label tables,
  confusion matrices, label-audit caveats, web search, limitations, next steps, operational notes
  and code layout.
- Raw results: [retrieval](eval/results/retrieval.md) ·
  [answers](eval/results/rag.md) · [label audit](eval/results/rag_label_audit.md) ·
  [web search](eval/results/web_retrieval.md) · [latency](eval/results/latency.md) ·
  [label-stratified analysis](eval/results/analysis.md) ·
  [judge consistency](eval/results/judge_agreement.md)

## License

[MIT](LICENSE)
