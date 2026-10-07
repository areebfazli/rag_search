# Scientific Claim Checker (Semantic Search with RAG Engine)

[![CI](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml/badge.svg)](https://github.com/areebfazli/rag_search/actions/workflows/ci.yml)

You type a scientific or health claim, such as *"Vitamin D deficiency increases the risk of
multiple sclerosis"*. The tool finds the research papers most relevant to it, reads them, and
says whether they **support** the claim, **refute** it, or **don't contain enough evidence**,
citing the exact papers it relied on. It searches a built-in collection of 5,183 biomedical
paper abstracts (the SciFact benchmark) and can optionally search the web through Semantic
Scholar and PubMed. It runs on free AI models by default, so using it costs $0. Search and
verdict numbers below are scored against expert annotators' answers; faithfulness is scored by a
second AI model; speed and cost are measured.

![Search UI: results for "Vitamin D deficiency is associated with increased risk of multiple sclerosis" over 5,183 SciFact abstracts](docs/ui.png)

## How it works

Keyword search (BM25) and meaning-based search (`bge-small` embeddings) run side by side and their
rankings are combined (Reciprocal Rank Fusion). The top 5 abstracts go to a free language model
(Ling 3.0 Flash Sante) told to use only them; it gives a verdict with `[n]` citations, or holds
back when the papers lack evidence. A second model (Nemotron 3 Ultra) checks faithfulness.

## Results

All on the 300 SciFact test claims, default settings. The web row used the evaluation's web
settings, which differ from the API's, and predates the keyword-rewrite fixes in commit aa98231,
which change the query for 56 of the 300 claims and have not been re-measured on them yet
([details](docs/DETAILS.md#web-search-semantic-scholar-and-pubmed)).

| What is measured | Result | Source |
|---|---|---|
| Right paper among the top 10 results (Recall@10) | 0.855 | [`web_retrieval.md`](eval/results/web_retrieval.md) (local hybrid row) |
| Right paper among the top 100 results (Recall@100) | 0.965 | [`retrieval.md`](eval/results/retrieval.md) |
| Ranking quality (nDCG@10) | 0.724 | [`retrieval.md`](eval/results/retrieval.md) |
| Correct verdict (support / refute / not enough evidence) | **0.80** (239 of 300 = 0.797; 95% CI 0.748–0.838) | [`rag.md`](eval/results/rag.md), [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| Correct verdict under published label corrections (one-sided check) | 0.830 | [`rag_label_audit.md`](eval/results/rag_label_audit.md) |
| Answers that cite at least one paper | 282 of 300 (94%) | [`rag.json`](eval/results/rag.json) |
| Answers that stick to the papers (faithfulness, LLM judge, 193 answers) | 0.98 | [`rag.md`](eval/results/rag.md) |
| Search time per query (laptop CPU) | 0.124 s | [`latency.md`](eval/results/latency.md) |
| Cost of a full 300-claim evaluation run | $0 | [`rag.json`](eval/results/rag.json) (`cost`) |
| Web search (Semantic Scholar + PubMed): right paper in top 5 / top 100 | 0.228 / 0.384 (local search: 0.766 / 0.965) | [`web_retrieval.md`](eval/results/web_retrieval.md) |

**Key findings**

- **Combining keyword and meaning search may widen the pool of papers found** (top-100 recall
  0.942 → 0.965, but on only 11 claims that differ, 9 better and 2 worse: suggestive, not
  significant, sign test p = 0.065). Those extra papers are all on claims the dataset says have
  no evidence, so it does not find more evidence; it clearly beats keyword search alone on
  ranking (p = 0.0007).
- **A heavier re-ranking model didn't help.** The best one (`bge-reranker-base`) ties plain
  search (nDCG@10 +0.0001, p = 0.996) at 23.4 s per query, about 190× slower. It stays off.
- **Paid models were not better.** Paid GPT-6 Luna got fewer verdicts right (0.71 vs 0.78 for
  the free model, both without the verdict re-ask; p = 0.012). Paid gpt-oss-120b was
  indistinguishable on accuracy over the 50 claims tested (too few to rule a difference in or
  out) and cited papers far less often.
- **Some of the dataset's own answers are wrong.** A published 2026 audit corrects 11 of the 300
  test labels (about 4%) and calls 8 more debatable; scored against the corrections, accuracy is
  0.83 (0.84 with the debatable ones left out). The audit skipped the no-evidence claims, and of
  the 11 changes 10 favour the model and none goes against it, so this is a one-sided check; the
  headline keeps the original labels. A separate secondary check (three blind LLM annotators, not
  a human) sided with the model on 25 of 31 NEI-gold disagreements; see
  [details](docs/DETAILS.md#blind-re-labelling-of-the-nei-disagreements-llm-annotators-secondary).
- **Tried, measured, not adopted:** an evidence-first prompt, two verifier models, a
  "second look" step, majority voting and a wording prompt variant; none beat the current pipeline
  ([details](docs/DETAILS.md#tried-and-measured-not-adopted)).
- **Caveat:** the fixes behind 0.80 were chosen partly by looking at these same 300 test claims,
  so they are no longer a clean held-out set, and the last step (0.78 → 0.80) is borderline
  ([details](docs/DETAILS.md#how-much-to-trust-the-test-set)).

## Run it yourself

**You need:** `git` and `make`, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if
missing), ~2 GB of disk for the environment, models and index, and a free
[OpenRouter](https://openrouter.ai/) API key (only for answers; search works without one).
Optional: a free [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)
for web search (requests need an institutional email; without a key, web search uses a shared
pool that is often rate-limited), and your email as `SSR_NCBI_EMAIL`, which NCBI asks PubMed
users to send.

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
| `SSR_RAG_N=all make eval-rag` | Full answer-quality run on 300 claims → `eval/results/rag.md`; ~670 free requests, about 1.5 hours. Add `SSR_RAG_CHECK_QUOTA=1` to stop up front if today's free quota is short. Only this full run on the default models writes there (and only once every claim is scored); a smaller or modified run, including the default `make eval-rag` (50 claims), goes to `data/eval_runs/` |
| `make rag-compare A=a/rag.json B=b/rag.json` | Claim-by-claim significance test between two answer runs |
| `make rag-audit` | Re-score answers under the published label corrections (offline, no LLM) |
| `make web-eval` | Web search (Semantic Scholar + PubMed) vs local search (a few thousand requests at 1 per second; set the S2 key) |
| `make latency` | Per-query search speed for every configuration |
| `make test` / `make lint` | Unit tests / ruff |

OpenRouter's free models cost $0, but the free tier allows 1,000 requests a day only on accounts
that have bought at least $10 of credits at some point (far fewer otherwise). If a run hits the
limit it stops, and re-running the same command resumes where it left off.

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
