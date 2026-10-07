.PHONY: install index api eval eval-rag rag-compare rag-audit analysis latency web-eval test lint ci

install:  ## create env + install deps
	uv sync

index:  ## build dense (Qdrant) + BM25 indices (one-time)
	uv run python -m app.ingest.build_index

api:  ## serve UI + API at http://localhost:8000
	uv run uvicorn app.api.main:app --reload

eval:  ## reproduce the retrieval metrics table
	uv run python -m app.eval.retrieval_eval

analysis:  ## label-stratified re-score of the cached eval runs -> eval/results/analysis.{md,json}
	uv run --locked python -m app.eval.analysis

latency:  ## per-query retrieval latency for every eval config -> eval/results/latency.{md,json}
	uv run --locked python -m app.eval.latency

eval-rag:  ## RAG answer-quality eval (needs SSR_OPENROUTER_API_KEY in .env; $0 on the default free models; an optional paid generator costs ~$0.005-0.10/run depending on model and SSR_RAG_N, capped by SSR_RAG_MAX_SPEND_USD). Default SSR_RAG_N=50 -> data/eval_runs/; only SSR_RAG_N=all on the test split with the default models, and every claim scored, writes eval/results/rag.{md,json}
	uv run python -m app.eval.rag_eval

web-eval:  ## S2 web search vs local hybrid on gold qrels -> eval/results/web_retrieval.{md,json} (set SSR_S2_API_KEY; pooled S2 + PubMed, a few thousand requests at 1 req/s)
	uv run --locked python -m app.eval.web_eval

rag-compare:  ## paired McNemar of two RAG runs: make rag-compare A=path/rag.json B=path/rag.json [ARGS=--labels=audit]
	uv run --locked python -m app.eval.rag_compare $(A) $(B) $(ARGS)

rag-audit:  ## re-score rag.json under the SciFact label audit (offline, no LLM) -> eval/results/rag_label_audit.{md,json}; RUNS=other/rag.json writes next to each
	uv run --locked python -m app.eval.label_audit $(RUNS) $(ARGS)

# --locked: run against uv.lock exactly as committed, and fail (rather than silently
# re-resolve and rewrite it) if pyproject.toml has drifted. `make install` is the step
# that is allowed to update the lock.
test:  ## run unit tests
	uv run --locked pytest -q

lint:  ## lint
	uv run --locked ruff check app tests

ci:  ## what CI runs: locked install, then lint + test
	uv sync --locked
	$(MAKE) lint test
