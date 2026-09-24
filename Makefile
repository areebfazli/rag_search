.PHONY: install index api eval eval-rag rag-compare analysis latency test lint ci

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

eval-rag:  ## RAG answer-quality eval (needs SSR_OPENROUTER_API_KEY in .env; ~$0.01-0.02/run)
	uv run python -m app.eval.rag_eval

rag-compare:  ## paired McNemar of two RAG runs: make rag-compare A=path/rag.json B=path/rag.json
	uv run --locked python -m app.eval.rag_compare $(A) $(B)

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
