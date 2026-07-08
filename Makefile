.PHONY: install index api eval test lint

install:  ## create env + install deps
	uv sync

index:  ## build dense (Qdrant) + BM25 indices (one-time)
	uv run python -m app.ingest.build_index

api:  ## serve UI + API at http://localhost:8000
	uv run uvicorn app.api.main:app --reload

eval:  ## reproduce the retrieval metrics table
	uv run python -m app.eval.retrieval_eval

test:  ## run unit tests
	uv run pytest -q

lint:  ## lint
	uv run ruff check app tests
