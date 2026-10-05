.PHONY: install index api eval eval-rag rag-compare rag-audit verify-eval verify-export verify-train verify-combine analysis latency web-eval test lint ci

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

eval-rag:  ## RAG answer-quality eval (needs SSR_OPENROUTER_API_KEY in .env; $0 on the default free models; an optional paid generator costs ~$0.005-0.10/run depending on model and SSR_RAG_N, capped by SSR_RAG_MAX_SPEND_USD)
	uv run python -m app.eval.rag_eval

web-eval:  ## S2 web search vs local hybrid on gold qrels -> eval/results/web_retrieval.{md,json} (set SSR_S2_API_KEY; pooled S2 + PubMed, a few thousand requests at 1 req/s)
	uv run --locked python -m app.eval.web_eval

verify-eval:  ## NLI (DeBERTa) verdicts, no LLM -> data/eval_runs/verify_*/ (SSR_RAG_DATASET, SSR_RAG_N; ARGS=--tune on train only)
	uv run --locked python -m app.eval.verify_eval $(ARGS)

verify-export:  ## verifier training data + Kaggle GPU notebook -> data/kaggle/ (see data/kaggle/README.md)
	HF_HUB_OFFLINE=1 uv run --locked python -m app.verify.kaggle_export $(ARGS)

verify-train:  ## fine-tune the PubMedBERT evidence verifier on CPU (~3 h, resumable; Kaggle GPU is the usual route) -> data/models/verifier/model/
	mkdir -p data/models/verifier
	HF_HUB_OFFLINE=1 nohup uv run --locked python -m app.verify.train $(ARGS) > data/models/verifier/nohup.out 2>&1 &
	@echo "training in the background: tail -f data/models/verifier/train.log"

verify-combine:  ## verifier x Ling rules R0-R4: tune on the 100 held-out train claims, then test once on 300 (STAGE=all|validate|tune|test) -> data/eval_runs/verify_*/
	HF_HUB_OFFLINE=1 uv run --locked python -m app.eval.verify_combine $(STAGE) $(ARGS)

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
