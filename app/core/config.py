"""Central configuration. All settings are overridable via SSR_-prefixed env vars
(or a .env file), so the same code runs against OpenRouter's free tier, Groq, a local
Ollama or another hosted API with no edits — just a different provider / base URL.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SSR_", env_file=".env", extra="ignore")

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge-small is asymmetric: prepend this to QUERIES only, never to documents.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Reranker ---
    # CPU-friendly default (22M). Note the eval found that reranking does NOT pay off on
    # SciFact with either model tested: this one costs 0.027 nDCG@10 vs plain hybrid, and
    # BAAI/bge-reranker-base (278M) merely ties it at 32.3 s/query. Swap via
    # SSR_RERANKER_MODEL, but measure before assuming a bigger reranker helps.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector store (Qdrant local/embedded: on-disk path, or ":memory:") ---
    qdrant_location: str = "./data/qdrant"
    qdrant_collection: str = "corpus"
    bm25_path: str = "./data/bm25s"

    # --- Data (BEIR/SciFact ships gold qrels for honest, comparable metrics) ---
    corpus_dataset: str = "beir/scifact"
    eval_dataset: str = "beir/scifact/test"

    # --- API ---
    # Trust X-Forwarded-For for rate-limit keying. Enable ONLY behind a proxy you
    # control (Fly/Render), which sets XFF and strips client-supplied values —
    # trusting it while directly exposed lets clients spoof their IP to evade limits.
    trust_proxy: bool = False
    # How many proxies you actually run in front of this service. Each appends the
    # peer it saw, so the real client sits `trusted_proxy_hops` from the right of the
    # X-Forwarded-For list. This MUST equal your real hop count:
    #   too low  -> keys on one of your own proxies: every user behind it shares a bucket.
    #   too high -> indexes past your proxies' entries into the part of the list the
    #               CLIENT supplied. The client then picks its own key and can rotate it
    #               per request, so the rate limit is bypassed entirely. Only the
    #               rightmost `hops` entries are trustworthy; the count is the trust
    #               boundary, so over-stating it is a security bug, not a mis-tuning.
    # The default of 1 (plain rightmost) is the safe value and cannot be over-indexed.
    trusted_proxy_hops: int = Field(default=1, ge=1)

    # --- Retrieval knobs ---
    # dense_top_k is the first-stage candidate depth used by BOTH retrievers in
    # hybrid mode (SearchService.candidate_k); rerank_top_k is the default number
    # of results returned when the caller doesn't pass top_k.
    dense_top_k: int = Field(default=100, ge=1)
    rerank_top_k: int = Field(default=8, ge=1)
    # RRF divides by (rrf_k + rank); a non-positive value divides by zero at rank
    # |rrf_k| instead of failing at startup.
    rrf_k: int = Field(default=60, ge=1)
    # A cross-encoder scores every (query, passage) pair with a full forward pass, so
    # its cost is linear in the candidate count — reranking the whole 100-doc fused
    # pool costs ~20s on CPU and, held under the retrieval lock, is a trivial DoS
    # vector. Rerank only this top slice; the tail keeps its fused order, so Recall
    # at depths beyond the slice is unchanged. Constrained to >=1: a 0 would silently
    # turn hybrid_rerank into plain hybrid.
    rerank_candidates: int = Field(default=32, ge=1)
    # Cross-encoder batch size. Peak activation memory scales with batch x seq_len, so
    # a small batch keeps the footprint flat on memory-constrained hosts; on CPU the
    # throughput cost is minor, and it is a large win when the alternative is swapping.
    rerank_batch_size: int = Field(default=8, ge=1)

    # --- LLM providers (generator and RAG-eval judge) ---
    # Each role picks a provider; the PROVIDER decides the base URL, the API key and
    # which model setting applies (app/core/llm_endpoints.py is the single resolver):
    #   "openrouter" -> base URL is hard-coded to https://openrouter.ai/api/v1 (a
    #                   leftover SSR_LLM_BASE_URL is ignored), key = SSR_OPENROUTER_API_KEY,
    #                   model = openrouter_llm_model / openrouter_judge_model. Every request
    #                   must pass llm_endpoints.enforce_spend_policy: `:free` ids, or an
    #                   allowlisted paid GENERATOR id with pinned, capped routing.
    #   "groq"       -> the generic OpenAI-compatible endpoint: SSR_LLM_BASE_URL (Groq
    #                   by default; Ollama/OpenAI work too), key = SSR_LLM_API_KEY,
    #                   model = llm_model / judge_model. Refuses an openrouter.ai base URL,
    #                   so the Groq key can never be sent to OpenRouter.
    # So each key only ever travels to its own provider, and a Groq model name pinned in
    # .env (SSR_LLM_MODEL) cannot leak into an OpenRouter run: the harnesses print which
    # settings were set but ignored for the provider in use.
    llm_provider: Literal["groq", "openrouter"] = "openrouter"
    judge_provider: Literal["groq", "openrouter"] = "openrouter"
    openrouter_api_key: str = ""
    # Generator: FREE by default, so the whole project runs on free models. An id that is
    # not in the paid allowlist below gets `:free` appended (as the judge does) and is sent
    # with llm_endpoints.FREE_ROUTING ($0 max_price, no fallbacks) — it never touches the
    # allowlist or PAID_ROUTING. Ling 3.0 Flash Sante: on the 2026-09-24 bench of 10 free
    # models it served 22/22 calls with the fastest p50 (1.1 s); it is health/medicine-tuned
    # (SciFact is biomedical), has no expiration date, and is a different family from the
    # Nemotron judge. It reasons by default with no parameter sent (108-1,847 hidden
    # reasoning tokens per claim on a 2026-09-24 smoke test), and `reasoning.effort=low`
    # did not shorten it, so llm_reasoning_effort="auto" sends it nothing; the 2048-token
    # budget + one 2x retry below absorbs the long tail (see llm_max_completion_tokens).
    # The paid path is still available: SSR_OPENROUTER_LLM_MODEL=openai/gpt-oss-120b (the
    # model the earlier committed numbers were generated with) is allowlisted, and is always
    # sent with the pinned, price-capped, no-fallback PAID_ROUTING (~$0.0003/query).
    openrouter_llm_model: str = "inclusionai/ling-3.0-flash-sante:free"
    # Judge: must be a `:free` id (appended if missing); a paid judge is refused. Free
    # tier: 20 req/min and, with >= $10 credits ever bought, 1,000 req/day account-wide.
    # Nemotron 3 Ultra: on a 2026-09-24 bench of 10 free models it was the only one with
    # 22/22 calls served (no 429s), 100% parseable verdicts, and exact agreement with the
    # previous qwen3.8-27b judge on answered rows; qwen3.8-27b:free was 0/4 (upstream 429s).
    # Backup: inclusionai/ling-3.0-flash-sante:free (also 22/22, different family).
    openrouter_judge_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    # The ONLY paid model ids that may ever be sent to OpenRouter (generator role only).
    # A configured generator id outside it is normalised to its `:free` variant; any other
    # non-`:free` id reaching a request is refused before any network call.
    openrouter_paid_model_allowlist: tuple[str, ...] = ("openai/gpt-oss-120b",)
    # Hard per-run spend ceiling for rag_eval (USD, OpenRouter-reported cost; unreported
    # cost counts at the max_price caps). The run stops, writing nothing, before a query
    # that could take the total past it.
    rag_max_spend_usd: float = Field(default=0.10, ge=0.0)

    # --- LLM generation, "groq" provider (generic OpenAI-compatible endpoint) ---
    # Used only when llm_provider / judge_provider is "groq". Override via SSR_ env vars
    # for Ollama/another OpenAI-compatible backend. Key via SSR_LLM_API_KEY.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    # Completion budget per generation call (sent as `max_tokens`, the name Groq, Ollama
    # and OpenAI chat models all honour). On a reasoning model the HIDDEN reasoning
    # tokens are spent from this same budget before any answer text, so it must cover
    # reasoning + a 2-4 sentence cited answer + the verdict line — the old 400 let
    # gpt-oss-120b's default (medium) reasoning truncate 7 of 50 eval answers, two to
    # empty strings. A reply that still hits the cap is retried once at 2x (see
    # generator.LLMGenerator.generate). 2048 (retry 4096) since the free Ling generator:
    # it reasons far longer than gpt-oss, and at 1024/2048 it needed the retry on 36 of
    # 50 eval claims and still truncated 7. Free, so the larger budget costs nothing;
    # for paid gpt-oss it only raises the worst case, which SSR_RAG_MAX_SPEND_USD caps.
    llm_max_completion_tokens: int = Field(default=2048, ge=16)
    # Reasoning effort, sent only when it resolves to a value — as `reasoning_effort` on
    # the groq provider, and as OpenRouter's unified `reasoning: {"effort": ...}` object
    # on openrouter (docs: openrouter.ai/docs/use-cases/reasoning-tokens):
    #   "auto"     -> "medium" for gpt-oss models (Groq and Ollama both accept it), and NOT
    #                 sent for any other model — a non-reasoning model can reject the
    #                 unknown parameter with a 400, so swapping SSR_LLM_MODEL stays safe.
    #                 The default free Ling generator gets nothing: it reasons on its own.
    #   "" / "off" -> never sent.
    #   any other  -> sent as-is to whatever model is configured (provider vocabularies
    #                 differ: gpt-oss takes low|medium|high, qwen3 also none|default).
    # "medium" (gpt-oss's own default) rather than "low": on a live spot check, "low"
    # dropped the required Verdict line on a claim that "medium" answered correctly,
    # and verdict compliance is scored. Medium's longer reasoning (~900 tokens seen) is
    # why the budget grew from the old 400 (now 2048 with a one-shot 2x retry).
    llm_reasoning_effort: str = "auto"
    # RAG-eval judge on the "groq" provider (openrouter_judge_model is the same model's
    # free OpenRouter variant). Deliberately a different model FAMILY from the generator,
    # not just a smaller size: same-family judging compounds shared preferences.
    judge_model: str = "qwen/qwen3.8-27b"


settings = Settings()
