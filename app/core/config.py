"""Central configuration. All settings are overridable via SSR_-prefixed env vars
(or a .env file), so the same code runs against Mistral's free tier (the default
generator), OpenRouter's free tier (the default judge) or any OpenAI-compatible endpoint
(Groq, a local Ollama, OpenAI) with no edits — just a different provider / base URL.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
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
    # BAAI/bge-reranker-base (278M) merely ties it at 23.4 s/query. Swap via
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
    #   "mistral"       -> generator only (the default). Base URL is hard-coded to
    #                      https://api.mistral.ai/v1, key = SSR_MISTRAL_API_KEY, model =
    #                      mistral_llm_model, with mistral_llm_fallback_model tried once
    #                      when the primary call fails with an availability error
    #                      (generator.LLMGenerator.generate).
    #   "openrouter"    -> base URL is hard-coded to https://openrouter.ai/api/v1 (a
    #                      leftover SSR_LLM_BASE_URL is ignored), key =
    #                      SSR_OPENROUTER_API_KEY, model = openrouter_llm_model /
    #                      openrouter_judge_model. FREE models only: every request must
    #                      pass llm_endpoints.enforce_spend_policy (`:free` ids, $0
    #                      max_price, no fallbacks).
    #   "openai_compat" -> any OpenAI-compatible endpoint (Groq by default; Ollama and
    #                      OpenAI work too): SSR_LLM_BASE_URL, key = SSR_LLM_API_KEY,
    #                      model = llm_model / judge_model. Refuses an openrouter.ai or
    #                      api.mistral.ai base URL, so SSR_LLM_API_KEY can never be sent
    #                      to either.
    # So each key only ever travels to its own provider, and a model name pinned in .env
    # (SSR_LLM_MODEL) cannot leak into another provider's run: the harnesses print which
    # settings were set but ignored for the provider in use (names only, never values).
    # The API keys are SecretStr, so repr(settings), tracebacks and logs show '**********';
    # only llm_endpoints unwraps them (secret_value) when it builds an LLMEndpoint.
    # A run with a non-default generator or judge is never canonical (rag_eval.output_dir).
    llm_provider: Literal["mistral", "openai_compat", "openrouter"] = "mistral"
    judge_provider: Literal["openai_compat", "openrouter"] = "openrouter"
    # Generator: Mistral's free tier ("Experiment" plan). ministral-14b-latest: 30 req/min
    # and 937.5k tokens/min on that tier, accepts temperature + max_tokens, and is not a
    # reasoning model (no hidden reasoning tokens, no reasoning field sent). A different
    # family from the Nemotron judge. mistral-small/medium/large are not available on the
    # free tier (429 "limit 0" / 403 tier_not_allowed). Chosen 2026-10 after the free
    # OpenRouter Ling generator was withdrawn upstream.
    mistral_api_key: SecretStr = SecretStr("")
    mistral_llm_model: str = "ministral-14b-latest"
    # Tried once, only when the primary call fails with an availability error (429 after
    # the SDK retries, 5xx, 403/404, timeout); "" disables it. The answer records which
    # model replied (model_used); a canonical rag_eval run refuses any fallback answer.
    mistral_llm_fallback_model: str = "ministral-8b-latest"
    openrouter_api_key: SecretStr = SecretStr("")
    # Generator on the openrouter provider (an alternative; not the default): a free model
    # (`:free` is appended if missing), sent with llm_endpoints.FREE_ROUTING ($0 max_price,
    # no fallbacks). NOTE: this Ling 3.0 Flash Sante id was the default generator until it
    # was withdrawn from OpenRouter (2026-10; requests return 404); it is kept as the
    # recorded id of the earlier canonical run. Set SSR_OPENROUTER_LLM_MODEL to a live
    # `:free` model to use this provider.
    openrouter_llm_model: str = "inclusionai/ling-3.0-flash-sante:free"
    # Judge: a free model too (`:free` appended if missing). Free tier: 20 req/min and,
    # with >= $10 credits ever bought, 1,000 req/day account-wide. Nemotron 3 Ultra: on a
    # 2026-09-24 bench of 10 free models it was the only one with 22/22 calls served (no
    # 429s), 100% parseable verdicts, and exact agreement with the previous qwen3.8-27b
    # judge on answered rows; qwen3.8-27b:free was 0/4 (upstream 429s).
    openrouter_judge_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"

    # --- LLM generation, "openai_compat" provider (any OpenAI-compatible endpoint) ---
    # Used only when llm_provider / judge_provider is "openai_compat". The defaults point
    # at Groq; set SSR_LLM_BASE_URL / SSR_LLM_MODEL for Ollama, OpenAI or another backend.
    # Key via SSR_LLM_API_KEY.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: SecretStr = SecretStr("")
    # Completion budget per generation call (sent as `max_tokens`, the name OpenRouter,
    # Groq, Ollama and OpenAI chat models all honour). On a reasoning model the HIDDEN
    # reasoning tokens are spent from this same budget before any answer text, so it must
    # cover reasoning + a 2-4 sentence cited answer + the verdict line. A reply that still
    # hits the cap is retried once at 2x (see generator.LLMGenerator.generate). 2048
    # (retry 4096) since the earlier free Ling generator, a reasoning model: at 1024 it
    # needed the retry on 36 of 50 eval claims and still truncated 7. The default
    # ministral generator does not reason, so it rarely comes near the budget.
    llm_max_completion_tokens: int = Field(default=2048, ge=16)
    # Reasoning effort, sent only when it resolves to a value — as `reasoning_effort` on
    # the openai_compat and mistral providers, and as OpenRouter's unified `reasoning: {"effort": ...}`
    # object on openrouter (docs: openrouter.ai/docs/use-cases/reasoning-tokens):
    #   "auto" / "" / "off" -> never sent: the model's own default applies (the default
    #                          ministral generator does not reason). A non-reasoning
    #                          model can reject the unknown parameter with a 400, so
    #                          swapping the model stays safe.
    #   any other           -> sent as-is to whatever model is configured (provider
    #                          vocabularies differ: gpt-oss takes low|medium|high).
    llm_reasoning_effort: str = "auto"
    # SSR_LLM_REASK: when a CLAIM's reply has no parseable verdict (cut off to nothing
    # after the retry, or prose without a verdict line), make exactly one extra
    # verdict-only call (prompts.reask_messages, generator.REASK_MAX_TOKENS) and take its
    # verdict. Never for a question. Measured on the 300 test claims (post-hoc re-ask
    # experiment, frozen on train, on the earlier Ling generator): verdict accuracy
    # 0.7767 -> 0.8000, 7 fixed / 0 broken, p=0.016; it fired on 15 of 300. A rag_eval run
    # with it off is never canonical.
    llm_reask: bool = True
    # RAG-eval judge on the "openai_compat" provider. Deliberately a different model FAMILY from the generator,
    # not just a smaller size: same-family judging compounds shared preferences.
    judge_model: str = "qwen/qwen3.8-27b"


def secret_value(value: SecretStr | str) -> str:
    """The raw string behind a key setting. Accepts a plain str as well, since attribute
    assignment on a BaseSettings instance (e.g. pytest monkeypatch) is not validated.
    Compare the result, never the SecretStr: SecretStr("") is truthy."""
    return value.get_secret_value() if isinstance(value, SecretStr) else value


settings = Settings()
