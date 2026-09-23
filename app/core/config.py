"""Central configuration. All settings are overridable via SSR_-prefixed env vars
(or a .env file), so the same code runs against a local Ollama or a hosted API
with no edits — just a different SSR_LLM_BASE_URL.
"""
from __future__ import annotations

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

    # --- LLM generation (OpenAI-compatible endpoint) ---
    # Default backend: Groq (free tier, OpenAI-compatible). Override via SSR_ env
    # vars for Ollama/another provider. Supply the key through SSR_LLM_API_KEY.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    # Completion budget per generation call (sent as `max_tokens`, the name Groq, Ollama
    # and OpenAI chat models all honour). On a reasoning model the HIDDEN reasoning
    # tokens are spent from this same budget before any answer text, so it must cover
    # reasoning + a 2-4 sentence cited answer + the verdict line — the old 400 let
    # gpt-oss-120b's default (medium) reasoning truncate 7 of 50 eval answers, two to
    # empty strings. A reply that still hits the cap is retried once at 2x (see
    # generator.LLMGenerator.generate).
    llm_max_completion_tokens: int = Field(default=1024, ge=16)
    # Reasoning effort, sent as `reasoning_effort` only when it resolves to a value:
    #   "auto"     -> "medium" for gpt-oss models (Groq and Ollama both accept it), and NOT
    #                 sent for any other model — a non-reasoning model can reject the
    #                 unknown parameter with a 400, so swapping SSR_LLM_MODEL stays safe.
    #   "" / "off" -> never sent.
    #   any other  -> sent as-is to whatever model is configured (provider vocabularies
    #                 differ: gpt-oss takes low|medium|high, qwen3 also none|default).
    # "medium" (gpt-oss's own default) rather than "low": on a live spot check, "low"
    # dropped the required Verdict line on a claim that "medium" answered correctly,
    # and verdict compliance is scored. Medium's longer reasoning (~900 tokens seen) is
    # why llm_max_completion_tokens is 1024 with a one-shot 2x retry, not the old 400.
    llm_reasoning_effort: str = "auto"
    # RAG-eval judge. Deliberately a different model FAMILY from the generator, not just
    # a smaller size: same-family judging compounds shared preferences, and a separate
    # model also draws on a separate provider rate-limit bucket.
    judge_model: str = "qwen/qwen3.8-27b"


settings = Settings()
