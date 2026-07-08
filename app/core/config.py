"""Central configuration. All settings are overridable via SSR_-prefixed env vars
(or a .env file), so the same code runs against a local Ollama or a hosted API
with no edits — just a different SSR_LLM_BASE_URL.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SSR_", env_file=".env", extra="ignore")

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge-small is asymmetric: prepend this to QUERIES only, never to documents.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_dim: int = 384

    # --- Reranker ---
    # CPU-friendly default (22M). mxbai-rerank-base-v2 is the GPU-quality upgrade
    # documented in the design notes; swap via SSR_RERANKER_MODEL.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector store (Qdrant local/embedded: on-disk path, or ":memory:") ---
    qdrant_location: str = "./data/qdrant"
    qdrant_collection: str = "corpus"
    bm25_path: str = "./data/bm25s"

    # --- Data (BEIR/SciFact ships gold qrels for honest, comparable metrics) ---
    corpus_dataset: str = "beir/scifact"
    eval_dataset: str = "beir/scifact/test"

    # --- Retrieval knobs ---
    dense_top_k: int = 100
    lexical_top_k: int = 100
    fusion_top_k: int = 50
    rerank_top_k: int = 8
    rrf_k: int = 60

    # --- LLM generation (OpenAI-compatible endpoint) ---
    # Default backend: Groq (free tier, OpenAI-compatible). Override via SSR_ env
    # vars for Ollama/another provider. Supply the key through SSR_LLM_API_KEY.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str = ""


settings = Settings()
