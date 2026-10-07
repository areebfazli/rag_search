"""Central configuration. All settings are overridable via SSR_-prefixed env vars
(or a .env file), so the same code runs against OpenRouter's free tier, Groq, a local
Ollama or another hosted API with no edits — just a different provider / base URL.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
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

    # --- Semantic Scholar web search (optional modes `web` / `hybrid_web`) ---
    # Never used by the default mode. Without a key, requests go to S2's shared public
    # pool ("1000 requests per second shared among all unauthenticated users", and "may
    # be further throttled during periods of heavy use" — it 429s often); a key, sent as
    # the `x-api-key` header only when set, has an "introductory rate limit of 1 RPS on
    # all endpoints" (semanticscholar.org/product/api). The key only ever goes to
    # api.semanticscholar.org (the base URL is fixed in app/retrieve/semantic_scholar.py).
    s2_api_key: str = ""
    # Process-wide request spacing for every S2 call, retries included — the documented
    # per-key rate, and a polite one for the shared pool. Per process: with N uvicorn
    # workers, set it to 1/N.
    s2_rate_per_s: float = Field(default=1.0, gt=0)
    s2_timeout_s: float = Field(default=10.0, gt=0)
    # Retries after the first attempt, on 429 / 5xx / network errors, honouring
    # Retry-After. The API caps this at 1 (SemanticScholarRetriever.for_api).
    s2_max_retries: int = Field(default=3, ge=0)
    # API only: the longest a request waits for a rate-limiter slot or a single retry
    # sleep before giving up — hybrid_web then degrades to local results, web returns
    # 503. Bounded so a flood of web-mode requests cannot park every worker thread.
    s2_max_wait_s: float = Field(default=5.0, ge=0)
    # Raw-response cache (gitignored under data/). web_eval always uses it; the API only
    # when s2_api_cache is on — arbitrary public queries would otherwise grow it unbounded.
    s2_cache_dir: str = "./data/s2_cache"
    s2_api_cache: bool = False
    # Web-mode pipeline (app/retrieve/web_search.py), both deterministic and LLM-free:
    # send S2 a keyword rewrite of the claim (a 6-term query + a 3-term one, results
    # pooled) instead of the raw sentence, and re-rank the S2 candidates locally with
    # RRF(bge-small dense, BM25 over the candidates). Defaults follow web_eval's measured
    # result (eval/results/web_retrieval.md): on the 300 SciFact test claims (rules
    # frozen on 100 train claims), rewrite+rerank lifts gold-paper Recall@5 from 0.040 to
    # 0.185 and nDCG@10 from 0.035 to 0.159 vs the raw-claim query (45 better / 1 worse,
    # p<0.0001). Rerank alone does nothing (p=1.0); the rewrite is what lets it work.
    s2_query_rewrite: bool = True
    s2_rerank: bool = True
    # Extra web CANDIDATE sources (web_search.WebSearch.search_pooled), pooled with the S2
    # rewrite results and re-ranked together by the same local RRF. Chosen on 100 train
    # claims (app/eval/web_pool_eval.py), frozen, then measured once on the 300 test claims
    # (eval/results/web_retrieval.md, "Pooled ... (live)"): vs rewrite+rerank, Recall@5
    # 0.185 -> 0.228 (p=0.012), Recall@100 0.244 -> 0.384 (46 better / 0 worse, p<0.0001),
    # nDCG@10 0.159 -> 0.203 (p=0.002). Hence ON by default:
    #   web_pubmed: PubMed E-utilities Best Match (app/retrieve/pubmed.py; keyless — NCBI
    #     allows 3 req/s per IP; `tool` param only, no email), on web_pubmed_queries
    #   web_snippets: S2 snippet search (app/retrieve/s2_extra.py) on web_snippet_queries;
    #     papers quoting the claim verbatim or mentioning SciFact are excluded
    #   web_dense_cap: embed/re-rank only the top N of a cheap BM25 pre-rank (0 = all) —
    #     embedding is ~60 ms per candidate on CPU, the main latency cost of a big pool
    # Off by default (more S2 requests for no significant gain):
    #   s2_multi_query: extra deterministic S2 keyword queries (query_rewrite.multi_queries);
    #     the "offline" test row (3 of them + rewrite snippets) vs live: R@100 +0.013, p=0.17
    #   web_citation_seeds: references+citations of the top-N pre-ranked S2 hits (train
    #     only: +0 Recall@100 on top of the live pool, ~2-3 extra S2 requests)
    s2_multi_query: int = Field(default=0, ge=0, le=5)
    web_pubmed: bool = True
    web_pubmed_queries: str = "rewrite,claim"  # comma list of "rewrite" / "claim"
    web_snippets: bool = True
    web_snippet_queries: str = "claim"
    web_citation_seeds: int = Field(default=0, ge=0, le=10)
    web_citation_cap: int = Field(default=30, ge=1, le=1000)
    web_dense_cap: int = Field(default=50, ge=0)
    pubmed_rate_per_s: float = Field(default=2.0, gt=0, le=3.0)
    # API guards for web modes (web / hybrid_web) only, on top of the per-endpoint limits
    # (/search 30/minute, /answer 10/minute), which still apply. One web request fans out
    # to several S2 + PubMed calls, so: a stricter per-IP limit, ONE bucket shared by
    # /search and /answer (a limits-library string, e.g. "6/minute"), and a process-wide
    # cap on concurrent web retrievals — when every slot is busy the request gets 503 +
    # Retry-After at once instead of queueing a worker thread. Per process, like the
    # rate limiter's in-memory storage. Local modes never touch either.
    web_rate_limit: str = "6/minute"
    web_max_concurrent: int = Field(default=2, ge=1)

    @field_validator("web_rate_limit")
    @classmethod
    def _valid_web_rate_limit(cls, v: str) -> str:
        """Fail at startup, not as a 500 on the first web request."""
        from limits import parse

        try:
            parse(v)
        except ValueError as e:
            raise ValueError(f"SSR_WEB_RATE_LIMIT is not a rate limit string: {v!r}") from e
        return v

    # API web-request bounds (WebSearch / service.web_extras). The eval sets neither, so
    # the committed web numbers (eval/results/web_retrieval.*) were measured WITHOUT them:
    # no deadline, and dense cap 50 (web_pool_live) / 100 (web_pool_offline) per
    # web_eval.POOL_VARIANTS — never the API's 30. API results can therefore differ.
    #   web_deadline_s: per-request budget for a pooled web search (0 = none); an extra
    #     source still pending or running when it runs out is abandoned and reported in
    #     `warnings`, and its HTTP calls stop. The base S2 failure path never waits.
    #     Unmeasured on the gold qrels (API-only latency guard).
    #   web_api_dense_cap: the API embeds at most this many candidates per request (the
    #     rerank runs under the global retrieval lock). Unmeasured at 30: the eval rows
    #     use 50 / 100. 0 = no extra cap.
    web_deadline_s: float = Field(default=20.0, ge=0)
    web_api_dense_cap: int = Field(default=30, ge=0)
    # NCBI E-utilities identification (NBK25497, https://www.ncbi.nlm.nih.gov/books/
    # NBK25497/): `email` "should be a complete and valid e-mail address of the software
    # developer", and an `api_key` raises the limit from 3 to 10 requests/second. This
    # app still caps PubMed at 3 req/s (pubmed_rate_per_s, default 2, bound le=3.0): to
    # use the key's higher limit, set SSR_PUBMED_RATE_PER_S and raise that bound (the
    # PubMedSource limiter itself allows up to 10 with a key). Both
    # optional and empty by default; sent only to eutils.ncbi.nlm.nih.gov, only when set,
    # and never part of a cache key or cache file.
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    # API only: drop S2 snippets that look like a dataset dump ("Claim: ..." / "Evidence:"
    # formatting) whose claim text overlaps a SciFact claim (token Jaccard >= 0.6)
    # (s2_extra.DatasetSnippetFilter). OFF in web_eval (the committed numbers). Measured
    # offline on the cached 300-claim test pools (2026-10-05, filter on vs off): it drops
    # 2 snippet papers (2 claims) from web_pool_live and 4 drops covering 3 distinct
    # papers across 4 claims from web_pool_offline, none of them gold; R@5 / R@10 / R@100 / nDCG@10 are identical
    # (live .2283 / .2661 / .3843 / .2030). A contamination guard, not a recall change.
    web_snippet_dataset_filter: bool = True

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
    # Paid paths are still available (opt-in, generator only): SSR_OPENROUTER_LLM_MODEL=
    # openai/gpt-oss-120b (the model the earlier committed numbers were generated with,
    # ~$0.0003/query) or openai/gpt-6-luna are allowlisted, and each is always sent with
    # its own pinned, price-capped, no-fallback entry in llm_endpoints.PAID_ROUTES. A run
    # with a non-default generator or judge is never canonical (rag_eval.output_dir).
    openrouter_llm_model: str = "inclusionai/ling-3.0-flash-sante:free"
    # Judge: must be a `:free` id (appended if missing); a paid judge is refused. Free
    # tier: 20 req/min and, with >= $10 credits ever bought, 1,000 req/day account-wide.
    # Nemotron 3 Ultra: on a 2026-09-24 bench of 10 free models it was the only one with
    # 22/22 calls served (no 429s), 100% parseable verdicts, and exact agreement with the
    # previous qwen3.8-27b judge on answered rows; qwen3.8-27b:free was 0/4 (upstream 429s).
    # Backup: inclusionai/ling-3.0-flash-sante:free (also 22/22, different family).
    openrouter_judge_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    # The ONLY paid model ids that may ever be sent to OpenRouter (generator role only),
    # and each must also have a routing entry in llm_endpoints.PAID_ROUTES (an id with
    # none is refused). A configured generator id outside it is normalised to its `:free`
    # variant; any other non-`:free` id reaching a request is refused before any network
    # call.
    openrouter_paid_model_allowlist: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-6-luna")
    # Hard per-run spend ceiling for rag_eval (USD, OpenRouter-reported cost; unreported
    # cost counts at the generator's own max_price caps). The run stops, writing nothing,
    # before a query that could take the total past it.
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
    #   "auto"     -> "medium" for gpt-oss models (Groq and Ollama both accept it) and for
    #                 openai/gpt-6-luna (a reasoning model; OpenRouter lists `reasoning`), NOT
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
    # SSR_LLM_REASK: when a CLAIM's reply has no parseable verdict (cut off to nothing
    # after the retry, or prose without a verdict line), make exactly one extra
    # verdict-only call (prompts.reask_messages, generator.REASK_MAX_TOKENS) and take its
    # verdict. Never for a question. Measured on the 300 test claims (post-hoc re-ask
    # experiment, frozen on train): verdict accuracy 0.7767 -> 0.8000, 7 fixed /
    # 0 broken, p=0.016; it fired on 15 of 300. With a paid generator it costs one more
    # paid call on those replies. A rag_eval run with it off is never canonical.
    llm_reask: bool = True
    # RAG-eval judge on the "groq" provider (openrouter_judge_model is the same model's
    # free OpenRouter variant). Deliberately a different model FAMILY from the generator,
    # not just a smaller size: same-family judging compounds shared preferences.
    judge_model: str = "qwen/qwen3.8-27b"


settings = Settings()
