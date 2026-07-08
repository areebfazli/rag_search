# Semantic Search + RAG Engine

Hybrid retrieval (**BM25 + dense embeddings**, fused with Reciprocal Rank Fusion), a **cross-encoder reranker**, and grounded **RAG** answers — with an evaluation harness that measures every stage on **gold relevance labels**.

## Pipeline

```
query ─┬─► BM25 (bm25s)        top-100 ─┐
       └─► dense (Qdrant)      top-100 ─┴─► RRF ─► cross-encoder rerank ─► top-8
                                                          │
                                    context + grounded prompt ─► LLM ─► answer + citations
```

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Lexical | `bm25s` |
| Fusion | Reciprocal Rank Fusion (hand-rolled) |
| Vector store | Qdrant (embedded local mode) |
| Reranker | `mixedbread-ai/mxbai-rerank-base-v2` |
| API | FastAPI |
| Retrieval eval | `ranx` on BEIR/SciFact gold qrels |
| RAG eval | `ragas` |

## Evaluation (headline artifact)

Measured on BEIR/SciFact — 300 test queries, gold relevance judgments (`uv run python -m app.eval.retrieval_eval`):

| Config | nDCG@10 | Recall@100 | MRR@10 | MAP@100 |
|---|---|---|---|---|
| BM25 | 0.6863 | 0.9127 | 0.6492 | 0.6439 |
| Dense (bge-small) | 0.7127 | 0.9417 | 0.6822 | 0.6736 |
| **Hybrid (RRF)** | **0.7241** | **0.9650** | **0.6886** | **0.6816** |
| Hybrid + rerank (MS-MARCO MiniLM) | 0.6715 | 0.9650 | 0.6367 | 0.6321 |

**Findings.** Hybrid fusion (BM25 + dense via RRF) is the best configuration: Recall@100 climbs 0.913 → 0.942 → 0.965 and nDCG@10 improves monotonically to 0.724 as dense retrieval and then fusion are added. The off-the-shelf MS-MARCO MiniLM cross-encoder *reduced* quality on SciFact (0.724 → 0.671 nDCG@10) — its own ranking ceiling on scientific-claim text sits below the hybrid fusion, so reordering with it hurts. Reranking only pays off with a domain-appropriate model (e.g. `bge-reranker`); a concrete reminder that every stage must be measured, not assumed. **Hybrid is therefore the default.**

## Grounded answers (RAG)

`GET /answer?q=...` runs the hybrid retrieval, then generates a grounded answer with an OpenAI-compatible LLM (Groq `llama-3.3-70b-versatile` by default; switch providers via `SSR_LLM_BASE_URL`). Answers cite sources as `[n]` mapped back to document ids, and the model is instructed to **abstain when the retrieved context lacks the evidence** rather than hallucinate.

Example — `/answer?q=Can aspirin reduce the risk of colorectal cancer?`:
> "Aspirin has been shown to reduce the risk of colorectal cancer [1][2][3] … a pooled analysis of four randomized trials showed a 34% reduction in 20-year colorectal cancer mortality [3]."

**Answer-quality eval** (`app/eval/rag_eval.py`) is an abstention-aware LLM-as-judge measuring faithfulness + context relevance. On a 70B-judged sample: faithfulness ≈ 0.8–1.0 on answered questions, with correct abstention when the corpus lacks evidence (see `eval/results/rag.md`).

## Quickstart

```bash
uv sync                                   # Python 3.12 env + deps
uv run python -m app.ingest.build_index   # build dense (Qdrant) + BM25 indices (one-time)
uv run uvicorn app.api.main:app --reload  # serve UI + API at http://localhost:8000
uv run python -m app.eval.retrieval_eval  # reproduce the metrics table above
uv run pytest                             # tests
```

Open <http://localhost:8000> for the search UI, or query the API directly:
`GET /search?q=...&mode=hybrid&top_k=8` — modes: `bm25`, `dense`, `hybrid`, `hybrid_rerank`.
(`make install|index|api|eval|test` wrap these.)
