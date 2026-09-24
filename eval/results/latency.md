# Retrieval latency — BEIR/SciFact (300 test queries)

Per-query wall-clock of `SearchService.retrieve` (in-process; excludes HTTP and the API's retrieval lock), seconds, after 3 untimed warm-up queries per config.

| Config | Mode | top_k | n | mean | p50 | p90 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| BM25 | bm25 | 100 | 300 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| Dense (bge-small) | dense | 100 | 300 | 0.124 | 0.140 | 0.155 | 0.157 | 0.171 |
| Hybrid (RRF) | hybrid | 100 | 300 | 0.124 | 0.138 | 0.156 | 0.159 | 0.220 |
| BM25 | bm25 | 8 | 300 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| Dense (bge-small) | dense | 8 | 300 | 0.101 | 0.098 | 0.150 | 0.153 | 0.166 |
| Hybrid + rerank (MS-MARCO MiniLM) | hybrid_rerank | 100 | 40 | 3.936 | 3.775 | 5.368 | 5.626 | 5.690 |
| Hybrid + rerank (bge-reranker-base) | hybrid_rerank | 100 | 40 | 23.377 | 22.659 | 28.553 | 29.331 | 35.766 |

First-stage rows cover every test query; rerank rows a fixed sample of 40 (seed 13). Reranking scores the top 32 of 100 fused candidates (batch 8), exactly as in `retrieval.md`. API-shaped rows (top_k=8) are shown for bm25/dense only: for hybrid and rerank the API shape does identical work (candidate pool 100, fixed slice).

## Cold start (not in the table above)

- SearchService construction (embedder + Qdrant + BM25 load): 4.19 s
- Cross-encoder load (rerank_minilm): 4.27 s
- Cross-encoder load (rerank_bge): 6.16 s
- First call per config (first warm-up query): bm25 0.004 s, dense 0.220 s, hybrid 0.170 s, bm25_api 0.001 s, dense_api 0.029 s, rerank_minilm 1.787 s, rerank_bge 24.530 s

## Environment

- CPU: Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz — 4 physical / 8 logical cores; load average (1/5/15 min) (1.11, 1.25, 1.37) at start, (5.43, 5.39, 4.59) at end
- torch 2.13.0+cpu, 4 threads; Python 3.12.13; Linux-7.0.10-101.fc43.x86_64-x86_64-with-glibc2.42
- git 0277fb69cd57418243b11fff4b4790efa13229bc; run 2026-09-24T13:50:38+00:00; total wall-clock 21.8 min
