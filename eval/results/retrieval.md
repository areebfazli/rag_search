# Retrieval evaluation — BEIR/SciFact (300 test queries)

| Config | nDCG@10 | Recall@100 | MRR@10 | MAP@100 |
|---|---|---|---|---|
| BM25 | 0.6863 | 0.9127 | 0.6492 | 0.6439 |
| Dense (bge-small) | 0.7127 | 0.9417 | 0.6822 | 0.6736 |
| Hybrid (RRF) | 0.7241 | **0.9650** | 0.6886 | 0.6816 |
| Hybrid + rerank (MS-MARCO MiniLM) | 0.6975 | **0.9650** | 0.6632 | 0.6558 |
| Hybrid + rerank (bge-reranker-base) | **0.7242** | **0.9650** | **0.6901** | **0.6834** |

Reranked slice: top-32 of 100 fused candidates (the tail keeps its fused order, so Recall@100 is unchanged by reranking).

## Significance

Paired two-sided Student's t-test on per-query nDCG@10, p < 0.05.

| Comparison (nDCG@10) | Δ | p | W/T/L | Significant |
|---|---|---|---|---|
| hybrid vs dense — Does fusing BM25 into dense retrieval help? | +0.0114 | 0.2578 | 53/213/34 | no |
| hybrid vs bm25 — Does the hybrid beat lexical alone? | +0.0378 | 0.0007 | 72/202/26 | yes |
| rerank_minilm vs hybrid — Does the MS-MARCO cross-encoder help? | -0.0266 | 0.0555 | 45/192/63 | no |
| rerank_bge vs hybrid — Does a domain-appropriate cross-encoder help? | +0.0001 | 0.9964 | 48/196/56 | no |
| rerank_bge vs rerank_minilm — Does the reranker choice matter? | +0.0266 | 0.0376 | 65/190/45 | yes |
