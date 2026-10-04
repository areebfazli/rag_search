# Web retrieval (Semantic Scholar) — BEIR/SciFact (300 test claims)

S2 responses fetched 2026-09-29T18:11:51+00:00 → 2026-10-04T16:13:20+00:00 (UTC). S2's index and ranking change over time; re-running re-scores this cached snapshot.

S2 searches its whole graph (~200M papers); the local index holds 5,183. This measures whether open-web retrieval surfaces the gold paper among everything, not a like-for-like ranker comparison.

Variants (app/retrieve/web_search.py; no LLM anywhere): *raw* sends the claim sentence as the S2 query; *rewrite* sends a keyword query built from it (app/retrieve/query_rewrite.py: ≤6 content terms, plus a 3-term fallback query whose results are pooled with the first's, unless the first alone fills the page); *rerank* re-orders the S2 candidates (≤100 per query) locally by RRF of bge-small similarity to the claim and BM25 over the candidates. Rewrite rules were chosen on a seeded 100-claim beir/scifact/train sample and frozen before the test run.

*Pooled* rows (app/retrieve/web_search.py `search_pooled`) widen CANDIDATE GENERATION: the rewrite's S2 results are pooled with S2 snippet search (app/retrieve/s2_extra.py; papers quoting the claim verbatim or mentioning SciFact are excluded) and PubMed Best Match (app/retrieve/pubmed.py), external ids resolved to S2 corpus ids with S2 paper/batch, deduped, and re-ranked by the same local RRF — with only the top N of a BM25 pre-rank embedded (latency bound). Sources and settings chosen on the train sample with app/eval/web_pool_eval.py and frozen before this run.

| Config | Recall@5 | Recall@10 | Recall@100 | nDCG@10 |
|---|---|---|---|---|
| Local hybrid (RRF, 5,183 docs) | 0.7664 | 0.8554 | 0.9650 | 0.7241 |
| S2, raw claim as query (web mode as before) | 0.0400 | 0.0433 | 0.0558 | 0.0354 |
| S2, raw claim as query (web mode as before), incl. no-abstract papers | 0.0467 | 0.0500 | 0.0625 | 0.0375 |
| S2, keyword rewrite | 0.1017 | 0.1150 | 0.2358 | 0.0894 |
| S2, keyword rewrite, incl. no-abstract papers | 0.1250 | 0.1350 | 0.2758 | 0.1034 |
| S2, raw claim + local rerank | 0.0400 | 0.0483 | 0.0558 | 0.0401 |
| S2, raw claim + local rerank, incl. no-abstract papers | 0.0467 | 0.0550 | 0.0625 | 0.0443 |
| S2, keyword rewrite + local rerank | 0.1850 | 0.2017 | 0.2442 | 0.1591 |
| S2, keyword rewrite + local rerank, incl. no-abstract papers | 0.2150 | 0.2417 | 0.3175 | 0.1803 |
| Pooled: S2 rewrite + S2 snippets + PubMed, dense rerank of top 50 (live) | 0.2283 | 0.2661 | 0.3843 | 0.2030 |
| Pooled: + 3 extra S2 queries + rewrite snippets, dense rerank of top 100 (offline) | 0.2261 | 0.2639 | 0.3969 | 0.2028 |

## Significance (paired two-sided Student's t-test, p < 0.05)

| Comparison | Metric | Δ | p | W/T/L | Significant |
|---|---|---|---|---|---|
| web vs hybrid | Recall@5 | -0.7264 | 0.0000 | 0/77/223 | yes |
| web vs hybrid | Recall@10 | -0.8121 | 0.0000 | 0/53/247 | yes |
| web vs hybrid | Recall@100 | -0.9092 | 0.0000 | 0/26/274 | yes |
| web vs hybrid | nDCG@10 | -0.6887 | 0.0000 | 0/48/252 | yes |
| web_any vs hybrid | Recall@5 | -0.7197 | 0.0000 | 0/79/221 | yes |
| web_any vs hybrid | Recall@10 | -0.8054 | 0.0000 | 0/55/245 | yes |
| web_any vs hybrid | Recall@100 | -0.9025 | 0.0000 | 0/28/272 | yes |
| web_any vs hybrid | nDCG@10 | -0.6866 | 0.0000 | 0/48/252 | yes |
| web_rewrite vs hybrid | Recall@5 | -0.6647 | 0.0000 | 0/95/205 | yes |
| web_rewrite vs hybrid | Recall@10 | -0.7404 | 0.0000 | 0/74/226 | yes |
| web_rewrite vs hybrid | Recall@100 | -0.7292 | 0.0000 | 0/79/221 | yes |
| web_rewrite vs hybrid | nDCG@10 | -0.6347 | 0.0000 | 2/56/242 | yes |
| web_rewrite_any vs hybrid | Recall@5 | -0.6414 | 0.0000 | 0/102/198 | yes |
| web_rewrite_any vs hybrid | Recall@10 | -0.7204 | 0.0000 | 0/80/220 | yes |
| web_rewrite_any vs hybrid | Recall@100 | -0.6892 | 0.0000 | 0/91/209 | yes |
| web_rewrite_any vs hybrid | nDCG@10 | -0.6207 | 0.0000 | 2/59/239 | yes |
| web_rerank vs hybrid | Recall@5 | -0.7264 | 0.0000 | 0/77/223 | yes |
| web_rerank vs hybrid | Recall@10 | -0.8071 | 0.0000 | 0/54/246 | yes |
| web_rerank vs hybrid | Recall@100 | -0.9092 | 0.0000 | 0/26/274 | yes |
| web_rerank vs hybrid | nDCG@10 | -0.6840 | 0.0000 | 1/49/250 | yes |
| web_rerank_any vs hybrid | Recall@5 | -0.7197 | 0.0000 | 0/79/221 | yes |
| web_rerank_any vs hybrid | Recall@10 | -0.8004 | 0.0000 | 0/56/244 | yes |
| web_rerank_any vs hybrid | Recall@100 | -0.9025 | 0.0000 | 0/28/272 | yes |
| web_rerank_any vs hybrid | nDCG@10 | -0.6798 | 0.0000 | 1/50/249 | yes |
| web_rewrite_rerank vs hybrid | Recall@5 | -0.5814 | 0.0000 | 0/120/180 | yes |
| web_rewrite_rerank vs hybrid | Recall@10 | -0.6537 | 0.0000 | 0/100/200 | yes |
| web_rewrite_rerank vs hybrid | Recall@100 | -0.7208 | 0.0000 | 0/81/219 | yes |
| web_rewrite_rerank vs hybrid | nDCG@10 | -0.5650 | 0.0000 | 2/72/226 | yes |
| web_rewrite_rerank_any vs hybrid | Recall@5 | -0.5514 | 0.0000 | 0/129/171 | yes |
| web_rewrite_rerank_any vs hybrid | Recall@10 | -0.6137 | 0.0000 | 0/112/188 | yes |
| web_rewrite_rerank_any vs hybrid | Recall@100 | -0.6475 | 0.0000 | 0/103/197 | yes |
| web_rewrite_rerank_any vs hybrid | nDCG@10 | -0.5438 | 0.0000 | 2/73/225 | yes |
| web_pool_live vs hybrid | Recall@5 | -0.5381 | 0.0000 | 0/133/167 | yes |
| web_pool_live vs hybrid | Recall@10 | -0.5893 | 0.0000 | 0/119/181 | yes |
| web_pool_live vs hybrid | Recall@100 | -0.5807 | 0.0000 | 0/122/178 | yes |
| web_pool_live vs hybrid | nDCG@10 | -0.5211 | 0.0000 | 2/82/216 | yes |
| web_pool_offline vs hybrid | Recall@5 | -0.5403 | 0.0000 | 0/133/167 | yes |
| web_pool_offline vs hybrid | Recall@10 | -0.5915 | 0.0000 | 0/118/182 | yes |
| web_pool_offline vs hybrid | Recall@100 | -0.5681 | 0.0000 | 0/126/174 | yes |
| web_pool_offline vs hybrid | nDCG@10 | -0.5213 | 0.0000 | 1/83/216 | yes |
| web_rewrite vs web | Recall@5 | +0.0617 | 0.0001 | 21/277/2 | yes |
| web_rewrite vs web | Recall@10 | +0.0717 | 0.0000 | 23/276/1 | yes |
| web_rewrite vs web | Recall@100 | +0.1800 | 0.0000 | 56/243/1 | yes |
| web_rewrite vs web | nDCG@10 | +0.0540 | 0.0001 | 24/271/5 | yes |
| web_rewrite_any vs web_any | Recall@5 | +0.0783 | 0.0000 | 27/270/3 | yes |
| web_rewrite_any vs web_any | Recall@10 | +0.0850 | 0.0000 | 29/268/3 | yes |
| web_rewrite_any vs web_any | Recall@100 | +0.2133 | 0.0000 | 66/233/1 | yes |
| web_rewrite_any vs web_any | nDCG@10 | +0.0659 | 0.0000 | 31/263/6 | yes |
| web_rerank vs web | Recall@5 | +0.0000 | 1.0000 | 1/298/1 | no |
| web_rerank vs web | Recall@10 | +0.0050 | 0.4063 | 3/296/1 | no |
| web_rerank vs web | Recall@100 | +0.0000 | nan | 0/300/0 | no |
| web_rerank vs web | nDCG@10 | +0.0047 | 0.3817 | 5/293/2 | no |
| web_rerank_any vs web_any | Recall@5 | +0.0000 | 1.0000 | 1/298/1 | no |
| web_rerank_any vs web_any | Recall@10 | +0.0050 | 0.4063 | 3/296/1 | no |
| web_rerank_any vs web_any | Recall@100 | +0.0000 | nan | 0/300/0 | no |
| web_rerank_any vs web_any | nDCG@10 | +0.0068 | 0.2436 | 8/289/3 | no |
| web_rewrite_rerank vs web | Recall@5 | +0.1450 | 0.0000 | 45/254/1 | yes |
| web_rewrite_rerank vs web | Recall@10 | +0.1583 | 0.0000 | 48/252/0 | yes |
| web_rewrite_rerank vs web | Recall@100 | +0.1883 | 0.0000 | 59/240/1 | yes |
| web_rewrite_rerank vs web | nDCG@10 | +0.1238 | 0.0000 | 52/246/2 | yes |
| web_rewrite_rerank_any vs web_any | Recall@5 | +0.1683 | 0.0000 | 52/247/1 | yes |
| web_rewrite_rerank_any vs web_any | Recall@10 | +0.1917 | 0.0000 | 58/242/0 | yes |
| web_rewrite_rerank_any vs web_any | Recall@100 | +0.2550 | 0.0000 | 79/220/1 | yes |
| web_rewrite_rerank_any vs web_any | nDCG@10 | +0.1428 | 0.0000 | 63/234/3 | yes |
| web_pool_live vs web_rewrite_rerank | Recall@5 | +0.0433 | 0.0121 | 20/273/7 | yes |
| web_pool_live vs web_rewrite_rerank | Recall@10 | +0.0644 | 0.0003 | 25/270/5 | yes |
| web_pool_live vs web_rewrite_rerank | Recall@100 | +0.1401 | 0.0000 | 46/254/0 | yes |
| web_pool_live vs web_rewrite_rerank | nDCG@10 | +0.0439 | 0.0023 | 30/249/21 | yes |
| web_pool_offline vs web_rewrite_rerank | Recall@5 | +0.0411 | 0.0198 | 21/271/8 | yes |
| web_pool_offline vs web_rewrite_rerank | Recall@10 | +0.0622 | 0.0006 | 26/268/6 | yes |
| web_pool_offline vs web_rewrite_rerank | Recall@100 | +0.1527 | 0.0000 | 51/247/2 | yes |
| web_pool_offline vs web_rewrite_rerank | nDCG@10 | +0.0437 | 0.0020 | 31/250/19 | yes |
| web_pool_offline vs web_pool_live | Recall@5 | -0.0022 | 0.7061 | 2/296/2 | no |
| web_pool_offline vs web_pool_live | Recall@10 | -0.0022 | 0.7061 | 2/296/2 | no |
| web_pool_offline vs web_pool_live | Recall@100 | +0.0126 | 0.1735 | 9/288/3 | no |
| web_pool_offline vs web_pool_live | nDCG@10 | -0.0002 | 0.9576 | 14/270/16 | no |

## Id mapping check

SciFact doc ids looked up as S2 `CorpusId:<id>`: 264/283 gold docs found, 256 with a matching title (match rate 0.905, threshold 0.9).

Queries: 300 · S2 requests this run: 0 · authenticated: True · raw-query web hits dropped for having no abstract: 2608 · rewrite fallback query sent for 273/300 claims · rewritten claims with no content terms (sent raw): 0 · candidate embeddings cached/computed this run: 113961/16018 · PubMed requests this run: 0.
