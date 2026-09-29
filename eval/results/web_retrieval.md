# Web retrieval (Semantic Scholar) — BEIR/SciFact (300 test claims)

S2 responses fetched 2026-09-29T18:11:51+00:00 → 2026-09-29T18:52:51+00:00 (UTC). S2's index and ranking change over time; re-running re-scores this cached snapshot.

S2 searches its whole graph (~200M papers); the local index holds 5,183. This measures whether open-web retrieval surfaces the gold paper among everything, not a like-for-like ranker comparison.

| Config | nDCG@10 | Recall@10 | Recall@100 |
|---|---|---|---|
| Local hybrid (RRF, 5,183 docs) | 0.7241 | 0.8554 | 0.9650 |
| Semantic Scholar search (web mode) | 0.0354 | 0.0433 | 0.0558 |
| Semantic Scholar search, incl. no-abstract papers | 0.0375 | 0.0500 | 0.0625 |

## Significance (paired two-sided Student's t-test, p < 0.05)

| Comparison | Metric | Δ | p | W/T/L | Significant |
|---|---|---|---|---|---|
| web vs hybrid | nDCG@10 | -0.6887 | 0.0000 | 0/48/252 | yes |
| web vs hybrid | Recall@10 | -0.8121 | 0.0000 | 0/53/247 | yes |
| web vs hybrid | Recall@100 | -0.9092 | 0.0000 | 0/26/274 | yes |
| web_any vs hybrid | nDCG@10 | -0.6866 | 0.0000 | 0/48/252 | yes |
| web_any vs hybrid | Recall@10 | -0.8054 | 0.0000 | 0/55/245 | yes |
| web_any vs hybrid | Recall@100 | -0.9025 | 0.0000 | 0/28/272 | yes |

## Id mapping check

SciFact doc ids looked up as S2 `CorpusId:<id>`: 264/283 gold docs found, 256 with a matching title (match rate 0.905, threshold 0.9).

Queries: 300 · S2 requests this run: 456 · authenticated: True · web hits dropped for having no abstract: 2608.
