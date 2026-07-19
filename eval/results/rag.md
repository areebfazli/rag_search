# RAG answer quality — LLM-as-judge, scored against gold qrels

49 SciFact claims (random sample, seed=13) · top_k=5 · generator=llama-3.3-70b-versatile · judge=llama-3.1-8b-instant

1 query skipped (pipeline errors).

## Answer quality

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.83 |
| Context relevance (all) | 0.72 |

## Abstention, scored against gold qrels

`evidence` = a gold-relevant document was actually retrieved into the top-5.
Abstaining is only correct when there was no evidence to use.

| Metric | Score |
|---|---|
| Evidence retrieved (gold doc in top-k) | 0.82 |
| Answered (model attempted an answer) | 0.73 |
| Abstention precision (abstained & no evidence) | 0.38 |
| Abstention recall (no evidence & abstained) | 0.56 |
| False abstention (had evidence, still abstained) | 0.20 |
| Answered without evidence (hallucination risk) | 0.11 |
