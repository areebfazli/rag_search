# RAG answer quality — LLM-as-judge

10 SciFact questions · generator=llama-3.3-70b-versatile · judge=llama-3.1-8b-instant

| Metric | Score |
|---|---|
| Answered (context had the evidence) | 0.50 |
| Faithfulness (over answered) | 0.7600 |
| Context relevance (all) | 0.6100 |

Many SciFact claims have no supporting passage in the top-k, so the system correctly abstains on ~50% of them rather than hallucinating.
