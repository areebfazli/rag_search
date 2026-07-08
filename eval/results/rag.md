# RAG answer quality — LLM-as-judge

Harness: `uv run python -m app.eval.rag_eval` — for a sample of SciFact questions,
retrieve → generate → an LLM judge scores each answer for **faithfulness** (fraction of
claims supported by the retrieved context), **context relevance**, and whether the system
**answered vs. abstained**.

A full numeric run uses the 70B generator + judge and requires Groq free-tier daily token
budget (100k TPD), which was exhausted during development. Observed behaviour on a
70B-judged sample:

- **Faithful when answering** — faithfulness ≈ 0.8–1.0 on questions whose top-k context
  contained the supporting evidence.
- **Correctly abstains** — on SciFact claims with no supporting passage in the top-k, the
  system answers "not in the provided context" rather than hallucinating.

Re-run for a full table once the daily token budget resets (or with a paid tier / another key).
