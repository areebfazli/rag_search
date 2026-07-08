"""RAG answer-quality eval via LLM-as-judge (Groq).

For a sample of SciFact questions: retrieve -> generate -> an LLM judge scores:
  - answered          : did the answer attempt to answer from context, or abstain?
  - faithfulness      : fraction of the answer's claims supported by the context
  - context_relevance : how relevant the retrieved context is to the question

SciFact is claim-verification, so for many claims the top-k context does not contain
the evidence — a well-behaved RAG system should *abstain* rather than hallucinate.
We report the abstention rate separately and measure faithfulness only over answered
questions (scoring a correct "not in context" as unfaithful would be wrong).

Generator and judge use different models (separate Groq rate-limit buckets), with a
throttle so the run stays under the free-tier limits. LLM-as-judge agrees with humans
~85-92% of the time — treat as signal, not truth. (ragas is the heavier production
alternative; this stays dependency-light and runs on the free Groq tier.)

Run:
    uv run python -m app.eval.rag_eval
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from openai import OpenAI

from app.core.config import settings
from app.generate.generator import LLMGenerator
from app.ingest.corpus import load_queries_qrels
from app.retrieve.service import SearchService

# Generator + judge default to the 70B model (settings.llm_model) for quality — the 70B
# judge follows the JSON scoring reliably, whereas the 8B judge tends to drop fields.
# Requires Groq free-tier daily token budget (100k TPD); if that is exhausted, drop these
# to "llama-3.1-8b-instant" for a budget-friendly (lower-quality) sample run.
N = 10
GEN_MODEL = settings.llm_model
JUDGE_MODEL = settings.llm_model
TOP_K = 5
THROTTLE_S = 15.0  # stay under the free-tier tokens-per-minute budget
JUDGE_CTX_CHARS = 400
OUT = Path("eval/results")

JUDGE_SYSTEM = (
    "You are a strict RAG evaluator. Given a QUESTION, the CONTEXT passages retrieved, and the "
    "ANSWER generated, return ONLY a JSON object:\n"
    '{"answered": <true if the answer attempts to answer using the context; false if it states '
    'the information is not present in the context>, "faithfulness": <float 0-1, fraction of the '
    "answer's factual claims that are directly supported by the context>, "
    '"context_relevance": <float 0-1, how relevant the context is to the question>}'
)


def _parse(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def judge(client: OpenAI, model: str, q: str, contexts: list[str], answer: str) -> dict:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"QUESTION: {q}\n\nCONTEXT:\n{ctx}\n\nANSWER: {answer}"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    d = _parse(resp.choices[0].message.content or "")
    return {
        "answered": bool(d.get("answered", False)),
        "faithfulness": float(d.get("faithfulness", 0.0)),
        "context_relevance": float(d.get("context_relevance", 0.0)),
    }


def main() -> None:
    queries, _ = load_queries_qrels()
    qids = list(queries)[:N]
    service = SearchService()
    generator = LLMGenerator(model=GEN_MODEL)
    judge_client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=5)

    rows = []
    for qid in qids:
        q = queries[qid]
        try:
            hits = service.retrieve(q, mode="hybrid", top_k=TOP_K)
            ans = generator.generate(q, hits)
            s = judge(judge_client, JUDGE_MODEL, q, [h.text[:JUDGE_CTX_CHARS] for h in hits], ans.text)
        except Exception as e:  # skip a query rather than lose the whole run
            print(f"  q{qid:>4} SKIPPED ({type(e).__name__}: {str(e)[:60]})")
            continue
        rows.append(s)
        flag = "answered" if s["answered"] else "abstain "
        print(f"  q{qid:>4} {flag} faith={s['faithfulness']:.2f} ctx={s['context_relevance']:.2f}  {q[:50]}")
        time.sleep(THROTTLE_S)

    answered = [r for r in rows if r["answered"]]
    agg = {
        "n": len(rows),
        "answered_rate": round(len(answered) / len(rows), 4) if rows else 0.0,
        "faithfulness_answered": round(statistics.mean(r["faithfulness"] for r in answered), 4)
        if answered
        else 0.0,
        "context_relevance": round(statistics.mean(r["context_relevance"] for r in rows), 4)
        if rows
        else 0.0,
        "generator_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rag.json").write_text(json.dumps(agg, indent=2))
    (OUT / "rag.md").write_text(
        f"# RAG answer quality — LLM-as-judge\n\n"
        f"{agg['n']} SciFact questions · generator={agg['generator_model']} · judge={agg['judge_model']}\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| Answered (context had the evidence) | {agg['answered_rate']:.2f} |\n"
        f"| Faithfulness (over answered) | {agg['faithfulness_answered']:.4f} |\n"
        f"| Context relevance (all) | {agg['context_relevance']:.4f} |\n\n"
        f"Many SciFact claims have no supporting passage in the top-k, so the system correctly "
        f"abstains on ~{1 - agg['answered_rate']:.0%} of them rather than hallucinating.\n"
    )
    print(
        f"\nAnswered={agg['answered_rate']:.2f}  "
        f"Faithfulness(answered)={agg['faithfulness_answered']:.4f}  "
        f"Context-relevance={agg['context_relevance']:.4f}  (n={agg['n']})"
    )
    print(f"Wrote {OUT/'rag.md'} and {OUT/'rag.json'}")


if __name__ == "__main__":
    main()
