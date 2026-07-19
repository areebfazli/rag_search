"""RAG answer-quality eval via LLM-as-judge (Groq), scored against gold qrels.

For a random sample of SciFact claims: retrieve -> generate -> an LLM judge scores
faithfulness and context relevance, and decides whether the answer *attempted* an
answer or abstained.

SciFact is claim-verification, so for many claims the top-k context genuinely does
not contain the evidence — a well-behaved RAG system should *abstain* rather than
hallucinate. Crucially, "it abstained" is not by itself a good outcome: an
over-conservative model that abstains on answerable questions looks identical on
abstention rate alone. So we cross the judge's answered/abstained call with the gold
qrels (was a relevant doc actually retrieved?) and report abstention precision and
recall, plus the dangerous quadrant — answered with no evidence retrieved.

Generator and judge use different models (separate Groq rate-limit buckets), with a
throttle so the run stays under the free-tier limits. LLM-as-judge agrees with humans
~85-92% of the time — treat as signal, not truth.

Run:
    uv run python -m app.eval.rag_eval
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

from openai import OpenAI

from app.core.config import settings
from app.core.interfaces import hit_passage
from app.generate.generator import LLMGenerator
from app.ingest.corpus import load_queries_qrels
from app.retrieve.service import SearchService

# Generator uses the 70B model (settings.llm_model) for answer quality; the judge uses a
# DIFFERENT, lighter model so it isn't grading its own output (avoids self-evaluation
# bias) and draws from a separate Groq rate-limit bucket. Both need free-tier budget.
N = 50
SEED = 13  # fixed sample: reproducible, and not just the first N ids in dataset order
GEN_MODEL = settings.llm_model
JUDGE_MODEL = "llama-3.1-8b-instant"
TOP_K = 5
THROTTLE_S = 15.0  # stay under the free-tier tokens-per-minute budget
OUT = Path("eval/results")

JUDGE_SYSTEM = (
    "You are a strict RAG evaluator. Given a QUESTION, the CONTEXT passages retrieved, and the "
    "ANSWER generated, return ONLY a JSON object:\n"
    '{"answered": <true if the answer attempts to answer using the context; false if it states '
    'the information is not present in the context>, "faithfulness": <float 0-1, fraction of the '
    "answer's factual claims that are directly supported by the context>, "
    '"context_relevance": <float 0-1, how relevant the context is to the question>}'
)


_SCORE_KEYS = frozenset({"answered", "faithfulness", "context_relevance"})


class JudgeParseError(RuntimeError):
    """The judge returned something we could not read as JSON."""


def _parse(s: str) -> dict:
    """Extract the first JSON object in the response that carries a score field.

    A greedy ``\\{.*\\}`` span runs from the first brace to the last, so any prose
    containing another brace after the object makes the whole parse fail and silently
    drops the query. raw_decode stops at the end of the first valid object.

    Requiring an expected key matters as much as the decode: a judge that wraps its
    verdict (``{"result": {"answered": true, ...}}``) would otherwise return the
    *outer* object, whose ``.get("answered")`` is None — scoring a real answer as an
    abstention with zero faithfulness, and doing it silently, since a successful parse
    never increments the failure counter. Skipping wrappers finds the inner object.
    """
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and _SCORE_KEYS & obj.keys():
                return obj
    raise JudgeParseError(s[:120])


def _as_bool(v: object) -> bool:
    """Coerce the judge's ``answered`` field explicitly.

    A small judge model routinely emits ``"answered": "false"`` as a JSON *string*,
    and ``bool("false")`` is True — which would count a correct abstention as an
    answer and corrupt both headline metrics at once.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def _as_score(v: object) -> float:
    """Coerce a 0-1 judge score; ``null`` and out-of-range values are common.

    NaN must be rejected before the clamp, not by it: ``min(1.0, nan)`` is ``nan``
    and ``max(0.0, nan)`` is ``1.0``, so a NaN would silently become a *perfect*
    score. json.loads accepts bare ``NaN``, so the judge can really emit one.
    """
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return max(0.0, min(1.0, f))


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
        "answered": _as_bool(d.get("answered", False)),
        "faithfulness": _as_score(d.get("faithfulness", 0.0)),
        "context_relevance": _as_score(d.get("context_relevance", 0.0)),
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def aggregate(rows: list[dict]) -> dict:
    """Cross the judge's answered/abstained call with whether gold evidence was
    actually retrieved, so abstention is scored rather than assumed correct."""
    answered = [r for r in rows if r["answered"]]
    with_ev = [r for r in rows if r["evidence"]]
    without_ev = [r for r in rows if not r["evidence"]]
    abstained = [r for r in rows if not r["answered"]]
    # The two error quadrants: abstaining despite retrieving the gold doc, and
    # answering when nothing relevant was retrieved (the hallucination risk).
    false_abstentions = [r for r in abstained if r["evidence"]]
    answered_no_ev = [r for r in answered if not r["evidence"]]
    return {
        "n": len(rows),
        "evidence_rate": _rate(len(with_ev), len(rows)),
        "answered_rate": _rate(len(answered), len(rows)),
        "abstention_precision": _rate(len(abstained) - len(false_abstentions), len(abstained)),
        "abstention_recall": _rate(
            len([r for r in without_ev if not r["answered"]]), len(without_ev)
        ),
        "false_abstention_rate": _rate(len(false_abstentions), len(with_ev)),
        "answered_without_evidence_rate": _rate(len(answered_no_ev), len(answered)),
        "faithfulness_answered": round(statistics.mean(r["faithfulness"] for r in answered), 4)
        if answered
        else None,
        "context_relevance": round(statistics.mean(r["context_relevance"] for r in rows), 4)
        if rows
        else None,
        "generator_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL,
        "top_k": TOP_K,
        "sample_seed": SEED,
    }


def _markdown(agg: dict, skipped: int, parse_failures: int) -> str:
    def pct(key: str) -> str:
        v = agg[key]
        return "n/a" if v is None else f"{v:.2f}"

    notes = []
    if skipped:
        notes.append(f"{skipped} quer{'y' if skipped == 1 else 'ies'} skipped (pipeline errors)")
    if parse_failures:
        notes.append(
            f"{parse_failures} judge repl{'y' if parse_failures == 1 else 'ies'} unparseable"
        )
    note_line = f"\n{'; '.join(notes)}.\n" if notes else ""
    return (
        f"# RAG answer quality — LLM-as-judge, scored against gold qrels\n\n"
        f"{agg['n']} SciFact claims (random sample, seed={agg['sample_seed']}) · "
        f"top_k={agg['top_k']} · generator={agg['generator_model']} · "
        f"judge={agg['judge_model']}\n"
        f"{note_line}\n"
        f"## Answer quality\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| Faithfulness (over answered) | {pct('faithfulness_answered')} |\n"
        f"| Context relevance (all) | {pct('context_relevance')} |\n\n"
        f"## Abstention, scored against gold qrels\n\n"
        f"`evidence` = a gold-relevant document was actually retrieved into the top-{agg['top_k']}.\n"
        f"Abstaining is only correct when there was no evidence to use.\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| Evidence retrieved (gold doc in top-k) | {pct('evidence_rate')} |\n"
        f"| Answered (model attempted an answer) | {pct('answered_rate')} |\n"
        f"| Abstention precision (abstained & no evidence) | {pct('abstention_precision')} |\n"
        f"| Abstention recall (no evidence & abstained) | {pct('abstention_recall')} |\n"
        f"| False abstention (had evidence, still abstained) | {pct('false_abstention_rate')} |\n"
        f"| Answered without evidence (hallucination risk) | {pct('answered_without_evidence_rate')} |\n"
    )


def main() -> None:
    queries, qrels = load_queries_qrels()
    qids = sorted(queries)
    random.Random(SEED).shuffle(qids)  # fixed random sample, not the first N in id order
    qids = qids[:N]

    service = SearchService()
    generator = LLMGenerator(model=GEN_MODEL)
    judge_client = OpenAI(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=5, timeout=30.0
    )

    rows: list[dict] = []
    skipped = parse_failures = 0
    skip_reasons: dict[str, int] = {}
    for n, qid in enumerate(qids, start=1):
        q = queries[qid]
        gold = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        try:
            hits = service.retrieve(q, mode="hybrid", top_k=TOP_K)
            ans = generator.generate(q, hits)
            # Judge must see the SAME context the generator saw (title + full text) —
            # a truncated view would misscore claims grounded in the cut-off part.
            s = judge(judge_client, JUDGE_MODEL, q, [hit_passage(h) for h in hits], ans.text)
        except JudgeParseError as e:
            parse_failures += 1
            print(f"  [{n}/{len(qids)}] q{qid:>4} JUDGE-UNPARSEABLE ({str(e)[:50]})", flush=True)
            time.sleep(THROTTLE_S)
            continue
        except Exception as e:  # skip a query rather than lose the whole run
            skipped += 1
            # Record WHY, not just how many: a run thinned by rate limits and one
            # thinned by a broken index are very different, and the artifact should
            # say which without anyone having to still have the console log.
            skip_reasons[type(e).__name__] = skip_reasons.get(type(e).__name__, 0) + 1
            print(f"  [{n}/{len(qids)}] q{qid:>4} SKIPPED ({type(e).__name__}: {str(e)[:50]})", flush=True)
            time.sleep(THROTTLE_S)  # a failure is often the rate limit — back off too
            continue
        s["evidence"] = bool(gold & {h.doc_id for h in hits})
        rows.append(s)
        verdict = "answered" if s["answered"] else "abstain "
        ev = "eV" if s["evidence"] else "--"
        print(
            f"  [{n}/{len(qids)}] q{qid:>4} {verdict} {ev} "
            f"faith={s['faithfulness']:.2f} ctx={s['context_relevance']:.2f}  {q[:45]}",
            flush=True,
        )
        time.sleep(THROTTLE_S)

    agg = aggregate(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rag.json").write_text(
        json.dumps(
            {
                **agg,
                "skipped": skipped,
                "skip_reasons": skip_reasons,
                "judge_parse_failures": parse_failures,
            },
            indent=2,
        )
    )
    (OUT / "rag.md").write_text(_markdown(agg, skipped, parse_failures))
    print(
        f"\nn={agg['n']}  evidence={agg['evidence_rate']}  answered={agg['answered_rate']}  "
        f"abstention_precision={agg['abstention_precision']}  "
        f"faithfulness={agg['faithfulness_answered']}  ctx={agg['context_relevance']}"
    )
    print(f"Wrote {OUT/'rag.md'} and {OUT/'rag.json'}")


if __name__ == "__main__":
    main()
