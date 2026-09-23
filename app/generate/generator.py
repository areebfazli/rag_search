"""LLM answer generation over the OpenAI-compatible interface.

Points at whatever SSR_LLM_BASE_URL is configured (Groq by default; swap to Ollama
or another provider with no code change). Maps the model's [n] citations back to
document ids so answers are traceable to sources, and parses the optional final
verdict line (claims only) into Answer.verdict.
"""
from __future__ import annotations

import re

from openai import OpenAI

from app.core.config import settings
from app.core.interfaces import Answer, SearchHit
from app.generate.prompts import SYSTEM, VERDICTS, build_user_prompt

_CITE = re.compile(r"\[(\d+)\]")
# A well-formed verdict line: the whole line, tolerating the markdown emphasis, quotes
# and trailing period models like to add, but nothing else (no trailing citations, no
# prose) — anything looser starts matching sentences that merely mention a verdict.
_MARKUP = r"[\s*_`'\"]*"
_VERDICT_ALT = "|".join(v.replace(" ", r"\s+") for v in VERDICTS)
_VERDICT_LINE = re.compile(
    rf"^{_MARKUP}verdict{_MARKUP}:{_MARKUP}({_VERDICT_ALT}){_MARKUP}\.?{_MARKUP}$",
    re.IGNORECASE,
)
# Any line that *starts* like a verdict — used to detect a second, competing one.
_VERDICT_LIKE = re.compile(rf"^{_MARKUP}verdict{_MARKUP}:", re.IGNORECASE)


def map_citations(text: str, hits: list[SearchHit]) -> list[str]:
    """Map [n] markers in the answer to hit doc_ids — 1-based, deduped, ordered,
    ignoring out-of-range indices the model may hallucinate."""
    cited = sorted({int(n) for n in _CITE.findall(text)})
    return [hits[n - 1].doc_id for n in cited if 1 <= n <= len(hits)]


def split_verdict(text: str) -> tuple[str, str | None]:
    """Split the optional trailing ``Verdict: ...`` line off an answer.

    Returns (display_text, verdict). The verdict counts only if the LAST non-empty line
    is a well-formed verdict line and no earlier line also looks like one. That is what
    keeps a verdict string quoted from a passage (or planted in one) from being read as
    the model's own: mid-line mentions never match, and an echoed verdict line followed
    by the real one is ambiguous, so it yields None rather than a guess.

    A valid line is stripped from the display text — it is a machine field, surfaced
    separately, and the prose already says the same thing in words. A malformed one is
    left in place, so nothing the model wrote is silently dropped. If the verdict was
    the ENTIRE reply, the text is kept as-is rather than serving an empty answer.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return text, None
    m = _VERDICT_LINE.match(lines[-1])
    if m is None or any(_VERDICT_LIKE.match(line) for line in lines[:-1]):
        return text, None
    verdict = " ".join(m.group(1).upper().split())
    body = "\n".join(lines[:-1]).rstrip()
    return (body or text.strip()), verdict


class LLMGenerator:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm_model
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            max_retries=5,  # backoff through transient Groq free-tier 429s
            timeout=30.0,  # don't let a hung request tie up a worker for minutes
        )

    def generate(self, query: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(text="No relevant documents were found.", citations=[], hits=[])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(query, hits)},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        text, verdict = split_verdict((resp.choices[0].message.content or "").strip())
        # Citations come from the displayed text, so every listed source is one the
        # reader can see referenced (a valid verdict line carries no citations anyway).
        return Answer(text=text, citations=map_citations(text, hits), hits=hits, verdict=verdict)
