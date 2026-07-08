"""LLM answer generation over the OpenAI-compatible interface.

Points at whatever SSR_LLM_BASE_URL is configured (Groq by default; swap to Ollama
or another provider with no code change). Maps the model's [n] citations back to
document ids so answers are traceable to sources.
"""
from __future__ import annotations

import re

from openai import OpenAI

from app.core.config import settings
from app.core.interfaces import Answer, SearchHit
from app.generate.prompts import SYSTEM, build_user_prompt

_CITE = re.compile(r"\[(\d+)\]")


def map_citations(text: str, hits: list[SearchHit]) -> list[str]:
    """Map [n] markers in the answer to hit doc_ids — 1-based, deduped, ordered,
    ignoring out-of-range indices the model may hallucinate."""
    cited = sorted({int(n) for n in _CITE.findall(text)})
    return [hits[n - 1].doc_id for n in cited if 1 <= n <= len(hits)]


class LLMGenerator:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm_model
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            max_retries=5,  # backoff through transient Groq free-tier 429s
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
        text = (resp.choices[0].message.content or "").strip()
        return Answer(text=text, citations=map_citations(text, hits), hits=hits)
