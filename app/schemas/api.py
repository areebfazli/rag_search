from __future__ import annotations

from pydantic import BaseModel


class Hit(BaseModel):
    doc_id: str
    score: float
    title: str = ""
    text: str = ""
    # Additive: "local" for the SciFact index, "semantic_scholar" for web hits (modes
    # web / hybrid_web). url/year are set for web hits only; url is always http(s).
    url: str | None = None
    year: int | None = None
    source: str = "local"


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[Hit]
    # e.g. hybrid_web fell back to local results because Semantic Scholar failed.
    warnings: list[str] = []


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[str]
    hits: list[Hit]
    # Additive + optional, so existing clients are unaffected. Set when the model ended
    # with a well-formed verdict line or a trailing "Verdict: X" on its last line (either
    # is stripped from `answer`), or — for a claim, never a question — stated its stance
    # in its first sentence (generator.parse_verdict): SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE.
    verdict: str | None = None
    warnings: list[str] = []
