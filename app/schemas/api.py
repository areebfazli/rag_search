from __future__ import annotations

from pydantic import BaseModel, Field


class Hit(BaseModel):
    doc_id: str
    score: float
    title: str = ""
    text: str = ""


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[Hit]


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
    # Additive + optional: which step produced `verdict` — "line" | "inline" | "stance"
    # (parsed from the answer), "reask" (the claim's answer had no verdict, so one
    # verdict-only re-check of the same passages supplied it), or None with no verdict.
    verdict_source: str | None = None
    # Human-readable notes, de-duplicated: e.g. the verdict re-ask failed (only the error
    # CLASS name is shown, never its message) so the first answer is served without a
    # verdict.
    warnings: list[str] = Field(default_factory=list)
