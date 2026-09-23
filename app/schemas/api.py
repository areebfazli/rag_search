from __future__ import annotations

from pydantic import BaseModel


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
    # Additive + optional, so existing clients are unaffected. Set only when the input
    # was a claim and the model ended with a well-formed verdict line (which is then
    # stripped from `answer`): SUPPORTED | REFUTED | NOT ENOUGH EVIDENCE.
    verdict: str | None = None
