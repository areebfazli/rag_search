"""Core data types and the swappable component contracts.

Retriever / Reranker / Generator are Protocols so every stage of the pipeline is
substitutable (e.g. dense vs lexical vs hybrid retriever; local vs hosted
generator) without touching call sites — the design point of the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SearchHit:
    doc_id: str
    score: float
    text: str = ""
    metadata: dict = field(default_factory=dict)


def hit_passage(hit: "SearchHit") -> str:
    """Title + text — the same CONTENT that was embedded / BM25-indexed, joined with
    ". " for single-line model input (indexing joins with "\\n\\n"; see
    corpus.document_passage). Used for reranking and answer context, since the
    vector-store payload keeps text only."""
    title = hit.metadata.get("title", "").strip()
    return f"{title}. {hit.text}".strip() if title else hit.text


@dataclass
class Answer:
    text: str
    citations: list[str]
    hits: list[SearchHit]


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[SearchHit]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, hits: list[SearchHit]) -> Answer: ...
