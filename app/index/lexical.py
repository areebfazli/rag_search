"""BM25 lexical retrieval via bm25s (fast, NumPy-based) with English stemming.

bm25s returns corpus *indices*; we map those back to application doc ids via the
stored doc list, and persist both the index and that list for reuse.
"""
from __future__ import annotations

import json
from pathlib import Path

import bm25s
import Stemmer

from app.core.config import settings
from app.core.interfaces import SearchHit


class LexicalIndex:
    def __init__(self):
        self.retriever: bm25s.BM25 | None = None
        self.stemmer = Stemmer.Stemmer("english")
        self.docs: list[dict] = []

    def build(self, docs: list[dict], texts: list[str]) -> None:
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemmer)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.docs = docs

    def save(self, path: str | None = None) -> None:
        p = Path(path or settings.bm25_path)
        p.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(p))
        (p / "docs.json").write_text(json.dumps(self.docs))

    def load(self, path: str | None = None) -> "LexicalIndex":
        p = Path(path or settings.bm25_path)
        self.retriever = bm25s.BM25.load(str(p))
        self.docs = json.loads((p / "docs.json").read_text())
        return self

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        q_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        k = min(top_k, len(self.docs))
        idxs, scores = self.retriever.retrieve(q_tokens, k=k)
        hits: list[SearchHit] = []
        for idx, score in zip(idxs[0], scores[0]):
            doc = self.docs[int(idx)]
            hits.append(
                SearchHit(
                    doc_id=doc["doc_id"],
                    score=float(score),
                    text=doc["text"],
                    metadata={"title": doc.get("title", "")},
                )
            )
        return hits
