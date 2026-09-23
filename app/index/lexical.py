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

# PyStemmer's Stemmer object does not expose its algorithm name, so keep it here where
# both the tokenizer and the index manifest (build_index) can read the same value.
STEMMER_LANGUAGE = "english"


class LexicalIndex:
    def __init__(self):
        self.retriever: bm25s.BM25 | None = None
        self.stemmer = Stemmer.Stemmer(STEMMER_LANGUAGE)
        self.docs: list[dict] = []

    def build(self, docs: list[dict], texts: list[str]) -> None:
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemmer)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.docs = docs

    def describe(self) -> dict:
        """What shaped this index, for data/index_manifest.json. The BM25 parameters
        are read off the constructed retriever rather than restated, so a bm25s default
        change shows up in the manifest instead of being silently misreported."""
        r = self.retriever
        return {
            "bm25s_version": bm25s.__version__,
            "stemmer": STEMMER_LANGUAGE,
            "k1": r.k1,
            "b": r.b,
            "delta": r.delta,
            "method": r.method,
            "idf_method": r.idf_method,
        }

    def save(self, path: str | None = None) -> None:
        p = Path(path or settings.bm25_path)
        p.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(p))
        (p / "docs.json").write_text(json.dumps(self.docs))

    def load(self, path: str | None = None) -> "LexicalIndex":
        # Security: only load an index you built yourself. bm25s deserializes on-disk
        # arrays, so pointing this at an untrusted index dir is a code-exec surface.
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
