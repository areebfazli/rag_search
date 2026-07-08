"""Second-stage cross-encoder reranker.

A cross-encoder scores (query, document) jointly, which is far more accurate than
the first-stage bi-encoder / BM25 but too expensive to run over the whole corpus —
so it only reorders the small fused candidate set (recall -> precision).
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.interfaces import SearchHit


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [
            SearchHit(hits[i].doc_id, float(scores[i]), hits[i].text, hits[i].metadata)
            for i in order[:top_k]
        ]
