"""Reciprocal Rank Fusion — combine lexical + dense rankings into one.

    RRF(d) = sum_r 1 / (k + rank_r(d))          (rank is 1-based)

RRF fuses on rank *position*, not score, so it needs no calibration between
BM25 scores and cosine similarities that live on completely different scales —
which is exactly why it's the default fusion for hybrid search.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from app.core.interfaces import SearchHit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked doc-id lists into a single (doc_id, score) list, best first.

    `rankings`: each element is a list of doc_ids already sorted best -> worst.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fuse_hits(
    hit_lists: Sequence[Sequence[SearchHit]], k: int = 60, top_k: int | None = None
) -> list[SearchHit]:
    """RRF over multiple SearchHit lists, preserving text/metadata.

    A doc may appear in only one ranker's list (e.g. dense finds it, BM25 doesn't),
    so we backfill text/metadata from whichever list carries it, keeping fused hits
    self-contained for downstream reranking/generation.
    """
    by_id: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for h in hits:
            existing = by_id.get(h.doc_id)
            if existing is None:
                by_id[h.doc_id] = SearchHit(h.doc_id, 0.0, h.text, dict(h.metadata))
            else:
                if not existing.text and h.text:
                    existing.text = h.text
                if h.metadata:
                    existing.metadata = {**h.metadata, **existing.metadata}

    fused = reciprocal_rank_fusion([[h.doc_id for h in hits] for hits in hit_lists], k=k)
    result = []
    for doc_id, score in fused:
        hit = by_id[doc_id]
        hit.score = score
        result.append(hit)
    return result[:top_k] if top_k else result
