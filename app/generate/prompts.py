"""Grounded RAG prompt: answer strictly from retrieved context, with [n] citations."""
from __future__ import annotations

from app.core.interfaces import SearchHit

SYSTEM = (
    "You are a careful scientific assistant. Answer the user's question using ONLY the "
    "provided context passages. Cite every claim with bracketed passage numbers like [1] or "
    "[2][3]. If the context does not contain enough information to answer, say so explicitly "
    "instead of guessing. Keep the answer to 2-4 sentences."
)


def build_user_prompt(query: str, hits: list[SearchHit]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {h.metadata.get('title', '').strip()}\n{h.text.strip()}"
        for i, h in enumerate(hits)
    )
    return f"Context passages:\n{context}\n\nQuestion: {query}\n\nAnswer (cite with [n]):"
