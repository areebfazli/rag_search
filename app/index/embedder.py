"""Dense embedding via sentence-transformers.

bge-small is asymmetric: the query prefix is prepended to queries only, never to
documents. Embeddings are L2-normalized so cosine reduces to a dot product,
matching Qdrant's COSINE distance.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None, query_prefix: str | None = None):
        self.model = SentenceTransformer(model_name or settings.embedding_model)
        self.query_prefix = (
            settings.embedding_query_prefix if query_prefix is None else query_prefix
        )

    @property
    def dim(self) -> int:
        # get_sentence_embedding_dimension was renamed to get_embedding_dimension
        # in sentence-transformers 5.x; support both.
        fn = getattr(self.model, "get_embedding_dimension", None)
        return fn() if fn else self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)
