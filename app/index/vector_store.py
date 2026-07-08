"""Qdrant wrapper using embedded local mode — no server or Docker required.

Point ids are sequential integers; the original string doc_id lives in the
payload alongside the text, so retrieval speaks the application's doc ids.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.core.interfaces import SearchHit


class VectorStore:
    def __init__(self, location: str | None = None, collection: str | None = None):
        self.client = QdrantClient(path=location or settings.qdrant_location)
        self.collection = collection or settings.qdrant_collection

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def upsert(self, docs: list[dict], vectors, batch: int = 256) -> None:
        points = [
            models.PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "doc_id": docs[i]["doc_id"],
                    "title": docs[i].get("title", ""),
                    "text": docs[i]["text"],
                },
            )
            for i in range(len(docs))
        ]
        for start in range(0, len(points), batch):
            self.client.upsert(self.collection, points[start : start + batch])

    def search(self, vector, top_k: int) -> list[SearchHit]:
        points = self.client.query_points(
            self.collection, query=vector.tolist(), limit=top_k, with_payload=True
        ).points
        return [
            SearchHit(
                doc_id=p.payload["doc_id"],
                score=float(p.score),
                text=p.payload.get("text", ""),
                metadata={"title": p.payload.get("title", "")},
            )
            for p in points
        ]
