"""Load the BEIR/SciFact corpus, queries, and gold qrels via ir_datasets.

SciFact documents are short scientific abstracts and BEIR relevance judgments are
at the document level, so we index whole documents (title + abstract) as single
passages — no chunking. A chunker would slot in here for longer corpora.
"""
from __future__ import annotations

import ir_datasets

from app.core.config import settings


def load_documents(dataset: str | None = None) -> list[dict]:
    ds = ir_datasets.load(dataset or settings.corpus_dataset)
    docs: list[dict] = []
    for d in ds.docs_iter():
        docs.append(
            {
                "doc_id": d.doc_id,
                "title": getattr(d, "title", "") or "",
                "text": getattr(d, "text", "") or "",
            }
        )
    return docs


def document_passage(doc: dict) -> str:
    """The text we embed and BM25-index for a document (title + abstract)."""
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    return f"{title}\n\n{text}".strip() if title else text


def load_queries_qrels(
    dataset: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries, qrels). qrels is {query_id: {doc_id: relevance}} — the exact
    nested-dict shape ranx.Qrels ingests."""
    ds = ir_datasets.load(dataset or settings.eval_dataset)
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = int(qr.relevance)
    return queries, qrels
