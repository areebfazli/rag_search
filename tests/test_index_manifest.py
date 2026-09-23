"""Tests for data/index_manifest.json, which the retrieval eval hashes into its cache
signature. Built against fakes for the embedder and vector store (no model download,
no Qdrant); only bm25s runs for real, since its parameters are what gets recorded.
"""
import json

import bm25s
import numpy as np

from app.core.config import settings
from app.index.lexical import STEMMER_LANGUAGE, LexicalIndex
from app.ingest import build_index as bi

DOCS = [
    {"doc_id": "d1", "title": "Aspirin", "text": "Aspirin reduces fever."},
    {"doc_id": "d2", "title": "Statins", "text": "Statins lower cholesterol."},
]


class FakeEmbedder:
    dim = 7

    def encode_documents(self, texts):
        return np.zeros((len(texts), self.dim))


class FakeStore:
    def recreate(self, dim):
        pass

    def upsert(self, docs, vectors):
        pass


def _build(tmp_path, monkeypatch):
    path = tmp_path / "index_manifest.json"
    monkeypatch.setattr(bi, "MANIFEST_PATH", path)
    monkeypatch.setattr(bi, "load_documents", lambda: [dict(d) for d in DOCS])
    monkeypatch.setattr(bi, "Embedder", FakeEmbedder)
    monkeypatch.setattr(bi, "VectorStore", FakeStore)
    monkeypatch.setattr(settings, "bm25_path", str(tmp_path / "bm25s"))
    bi.main()
    return path


def test_build_writes_a_manifest_describing_the_index(tmp_path, monkeypatch):
    m = json.loads(_build(tmp_path, monkeypatch).read_text())
    assert m["corpus_dataset"] == settings.corpus_dataset
    assert m["doc_count"] == len(DOCS)
    assert m["embedding_model"] == settings.embedding_model
    assert m["embedding_dim"] == FakeEmbedder.dim
    default = bm25s.BM25()  # the parameters must be the ones bm25s actually used
    assert m["bm25"] == {
        "bm25s_version": bm25s.__version__,
        "stemmer": STEMMER_LANGUAGE,
        "k1": default.k1,
        "b": default.b,
        "delta": default.delta,
        "method": default.method,
        "idf_method": default.idf_method,
    }


def test_manifest_bytes_are_deterministic(tmp_path, monkeypatch):
    # The eval hashes raw bytes, so an identical rebuild must not shift the signature.
    first = _build(tmp_path, monkeypatch).read_bytes()
    assert _build(tmp_path, monkeypatch).read_bytes() == first


def test_failed_build_leaves_no_stale_manifest(tmp_path, monkeypatch):
    path = _build(tmp_path, monkeypatch)

    class Boom(FakeStore):
        def upsert(self, docs, vectors):
            raise RuntimeError("disk full")

    monkeypatch.setattr(bi, "VectorStore", Boom)
    try:
        bi.main()
    except RuntimeError:
        pass
    # The old manifest would otherwise vouch for a half-rebuilt index.
    assert not path.exists()


def test_describe_reads_parameters_off_the_retriever():
    idx = LexicalIndex()
    idx.build(DOCS, [d["text"] for d in DOCS])
    idx.retriever.k1 = 0.9  # a non-default value must be reported, not the default
    assert idx.describe()["k1"] == 0.9
