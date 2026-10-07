"""LexicalIndex.search: docs sharing no term with the query (BM25 score 0) are not hits."""
from app.core.interfaces import SearchHit
from app.index.lexical import LexicalIndex
from app.retrieve.web_search import bm25_order

DOCS = [
    {"doc_id": "a", "title": "", "text": "aspirin inhibits platelet aggregation"},
    {"doc_id": "b", "title": "", "text": "statins lower cholesterol"},
    {"doc_id": "c", "title": "", "text": "vitamin d and bone density"},
]


def _index() -> LexicalIndex:
    lex = LexicalIndex()
    lex.build(DOCS, [d["text"] for d in DOCS], show_progress=False)
    return lex


def test_only_matching_docs_are_returned():
    hits = _index().search("aspirin platelet", top_k=3)
    assert [h.doc_id for h in hits] == ["a"]
    assert all(h.score > 0 for h in hits)


def test_a_query_with_no_known_token_returns_nothing_instead_of_k_arbitrary_docs():
    lex = _index()
    for q in ("5-HT2A", "α", "the of and", ""):
        assert lex.search(q, top_k=3) == []
    assert lex.search("aspirin", top_k=0) == []


def test_keep_unmatched_returns_the_zero_score_tail():
    hits = _index().search("aspirin", top_k=3, keep_unmatched=True)
    assert [h.doc_id for h in hits][0] == "a" and len(hits) == 3
    assert [h.score for h in hits[1:]] == [0.0, 0.0]


def test_web_bm25_order_still_ranks_the_whole_pool():
    # The web rerank fuses BM25 over the candidate pool with dense similarity; it was
    # measured with every candidate in the BM25 list (zero-score ones included).
    pool = [SearchHit(d["doc_id"], 1.0, d["text"], {"title": ""}) for d in DOCS]
    out = bm25_order("statins", pool)
    assert out[0].doc_id == "b" and sorted(h.doc_id for h in out) == ["a", "b", "c"]
