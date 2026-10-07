"""LexicalIndex.search: docs sharing no term with the query (BM25 score 0) are not hits."""
from app.index.lexical import LexicalIndex

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
