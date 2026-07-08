from app.core.interfaces import SearchHit, hit_passage
from app.generate.generator import map_citations


def _hits(n):
    return [SearchHit(f"doc{i}", 1.0, f"text{i}") for i in range(n)]


def test_map_citations_basic():
    hits = _hits(3)
    assert map_citations("A [1] and B [3].", hits) == ["doc0", "doc2"]


def test_map_citations_dedupes_and_orders():
    hits = _hits(3)
    assert map_citations("[3][1][1][2]", hits) == ["doc0", "doc1", "doc2"]


def test_map_citations_ignores_out_of_range():
    hits = _hits(2)
    assert map_citations("claim [5] and [2]", hits) == ["doc1"]


def test_map_citations_none():
    assert map_citations("no citations here", _hits(3)) == []


def test_hit_passage_prepends_title():
    h = SearchHit("d", 1.0, "abstract body", {"title": "My Title"})
    assert hit_passage(h) == "My Title. abstract body"


def test_hit_passage_without_title():
    h = SearchHit("d", 1.0, "abstract body", {})
    assert hit_passage(h) == "abstract body"
