from app.core.interfaces import SearchHit
from app.retrieve.fusion import fuse_hits, reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # "b" ranks high in both lists -> should win the fusion.
    lexical = ["a", "b", "c"]
    dense = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([lexical, dense], k=60)
    assert fused[0][0] == "b"


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["x"]], k=60))
    assert abs(fused["x"] - 1.0 / 61) < 1e-9


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [d for d, _ in fused] == ["a", "b", "c"]


def test_fuse_hits_backfills_text_and_metadata():
    lex = [SearchHit("b", 2.0, text="")]
    dns = [SearchHit("b", 0.9, text="hello", metadata={"src": "arxiv"})]
    fused = fuse_hits([lex, dns])
    assert fused[0].doc_id == "b"
    assert fused[0].text == "hello"
    assert fused[0].metadata["src"] == "arxiv"


def test_fuse_hits_top_k():
    a = [SearchHit("a", 1.0), SearchHit("b", 1.0), SearchHit("c", 1.0)]
    b = [SearchHit("a", 1.0), SearchHit("b", 1.0)]
    fused = fuse_hits([a, b], top_k=2)
    assert len(fused) == 2
    assert fused[0].doc_id == "a"  # appears top of both
