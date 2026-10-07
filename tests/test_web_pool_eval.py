"""web_pool_eval scoring: each pool is scored only on the claims its sources were queried for."""
import pytest

from app.eval.web_pool_eval import baseline_of, score_on_subsets


def test_baseline_of_keeps_the_view():
    assert baseline_of("s2+oa_sem") == "s2" and baseline_of("s2+oa_sem_any") == "s2_any"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # tiny toy runs: degenerate t-tests
def test_a_partial_source_is_scored_on_its_own_claims_against_the_same_baseline():
    qrels = {f"q{i}": {f"g{i}": 1} for i in range(4)}
    # s2 finds the gold for q0 and q2; OpenAlex ran on q0, q1 only and found both.
    runs = {
        "s2": {"q0": {"g0": 1.0}, "q1": {"x": 1.0}, "q2": {"g2": 1.0}, "q3": {"x": 1.0}},
        "oa_sem": {"q0": {"g0": 1.0}, "q1": {"g1": 1.0}},
    }
    subsets = {"s2": ["q0", "q1", "q2", "q3"], "oa_sem": ["q0", "q1"]}
    out = score_on_subsets(qrels, runs, subsets)
    assert out["s2"]["scores"]["recall@100"] == pytest.approx(0.5)
    assert "vs_s2_p" not in out["s2"]
    # Scored on its 2 claims (not as misses on the 2 it never saw: that would be 0.5)...
    assert out["oa_sem"]["scores"]["recall@100"] == pytest.approx(1.0)
    # ...against s2 on exactly those 2 claims.
    assert out["oa_sem"]["baseline_scores"]["recall@100"] == pytest.approx(0.5)
    assert set(out["oa_sem"]["vs_s2_p"]) == {"recall@5", "recall@10", "recall@100", "ndcg@10"}


def test_an_empty_subset_is_not_scored():
    out = score_on_subsets({"q0": {"g": 1}}, {"s2": {"q0": {"g": 1.0}}, "oa": {}}, {"s2": ["q0"], "oa": []})
    assert "oa" not in out and "s2" in out


@pytest.mark.parametrize("bad", ["../x", "../../eval/results", "a/b", "x y", "a.b", "-" * 65, "ü"])
def test_a_pool_tag_that_could_leave_eval_runs_is_refused_before_any_work(monkeypatch, bad):
    import app.eval.web_pool_eval as P

    monkeypatch.setenv("SSR_POOL_TAG", bad)
    with pytest.raises(ValueError, match="SSR_POOL_TAG"):
        P.pool_tag()
    # main() refuses it first: nothing is loaded, fetched or written.
    monkeypatch.setattr(P, "Sources", lambda *a, **k: pytest.fail("work started"))
    with pytest.raises(ValueError, match="SSR_POOL_TAG"):
        P.main()


@pytest.mark.parametrize("ok", ["", "dense50", "screen_c5-fix", "A1"])
def test_a_plain_pool_tag_is_kept(monkeypatch, ok):
    from app.eval.web_pool_eval import pool_tag

    monkeypatch.setenv("SSR_POOL_TAG", ok)
    assert pool_tag() == ok


def _paper(cid, abstract="An abstract."):
    return {"corpusId": cid, "title": f"T{cid}", "abstract": abstract, "url": None, "year": 2020}


class _FakeS2:
    def __init__(self, by_query):
        self.by_query = by_query
        self.calls: list[str] = []

    def fetch(self, query, limit):
        self.calls.append(query)
        return {"response": {"data": self.by_query.get(query, [])[:limit]}, "fetched_at": None}


def _sources(s2, rarity=None):
    from app.eval.web_pool_eval import SOURCE_NAMES, Sources

    src = object.__new__(Sources)  # no network clients: only the S2 fake is used
    src.s2, src.rarity = s2, rarity
    src.logical = {n: 0 for n in SOURCE_NAMES + ["resolve"]}
    return src


def test_s2_base_sends_the_fallback_exactly_when_the_live_pipeline_does():
    from app.retrieve.query_rewrite import rewrite_queries

    claim = (
        "Citrullinated proteins externalized in neutrophil extracellular traps act "
        "indirectly to disrupt the inflammatory cycle in mice."
    )
    primary, fallback = rewrite_queries(claim)
    full = _FakeS2({primary: [_paper(i) for i in range(100)], fallback: [_paper(500)]})
    src = _sources(full)
    hits = src.s2_base(claim)
    assert full.calls == [primary] and len(hits) == 100  # page full of abstracts: no fallback
    assert src.logical["s2"] == 1 and all(h.metadata["source"] == "s2" for h in hits)
    # Title-only papers do not count towards a full page, but are kept in the pool.
    thin = _FakeS2({primary: [_paper(i, abstract="") for i in range(100)], fallback: [_paper(500)]})
    hits = _sources(thin).s2_base(claim)
    assert thin.calls == [primary, fallback] and len(hits) == 101


def test_s2_deep_never_caches_an_error_body(tmp_path):
    from app.retrieve.semantic_scholar import S2Error, ResponseCache

    class ErrS2:
        def _request(self, method, path, params, json_body=None):
            return {"message": "Too Many Requests"}  # HTTP 200, error body

    src = _sources(ErrS2())
    src.cache = ResponseCache(tmp_path)
    with pytest.raises(S2Error):
        src.s2_deep("aspirin platelet")
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_fetch_sources_filter(monkeypatch):
    from app.eval.web_pool_eval import SOURCE_NAMES, fetch_sources

    monkeypatch.delenv("SSR_POOL_SOURCES", raising=False)
    assert fetch_sources() == set(SOURCE_NAMES)
    monkeypatch.setenv("SSR_POOL_SOURCES", "snip_claim, pm_rw")
    assert fetch_sources() == {"s2", "snip_claim", "pm_rw"}  # s2 is every row's baseline
    monkeypatch.setenv("SSR_POOL_SOURCES", "s2,nope")
    with pytest.raises(ValueError):
        fetch_sources()
