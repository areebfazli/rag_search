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
