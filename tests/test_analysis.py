import json

import pytest

from app.eval import analysis
from app.eval.retrieval_eval import CONFIGS, _signature
from app.retrieve.fusion import reciprocal_rank_fusion


def _as_run(ranking):
    return {d: float(len(ranking) - i) for i, d in enumerate(ranking)}


def _write_caches(tmp_path, n, complete=True, size=None):
    for key, _, mode, reranker in CONFIGS:
        if key not in analysis.KEYS:
            continue
        sig = _signature(mode, reranker, n)
        run = {str(i): {"d": 1.0} for i in range(size if size is not None else n)}
        (tmp_path / f"{key}-{sig}.json").write_text(
            json.dumps({"signature": sig, "complete": complete, "run": run})
        )


def test_loads_complete_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "CACHE", tmp_path)
    _write_caches(tmp_path, 3)
    runs = analysis.load_cached_runs(3)
    assert set(runs) == set(analysis.KEYS) and all(len(r) == 3 for r in runs.values())


def test_missing_cache_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "CACHE", tmp_path)
    with pytest.raises(FileNotFoundError, match="make eval"):
        analysis.load_cached_runs(3)


def test_other_sample_size_is_not_served(tmp_path, monkeypatch):
    # The signature encodes n_queries, so a 300-query analysis can't read a smoke run.
    monkeypatch.setattr(analysis, "CACHE", tmp_path)
    _write_caches(tmp_path, 20)
    with pytest.raises(FileNotFoundError):
        analysis.load_cached_runs(300)


@pytest.mark.parametrize("kw", [{"complete": False}, {"size": 2}])
def test_partial_cache_is_rejected(tmp_path, monkeypatch, kw):
    monkeypatch.setattr(analysis, "CACHE", tmp_path)
    _write_caches(tmp_path, 3, **kw)
    with pytest.raises(ValueError):
        analysis.load_cached_runs(3)


def test_hybrid_replay_check():
    dense, bm25 = ["a", "b", "c"], ["c", "d", "a"]
    fused = [d for d, _ in reciprocal_rank_fusion([dense, bm25])]
    runs = {"dense": {"q": _as_run(dense)}, "bm25": {"q": _as_run(bm25)}}
    analysis.check_hybrid_replay({**runs, "hybrid": {"q": _as_run(fused)}})
    with pytest.raises(ValueError, match="inconsistent"):
        analysis.check_hybrid_replay({**runs, "hybrid": {"q": _as_run(fused[::-1])}})


def test_weighted_rrf_formula_and_missing_docs():
    # "a": dense rank 1, BM25 rank 2; "b": dense only; "c": BM25 only (rank 1).
    fused = dict(analysis.weighted_rrf([["a", "b"], ["c", "a"]], (0.3, 0.7), k=10))
    assert fused["a"] == pytest.approx(0.3 / 11 + 0.7 / 12)
    assert fused["b"] == pytest.approx(0.3 / 12)  # absent from BM25: no BM25 term
    assert fused["c"] == pytest.approx(0.7 / 11)


@pytest.mark.parametrize("k", analysis.K_SWEEP)
def test_equal_weight_matches_production_rrf_ties_included(k):
    # Mirror-image lists tie every doc pair, so this pins the tie-break order too.
    dense, bm25 = ["a", "b", "c", "d"], ["d", "c", "b", "a"]
    prod = [d for d, _ in reciprocal_rank_fusion([dense, bm25], k=k)]
    ours = [d for d, _ in analysis.weighted_rrf([dense, bm25], (0.5, 0.5), k)]
    assert ours == prod


def _sweep_fixture(monkeypatch):
    """Two queries whose top-2 (DEPTH patched to 2) depends on the dense weight."""
    monkeypatch.setattr(analysis, "DEPTH", 2)
    dense = {"q1": ["g", "x", "y"], "q2": ["x", "h", "y"]}
    bm25 = {"q1": ["x", "y", "g"], "q2": ["h", "y", "x"]}
    runs = {
        "dense": {q: _as_run(r) for q, r in dense.items()},
        "bm25": {q: _as_run(r) for q, r in bm25.items()},
    }
    runs["hybrid"] = {
        q: _as_run(
            [d for d, _ in reciprocal_rank_fusion([dense[q], bm25[q]], k=analysis.BASE_K)][:2]
        )
        for q in dense
    }
    relevant = {"q1": {"g"}, "q2": {"h"}}
    qids = ["q1", "q2"]
    committed = analysis.score_and_compare(runs, qids, relevant, ["hybrid"], [])["scores"][
        "hybrid"
    ]
    return runs, qids, relevant, committed


def test_rrf_sensitivity_grid_and_truncation(monkeypatch):
    runs, qids, relevant, committed = _sweep_fixture(monkeypatch)
    sens = analysis.rrf_sensitivity(runs, qids, relevant, committed)
    grid = [(r["k"], r["dense_weight"]) for r in sens["rows"]]
    assert grid == [(k, 0.5) for k in analysis.K_SWEEP] + [
        (analysis.BASE_K, w) for w in analysis.DENSE_WEIGHTS if w != 0.5
    ]
    rows = {(r["k"], r["dense_weight"]): r for r in sens["rows"]}
    base = rows[(analysis.BASE_K, 0.5)]
    assert base["baseline"] and base["vs_baseline"] is None
    assert base["scores"] == committed
    # Every fused list is cut to DEPTH=2 before scoring: q1's gold "g" (dense 1, BM25 3)
    # survives at equal weight but not when BM25 dominates (w=0.3) — "x"/"y" outrank it.
    low = rows[(analysis.BASE_K, 0.3)]
    assert base["scores"]["recall@100"] == 1.0
    assert low["scores"]["recall@100"] == 0.5
    assert low["vs_baseline"]["recall@100"]["delta"] == pytest.approx(-0.5)
    assert low["vs_baseline"]["recall@100"]["win_tie_loss"] == {"W": 0, "T": 1, "L": 1}
    assert sens["union_recall@100_ceiling"] == 1.0


def test_rrf_sensitivity_baseline_must_be_committed_hybrid(monkeypatch):
    runs, qids, relevant, committed = _sweep_fixture(monkeypatch)
    with pytest.raises(ValueError, match="committed hybrid row"):
        analysis.rrf_sensitivity(runs, qids, relevant, {**committed, "ndcg@10": 0.123})
    runs["hybrid"]["q1"] = _as_run(["g", "x"])  # true equal-weight order is x, g
    with pytest.raises(ValueError, match="differs from the cached hybrid run"):
        analysis.rrf_sensitivity(runs, qids, relevant, committed)


@pytest.mark.parametrize(
    "wins,losses,want",
    [
        (9, 2, 134 / 2048),  # hybrid vs dense R@100: 2 * (C(11,0)+C(11,1)+C(11,2)) / 2^11
        (2, 9, 134 / 2048),  # symmetric
        (5, 0, 2 / 32),  # 5-0 is not significant at 0.05 two-sided
        (1, 13, 2 * 15 / 2**14),
        (3, 3, 1.0),  # capped at 1
    ],
)
def test_sign_test_p_exact(wins, losses, want):
    assert analysis.sign_test_p(wins, losses) == pytest.approx(want)


def test_sign_test_p_no_discordant_is_nan():
    assert analysis.sign_test_p(0, 0) != analysis.sign_test_p(0, 0)  # NaN


def _comparison(p, wins, ties, losses):
    return {
        "delta": 0.01,
        "p": p,
        "win_tie_loss": {"W": wins, "T": ties, "L": losses},
        "sign_p": analysis.sign_test_p(wins, losses),
    }


def test_recall_sig_marker_follows_sign_test_not_t_test():
    # t-test p 0.03 but a 9-2 split: sign p 0.065 -> recall row not marked; nDCG uses t-test.
    c = _comparison(0.03, 9, 289, 2)
    assert not analysis._is_sig(c, "recall@100")
    assert analysis._is_sig(c, "ndcg@10")
    assert analysis._is_sig(_comparison(0.2, 18, 281, 1), "recall@100")


def test_score_and_compare_reports_sign_p():
    # 3 queries: hybrid finds gold on q1, q2; dense on none -> W/T/L 2/1/0 at Recall@100.
    runs = {
        "hybrid": {"q1": {"g": 2.0, "x": 1.0}, "q2": {"h": 2.0}, "q3": {"x": 1.0}},
        "dense": {"q1": {"x": 1.0}, "q2": {"y": 1.0}, "q3": {"x": 1.0}},
    }
    relevant = {"q1": {"g"}, "q2": {"h"}, "q3": {"z"}}
    blob = analysis.score_and_compare(
        runs, ["q1", "q2", "q3"], relevant, ["hybrid", "dense"], [("hybrid", "dense")]
    )
    c = blob["comparisons"]["hybrid_vs_dense"]["recall@100"]
    assert c["win_tie_loss"] == {"W": 2, "T": 1, "L": 0}
    assert c["sign_p"] == pytest.approx(0.5)
