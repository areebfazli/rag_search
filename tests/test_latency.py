"""Tests for the latency benchmark: stats, warm-up exclusion, and the output guard.

A fake service advances a fake clock by a per-query cost, so every timing is exact and
no model, index or Qdrant lock is involved.
"""
import pytest

from app.core.config import settings
from app.eval import latency as lat
from app.eval import retrieval_eval as re_mod


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeService:
    """retrieve() 'takes' costs[text] seconds on the fake clock and logs each call."""

    def __init__(self, clock, costs):
        self.clock, self.costs, self.calls = clock, costs, []

    def retrieve(self, query, mode="hybrid", top_k=None, candidate_k=None):
        self.calls.append((query, mode, top_k, candidate_k))
        self.clock.now += self.costs[query]
        return []


# --- statistics ------------------------------------------------------------------


def test_percentile_matches_linear_interpolation():
    xs = [4.0, 1.0, 3.0, 2.0]  # unsorted on purpose
    assert lat.percentile(xs, 0) == 1.0
    assert lat.percentile(xs, 100) == 4.0
    assert lat.percentile(xs, 50) == pytest.approx(2.5)
    assert lat.percentile(xs, 90) == pytest.approx(3.7)  # numpy.percentile(xs, 90)
    assert lat.percentile([7.0], 95) == 7.0


def test_percentile_of_nothing_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        lat.percentile([], 50)


def test_summarize_reports_every_field():
    s = lat.summarize([float(i) for i in range(1, 11)])  # 1..10
    assert s["n"] == 10
    assert s["mean"] == pytest.approx(5.5)
    assert s["p50"] == pytest.approx(5.5)
    assert s["p90"] == pytest.approx(9.1)
    assert s["p95"] == pytest.approx(9.55)
    assert s["max"] == 10.0


# --- timing ----------------------------------------------------------------------


def test_warmup_is_run_but_excluded_from_measured_latencies():
    clock = FakeClock()
    # The warm-up call is very expensive (a model load, say); it must not leak in.
    svc = FakeService(clock, {"warm": 50.0, "a": 1.0, "b": 2.0})
    cfg = lat._config("hybrid", lat.DEPTH, "all")
    per_query, warm = lat.time_config(svc, [("q1", "a"), ("q2", "b")], cfg, ["warm"], clock)

    assert per_query == {"q1": 1.0, "q2": 2.0}
    assert warm == [50.0]
    assert [c[0] for c in svc.calls] == ["warm", "a", "b"]  # warm-up runs first
    assert lat.summarize(list(per_query.values()))["max"] == 2.0


def test_configs_pass_the_eval_and_api_shapes_through_to_retrieve():
    clock = FakeClock()
    svc = FakeService(clock, {"a": 0.1})
    eval_cfg = lat._config("rerank_bge", lat.DEPTH, "rerank")
    api_cfg = lat._config("dense", lat.API_TOP_K, "all", "_api")
    lat.time_config(svc, [("q", "a")], eval_cfg, [], clock)
    lat.time_config(svc, [("q", "a")], api_cfg, [], clock)
    # Eval shape mirrors retrieval_eval (top_k = candidate_k = DEPTH); API shape
    # leaves candidate_k to the service default, as app/api/main.py does.
    assert svc.calls == [
        ("a", "hybrid_rerank", lat.DEPTH, lat.DEPTH),
        ("a", "dense", lat.API_TOP_K, None),
    ]


def test_rerank_configs_use_the_same_reranker_models_as_the_eval():
    ours = {c["key"]: c["reranker"] for c in lat.LAT_CONFIGS if c["reranker"]}
    theirs = {key: rr for key, _, _, rr in re_mod.CONFIGS if rr}
    assert ours == theirs


def test_rerank_sample_is_seeded_and_warmup_comes_from_outside_it():
    queries = {f"q{i}": f"text{i}" for i in range(50)}
    s1, w1 = lat.pick_queries(queries, n_sample=10, n_warmup=3)
    s2, w2 = lat.pick_queries(queries, n_sample=10, n_warmup=3)
    assert s1 == s2 and w1 == w2  # deterministic
    assert len(s1) == 10 and len(w1) == 3
    assert not set(w1) & {text for _, text in s1}


def test_tiny_query_set_still_gets_a_warmup():
    queries = {"q1": "a", "q2": "b"}
    sample, warm = lat.pick_queries(queries, n_sample=40, n_warmup=3)
    assert len(sample) == 2  # capped at what exists
    assert warm  # falls back to reusing sampled queries rather than skipping warm-up


# --- output-path guard ------------------------------------------------------------


def test_only_the_full_canonical_run_writes_the_committed_artifact():
    out, canonical = lat.output_dir(0, 300, lat.CANONICAL_DATASET)
    assert canonical and out == re_mod.OUT


def test_a_limited_run_never_writes_to_eval_results(monkeypatch):
    monkeypatch.setattr(settings, "eval_dataset", lat.CANONICAL_DATASET)
    out, canonical = lat.output_dir(20, 20)
    assert not canonical
    assert out.parent == re_mod.RUNS and out != re_mod.OUT
    assert out.name.startswith("latency_") and out.name.endswith("_20")


def test_a_different_dataset_never_writes_to_eval_results():
    out, canonical = lat.output_dir(0, 300, "beir/scifact/train")
    assert not canonical and out.parent == re_mod.RUNS


def test_markdown_renders_every_config_row():
    cfgs = [
        {**lat._config("bm25", lat.DEPTH, "all"), "stats": lat.summarize([0.01, 0.02]),
         "first_call_s": 0.5},
        {**lat._config("rerank_bge", lat.DEPTH, "rerank"), "stats": lat.summarize([30.0]),
         "first_call_s": 40.0},
    ]
    env = {k: 0 for k in ("candidate_depth", "rerank_candidates", "rerank_batch_size",
                          "api_top_k", "rerank_sample", "seed", "warmup_queries",
                          "logical_cores", "physical_cores", "torch_num_threads")}
    env.update(cpu_model="cpu", torch_version="t", python_version="p", platform="x",
               git_sha="abc", git_dirty=False, date="d",
               loadavg_start=(0.1, 0.1, 0.1), loadavg_end=(0.2, 0.2, 0.2))
    report = {"environment": env, "configs": cfgs, "total_runtime_s": 60.0,
              "cold_start": {"service_init_s": 1.0, "reranker_load_s": {"rerank_bge": 2.0}}}
    md = lat.to_markdown(report, "T")
    assert "| BM25 | bm25 | 100 | 2 |" in md
    assert "| Hybrid + rerank (bge-reranker-base) | hybrid_rerank | 100 | 1 | 30.000" in md
