"""Tests for the retrieval eval harness — the module that produces the headline table.

Everything here runs against a fake service and a temp cache dir: no models, no index,
no exclusive Qdrant lock. The cases that matter are the ones where a bug would be
*silent* — a stale cache served as fresh, or a corrupt file aborting a multi-hour run.
"""
import json

import pytest

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.eval import retrieval_eval as re_mod


class FakeService:
    """Returns deterministic hits and counts how many queries were actually run."""

    def __init__(self):
        self.calls = 0

    def retrieve(self, query, mode="hybrid", top_k=None, candidate_k=None):
        self.calls += 1
        return [SearchHit(f"doc{query}-{i}", 1.0 - i / 10, f"t{i}") for i in range(3)]


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "CACHE", tmp_path)
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    return tmp_path


QUERIES = {"q1": "alpha", "q2": "beta", "q3": "gamma"}


def _run(service, cache_dir, queries=QUERIES):
    return re_mod.cached_run(service, queries, "hybrid", "hybrid", None)


# --- caching ---------------------------------------------------------------


def test_second_run_hits_the_cache_and_recomputes_nothing(cache):
    first = FakeService()
    _run(first, cache)
    assert first.calls == len(QUERIES)

    second = FakeService()
    result = _run(second, cache)
    assert second.calls == 0  # served entirely from disk
    assert set(result) == set(QUERIES)


def test_changing_a_setting_that_affects_results_invalidates_the_cache(cache, monkeypatch):
    _run(FakeService(), cache)
    # rrf_k changes fusion, so a cached run under the old value must not be reused.
    monkeypatch.setattr(settings, "rrf_k", settings.rrf_k + 1)
    again = FakeService()
    _run(again, cache)
    assert again.calls == len(QUERIES)


def test_resume_only_runs_the_missing_queries(cache):
    # Simulate an interrupted run: one query done, not marked complete.
    sig = re_mod._signature("hybrid", None, len(QUERIES))
    (cache / f"hybrid-{sig}.json").write_text(
        json.dumps({"signature": sig, "complete": False, "run": {"q1": {"doc": 1.0}}})
    )
    service = FakeService()
    result = _run(service, cache)
    assert service.calls == 2  # q2 and q3 only — q1 came back from the checkpoint
    assert set(result) == set(QUERIES)


# --- corrupt checkpoints must degrade to "recompute", never abort -----------


@pytest.mark.parametrize(
    "content",
    [
        "",  # truncated to nothing
        '{"signature": "x", "run": {"q1": {"d": 1.0}}',  # truncated mid-write
        "null",  # valid JSON, not a dict -> blob["run"] raises TypeError
        "[1, 2, 3]",  # valid JSON list
        '"a string"',
        '{"signature": "x"}',  # dict with no run key
        '{"signature": "x", "run": null}',  # run present but wrong type
    ],
)
def test_corrupt_checkpoint_recomputes_instead_of_crashing(cache, content):
    # A multi-hour sweep must not be stranded behind a manual `rm`. Note several of
    # these decode fine and only fail on *shape* — catching JSON errors is not enough.
    sig = re_mod._signature("hybrid", None, len(QUERIES))
    (cache / f"hybrid-{sig}.json").write_text(content)
    service = FakeService()
    result = _run(service, cache)
    assert service.calls == len(QUERIES)
    assert set(result) == set(QUERIES)


def test_checkpoint_write_is_atomic(cache):
    # Written via a temp file + os.replace, so a kill mid-write leaves the previous
    # good file rather than a truncated one, and no .tmp is left behind on success.
    _run(FakeService(), cache)
    assert not list(cache.glob("*.tmp"))
    sig = re_mod._signature("hybrid", None, len(QUERIES))
    blob = json.loads((cache / f"hybrid-{sig}.json").read_text())
    assert blob["complete"] is True


# --- JSON artifact validity ------------------------------------------------


def test_json_safe_replaces_non_finite_with_null():
    # ranx returns NaN for a paired test over two identical score vectors — exactly
    # what Recall@100 is between reranked configs. Bare NaN is not valid JSON.
    out = re_mod._json_safe(
        {"p": float("nan"), "nested": [1.0, float("inf")], "deep": {"q": float("-inf")}}
    )
    assert out == {"p": None, "nested": [1.0, None], "deep": {"q": None}}
    # allow_nan=False proves nothing non-finite survived.
    json.dumps(out, allow_nan=False)


def test_json_safe_leaves_ordinary_values_alone():
    payload = {"a": 0.5, "b": [1, 2], "c": "text", "d": True, "e": None}
    assert re_mod._json_safe(payload) == payload


# --- index manifest in the cache signature ---------------------------------


@pytest.fixture()
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "index_manifest.json"
    monkeypatch.setattr(re_mod, "MANIFEST_PATH", path)
    return path


def _legacy_signature(mode, reranker, n_queries):
    """The pre-manifest formula, restated on purpose: the existing data/eval_cache/
    entries behind the committed tables are keyed by exactly this."""
    import hashlib

    payload = json.dumps(
        {
            "mode": mode,
            "reranker": reranker,
            "depth": re_mod.DEPTH,
            "rerank_candidates": settings.rerank_candidates,
            "rrf_k": settings.rrf_k,
            "dataset": settings.eval_dataset,
            "embedding_model": settings.embedding_model,
            "embedding_query_prefix": settings.embedding_query_prefix,
            "n_queries": n_queries,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def test_missing_manifest_keeps_the_legacy_signature(manifest):
    # An index built before manifests existed must not silently invalidate hours of
    # cached cross-encoder runs: the sentinel leaves the signature byte-identical.
    assert not manifest.exists()
    assert re_mod._index_fingerprint() is re_mod.NO_MANIFEST
    for _, _, mode, reranker in re_mod.CONFIGS:
        assert re_mod._signature(mode, reranker, 300) == _legacy_signature(mode, reranker, 300)


def test_manifest_contents_are_part_of_the_signature(manifest):
    legacy = re_mod._signature("hybrid", None, 3)
    manifest.write_text('{"doc_count": 5183}')
    first = re_mod._signature("hybrid", None, 3)
    assert first != legacy
    assert re_mod._signature("hybrid", None, 3) == first  # stable for identical bytes
    manifest.write_text('{"doc_count": 5184}')  # a reindex of a different corpus
    assert re_mod._signature("hybrid", None, 3) != first


def test_reindex_invalidates_the_cache(cache, manifest):
    manifest.write_text('{"embedding_model": "a"}')
    _run(FakeService(), cache)
    manifest.write_text('{"embedding_model": "b"}')
    again = FakeService()
    _run(again, cache)
    assert again.calls == len(QUERIES)  # not served from the old index's cache


# --- only the canonical run may write to eval/results ----------------------


@pytest.fixture()
def sandbox(tmp_path, monkeypatch, manifest):
    """main() against fakes, with every output path redirected under tmp_path."""
    out, runs, cache_dir = tmp_path / "results", tmp_path / "runs", tmp_path / "cache"
    monkeypatch.setattr(re_mod, "OUT", out)
    monkeypatch.setattr(re_mod, "RUNS", runs)
    monkeypatch.setattr(re_mod, "CACHE", cache_dir)
    monkeypatch.setattr(re_mod, "SearchService", FakeService)
    # cached_run instantiates a real cross-encoder for the rerank configs; FakeService
    # ignores it, and loading one would pull model weights (and fail offline in CI).
    import app.rerank.cross_encoder as ce

    monkeypatch.setattr(ce, "CrossEncoderReranker", lambda model: None)
    qrels = {q: {f"doc{t}-0": 1} for q, t in QUERIES.items()}
    monkeypatch.setattr(re_mod, "load_queries_qrels", lambda: (dict(QUERIES), qrels))
    monkeypatch.setattr(settings, "eval_dataset", re_mod.CANONICAL_DATASET)
    monkeypatch.delenv("SSR_EVAL_LIMIT", raising=False)
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    return out, runs


def test_canonical_run_writes_to_results(sandbox):
    out, runs = sandbox
    re_mod.main()
    assert (out / "retrieval.md").read_text().startswith(
        f"# Retrieval evaluation — BEIR/SciFact ({len(QUERIES)} test queries)\n"
    )
    assert json.loads((out / "retrieval.json").read_text())["n_queries"] == len(QUERIES)
    assert not runs.exists()


def test_limited_run_never_touches_results(sandbox, monkeypatch):
    out, runs = sandbox
    monkeypatch.setenv("SSR_EVAL_LIMIT", "2")
    re_mod.main()
    assert not out.exists()
    run_dir = runs / "beir-scifact-test_2"
    header = (run_dir / "retrieval.md").read_text().splitlines()[0]
    assert "beir/scifact/test (2 queries" in header
    assert "SSR_EVAL_LIMIT=2" in header and "non-canonical" in header
    assert json.loads((run_dir / "retrieval.json").read_text())["n_queries"] == 2


def test_other_dataset_never_touches_results(sandbox, monkeypatch):
    out, runs = sandbox
    monkeypatch.setattr(settings, "eval_dataset", "beir/scifact/train")
    re_mod.main()
    assert not out.exists()
    header = (runs / f"beir-scifact-train_{len(QUERIES)}" / "retrieval.md").read_text()
    assert header.startswith(f"# Retrieval evaluation — beir/scifact/train ({len(QUERIES)} queries")
