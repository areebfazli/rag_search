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
