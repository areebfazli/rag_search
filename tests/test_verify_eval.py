"""Tests for the NLI verdict evaluator (app/eval/verify_eval.py): fakes only — no model,
no index, no Qdrant lock, no network."""
import argparse
import json
from pathlib import Path

import pytest

from app.core.interfaces import SearchHit
from app.eval import rag_compare, verify_eval
from app.eval.verify_eval import (
    UnsafeOutputError,
    assert_safe_output,
    best_tau,
    build_row,
    class_scores,
    curve,
    output_dir,
    resolve_settings,
    tune,
)
from app.ingest.corpus import ClaimLabel
from app.verify import nli
from app.verify.nli import PairScore


def P(s, c):
    return {"SUPPORT": s, "CONTRADICT": c, "NEI": round(1 - s - c, 6)}


# --- output guard ---------------------------------------------------------------------------


def test_output_dir_is_always_under_data_eval_runs():
    for ds in ("beir/scifact/test", "beir/scifact/train"):
        p = output_dir(ds, 300, "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "fp32")
        assert p.parts[:2] == ("data", "eval_runs")
        assert p.name == f"verify_{ds.replace('/', '-')}_300_MoritzLaurer-DeBERTa-v3-base-mnli-fever-anli"


def test_non_frozen_runs_get_a_distinct_name():
    plain = output_dir("beir/scifact/test", 300, "m/x", "fp32")
    assert output_dir("beir/scifact/test", 300, "m/x", "int8-dynamic") != plain
    custom = output_dir("beir/scifact/test", 300, "m/x", "fp32", {"tau": 0.3})
    assert custom.name.startswith(plain.name + "_custom-")


@pytest.mark.parametrize("bad", ["eval/results", "eval/results/verify", "data/eval_runs", "data/other", "/tmp/x"])
def test_assert_safe_output_refuses_the_committed_artifact_and_outside_paths(bad):
    with pytest.raises(UnsafeOutputError):
        assert_safe_output(Path(bad))


def test_assert_safe_output_refuses_a_runs_dir_that_escapes_via_dotdot():
    with pytest.raises(UnsafeOutputError):
        assert_safe_output(Path("data/eval_runs/../../eval/results/rag"))


# --- contamination guard ----------------------------------------------------------------------


def _args(**kw):
    base = {"tune": False, "tau": None, "rule": None, "k": None, "windowing": None}
    return argparse.Namespace(**{**base, **kw})


def test_tune_is_refused_on_the_test_split():
    with pytest.raises(SystemExit, match="only allowed on beir/scifact/train"):
        resolve_settings(_args(tune=True), "beir/scifact/test", int8=False)


def test_untuned_frozen_settings_refuse_a_test_run(monkeypatch):
    monkeypatch.setitem(nli.FROZEN, "tuned_on", None)
    with pytest.raises(SystemExit, match="not been tuned"):
        resolve_settings(_args(), "beir/scifact/test", int8=nli.FROZEN["int8"])
    # Explicit values for everything are allowed, and recorded as an override.
    chosen, overrides, source = resolve_settings(
        _args(tau=0.4, rule="max", k=5, windowing="truncate"), "beir/scifact/test", nli.FROZEN["int8"]
    )
    assert chosen["tau"] == 0.4 and "OVERRIDDEN" in source and overrides["tau"] == 0.4


def test_frozen_settings_are_used_as_is(monkeypatch):
    monkeypatch.setitem(nli.FROZEN, "tuned_on", "beir/scifact/train (809 claims)")
    chosen, overrides, source = resolve_settings(_args(), "beir/scifact/test", nli.FROZEN["int8"])
    assert chosen == {k: nli.FROZEN[k] for k in ("windowing", "rule", "k", "tau")}
    assert overrides == {} and source.startswith("frozen, tuned on beir/scifact/train")


# --- metrics + tuning -------------------------------------------------------------------------


def test_class_scores_macro_f1_by_hand():
    gold = ["SUPPORT", "SUPPORT", "CONTRADICT", "NEI"]
    pred = ["SUPPORT", "NEI", "CONTRADICT", "NEI"]
    cs = class_scores(gold, pred)
    assert cs["accuracy"] == 0.75
    # SUPPORT P=1 R=.5 F1=.6667; CONTRADICT 1; NEI P=.5 R=1 F1=.6667
    assert cs["macro_f1"] == pytest.approx((0.6667 + 1 + 0.6667) / 3, abs=1e-4)
    assert cs["per_label"]["SUPPORT"]["recall"] == 0.5


def test_tau_curve_and_plateau_median():
    probs = {"a": [P(0.9, 0.0)], "b": [P(0.55, 0.0)], "c": [P(0.3, 0.0)]}
    gold = {"a": "SUPPORT", "b": "SUPPORT", "c": "NEI"}
    pts = curve(probs, gold, "max", 5)
    by = {p["tau"]: p["accuracy"] for p in pts}
    assert by[0.0] == pytest.approx(2 / 3, abs=1e-4) and by[0.5] == 1.0
    assert by[0.6] == pytest.approx(2 / 3, abs=1e-4)
    b = best_tau(pts)
    # Perfect on τ in (0.30, 0.55]: 0.31..0.55 -> 25 values, median 0.43.
    assert b["accuracy"] == 1.0 and b["n_tied"] == 25 and b["tau"] == 0.43


def test_tune_picks_the_better_windowing_and_aggregation():
    gold = {"a": "SUPPORT", "b": "NEI"}
    probs = {
        # truncation missed the late evidence on "a"; windows found it
        "truncate": {"a": [P(0.2, 0.0), P(0.1, 0.0)], "b": [P(0.2, 0.0), P(0.1, 0.0)]},
        "window": {"a": [P(0.9, 0.0), P(0.1, 0.0)], "b": [P(0.2, 0.0), P(0.1, 0.0)]},
    }
    t = tune(probs, gold)
    assert t["best"]["windowing"] == "window" and t["best"]["accuracy"] == 1.0
    assert (t["best"]["rule"], t["best"]["k"]) == ("max", 5)  # the first on a tie
    assert len(t["table"]) == 2 * len(verify_eval.TUNE_AGGREGATIONS)


# --- rows are rag.json-compatible ---------------------------------------------------------------


def _row(qid, gold, probs, top=("d1", "d2"), rationale=("d1",)):
    label = ClaimLabel(gold, set(rationale) if gold != "NEI" else set())
    scores = [PairScore(p) for p in probs]
    return build_row(qid, label, set(top), list(top), scores, 0.5, "max", 5)


def test_row_schema_and_rag_compare_accepts_it():
    rows = [
        _row("1", "SUPPORT", [P(0.1, 0.1), P(0.8, 0.1)]),
        _row("2", "CONTRADICT", [P(0.1, 0.7), P(0.1, 0.1)]),
        _row("3", "NEI", [P(0.1, 0.1), P(0.2, 0.1)]),
    ]
    r1, r2, r3 = rows
    for k in rag_compare.REQUIRED_ROW_KEYS:
        assert k in r1
    assert (r1["predicted_label"], r1["answered"], r1["chosen_doc_id"], r1["chosen_rank"]) == ("SUPPORT", True, "d2", 2)
    assert r1["evidence"] is True and r1["cited_doc_ids"] == ["d2"]
    assert r1["verdict"] == "SUPPORTED" and r2["verdict"] == "REFUTED"
    assert (r3["predicted_label"], r3["answered"], r3["cited_doc_ids"], r3["evidence"]) == ("NEI", False, [], False)
    assert r3["abstention_class"] == "correct_abstention"
    assert [p["doc_id"] for p in r1["passage_probs"]] == ["d1", "d2"]

    blob = {**verify_eval.summarize(rows), "run": {"dataset": "beir/scifact/test", "generator_model": "nli:x"}, "rows": rows}
    agg = verify_eval.summarize(rows)
    assert agg["verdict_accuracy"] == 1.0 and agg["macro_f1"] == 1.0
    llm = {"n": 3, "run": {"dataset": "beir/scifact/test", "generator_model": "ling"}, "rows": [
        {"query_id": q, "gold_label": g, "predicted_label": "NEI", "answered": False, "evidence": e,
         "cited_doc_ids": []}
        for q, g, e in (("1", "SUPPORT", True), ("2", "CONTRADICT", True), ("3", "NEI", False))
    ]}
    res = rag_compare.compare(llm, blob)
    assert res["outcomes"]["verdict_correct"]["c"] == 2  # NLI right where the LLM was wrong
    assert ("generator_model", "ling", "nli:x") in res["settings_differ"]


# --- main, end to end, on fakes ---------------------------------------------------------------


class FakeVerifier:
    model_name = "fake/nli"
    dtype = "fp32"
    max_length = 512
    batch_size = 1

    def __init__(self, windowing):
        self.windowing = windowing
        self.pairs_scored = 0
        self.forward_seconds = 0.0
        self.windows_scored = 0

    def window_tag(self, title, abstract, claim):
        return "full"

    def score(self, items):
        self.pairs_scored += len(items)
        self.windows_scored += len(items)
        out = []
        for _t, abstract, _c in items:
            if "SUPPORTS" in abstract:
                out.append(PairScore(P(0.9, 0.05)))
            elif "REFUTES" in abstract:
                out.append(PairScore(P(0.05, 0.9)))
            else:
                out.append(PairScore(P(0.1, 0.1)))
        return out


class FakeService:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, mode="hybrid", top_k=None):
        assert (mode, top_k) == ("hybrid", 5)
        self.calls += 1
        n = int(query.split()[-1])
        return [SearchHit(f"d{n}", 1.0), *[SearchHit(f"x{i}", 0.5) for i in range(4)]]


@pytest.fixture()
def fake_world(tmp_path, monkeypatch):
    queries = {str(i): f"claim number {i}" for i in range(1, 7)}
    gold = {"1": "SUPPORT", "2": "SUPPORT", "3": "CONTRADICT", "4": "CONTRADICT", "5": "NEI", "6": "NEI"}
    text = {"SUPPORT": "This SUPPORTS it.", "CONTRADICT": "This REFUTES it.", "NEI": "Unrelated."}
    docs = [{"doc_id": f"d{q}", "title": "T", "text": text[g]} for q, g in gold.items()]
    docs += [{"doc_id": f"x{i}", "title": "X", "text": "Filler."} for i in range(4)]
    labels = {q: ClaimLabel(g, {f"d{q}"} if g != "NEI" else set()) for q, g in gold.items()}
    monkeypatch.setattr(verify_eval, "load_queries_qrels", lambda ds: (queries, {q: {f"d{q}": 1} for q in queries}))
    monkeypatch.setattr(verify_eval, "load_claim_labels", lambda dataset, query_ids: labels)
    monkeypatch.setattr(verify_eval, "load_documents", lambda: docs)
    monkeypatch.setattr(verify_eval, "scifact_source_zip", lambda: tmp_path / "missing.zip")
    monkeypatch.setattr(verify_eval, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(verify_eval, "FORBIDDEN_OUT", tmp_path / "eval" / "results")
    monkeypatch.setattr(verify_eval, "RETRIEVAL_CACHE", tmp_path / "retrieval_cache")
    monkeypatch.setattr(nli, "CACHE_DIR", tmp_path / "nli_cache")
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    monkeypatch.delenv("SSR_EVAL_LIMIT", raising=False)
    monkeypatch.setenv("SSR_RAG_N", "all")
    return tmp_path


def test_main_tunes_on_train_and_writes_only_under_runs(fake_world, monkeypatch):
    monkeypatch.setenv("SSR_RAG_DATASET", "beir/scifact/train")
    service = FakeService()
    out = verify_eval.main(["--tune"], verifier_factory=FakeVerifier, service_factory=lambda: service, log=lambda *a: None)
    assert out.parent == fake_world / "runs"
    assert not (fake_world / "eval").exists()
    blob = json.loads((out / "rag.json").read_text())
    assert blob["verdict_accuracy"] == 1.0 and blob["n"] == 6
    assert blob["run"]["settings_source"].startswith("tuned in-sample on beir/scifact/train")
    assert blob["run"]["sample_seed"] == 13 and blob["run"]["canonical"] is False
    assert (out / "tuning.json").exists() and (out / "tuning.md").exists()
    rag_compare.load_run(out / "rag.json")  # valid rag_compare input
    assert service.calls == 6

    # Re-running is free: retrieval and every pair come from the caches.
    scorer = {}

    def factory(w):
        scorer["v"] = FakeVerifier(w)
        return scorer["v"]

    verify_eval.main(["--tune"], verifier_factory=factory,
                     service_factory=lambda: pytest.fail("retrieval should be cached"), log=lambda *a: None)
    assert scorer["v"].pairs_scored == 0


def test_main_on_test_split_uses_frozen_settings(fake_world, monkeypatch):
    monkeypatch.setenv("SSR_RAG_DATASET", "beir/scifact/test")
    monkeypatch.setitem(nli.FROZEN, "tuned_on", "beir/scifact/train (fake)")
    out = verify_eval.main([], verifier_factory=FakeVerifier, service_factory=FakeService, log=lambda *a: None)
    blob = json.loads((out / "rag.json").read_text())
    assert blob["run"]["verifier"]["tau"] == nli.FROZEN["tau"]
    assert blob["run"]["overrides"] == {}
    assert not (out / "tuning.json").exists()
    assert out.name == "verify_beir-scifact-test_6_fake-nli"
