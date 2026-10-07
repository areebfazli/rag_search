"""verify_combine reads Ling's rows as the first pass left them (the verdict re-ask undone),
for the test run, the train tuning run and the rag_compare baseline. Fakes only — no
model, no corpus, no network."""
import json

import pytest

from app.eval import rag_eval, verify_combine, verify_eval
from app.ingest import corpus
from app.verify import nli
from app.verify import train_data as td
from app.verify.nli import PairScore
from tests.test_verify_trained import FROZEN_OK, make_trained

TRUNCATED = "[Answer truncated: the model ran out of its token budget.]"
REASK_RUN_KEYS = {"reask": {"enabled": True}, "reask_cache": "x", "reask_replies": {"cached": 1}}


def first_pass(qid: str, gold: str, label: str, verdict, answered: bool, answer: str) -> dict:
    """A rag_eval row as the first pass stores it."""
    return {
        "query_id": qid, "gold_label": gold, "predicted_label": label, "verdict": verdict,
        "verdict_source": "line" if verdict else None, "answered": answered,
        "answered_source": "verdict" if verdict else "judge", "judge_answered": answered,
        "abstention_class": rag_eval._abstention_class(answered, gold != "NEI"),
        "abstention_class_qrels": rag_eval._abstention_class(answered, True),
        "evidence": gold != "NEI", "evidence_qrels": True, "answer": answer,
        "cited_doc_ids": [], "retrieved_doc_ids": ["d1", "d2"], "faithfulness": 1.0,
        "context_relevance": 1.0, "truncated": verdict is None, "generation_attempts": 1,
    }


def reasked_blob(dataset: str) -> tuple[dict, list[dict]]:
    """(rag.json as rag_eval now writes it, its first-pass rows). Claim 1 had no verdict
    and the re-ask supplied SUPPORTED (gold SUPPORT: the re-ask made it correct); claim 2
    already had one (re-ask not attempted)."""
    fp = [first_pass("1", "SUPPORT", "NEI", None, False, TRUNCATED),
          first_pass("2", "SUPPORT", "SUPPORT", "SUPPORTED", True, "It holds [1].")]
    rows = [rag_eval.apply_reask(fp[0], {"raw": "Verdict: SUPPORTED [1]", "finish_reason": "stop"}),
            {**fp[1], "reask_attempted": False}]
    assert rows[0]["predicted_label"] == "SUPPORT" and rows[0]["verdict_source"] == "reask"
    run = {"dataset": dataset, "git_sha": "f" * 40, "generator_model": "ling", **REASK_RUN_KEYS}
    return {"n": 2, "run": run, "rows": rows}, fp


def test_ling_first_pass_undoes_the_reask_rows_and_run_settings():
    blob, fp = reasked_blob("beir/scifact/test")
    out = verify_combine.ling_first_pass(blob)
    assert out["rows"] == fp
    assert not any(k.startswith("reask") for k in out["run"]) and "first_pass" in out["run"]["ling_rows"]
    assert blob["rows"][0]["verdict_source"] == "reask"  # the input is not mutated
    # A pre-re-ask file (the one the committed verify_combined_* outputs were built from)
    # passes through unchanged.
    old = {"run": {"dataset": "beir/scifact/test"}, "rows": fp}
    assert verify_combine.ling_first_pass(old) == old


def test_train_tuning_rows_are_the_first_pass_reparsed(tmp_path, monkeypatch):
    blob, _ = reasked_blob(verify_combine.TRAIN_DATASET)
    path = tmp_path / "train_rag.json"
    path.write_text(json.dumps(blob))
    monkeypatch.setattr(td, "TUNING_RUN", path)
    monkeypatch.setattr(corpus, "load_queries_qrels", lambda ds: ({"1": "claim one", "2": "claim two"}, {}))
    rows, run, _ = verify_combine._ling_train_rows()
    by = {r["query_id"]: r for r in rows}
    # The re-ask's SUPPORT is gone: the first pass had no verdict, and none is parseable.
    # Its reply was cut off at the token budget, so it is NONE (never NEI), whatever the
    # judge's `answered` said (rag_eval.predicted_label).
    assert by["1"]["predicted_label"] == rag_eval.NO_VERDICT and by["1"]["verdict"] is None
    assert not any(k in r for r in rows for k in ("first_pass", "reask", "reask_attempted"))
    assert "reask" not in run and by["2"]["predicted_label"] == "SUPPORT"


def test_test_stage_combines_and_compares_against_the_first_pass(tmp_path, monkeypatch):
    blob, fp = reasked_blob(verify_combine.TEST_DATASET)
    test_run = tmp_path / "rag.json"
    test_run.write_text(json.dumps(blob))
    runs = tmp_path / "runs"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({**FROZEN_OK, "checkpoint_sha": "a" * 64}))  # R0: Ling alone
    monkeypatch.setattr(verify_eval, "RUNS", runs)
    monkeypatch.setattr(verify_combine, "TEST_RUN", test_run)
    monkeypatch.setattr(verify_combine, "N_TEST", 2)
    monkeypatch.setattr(verify_combine, "FROZEN_PATH", frozen)
    monkeypatch.setattr(verify_combine, "test_dir", lambda rule: runs / rule)
    monkeypatch.setattr(verify_combine, "R1_TEST_DIR", runs / "r1")
    monkeypatch.setattr(verify_combine, "load_verifier", lambda w="truncate": make_trained("a" * 64))
    monkeypatch.setattr(nli, "CACHE_DIR", tmp_path / "nli")
    monkeypatch.setattr(corpus, "load_queries_qrels", lambda ds: ({"1": "c1", "2": "c2"}, {}))
    monkeypatch.setattr(corpus, "load_documents", lambda: [])
    seen = []
    weak = [PairScore({"SUPPORT": 0.1, "CONTRADICT": 0.1, "NEI": 0.8})] * 2

    def fake_score_rows(verifier, cache, rows, queries, docs, log=print):
        seen.extend(rows)
        return {r["query_id"]: weak for r in rows}

    monkeypatch.setattr(verify_combine, "score_rows", fake_score_rows)
    verify_combine.stage_test(log=lambda *_: None)
    assert seen == fp  # the verifier is paired with the first-pass rows
    out = json.loads((runs / "R0" / "rag.json").read_text())
    assert [r["ling_predicted_label"] for r in out["rows"]] == ["NEI", "SUPPORT"]
    assert out["verdict_accuracy"] == 0.5  # the first pass got claim 1 wrong
    assert not any(k.startswith("reask") for k in out["run"]) and "ling_rows" in out["run"]
    # R0 is Ling's first pass, and so is the baseline: no discordant claims. Against the
    # re-asked rows it would show claim 1 as broken (b=1).
    cmp = json.loads((runs / "R0" / "compare.json").read_text())
    assert (cmp["outcomes"]["verdict_correct"]["b"], cmp["outcomes"]["verdict_correct"]["c"]) == (0, 0)


@pytest.mark.skipif(not verify_combine.TEST_RUN.exists(), reason="needs the committed rag.json")
def test_committed_rows_map_to_the_rows_the_combined_runs_were_built_from():
    # Opportunistic: when the committed combined test run exists, its Ling labels are
    # exactly the first pass of the committed rag.json (so a --rerun means the same).
    combined = verify_combine.test_dir("R3") / "rag.json"
    if not combined.exists():
        pytest.skip("no local verify_combined_R3_test_300 output")
    fp = {r["query_id"]: r for r in verify_combine.ling_first_pass(
        json.loads(verify_combine.TEST_RUN.read_text()))["rows"]}
    for r in json.loads(combined.read_text())["rows"]:
        assert r["ling_predicted_label"] == fp[r["query_id"]]["predicted_label"]
