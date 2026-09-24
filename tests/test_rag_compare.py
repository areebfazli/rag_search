"""Tests for the paired rag_eval comparison (exact McNemar). Pure files, no network."""
import json
import math

import pytest

from app.eval import rag_compare
from app.eval.rag_compare import CompareError, mcnemar_exact

# --- the exact McNemar p-value, against hand-computed binomial tails -----------------------


def test_no_discordant_pairs_is_p_one():
    assert mcnemar_exact(0, 0) == 1.0


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (1, 0, 1.0),  # n=1: 2 * C(1,0)/2 = 1
        (5, 0, 2 / 32),  # 2 * 0.5^5 = 0.0625: not significant
        (6, 0, 2 / 64),  # 0.03125: the smallest all-one-way n that is
        (3, 7, 2 * (1 + 10 + 45 + 120) / 1024),  # 2 * P[X<=3 | n=10] = 0.34375
        (7, 3, 0.34375),  # symmetric in b, c
        (10, 10, 1.0),  # 2 * P[X<=10 | n=20] > 1, capped
        (0, 12, 2 / 4096),
    ],
)
def test_mcnemar_exact_matches_hand_computed_tails(b, c, p):
    assert mcnemar_exact(b, c) == pytest.approx(p, rel=1e-12)


def test_mcnemar_exact_handles_large_n_without_overflow():
    p = mcnemar_exact(150, 150)
    assert p == 1.0
    assert 0 < mcnemar_exact(0, 300) < 1e-80 and math.isfinite(mcnemar_exact(0, 300))


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


# --- pairing, outcomes, settings diff --------------------------------------------------------


def row(qid, gold="SUPPORT", pred="SUPPORT", answered=True, evidence=True, cited=("d1",)):
    return {
        "query_id": qid,
        "gold_label": gold,
        "predicted_label": pred,
        "answered": answered,
        "evidence": evidence,
        "cited_doc_ids": list(cited),
    }


def run(rows, **run_fields):
    base = {
        "dataset": "beir/scifact/test",
        "generator_provider": "openrouter",
        "generator_model": "gen:free",
        "judge_provider": "openrouter",
        "judge_model": "judge:free",
        "prompt_hash": "p1",
        "judge_prompt_hash": "j1",
        "n_requested": len(rows),
        "sample_seed": 13,
    }
    return {"n": len(rows), "run": {**base, **run_fields}, "rows": rows}


def write(tmp_path, name, blob):
    p = tmp_path / name
    p.write_text(json.dumps(blob))
    return p


def test_verdict_table_counts_discordant_pairs():
    # q1 both right, q2 A only (b), q3/q4 B only (c), q5 both wrong.
    a = run([row("1"), row("2"), row("3", pred="NEI"), row("4", pred="NEI"), row("5", pred="NEI")])
    b = run([row("1"), row("2", pred="NEI"), row("3"), row("4"), row("5", pred="NONE")])
    out = rag_compare.compare(a, b)["outcomes"]["verdict_correct"]
    assert (out["both"], out["b"], out["c"], out["neither"]) == (1, 1, 2, 1)
    assert out["acc_a"] == 0.4 and out["acc_b"] == 0.6 and out["delta"] == 0.2
    assert out["p"] == pytest.approx(mcnemar_exact(1, 2)) == 1.0


def test_abstention_and_citation_outcomes():
    a = run([
        row("1", answered=True, evidence=True),  # right
        row("2", answered=True, evidence=False),  # answered without evidence: wrong
        row("3", answered=False, evidence=False, cited=()),  # correct abstention
    ])
    b = run([
        row("1", answered=False, evidence=True),  # false abstention: wrong
        row("2", answered=False, evidence=False),  # right
        row("3", answered=False, evidence=False, cited=()),
    ])
    res = rag_compare.compare(a, b)["outcomes"]
    ab = res["abstention_correct"]
    assert (ab["b"], ab["c"], ab["both"], ab["neither"]) == (1, 1, 1, 0)
    cit = res["has_citation"]
    assert (cit["both"], cit["neither"], cit["b"], cit["c"]) == (2, 1, 0, 0)
    assert cit["p"] == 1.0 and cit["delta"] == 0.0


def test_refuses_mismatched_claim_sets_unless_intersect():
    a = run([row("1"), row("2"), row("3")])
    b = run([row("2"), row("3"), row("4")])
    with pytest.raises(CompareError, match="claim sets differ: 1 only in A, 1 only in B"):
        rag_compare.compare(a, b)
    res = rag_compare.compare(a, b, intersect=True)
    assert res["pairing"] == {"n_paired": 2, "only_a": 1, "only_b": 1, "intersected": True}
    md = rag_compare.to_markdown(res, "A.json", "B.json")
    assert "INTERSECTION only: 1 claims only in A and 1 only in B were dropped" in md


def test_refuses_disjoint_runs_and_gold_label_clashes():
    with pytest.raises(CompareError, match="share no claims"):
        rag_compare.compare(run([row("1")]), run([row("2")]), intersect=True)
    with pytest.raises(CompareError, match="gold labels differ"):
        rag_compare.compare(run([row("1")]), run([row("1", gold="NEI")]))


def test_settings_diff_names_what_changed():
    a = run([row("1")])
    b = run([row("1")], generator_model="other:free", prompt_hash="p2", n_requested="all")
    res = rag_compare.compare(a, b)
    differ = {k: (va, vb) for k, va, vb in res["settings_differ"]}
    assert differ == {
        "generator_model": ("gen:free", "other:free"),
        "prompt_hash": ("p1", "p2"),
        "n_requested": (1, "all"),
    }
    assert "dataset" in res["settings_same"] and "judge_model" in res["settings_same"]


def test_old_rag_json_dataset_falls_back_to_label_source():
    a = run([row("1")])
    del a["run"]["dataset"]
    a["run"]["label_source"] = {"dataset": "beir/scifact/test"}
    assert "dataset" in rag_compare.compare(a, run([row("1")]))["settings_same"]


def test_load_run_rejects_bad_files(tmp_path):
    with pytest.raises(CompareError, match="cannot read"):
        rag_compare.load_run(tmp_path / "missing.json")
    with pytest.raises(CompareError, match="no `rows`"):
        rag_compare.load_run(write(tmp_path, "x.json", {"run": {}}))
    with pytest.raises(CompareError, match="duplicate query_id"):
        rag_compare.load_run(write(tmp_path, "d.json", run([row("1"), row("1")])))
    with pytest.raises(CompareError, match="lacks"):
        rag_compare.load_run(write(tmp_path, "k.json", run([{"query_id": "1"}])))


def test_main_prints_markdown_and_writes_out(tmp_path, capsys):
    a = write(tmp_path, "a.json", run([row("1"), row("2", pred="NEI")]))
    b = write(tmp_path, "b.json", run([row("1"), row("2")], judge_model="j2:free"))
    out = tmp_path / "reports" / "cmp.md"
    rag_compare.main([str(a), str(b), "--out", str(out)])
    printed = capsys.readouterr().out
    assert printed.startswith("# RAG run comparison")
    assert "| Verdict correct (predicted == gold) | 0.5000 | 1.0000 | +0.5000 | 0 | 1 |" in printed
    assert "| judge_model | `judge:free` | `j2:free` |" in printed
    assert out.read_text() in printed


def test_main_refuses_mismatched_sets_with_a_clear_exit(tmp_path):
    a = write(tmp_path, "a.json", run([row("1")]))
    b = write(tmp_path, "b.json", run([row("2")]))
    with pytest.raises(SystemExit, match="rag_compare: claim sets differ"):
        rag_compare.main([str(a), str(b)])
