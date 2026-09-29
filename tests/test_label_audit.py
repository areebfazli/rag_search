"""Tests for the SciFact label-audit re-score. Fakes only: no network, no data/ files."""
import hashlib
import json

import pytest

from app.eval import label_audit, rag_compare
from app.eval.label_audit import Audit, AuditError, Correction
from app.ingest.corpus import ClaimLabel


def corr(claim, orig, new, doc=None, etype="label_reversal"):
    return Correction(str(claim), doc, orig, new, etype)


LABELS = {
    "1": ClaimLabel("SUPPORT", {"d1"}),
    "2": ClaimLabel("CONTRADICT", {"d2"}),
    "3": ClaimLabel("SUPPORT", {"d3a", "d3b", "d3c"}),  # a multi-doc claim
    "4": ClaimLabel("NEI"),
    "5": ClaimLabel("SUPPORT", {"d5"}),  # debatable in the fake audit
}


# --- pair corrections -> claim labels ----------------------------------------------------------


def test_null_doc_flip_on_a_single_doc_claim():
    out = label_audit.apply_corrections(LABELS, [corr(1, "SUPPORT", "CONTRADICT")])
    assert out["1"] == ClaimLabel("CONTRADICT", {"d1"})
    assert out["2"] is LABELS["2"]  # untouched claims keep their object
    assert LABELS["1"].label == "SUPPORT"  # the input is not mutated


def test_null_doc_nei_empties_the_rationale_set():
    out = label_audit.apply_corrections(LABELS, [corr(2, "CONTRADICT", "NEI", etype="entity_mismatch")])
    assert out["2"] == ClaimLabel("NEI") and out["2"].rationale_doc_ids == set()


def test_one_doc_of_several_to_nei_keeps_the_claim_label():
    out = label_audit.apply_corrections(LABELS, [corr(3, "SUPPORT", "NEI", doc="d3c")])
    assert out["3"] == ClaimLabel("SUPPORT", {"d3a", "d3b"})
    assert label_audit.label_changed(LABELS["3"], out["3"])  # rationale set changed


def test_all_docs_of_a_multi_doc_claim_to_nei_is_nei():
    out = label_audit.apply_corrections(LABELS, [corr(3, "SUPPORT", "NEI")])
    assert out["3"] == ClaimLabel("NEI")


def test_one_doc_flipped_on_a_multi_doc_claim_mixes_labels_and_is_refused():
    with pytest.raises(AuditError, match="mix"):
        label_audit.apply_corrections(LABELS, [corr(3, "SUPPORT", "CONTRADICT", doc="d3a")])


def test_every_doc_flipped_one_by_one_gives_the_new_label():
    cs = [corr(3, "SUPPORT", "CONTRADICT", doc=d) for d in ("d3a", "d3b", "d3c")]
    assert label_audit.apply_corrections(LABELS, cs)["3"] == ClaimLabel("CONTRADICT", {"d3a", "d3b", "d3c"})


@pytest.mark.parametrize(
    ("c", "match"),
    [
        (corr(1, "CONTRADICT", "NEI"), "audit says original CONTRADICT, ours is SUPPORT"),
        (corr(1, "SUPPORT", "NEI", doc="dX"), "not one of its rationale docs"),
        (corr(99, "SUPPORT", "NEI"), "not in this split"),
        (corr(4, "SUPPORT", "NEI"), "no evidence pair"),
    ],
)
def test_mapping_refuses_to_guess(c, match):
    with pytest.raises(AuditError, match=match):
        label_audit.apply_corrections(LABELS, [c])


# --- parsing the audit files ------------------------------------------------------------------


def corrections_json(entries):
    return json.dumps({"corrections": entries})


def entry(claim, doc, orig, new, etype="label_reversal"):
    return {"claim_id": claim, "doc_id": doc, "original_label": orig, "corrected_label": new,
            "error_type": etype, "justification": "j"}


def test_per_document_entries_are_not_strict():
    strict, per_doc = label_audit.parse_corrections(corrections_json([
        entry(1, None, "SUPPORT", "CONTRADICT"),
        entry(3, None, "SUPPORT", "NEI", etype=label_audit.PER_DOCUMENT_ERROR),
    ]))
    assert [c.claim_id for c in strict] == ["1"] and [c.claim_id for c in per_doc] == ["3"]
    assert strict[0].doc_id is None


def test_int_doc_ids_become_strings():
    strict, _ = label_audit.parse_corrections(corrections_json([entry(1, 123, "SUPPORT", "NEI")]))
    assert strict[0].doc_id == "123"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"no": "corrections"}),
        corrections_json([{"claim_id": 1}]),
        corrections_json([entry(1, None, "SUPPORT", "MAYBE")]),
        corrections_json([entry(1, None, "NEI", "SUPPORT")]),
        corrections_json([entry(1, None, "SUPPORT", "SUPPORT")]),
        corrections_json([entry(1, None, "SUPPORT", "NEI"), entry(1, None, "SUPPORT", "CONTRADICT")]),
    ],
)
def test_malformed_corrections_are_refused(text):
    with pytest.raises(AuditError):
        label_audit.parse_corrections(text)


CSV = "id,claim,my_verdict\n5,c,DEBATABLE\n6,c,GOLD OK\n2,c,DEBATABLE\n"
TEX = "The 8 final debatable claim IDs are 5.\n"


def test_debatable_is_csv_minus_upgraded_and_must_match_the_paper():
    assert label_audit.parse_debatable([CSV], TEX, corrected_ids={"2"}) == {"5"}
    with pytest.raises(AuditError, match="disagree"):
        label_audit.parse_debatable([CSV], TEX, corrected_ids=set())  # 2 not upgraded
    with pytest.raises(AuditError, match="no 'final debatable"):
        label_audit.parse_debatable([CSV], "nothing here", corrected_ids={"2"})


def test_pinned_files_are_fetched_verified_and_never_overwritten(tmp_path, monkeypatch):
    blobs = {rel: rel.encode() for rel in label_audit.AUDIT_FILES}
    monkeypatch.setattr(label_audit, "AUDIT_FILES",
                        {rel: hashlib.sha256(b).hexdigest() for rel, b in blobs.items()})
    urls = []

    def opener(url):
        urls.append(url)
        return blobs[url.split(label_audit.AUDIT_COMMIT + "/", 1)[1]]

    with pytest.raises(AuditError, match="missing"):
        label_audit.ensure_audit_files(tmp_path, fetch=False)
    label_audit.ensure_audit_files(tmp_path, opener=opener)
    assert len(urls) == len(blobs) and all(label_audit.AUDIT_COMMIT in u for u in urls)
    label_audit.ensure_audit_files(tmp_path, opener=opener)  # present: no second download
    assert len(urls) == len(blobs)
    p = label_audit.audit_path(label_audit.CORRECTIONS_FILE, tmp_path)
    p.write_text("tampered")
    with pytest.raises(AuditError, match="sha256"):
        label_audit.ensure_audit_files(tmp_path, opener=opener)
    assert p.read_text() == "tampered"
    with pytest.raises(AuditError, match="sha256"):  # a bad download is not written
        label_audit.ensure_audit_files(tmp_path / "x", opener=lambda url: b"wrong")


# --- re-scoring -------------------------------------------------------------------------------

AUDIT = Audit(
    corrections=(corr(1, "SUPPORT", "CONTRADICT"), corr(2, "CONTRADICT", "NEI", etype="entity_mismatch")),
    per_document=(corr(3, "SUPPORT", "NEI", etype=label_audit.PER_DOCUMENT_ERROR),),
    debatable=frozenset({"5"}),
)


def row(qid, pred, answered=True, retrieved=()):
    lab = LABELS[qid]
    retrieved = list(retrieved)
    ev = bool(lab.rationale_doc_ids & set(retrieved))
    return {
        "query_id": qid, "gold_label": lab.label, "predicted_label": pred, "verdict": pred,
        "answered": answered, "evidence": ev, "evidence_qrels": ev,
        "rationale_doc_ids": sorted(lab.rationale_doc_ids), "retrieved_doc_ids": retrieved,
        "abstention_class": label_audit._abstention_class(answered, ev), "cited_doc_ids": retrieved[:1],
    }


def blob(rows, **agg):
    run = {"dataset": label_audit.AUDIT_DATASET, "git_sha": "abc", "canonical": True}
    return {"n": len(rows), **agg, "run": run, "rows": rows}


ROWS = [
    row("1", "CONTRADICT", retrieved=["d1"]),  # wrong -> right
    row("2", "NEI", answered=False, retrieved=["d2"]),  # wrong (false abstention) -> right
    row("3", "SUPPORT", retrieved=["d3c"]),  # per-document claim: never changed
    row("4", "NEI", answered=False),  # right either way
    row("5", "NEI", answered=False, retrieved=["d5"]),  # debatable, wrong
]


def test_three_tiers_strict_vs_debatable_excluded():
    rep = label_audit.audit_run(blob(ROWS, verdict_accuracy=0.4), LABELS, AUDIT, source="x/rag.json")
    t = rep["tiers"]
    assert (t["original"]["n"], t["original"]["correct"]) == (5, 2)
    assert (t["corrected_strict"]["n"], t["corrected_strict"]["correct"]) == (5, 4)
    assert (t["corrected_excl_debatable"]["n"], t["corrected_excl_debatable"]["correct"]) == (4, 4)
    assert t["corrected_excl_debatable"]["verdict_accuracy"] == 1.0
    s = t["corrected_strict"]["per_label"]
    assert s["CONTRADICT"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert s["NEI"] == {"n": 2, "correct": 2, "accuracy": 1.0}
    assert t["corrected_strict"]["confusion"]["CONTRADICT"]["CONTRADICT"] == 1
    assert rep["n_label_changed"] == 2 and rep["n_debatable_in_run"] == 1


def test_claim_relabelled_nei_loses_its_evidence_and_abstaining_becomes_correct():
    rep = label_audit.audit_run(blob(ROWS), LABELS, AUDIT)
    c2 = next(c for c in rep["changed_claims"] if c["query_id"] == "2")
    assert (c2["evidence_before"], c2["evidence_after"]) == (True, False)
    assert (c2["abstention_class_before"], c2["abstention_class_after"]) == (
        "false_abstention", "correct_abstention")
    assert rep["tiers"]["original"]["quadrants"]["false_abstention"] == 2  # claims 2 and 5
    assert rep["tiers"]["corrected_strict"]["quadrants"]["false_abstention"] == 1  # only 5
    assert rep["tiers"]["corrected_strict"]["abstention_recall"] == 1.0


def test_per_document_error_is_reported_not_applied(monkeypatch):
    monkeypatch.setattr(label_audit, "PER_DOCUMENT_DOCS", {"3": "d3c"})
    rep = label_audit.audit_run(blob(ROWS), LABELS, AUDIT)
    assert "3" not in {c["query_id"] for c in rep["changed_claims"]}
    (p,) = rep["per_document"]
    assert p["applied"] is False and p["claim_label_if_applied_per_document"] == "SUPPORT"
    assert p["rationale_docs"] == "3 -> 2" and p["evidence_flag_would_change"] is True


def test_report_never_replaces_the_original_numbers(tmp_path):
    src = blob(ROWS, verdict_accuracy=0.4)
    before = json.dumps(src, sort_keys=True)
    rep = label_audit.audit_run(src, LABELS, AUDIT, source="eval/results/rag.json")
    assert json.dumps(src, sort_keys=True) == before  # input untouched
    assert rep["headline"] == "original" and rep["secondary_artifact"] is True
    assert rep["tiers"]["original"]["verdict_accuracy"] == 0.4
    md = label_audit.to_markdown(rep)
    assert "headline stays the original SciFact labels" in md and "single annotator" in md
    assert md.index("Original SciFact labels (headline)") < md.index("Corrected labels — strict")
    out_json, out_md = label_audit.output_paths(tmp_path / "rag.json")
    assert (out_json.name, out_md.name) == ("rag_label_audit.json", "rag_label_audit.md")
    # A run whose stored headline the original tier can't reproduce is refused.
    with pytest.raises(AuditError, match="stores 0.9"):
        label_audit.audit_run(blob(ROWS, verdict_accuracy=0.9), LABELS, AUDIT)


def test_refuses_other_splits_and_rows_scored_on_other_labels():
    with pytest.raises(AuditError, match="only covers"):
        label_audit.audit_run({**blob(ROWS), "run": {"dataset": "beir/scifact/train"}}, LABELS, AUDIT)
    bad = [dict(ROWS[0], gold_label="NEI"), *ROWS[1:]]
    with pytest.raises(AuditError, match="different label source"):
        label_audit.audit_run(blob(bad), LABELS, AUDIT)


def test_main_refuses_non_canonical_runs_into_eval_results(tmp_path, monkeypatch):
    out = tmp_path / "eval" / "results"
    out.mkdir(parents=True)
    (out / "rag.json").write_text(json.dumps({**blob(ROWS), "run": {"dataset": label_audit.AUDIT_DATASET}}))
    monkeypatch.setattr(label_audit.rag_eval, "OUT", out)
    monkeypatch.setattr(label_audit, "load_audit", lambda fetch=True: AUDIT)
    monkeypatch.setattr(label_audit, "load_claim_labels", lambda ds: LABELS)
    with pytest.raises(SystemExit, match="only the canonical run"):
        label_audit.main([str(out / "rag.json")])
    assert sorted(p.name for p in out.iterdir()) == ["rag.json"]


def test_main_writes_next_to_the_input(tmp_path, monkeypatch):
    src = tmp_path / "run" / "rag.json"
    src.parent.mkdir()
    src.write_text(json.dumps(blob(ROWS, verdict_accuracy=0.4)))
    monkeypatch.setattr(label_audit, "load_audit", lambda fetch=True: AUDIT)
    monkeypatch.setattr(label_audit, "load_claim_labels", lambda ds: LABELS)
    label_audit.main([str(src)])
    assert json.loads(src.read_text())["verdict_accuracy"] == 0.4
    rep = json.loads((src.parent / "rag_label_audit.json").read_text())
    assert rep["tiers"]["corrected_strict"]["correct"] == 4


# --- rag_compare under corrected labels ----------------------------------------------------------


def test_rag_compare_pairs_under_the_audit_key(monkeypatch):
    a = blob(ROWS)
    b = blob([dict(r, predicted_label="SUPPORT" if r["query_id"] == "1" else r["predicted_label"])
              for r in ROWS])
    orig = rag_compare.compare(a, b)["outcomes"]["verdict_correct"]
    assert (orig["b"], orig["c"]) == (0, 1)  # under the original key, B's SUPPORT on 1 is right
    monkeypatch.setattr(label_audit, "load_audit", lambda fetch=True: AUDIT)
    monkeypatch.setattr("app.ingest.corpus.load_claim_labels", lambda ds: LABELS)
    ra, rb = rag_compare.relabel_runs(a, b, "corrected_strict")
    fixed = rag_compare.compare(ra, rb)["outcomes"]["verdict_correct"]
    assert (fixed["b"], fixed["c"]) == (1, 0)  # under the audit key it is A that is right
    xa, xb = rag_compare.relabel_runs(a, b, "corrected_excl_debatable")
    assert rag_compare.compare(xa, xb)["pairing"]["n_paired"] == 4
    res = rag_compare.compare(ra, rb)
    res["labels"] = "audit"
    assert "Answer key: **audit**" in rag_compare.to_markdown(res, "A", "B")
    assert "Answer key" not in rag_compare.to_markdown(rag_compare.compare(a, b), "A", "B")
