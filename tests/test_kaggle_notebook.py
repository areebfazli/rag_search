"""The Kaggle notebook's v2 helpers: class weights + oversampling, early stopping, the LR
schedule, the confusion matrix, and the stage-1 external-data path (spec validation,
label mapping, SciFact contamination re-check, dedup, caps) — pure Python, fake files,
a monkeypatched downloader. No network."""
import json

import pytest

from app.verify import kaggle_export as ke
from app.verify import kaggle_notebook as kn
from tests.test_kaggle_export import fake_spec

# --- class weights and oversampling -------------------------------------------------------------


def test_class_weight_modes():
    c = {"SUPPORT": 20, "CONTRADICT": 5, "NEI": 75}
    inv = kn.compute_class_weights(c, "inverse")
    assert inv == {"SUPPORT": round(100 / 60, 6), "CONTRADICT": round(100 / 15, 6), "NEI": round(100 / 225, 6)}
    assert sum(inv[k] * c[k] for k in c) == pytest.approx(100)  # balanced: each label weighs n/3
    sq = kn.compute_class_weights(c, "sqrt_inverse")
    assert sq["CONTRADICT"] == pytest.approx(inv["CONTRADICT"] ** 0.5, abs=1e-5)
    assert sq["NEI"] < 1 < sq["SUPPORT"] < sq["CONTRADICT"] < inv["CONTRADICT"]
    assert kn.compute_class_weights(c, "none") == {"SUPPORT": 1.0, "CONTRADICT": 1.0, "NEI": 1.0}
    assert kn.compute_class_weights({"SUPPORT": 3, "NEI": 3}, "inverse")["CONTRADICT"] == 0.0
    with pytest.raises(AssertionError):
        kn.compute_class_weights(c, "focal")
    rows = [{"label": lab} for lab in ["SUPPORT"] * 20 + ["CONTRADICT"] * 5 + ["NEI"] * 75]
    assert kn.class_weights(rows) == inv


def test_oversampling_duplicates_one_label_and_weights_follow_the_effective_set():
    rows = [{"qid": str(i), "label": lab} for i, lab in enumerate(["SUPPORT"] * 4 + ["CONTRADICT"] * 2 + ["NEI"] * 10)]
    out = kn.oversample(rows, {"CONTRADICT": 2})
    assert kn.label_counts(out) == {"SUPPORT": 4, "CONTRADICT": 4, "NEI": 10}
    assert out[: len(rows)] == rows and [r["qid"] for r in out[len(rows):]] == ["4", "5"]
    half = kn.oversample(rows, {"NEI": 1.5}, seed=1)
    assert kn.label_counts(half)["NEI"] == 15 and half == kn.oversample(rows, {"NEI": 1.5}, seed=1)
    assert kn.oversample(rows, {}) == rows
    w = kn.compute_class_weights(kn.label_counts(out))
    assert w["CONTRADICT"] == w["SUPPORT"] == round(18 / 12, 6)


# --- early stopping and the schedule ---------------------------------------------------------------


def test_early_stopping_keeps_the_best_and_stops_after_patience():
    es = kn.EarlyStopping(patience=3)
    steps = [es.step(s, e) for e, s in enumerate([0.577, 0.489, 0.588, 0.58, 0.588, 0.50], start=1)]
    assert steps == [(True, False), (False, False), (True, False), (False, False), (False, False), (False, True)]
    assert (es.best, es.best_epoch, es.stopped_epoch) == (0.588, 3, 6)  # a tie is not a gain
    es = kn.EarlyStopping(patience=1, min_delta=0.01)
    assert es.step(0.5, 1) == (True, False) and es.step(0.505, 2) == (False, True) and es.best == 0.5
    never = kn.EarlyStopping(patience=0)
    assert all(not never.step(0.1, e)[1] for e in range(1, 20))


def test_lr_schedules_warm_up_then_decay_to_zero():
    for sched in ("cosine", "linear"):
        f = lambda s, sched=sched: kn.lr_lambda(s, 10, 110, sched)  # noqa: E731
        assert f(0) == 0 and f(5) == 0.5 and f(10) == 1.0
        assert f(60) == pytest.approx(0.5) and f(110) == pytest.approx(0.0) and f(500) == pytest.approx(0.0)
        assert all(f(s) >= f(s + 1) for s in range(10, 110))
    assert kn.lr_lambda(35, 10, 110, "cosine") > kn.lr_lambda(35, 10, 110, "linear")  # cosine holds lr longer
    with pytest.raises(ValueError):
        kn.lr_lambda(50, 10, 110, "step")


def test_confusion_matrix_rows_gold_columns_predicted():
    gold = ["SUPPORT", "SUPPORT", "CONTRADICT", "NEI", "NEI", "NEI"]
    pred = ["SUPPORT", "NEI", "SUPPORT", "NEI", "NEI", "CONTRADICT"]
    m = kn.confusion_matrix(gold, pred)
    assert m == [[1, 0, 1], [1, 0, 0], [0, 1, 2]]
    text = kn.format_confusion(m)
    assert "gold SUPP" in text and "pred CONT" in text and len(text.splitlines()) == 4


# --- stage-1 external data --------------------------------------------------------------------------


RECORDS = [
    {"c": "Masks reduce transmission", "e": ["Masks cut", "spread."], "y": "SUPPORTED", "t": "Mask study"},
    {"c": "Vitamin C cures COVID", "e": "No effect was seen.", "y": "REFUTED", "t": ""},
    {"c": "Zinc helps", "e": "Unrelated text.", "y": "NEUTRAL", "t": None},
    {"c": "x", "e": "", "y": "SUPPORTED"},  # empty evidence
    {"c": "y", "e": "z", "y": "UNVERIFIED"},  # listed in drop_labels
    {"c": "w", "e": "v", "y": "mixture"},  # unknown label
]


def test_label_mapping_and_row_building():
    spec = fake_spec(columns={"claim": "c", "evidence": "e", "label": "y", "title": "t"}, drop_labels=["UNVERIFIED"])
    assert kn.map_label(spec, " SUPPORTED ") == "SUPPORT" and kn.map_label(spec, "UNVERIFIED") is None
    assert kn.map_label(fake_spec(label_map={0: "SUPPORT"}), 0) == "SUPPORT"  # int labels compare as text
    rows, rep = kn.build_external_rows(RECORDS, spec)
    assert [r["label"] for r in rows] == ["SUPPORT", "CONTRADICT", "NEI"]
    assert rows[0]["premise"] == "Mask study. Masks cut spread." and rows[1]["premise"] == "No effect was seen."
    assert rows[0]["source"] == "ext:fakever" and rows[0]["qid"] == "fakever:0"
    assert rep["dropped_empty"] == 1 and rep["dropped_by_label"] == {"UNVERIFIED": 1}
    assert rep["unknown_labels"] == {"mixture": 1} and rep["mapped"] == 3 and rep["rows_raw"] == 6
    only = kn.build_external_rows(RECORDS, {**spec, "where": {"y": ["REFUTED"]}})[1]
    assert only["rows_outside_where"] == 5 and only["mapped"] == 1
    with pytest.raises(SystemExit, match="no rows mapped"):
        kn.build_external_rows(RECORDS, fake_spec(label_map={"yes": "SUPPORT"}))


def test_spec_validation_requires_a_pinned_revision_and_known_labels():
    kn.validate_spec(fake_spec())
    for bad in (fake_spec(revision="main"), fake_spec(revision="abc123"), fake_spec(format="xlsx"),
                fake_spec(label_map={"A": "ENTAILMENT"}), fake_spec(columns={"claim": "c", "label": "y"})):
        with pytest.raises(AssertionError):
            kn.validate_spec(bad)
    no_license = fake_spec()
    del no_license["license"]
    with pytest.raises(AssertionError, match="license"):
        kn.validate_spec(no_license)


def test_contamination_filter_drops_scifact_claims_titles_abstracts_and_the_blocklist():
    sci_abstract = "Background: we studied " + "many things " * 40
    hashes = {k: set(v) for k, v in ke.build_hashes(
        ["Masks reduce transmission."], [{"title": "Zinc And Colds", "text": sci_abstract}],
        blocklist_texts=["Vitamin D prevents flu"]).items() if isinstance(v, list)}
    rows = [{"claim": c, "premise": p, "title": t, "evidence": e} for c, t, e, p in [
        ("MASKS reduce  transmission", "", "ev", "ev"),  # SciFact claim, other spelling
        ("Vitamin D prevents flu!", "", "ev", "ev"),  # blocklisted near-duplicate
        ("Zinc shortens colds", "zinc and colds", "ev", "zinc and colds. ev"),  # SciFact title
        ("Something new", "", sci_abstract.upper(), sci_abstract),  # SciFact abstract as evidence
        ("A clean claim", "Other title", "other evidence", "Other title. other evidence"),
    ]]
    kept, rep = kn.filter_contaminated(rows, hashes)
    assert [r["claim"] for r in kept] == ["A clean claim"]
    assert rep == {"claim_matches_scifact": 1, "claim_in_blocklist": 1, "title_matches_scifact": 1,
                   "evidence_matches_scifact_abstract": 1}


def test_dedup_and_cap_are_deterministic():
    rows = [{"claim": f"c{i % 5}", "premise": f"P{i % 5}."} for i in range(20)]
    kept, dropped = kn.dedup_rows(rows)
    assert len(kept) == 5 and dropped == 15 and kept == rows[:5]
    clash = [{"claim": "c", "premise": "p", "label": "SUPPORT"}, {"claim": "C ", "premise": "p.", "label": "NEI"},
             {"claim": "d", "premise": "p", "label": "NEI"}, {"claim": "d", "premise": "p", "label": "NEI"}]
    kept, dropped = kn.dedup_rows(clash)  # conflicting copies of (c, p) both go; (d, p) keeps one
    assert kept == [clash[2]] and dropped == 3
    big = [{"i": i} for i in range(100)]
    cap = kn.cap_rows(big, 10, seed=3)
    assert len(cap) == 10 and cap == kn.cap_rows(big, 10, seed=3) and cap == sorted(cap, key=lambda r: r["i"])
    assert kn.cap_rows(big, None, 3) == big and kn.cap_rows(big, 500, 3) == big


@pytest.mark.parametrize("fmt", ["csv", "tsv", "jsonl", "json"])
def test_prepare_external_reads_each_format_at_the_pinned_revision(tmp_path, monkeypatch, fmt):
    recs = [{"c": f"claim {i}", "e": f"evidence {i}", "y": ["SUPPORTED", "REFUTED", "NEUTRAL"][i % 3]} for i in range(9)]
    recs.append({"c": "claim 0", "e": "evidence 0", "y": "SUPPORTED"})  # exact duplicate
    path = tmp_path / f"data.{fmt}"
    if fmt in ("csv", "tsv"):
        sep = "," if fmt == "csv" else "\t"
        path.write_text("c{0}e{0}y\n".format(sep) + "".join(sep.join((r["c"], r["e"], r["y"])) + "\n" for r in recs))
    elif fmt == "jsonl":
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    else:
        path.write_text(json.dumps({"data": recs}))
    calls = []
    monkeypatch.setattr(kn, "hf_download", lambda repo, f, rev: calls.append((repo, f, rev)) or str(path))
    spec = fake_spec(format=fmt, files=["a", "b"], cap=4, **({"records_key": "data"} if fmt == "json" else {}))
    rows, (rep,) = kn.prepare_external([spec], {}, {"max_pairs": 3}, seed=13)
    assert calls == [(spec["repo_id"], "a", spec["revision"]), (spec["repo_id"], "b", spec["revision"])]
    assert rep["rows_raw"] == 20 and rep["duplicates_dropped"] == 11 and rep["before_cap"] == 9
    assert rep["used"] == 4 and len(rows) == 3  # per-dataset cap, then the stage-1 total cap
    assert rep["revision"] == spec["revision"] and rep["repo_id"] == spec["repo_id"] and rep["license"] == "test"


def test_prepare_external_stops_with_instructions_when_a_download_fails(monkeypatch):
    def boom(*a):
        raise OSError("offline")

    monkeypatch.setattr(kn, "hf_download", boom)
    with pytest.raises(SystemExit, match=r"Internet on, or set CONFIG\['stage1'\]\['enabled'\] = False"):
        kn.prepare_external([fake_spec()], {}, {"max_pairs": 10}, seed=13)


# --- the two real stage-1 specs, on fake records shaped like their files --------------------------


def real_spec(name: str) -> dict:
    (spec,) = [s for s in kn.EXTERNAL_DATASETS if s["name"] == name]
    return spec


def test_the_shipped_specs_are_valid_pinned_and_licensed():
    assert [s["name"] for s in kn.EXTERNAL_DATASETS] == ["healthver", "pubmedqa_l"]
    for s in kn.EXTERNAL_DATASETS:
        kn.validate_spec(s)
        assert len(s["revision"]) == 40 and s["license"] and s["url"].startswith("https://")
        assert set(s["label_map"].values()) == set(kn.LABELS)
    assert kn.CONFIG["stage1"]["epochs"] == 2 and kn.CONFIG["stage1"]["max_pairs"] == 12000


def test_healthver_rows_map_supports_refutes_neutral():
    recs = [{"Claim": "Masks prevent COVID-19", "Evidence": "Masks reduced infection.", "Label": "Supports"},
            {"Claim": "Garlic cures COVID-19", "Evidence": "No antiviral effect.", "Label": "Refutes"},
            {"Claim": "Zinc helps", "Evidence": "Zinc levels were measured.", "Label": "Neutral"}]
    rows, rep = kn.build_external_rows(recs, real_spec("healthver"))
    assert [r["label"] for r in rows] == ["SUPPORT", "CONTRADICT", "NEI"]
    assert rows[0]["premise"] == "Masks reduced infection." and rows[0]["source"] == "ext:healthver"
    assert rep["unknown_labels"] == {} and rep["mapped"] == 3


def pubmedqa_record(pubid, claim, decision):
    import numpy as np  # parquet struct columns arrive from pandas as dicts holding numpy arrays

    return {"pubid": np.int64(pubid), "claim": claim, "final_decision": decision,
            "context": {"contexts": np.array(["Background text.", "Results text."], dtype=object),
                        "labels": np.array(["BACKGROUND", "RESULTS"], dtype=object)},
            "long_answer": f"LEAKED CONCLUSION: the answer is {decision}."}


def test_pubmedqa_rows_use_the_nested_contexts_and_never_the_long_answer():
    recs = [pubmedqa_record(1, "Statins lower LDL.", "yes"), pubmedqa_record(2, "Aspirin prevents cancer.", "no"),
            pubmedqa_record(3, "Coffee affects sleep.", "maybe")]
    rows, rep = kn.build_external_rows(recs, real_spec("pubmedqa_l"))
    assert [r["label"] for r in rows] == ["SUPPORT", "CONTRADICT", "NEI"]
    assert rows[0]["premise"] == "Background text. Results text." and rows[0]["qid"] == "pubmedqa_l:1"
    assert all("LEAKED" not in r["premise"] and "LEAKED" not in r["claim"] for r in rows)
    assert kn.field({"a": {"b": {"c": 5}}}, "a.b.c") == 5 and kn.field({"a": {}}, "a.b.c") is None
    assert kn.field({"x.y": 1}, "x.y") == 1  # a literal dotted key wins


def test_the_crohns_claim_is_blocklisted_through_the_export_hashes():
    assert "It is Crohn's disease." in ke.EXTERNAL_BLOCKLIST_TEXTS
    hashes = {k: set(v) for k, v in ke.build_hashes([], [], ke.EXTERNAL_BLOCKLIST_TEXTS).items() if isinstance(v, list)}
    rows, _ = kn.build_external_rows([pubmedqa_record(9, "It is Crohn's disease.", "yes"),
                                      pubmedqa_record(10, "It is ulcerative colitis.", "no")],
                                     real_spec("pubmedqa_l"))
    kept, rep = kn.filter_contaminated(rows, hashes)
    assert [r["claim"] for r in kept] == ["It is ulcerative colitis."] and rep["claim_in_blocklist"] == 1
