"""Tests for the blind NEI re-labelling packet and scorer. Fakes only: no corpus, no
model, no network, nothing written outside tmp_path."""
import json
import re

import pytest

from app.core.paths import RESULTS
from app.eval import nei_relabel as nr

ANSWER = "ANSWERTEXT-must-not-leak"


def row(qid, gold, pred, docs=None, qrels=None):
    docs = docs or [f"d{qid}-{i}" for i in range(5)]
    return {
        "query_id": qid, "gold_label": gold, "predicted_label": pred,
        "verdict": {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED"}.get(pred, "NOT ENOUGH EVIDENCE"),
        "answer": f"{ANSWER} {qid}", "cited_doc_ids": [docs[0]], "retrieved_doc_ids": docs,
        "qrels_doc_ids": qrels if qrels is not None else [f"cited{qid}"],
        "rationale_doc_ids": [], "verdict_source": "line",
    }


def fake_rows():
    rows = []
    # 3 disagreements: gold NEI, model S/C. The first has its cited doc among the passages.
    rows.append(row("Q9101", "NEI", "SUPPORT", qrels=["dQ9101-2"]))
    rows.append(row("Q9102", "NEI", "CONTRADICT"))
    rows.append(row("Q9103", "NEI", "SUPPORT"))
    rows += [row(f"Q92{i:02d}", "NEI", "NEI") for i in range(6)]        # NEI controls pool
    rows += [row(f"Q93{i:02d}", "SUPPORT", "SUPPORT") for i in range(5)]
    rows += [row(f"Q94{i:02d}", "CONTRADICT", "CONTRADICT") for i in range(5)]
    rows += [row("Q9501", "SUPPORT", "NEI"), row("Q9502", "CONTRADICT", "SUPPORT")]  # never picked
    return rows


def blob(rows=None):
    rows = rows or fake_rows()
    acc = sum(r["predicted_label"] == r["gold_label"] for r in rows) / len(rows)
    return {"verdict_accuracy": round(acc, 4), "run": {"git_sha": "abc", "dataset": "beir/scifact/test"},
            "rows": rows}


def bare(doc_id):
    """A doc id with its claim-id letters removed, so passage text never contains an id."""
    return doc_id.replace("dQ", "")


def corpus(rows):
    claims = {r["query_id"]: f"Claim text for {r['query_id'][1:]} reduces risk." for r in rows}
    docs = {}
    for r in rows:
        for d in r["retrieved_doc_ids"]:
            docs[d] = {"title": f"  Title {bare(d)}  ", "text": f"Abstract body {bare(d)}.\n"}
    return claims, docs


def build(b=None, **kw):
    b = b or blob()
    claims, docs = corpus(b["rows"])
    kw.setdefault("n_nei_controls", 4)
    kw.setdefault("n_evidence_controls", 4)
    return nr.build_packet(b, claims, docs, **kw)


def page_data(html):
    m = re.search(r'<script type="application/json" id="packet-data">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


# --- selection ---------------------------------------------------------------------------


def test_selection_groups_and_counts():
    _, key = build()
    assert key["counts"] == {"disagreement": 3, "control_nei": 4, "control_evidence": 4}
    by = {e["claim_id"]: e for e in key["items"]}
    assert {c for c, e in by.items() if e["group"] == "disagreement"} == {"Q9101", "Q9102", "Q9103"}
    for e in key["items"]:
        if e["group"] == "control_nei":
            assert (e["gold_label"], e["model_label"]) == ("NEI", "NEI")
        if e["group"] == "control_evidence":
            assert e["gold_label"] == e["model_label"] != "NEI"
    ev = [by[c]["gold_label"] for c in by if by[c]["group"] == "control_evidence"]
    assert sorted(ev) == ["CONTRADICT", "CONTRADICT", "SUPPORT", "SUPPORT"]
    assert "Q9501" not in by and "Q9502" not in by
    assert by["Q9101"]["cited_doc_in_passages"] and not by["Q9102"]["cited_doc_in_passages"]
    assert len({e["item_id"] for e in key["items"]}) == len(key["items"])


def test_build_is_deterministic_and_seed_dependent():
    h1, k1 = build()
    h2, k2 = build()
    assert h1 == h2 and k1 == k2
    _, k3 = build(seed=7)
    assert [e["item_id"] for e in k3["items"]] != [e["item_id"] for e in k1["items"]]


def test_order_is_shuffled_not_grouped():
    rows = fake_rows() + [row(f"Q96{i:02d}", "NEI", "NEI") for i in range(30)]
    _, key = build(blob(rows), n_nei_controls=30)
    groups = [e["group"] for e in key["items"]]
    assert groups[:3] != ["disagreement"] * 3  # disagreements not simply first


def test_too_few_controls_is_refused():
    with pytest.raises(nr.RelabelError, match="only 6 available"):
        build(n_nei_controls=7)


def test_missing_doc_is_refused():
    b = blob()
    claims, docs = corpus(b["rows"])
    docs.pop("dQ9101-0")
    with pytest.raises(nr.RelabelError, match="not in the corpus"):
        nr.build_packet(b, claims, docs, n_nei_controls=1, n_evidence_controls=0)


# --- the page: content and blindness ----------------------------------------------------


def test_page_shows_claims_and_passages_in_rank_order():
    html, key = build()
    data = page_data(html)
    assert data["packet_id"] == key["packet_id"]
    assert [it["id"] for it in data["items"]] == [e["item_id"] for e in key["items"]]
    first = data["items"][0]
    entry = key["items"][0]
    assert first["claim"] == f"Claim text for {entry['claim_id'][1:]} reduces risk."
    assert [p["title"] for p in first["passages"]] == [f"Title {bare(d)}" for d in entry["retrieved_doc_ids"]]
    assert [p["text"] for p in first["passages"]] == [f"Abstract body {bare(d)}." for d in entry["retrieved_doc_ids"]]
    assert set(first) == {"id", "claim", "passages"}  # nothing else per item
    assert set(data) == {"packet_id", "labels_format", "items"}


def test_page_leaks_no_ids_labels_answers_or_citations():
    html, key = build()
    for e in key["items"]:
        assert e["claim_id"] not in html  # "Q9101" etc.; claim text uses the id without the Q
        for d in e["retrieved_doc_ids"]:
            assert d not in html
    assert ANSWER not in html
    for r in fake_rows():
        for d in r["qrels_doc_ids"]:
            assert d not in html
    low = html.lower()
    for word in ("gold", "verdict", "predict", "model", "cited", "citation", "disagree",
                 "control", "answer", "refuted", "supported", "key.json", "claim_id", "query_id"):
        assert word not in low, word
    # The label words appear only as the labeller's own choices, never as item data.
    data_txt = re.search(r'id="packet-data">(.*?)</script>', html, re.S).group(1)
    for lab in ("SUPPORT", "CONTRADICT", "NEI"):
        assert lab not in data_txt


def test_page_is_self_contained_and_renders_text_safely():
    html, _ = build()
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    assert "http://" not in html and "https://" not in html
    assert "innerHTML" not in html and "outerHTML" not in html and "insertAdjacentHTML" not in html
    assert "document.write" not in html and "eval(" not in html
    assert "textContent" in html and "localStorage" in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    # every localStorage access sits inside a try block
    script = html.rsplit("<script>", 1)[1]
    uses = list(re.finditer(r"localStorage", script))
    assert uses
    for m in uses:
        before = script[: m.start()]
        assert before.rfind("try {") > before.rfind("catch")


def test_hostile_passage_text_cannot_break_out():
    b = blob()
    claims, docs = corpus(b["rows"])
    evil = '</script><script>alert(1)</script><!-- & "x"'
    for d in docs:
        docs[d] = {"title": evil, "text": evil}
    claims = {k: evil for k in claims}
    html, _ = nr.build_packet(b, claims, docs, n_nei_controls=1, n_evidence_controls=0)
    assert "alert(1)</script>" not in html and "<!--" not in html
    assert html.count("</script>") == 2  # the data block and the app script, nothing injected
    assert page_data(html)["items"][0]["claim"] == evil  # round-trips intact as data


def test_csp_hashes_match_the_inline_script_and_style():
    import base64
    import hashlib

    html, _ = build()
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    for code in (script, style):
        h = base64.b64encode(hashlib.sha256(code.encode()).digest()).decode()
        assert f"'sha256-{h}'" in html


# --- writing ---------------------------------------------------------------------------------


def test_write_packet_and_refuse_a_different_existing_key(tmp_path):
    html, key = build()
    h, k = nr.write_packet(tmp_path, html, key)
    assert h.read_text() == html and json.loads(k.read_text()) == key
    nr.write_packet(tmp_path, html, key)  # same packet: fine
    _, other = build(seed=99)
    with pytest.raises(SystemExit, match="different packet"):
        nr.write_packet(tmp_path, html, other)
    nr.write_packet(tmp_path, html, other, force=True)


def test_never_writes_under_eval_results(tmp_path):
    html, key = build()
    with pytest.raises(SystemExit):
        nr.write_packet(RESULTS / "relabel_test_never_created", html, key)
    (tmp_path / "res").symlink_to(RESULTS, target_is_directory=True)
    with pytest.raises(SystemExit):
        nr.write_report(tmp_path / "res", {"x": 1})
    assert not (RESULTS / "relabel_test_never_created").exists()
    with pytest.raises(SystemExit):
        nr.main(["score", "--labels", "x.json", "--out", str(RESULTS)])


# --- scoring -----------------------------------------------------------------------------------


def export(key, labels):
    return {"format": nr.LABELS_FORMAT, "packet_id": key["packet_id"],
            "labels": {iid: {"label": lab, "passages": [1], "note": "n|ote\nx"} for iid, lab in labels.items()}}


def human_labels(key, dis, ctl):
    """dis: claim id -> human label for disagreements; ctl: default 'agree with gold'
    except overrides."""
    out = {}
    for e in key["items"]:
        c = e["claim_id"]
        out[e["item_id"]] = dis.get(c) if e["group"] == "disagreement" else ctl.get(c, e["gold_label"])
    return {k: v for k, v in out.items() if v is not None}


def test_kappa_known_values():
    assert nr.cohen_kappa(["SUPPORT", "NEI"], ["SUPPORT", "NEI"]) == 1.0
    a = ["SUPPORT"] * 20 + ["NEI"] * 30
    b = ["SUPPORT"] * 15 + ["NEI"] * 5 + ["SUPPORT"] * 10 + ["NEI"] * 20
    # po = 0.7, pe = 0.4*0.5 + 0.6*0.5 = 0.5 -> kappa 0.4
    assert nr.cohen_kappa(a, b) == pytest.approx(0.4)
    assert nr.cohen_kappa(["NEI"] * 3, ["NEI"] * 3) is None  # undefined
    assert nr.cohen_kappa([], []) is None


def test_score_counts_and_secondary_accuracy():
    b = blob()
    _, key = build(b)
    ctl_claims = [e["claim_id"] for e in key["items"] if e["group"] == "control_nei"]
    labels = human_labels(
        key,
        {"Q9101": "SUPPORT", "Q9102": "NEI", "Q9103": "CONTRADICT"},  # with model / gold / opposite
        {ctl_claims[0]: "SUPPORT"},  # one control the human labels differently from gold
    )
    rep = nr.score(key, nr.load_labels(export(key, labels), key), b)
    d = rep["disagreement_outcome"]
    assert (d["human_sides_with_model"], d["human_sides_with_gold_nei"], d["human_opposite_stance"]) == (1, 1, 1)
    assert d["sides_with_model_split"] == {"cited_doc_in_passages": 1, "cited_doc_not_in_passages": 0}
    assert rep["disagreement"]["human_vs_gold"]["agree"] == 1
    assert rep["disagreement"]["human_vs_model"]["agree"] == 1
    assert rep["disagreement"]["kappa_human_vs_gold"] is None  # gold is NEI throughout
    c = rep["controls"]
    assert c["n"] == 8 and c["human_vs_gold"]["agree"] == 7 == c["human_vs_model"]["agree"]
    assert c["confusion_gold_x_human"]["NEI"]["SUPPORT"] == 1

    sec = rep["secondary_verdict_accuracy"]
    n = len(b["rows"])
    base = sum(r["predicted_label"] == r["gold_label"] for r in b["rows"])
    assert sec["original_labels"]["correct"] == base and sec["stored_matches"]
    one_sided = sec["human_relabels_disagreements_only"]
    assert one_sided["correct"] == base + 1 and one_sided["labels_changed"] == 2 and one_sided["n"] == n
    two_sided = sec["human_relabels_all_sampled_items"]
    assert two_sided["correct"] == base + 1 - 1 and two_sided["labels_changed"] == 3
    assert "headline verdict accuracy stays on the original" in sec["caveat"]

    md = nr.to_markdown(rep)
    assert "SECONDARY" in md and "One annotator" in md
    assert "n\\|ote x" in md  # notes cannot break the table


def test_unlabelled_items_are_skipped_not_scored():
    b = blob()
    _, key = build(b)
    labels = human_labels(key, {"Q9101": "SUPPORT"}, {})
    first_ctl = next(e["item_id"] for e in key["items"] if e["group"] == "control_nei")
    labels.pop(first_ctl)
    exp = export(key, labels)
    exp["labels"][first_ctl] = {"label": None, "passages": [], "note": "skipped"}
    rep = nr.score(key, nr.load_labels(exp, key), b)
    assert rep["n_labelled"] == 1 + 7
    assert rep["disagreement_outcome"]["n_labelled"] == 1
    assert rep["secondary_verdict_accuracy"]["human_relabels_disagreements_only"]["labels_changed"] == 1


def test_label_aliases_and_bad_exports():
    assert nr.normalize_label("not enough evidence") == "NEI"
    assert nr.normalize_label("REFUTED") == "CONTRADICT"
    assert nr.normalize_label("") is None
    with pytest.raises(nr.RelabelError):
        nr.normalize_label("MAYBE")
    _, key = build()
    with pytest.raises(nr.RelabelError, match="packet"):
        nr.load_labels({"packet_id": "other", "labels": {}}, key)
    with pytest.raises(nr.RelabelError, match="not in the key"):
        nr.load_labels({"packet_id": key["packet_id"], "labels": {"it-bogus": {"label": "NEI"}}}, key)


def test_score_refuses_a_regenerated_run():
    b = blob()
    _, key = build(b)
    rows = [dict(r) for r in b["rows"]]
    for r in rows:
        if r["query_id"] == "Q9101":
            r["predicted_label"] = "NEI"
    with pytest.raises(nr.RelabelError, match="regenerated"):
        nr.score(key, {}, blob(rows))


def test_cli_build_and_score_end_to_end(tmp_path, monkeypatch):
    b = blob()
    rag = tmp_path / "rag.json"
    rag.write_text(json.dumps(b))
    monkeypatch.setattr(nr, "_load_corpus", lambda dataset: corpus(b["rows"]))
    out = tmp_path / "relabel"
    nr.main(["build", "--rag", str(rag), "--out", str(out), "--n-nei-controls", "3",
             "--n-evidence-controls", "2"])
    key = json.loads((out / "key.json").read_text())
    assert key["n_items"] == 3 + 3 + 2
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(export(key, human_labels(key, {"Q9101": "SUPPORT"}, {}))))
    nr.main(["score", "--labels", str(labels), "--rag", str(rag), "--out", str(out)])
    rep = json.loads((out / "report.json").read_text())
    assert rep["disagreement_outcome"]["human_sides_with_model"] == 1
    assert (out / "report.md").read_text().startswith("# Blind human re-labelling")


LLM_ANNOTATOR = "majority of 3 blind LLM annotators (2 Claude Opus, 1 Claude Sonnet)"


def _scored(**kw):
    b = blob()
    _, key = build(b)
    labels = human_labels(key, {"Q9101": "SUPPORT", "Q9102": "NEI", "Q9103": "CONTRADICT"}, {})
    return nr.score(key, nr.load_labels(export(key, labels), key), b, **kw)


def test_default_annotator_wording_is_unchanged():
    rep = _scored()
    assert rep["annotator"] == nr.DEFAULT_ANNOTATOR and rep["annotators_note"] is None
    assert rep["secondary_verdict_accuracy"]["caveat"] == nr.CAVEAT
    assert _scored(annotator="  ")["secondary_verdict_accuracy"]["caveat"] == nr.CAVEAT
    md = nr.to_markdown(rep)
    assert md.startswith("# Blind human re-labelling of SciFact NEI disagreements")
    assert "| Items | n | human vs gold | human vs model |" in md
    assert "Human relabels of the NEI disagreements only" in md
    assert "Human × gold on controls" in md and "| gold \\ human |" in md
    assert "Annotator:" not in md and "LLM" not in md


def test_llm_annotator_wording_is_neutral_and_caveated():
    rep = _scored(annotator=LLM_ANNOTATOR, annotators_note="Majority vote of 3.")
    assert rep["annotator"] == LLM_ANNOTATOR
    cav = rep["secondary_verdict_accuracy"]["caveat"]
    assert LLM_ANNOTATOR in cav and "LLMs, not a human" in cav and "weaker check than a human" in cav
    assert "generator is also an LLM" in cav and cav.endswith("Majority vote of 3.")
    assert "One annotator (the repo owner" not in cav
    assert "headline verdict accuracy stays on the original" in cav
    md = nr.to_markdown(rep)
    assert md.startswith("# Blind re-labelling of SciFact NEI disagreements")
    assert f"Annotator: {LLM_ANNOTATOR}." in md
    assert "| Items | n | annotator vs gold | annotator vs model |" in md
    assert "Annotator relabels of the NEI disagreements only" in md
    assert "the annotator sides with the **model**" in md
    assert "Annotator × gold on controls" in md and "| gold \\ annotator |" in md
    assert "human" not in md.lower().replace("not a human", "").replace("than a human", "")
    # counts do not depend on the annotator description
    base = _scored()
    assert rep["disagreement_outcome"] == base["disagreement_outcome"]


def test_cli_score_records_annotator(tmp_path, monkeypatch, capsys):
    b = blob()
    rag = tmp_path / "rag.json"
    rag.write_text(json.dumps(b))
    monkeypatch.setattr(nr, "_load_corpus", lambda dataset: corpus(b["rows"]))
    out = tmp_path / "relabel"
    nr.main(["build", "--rag", str(rag), "--out", str(out), "--n-nei-controls", "3",
             "--n-evidence-controls", "2"])
    key = json.loads((out / "key.json").read_text())
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(export(key, human_labels(key, {"Q9101": "SUPPORT"}, {}))))
    nr.main(["score", "--labels", str(labels), "--rag", str(rag), "--out", str(out),
             "--annotator", LLM_ANNOTATOR, "--annotators-note", "Extra."])
    rep = json.loads((out / "report.json").read_text())
    assert rep["annotator"] == LLM_ANNOTATOR and rep["annotators_note"] == "Extra."
    assert rep["secondary_verdict_accuracy"]["caveat"].endswith("Extra.")
    assert "annotator sides with the model" in capsys.readouterr().out
    assert (out / "report.md").read_text().startswith("# Blind re-labelling")


# --- default packet is unchanged by the evidence_errors set ---------------------------------

# sha256 of the default packet / reports on the fake rows above, recorded before the
# evidence_errors set was added: the first packet's page, key and reports must not change.
GOLDEN = {
    "html": "2e58ebf1b216b444ab3b75a43626a39f45d7073039c5d76cb67af15d34ed110b",
    "key": "1d50573260ada1b80ae46b03411b01d19321df82efab398b2c56d8e0b3b5e87e",
    "rep_default": "b57a4d4b66795c7a68b025c9bd213d0dd18ab5f072c5ca85a62fe3df60fdf9fc",
    "md_default": "de21d59814f246c10cc49a7895b8c00d8ef9933b3c1640a87460146572ead1b3",
    "rep_llm": "28fe92b265a2e0c7407e60c6ac3dfa23623c4e7397e0565f9a0677c353f3cf3a",
    "md_llm": "2a4bf72243c6c87fbca328bb16adc783da4c76274549c36f0c7ec0a3a378c031",
}


def _sha(s):
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()


def test_default_packet_and_reports_are_byte_identical():
    html, key = build()
    assert _sha(html) == GOLDEN["html"]
    assert _sha(json.dumps(key, indent=2)) == GOLDEN["key"]
    assert "set" not in key and "controls_exclude" not in key
    assert all("rationale_doc_in_passages" not in e for e in key["items"])
    for name, kw in (("default", {}), ("llm", {"annotator": LLM_ANNOTATOR, "annotators_note": "Note."})):
        rep = _scored(**kw)
        assert _sha(json.dumps(rep, indent=2, ensure_ascii=False)) == GOLDEN[f"rep_{name}"]
        assert _sha(nr.to_markdown(rep)) == GOLDEN[f"md_{name}"]
    # explicit default kind == implicit default
    _, key2 = build(kind=nr.NEI_SET)
    assert key2 == key


# --- evidence_errors set ----------------------------------------------------------------------


def evidence_rows():
    """fake_rows() plus more S/C-gold errors. Targets: Q9501 SUPPORT->NEI (rationale doc
    shown), Q9502 CONTRADICT->SUPPORT (rationale not shown), Q9503 SUPPORT->NONE,
    Q9504 SUPPORT->CONTRADICT (rationale shown)."""
    rows = fake_rows()
    rows += [row("Q9503", "SUPPORT", "NONE"), row("Q9504", "SUPPORT", "CONTRADICT")]
    for r in rows:
        if r["query_id"] in ("Q9501", "Q9504"):
            r["rationale_doc_ids"] = [r["retrieved_doc_ids"][3]]
            r["qrels_doc_ids"] = [r["retrieved_doc_ids"][3]]
        elif r["gold_label"] != "NEI":
            r["rationale_doc_ids"] = [f"ratQ{r['query_id']}"]
    return rows


EV_TARGETS = {"Q9501", "Q9502", "Q9503", "Q9504"}


def build_ev(b=None, **kw):
    b = b or blob(evidence_rows())
    kw.setdefault("kind", nr.EVIDENCE_SET)
    kw.setdefault("seed", nr.EVIDENCE_SEED)
    return build(b, **kw)


def test_evidence_selection_groups_and_counts():
    _, key = build_ev()
    assert key["set"] == nr.EVIDENCE_SET and key["seed"] == nr.EVIDENCE_SEED != nr.SEED
    assert key["counts"] == {"evidence_error": 4, "control_nei": 4, "control_evidence": 4}
    by = {e["claim_id"]: e for e in key["items"]}
    assert {c for c, e in by.items() if e["group"] == "evidence_error"} == EV_TARGETS
    assert not {"Q9101", "Q9102", "Q9103"} & set(by)  # NEI disagreements belong to packet 1
    for e in key["items"]:
        if e["group"] == "evidence_error":
            assert e["gold_label"] in ("SUPPORT", "CONTRADICT") and e["model_label"] != e["gold_label"]
        elif e["group"] == "control_nei":
            assert (e["gold_label"], e["model_label"]) == ("NEI", "NEI")
        else:
            assert e["gold_label"] == e["model_label"] != "NEI"
    assert by["Q9503"]["model_label"] == "NONE"
    assert by["Q9501"]["rationale_doc_in_passages"] and by["Q9504"]["rationale_doc_in_passages"]
    assert not by["Q9502"]["rationale_doc_in_passages"] and not by["Q9503"]["rationale_doc_in_passages"]
    # ids, samples and order differ from the default packet's seed
    _, k_default_seed = build_ev(seed=nr.SEED)
    assert {e["item_id"] for e in key["items"]}.isdisjoint(e["item_id"] for e in k_default_seed["items"])


def test_evidence_controls_exclude_earlier_packets():
    b = blob(evidence_rows())
    _, first = build(b)  # the NEI packet over the same rows
    _, key = build_ev(b, n_nei_controls=2, n_evidence_controls=2, exclude_keys=[first])
    first_claims = {e["claim_id"] for e in first["items"]}
    ctl = {e["claim_id"] for e in key["items"] if e["group"] != "evidence_error"}
    assert ctl and not ctl & first_claims
    assert key["controls_exclude"] == {"packets": [first["packet_id"]], "n_claims": len(first_claims)}
    # targets are never excluded
    _, k2 = build_ev(b, n_nei_controls=0, n_evidence_controls=0,
                     exclude_keys=[{"packet_id": "x", "items": [{"claim_id": "Q9501"}]}])
    assert {e["claim_id"] for e in k2["items"]} == EV_TARGETS
    with pytest.raises(nr.RelabelError, match="only 2 available"):  # 6 NEI controls - 4 used
        build_ev(b, n_nei_controls=3, exclude_keys=[first])


def test_unknown_set_is_refused():
    with pytest.raises(nr.RelabelError, match="unknown packet set"):
        build(kind="bogus")


def test_evidence_page_leaks_nothing_and_matches_the_default_ui():
    b = blob(evidence_rows())
    html, key = build_ev(b)
    for e in key["items"]:
        assert e["claim_id"] not in html
        for d in e["retrieved_doc_ids"]:
            assert d not in html
    assert ANSWER not in html
    for r in b["rows"]:
        for d in r["qrels_doc_ids"] + r["rationale_doc_ids"]:
            assert d not in html
    low = html.lower()
    for word in ("gold", "verdict", "predict", "model", "cited", "citation", "disagree",
                 "control", "answer", "refuted", "supported", "key.json", "claim_id", "query_id",
                 "rationale", "evidence_error", "evidence_errors", "relabel_evidence", "error"):
        assert word not in low, word
    data_txt = re.search(r'id="packet-data">(.*?)</script>', html, re.S).group(1)
    for lab in ("SUPPORT", "CONTRADICT", "NEI", "NONE"):
        assert lab not in data_txt
    data = page_data(html)
    assert set(data) == {"packet_id", "labels_format", "items"}
    assert all(set(it) == {"id", "claim", "passages"} for it in data["items"])
    assert all(set(p) == {"title", "text"} for it in data["items"] for p in it["passages"])
    # same page apart from the data block: nothing in the UI tells the two packets apart
    default_html, _ = build()
    strip = lambda h: re.sub(r'id="packet-data">.*?</script>', "", h, flags=re.S)  # noqa: E731
    assert strip(html) == strip(default_html)


def test_evidence_order_is_shuffled_not_grouped():
    _, key = build_ev()
    groups = [e["group"] for e in key["items"]]
    assert groups[:4] != ["evidence_error"] * 4 and groups[-4:] != ["evidence_error"] * 4


def test_export_items_is_the_blind_page_data(tmp_path):
    html, key = build_ev()
    nr.write_packet(tmp_path / "p", html, key)
    out = tmp_path / "blind" / "items.json"
    data = nr.export_items(tmp_path / "p", out)
    raw = out.read_text()
    assert json.loads(raw) == data == page_data(html)
    assert set(data) == {"packet_id", "labels_format", "items"} and data["packet_id"] == key["packet_id"]
    for e in key["items"]:
        assert e["claim_id"] not in raw and e["group"] not in raw
    for lab in ("SUPPORT", "CONTRADICT", "NEI", "NONE"):
        assert lab not in raw
    # a key for another packet next to the page is refused
    _, other = build_ev(seed=1)
    (tmp_path / "p" / "key.json").write_text(json.dumps(other))
    with pytest.raises(nr.RelabelError, match="different packet"):
        nr.export_items(tmp_path / "p", out)
    with pytest.raises(SystemExit):
        nr.export_items(tmp_path / "p", RESULTS / "never_created.json")


def ev_labels(key, tgt, ctl=None):
    """tgt: claim id -> label for targets; controls agree with gold unless in ctl."""
    ctl = ctl or {}
    out = {}
    for e in key["items"]:
        c = e["claim_id"]
        out[e["item_id"]] = tgt.get(c) if e["group"] == "evidence_error" else ctl.get(c, e["gold_label"])
    return {k: v for k, v in out.items() if v is not None}


def test_evidence_score_outcomes_and_secondary_accuracy():
    b = blob(evidence_rows())
    _, key = build_ev(b)
    ctl_nei = next(e["claim_id"] for e in key["items"] if e["group"] == "control_nei")
    labels = ev_labels(key, {
        "Q9501": "SUPPORT",      # with gold (model said NEI): a model judgement error
        "Q9502": "SUPPORT",      # with the model (gold CONTRADICT); rationale not shown
        "Q9503": "NEI",          # model NONE: any non-gold label is a third label
        "Q9504": "NEI",          # gold SUPPORT, model CONTRADICT: third label
    }, {ctl_nei: "SUPPORT"})
    rep = nr.score(key, nr.load_labels(export(key, labels), key), b)
    assert rep["set"] == nr.EVIDENCE_SET
    d = rep["target_outcome"]
    assert (d["with_gold"], d["with_model"], d["third_label"], d["n_labelled"], d["n_total"]) == (1, 1, 2, 4, 4)
    assert d["by_error_type"]["SUPPORT->NEI"] == {
        "n_total": 1, "n": 1, "with_gold": 1, "with_model": 0, "third_label": 0,
        "human_labels": {"SUPPORT": 1, "CONTRADICT": 0, "NEI": 0}}
    assert d["by_error_type"]["SUPPORT->NONE"]["third_label"] == 1
    assert d["by_rationale_shown"]["rationale_doc_in_passages"] == {
        "n": 2, "with_gold": 1, "with_model": 0, "third_label": 1}
    assert d["by_rationale_shown"]["rationale_doc_not_in_passages"] == {
        "n": 2, "with_gold": 0, "with_model": 1, "third_label": 1}
    # NONE stays visible in the model confusion table rather than dropping out
    assert rep["evidence_error"]["confusion_model_x_human"]["NONE"]["NEI"] == 1
    assert rep["evidence_error"]["human_vs_model"]["agree"] == 1
    c = rep["controls"]
    assert c["n"] == 8 and c["human_vs_gold"]["agree"] == 7
    assert c["confusion_gold_x_human"]["NEI"]["SUPPORT"] == 1

    sec = rep["secondary_verdict_accuracy"]
    base = sum(r["predicted_label"] == r["gold_label"] for r in b["rows"])
    assert sec["original_labels"]["correct"] == base and sec["stored_matches"]
    one = sec["human_relabels_targets_only"]
    assert one["correct"] == base + 1 and one["labels_changed"] == 3  # Q9502 flips to right
    two = sec["human_relabels_all_sampled_items"]
    assert two["correct"] == base + 1 - 1 and two["labels_changed"] == 4
    assert "headline verdict accuracy stays on the original" in sec["caveat"]
    per = {p["claim_id"]: p for p in rep["per_target"]}
    assert per["Q9501"]["outcome"] == "with_gold" and per["Q9502"]["outcome"] == "with_model"

    md = nr.to_markdown(rep)
    assert md.startswith("# Blind human re-labelling of SciFact SUPPORT/CONTRADICT errors")
    assert "sides with the **gold** label on 1" in md and "with the **model** on 1" in md
    assert "| SUPPORT->NONE | 1/1 | 0 | 0 | 1 |" in md
    assert "Human relabels of the S/C-gold errors only" in md and "One annotator" in md
    assert "n\\|ote x" in md


def test_evidence_score_llm_annotator_wording():
    b = blob(evidence_rows())
    _, key = build_ev(b)
    labels = ev_labels(key, {"Q9501": "SUPPORT", "Q9502": "SUPPORT"})
    rep = nr.score(key, nr.load_labels(export(key, labels), key), b,
                   annotator=LLM_ANNOTATOR, annotators_note="Majority vote of 3.")
    cav = rep["secondary_verdict_accuracy"]["caveat"]
    assert LLM_ANNOTATOR in cav and "LLMs, not a human" in cav and cav.endswith("Majority vote of 3.")
    md = nr.to_markdown(rep)
    assert md.startswith("# Blind re-labelling of SciFact SUPPORT/CONTRADICT errors")
    assert f"Annotator: {LLM_ANNOTATOR}." in md
    assert "the annotator sides with the **gold** label" in md
    assert "human" not in md.lower().replace("not a human", "").replace("than a human", "")
    d = rep["target_outcome"]
    assert (d["n_labelled"], d["with_gold"], d["with_model"]) == (2, 1, 1)
    per = {p["claim_id"]: p for p in rep["per_target"]}
    assert per["Q9503"]["human"] is None and per["Q9503"]["outcome"] is None


def test_cli_evidence_build_export_and_score(tmp_path, monkeypatch, capsys):
    b = blob(evidence_rows())
    rag = tmp_path / "rag.json"
    rag.write_text(json.dumps(b))
    monkeypatch.setattr(nr, "_load_corpus", lambda dataset: corpus(b["rows"]))
    first = tmp_path / "relabel"
    monkeypatch.setattr(nr, "DEFAULT_OUT", first)
    nr.main(["build", "--rag", str(rag), "--out", str(first), "--n-nei-controls", "2",
             "--n-evidence-controls", "2"])
    k1 = json.loads((first / "key.json").read_text())
    before = {p.name: p.read_bytes() for p in first.iterdir()}

    out = tmp_path / "relabel_evidence"
    nr.main(["build", "--set", "evidence_errors", "--rag", str(rag), "--out", str(out),
             "--n-nei-controls", "2", "--n-evidence-controls", "2"])
    key = json.loads((out / "key.json").read_text())
    assert key["set"] == "evidence_errors" and key["seed"] == nr.EVIDENCE_SEED
    assert key["controls_exclude"]["packets"] == [k1["packet_id"]]  # default: data/relabel/key.json
    assert key["counts"] == {"evidence_error": 4, "control_nei": 2, "control_evidence": 2}
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before  # packet 1 untouched
    # building the evidence set into packet 1's directory is refused (different packet)
    with pytest.raises(SystemExit, match="different packet"):
        nr.main(["build", "--set", "evidence_errors", "--rag", str(rag), "--out", str(first),
                 "--n-nei-controls", "2", "--n-evidence-controls", "2"])

    items = tmp_path / "blind2" / "items.json"
    nr.main(["export-items", "--set", "evidence_errors", "--packet", str(out), "--to", str(items)])
    assert json.loads(items.read_text())["packet_id"] == key["packet_id"]

    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(export(key, ev_labels(key, {"Q9501": "NEI", "Q9502": "CONTRADICT"}))))
    # scoring an evidence key without --set is refused, so packet 1's report is never overwritten
    with pytest.raises(SystemExit, match="--set evidence_errors"):
        nr.main(["score", "--labels", str(labels), "--key", str(out / "key.json"), "--rag", str(rag),
                 "--out", str(first)])
    nr.main(["score", "--set", "evidence_errors", "--labels", str(labels), "--rag", str(rag),
             "--out", str(out), "--annotator", LLM_ANNOTATOR])
    rep = json.loads((out / "report.json").read_text())
    assert rep["target_outcome"]["with_model"] == 1 and rep["target_outcome"]["with_gold"] == 1
    assert "on 2 labelled S/C-gold errors the annotator sides with gold 1" in capsys.readouterr().out
    assert (out / "report.md").read_text().startswith("# Blind re-labelling of SciFact SUPPORT/CONTRADICT")
    assert not (first / "report.json").exists()


def test_cli_explicit_exclude_key(tmp_path, monkeypatch):
    b = blob(evidence_rows())
    rag = tmp_path / "rag.json"
    rag.write_text(json.dumps(b))
    monkeypatch.setattr(nr, "_load_corpus", lambda dataset: corpus(b["rows"]))
    monkeypatch.setattr(nr, "DEFAULT_OUT", tmp_path / "missing")  # no implicit exclusion
    out = tmp_path / "ev"
    nr.main(["build", "--set", "evidence_errors", "--rag", str(rag), "--out", str(out),
             "--n-nei-controls", "2", "--n-evidence-controls", "2"])
    assert "controls_exclude" not in json.loads((out / "key.json").read_text())
    fake_key = tmp_path / "k.json"
    fake_key.write_text(json.dumps({"packet_id": "p0", "items": [{"claim_id": "Q9200"}]}))
    nr.main(["build", "--set", "evidence_errors", "--rag", str(rag), "--out", str(out), "--force",
             "--n-nei-controls", "2", "--n-evidence-controls", "2", "--exclude-key", str(fake_key)])
    key = json.loads((out / "key.json").read_text())
    assert key["controls_exclude"] == {"packets": ["p0"], "n_claims": 1}
    assert "Q9200" not in {e["claim_id"] for e in key["items"]}
