"""Tests for the fine-tuned verifier: training-data builder, TrainedVerifier, combination
rules and the verify_combine guards. Fakes only — no downloads, no model, no network."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.eval import rag_compare, verify_combine
from app.eval.verify_combine import (
    Signal,
    apply_rule,
    combined_row,
    rule_r1,
    rule_r2,
    rule_r3,
    rule_r4,
    tune_rules,
    verifier_signal,
)
from app.eval.verify_eval import UnsafeOutputError
from app.ingest.corpus import ClaimLabel
from app.verify import nli
from app.verify import train as tr
from app.verify import train_data as td
from app.verify.nli import LabelMappingError, PairCache, PairScore, score_pairs_cached
from app.verify.trained import (
    HASHED_FILES,
    WEIGHT_FILES,
    TrainedVerifier,
    checkpoint_sha,
    model_id,
    resolve_checkpoint,
    trained_label_order,
)
from tests.test_nli import FakeTokenizer

# --- a tiny SciFact-shaped world ---------------------------------------------------------------


def world():
    """Claims 1-8. 1/2 are a negated pair sharing doc a; 5 is NEI citing doc e."""
    queries = {str(i): f"claim {i}" for i in range(1, 9)}
    labels = {
        "1": ClaimLabel("SUPPORT", {"a"}),
        "2": ClaimLabel("CONTRADICT", {"a"}),
        "3": ClaimLabel("SUPPORT", {"b"}),
        "4": ClaimLabel("CONTRADICT", {"c"}),
        "5": ClaimLabel("NEI"),
        "6": ClaimLabel("SUPPORT", {"f"}),
        "7": ClaimLabel("NEI"),
        "8": ClaimLabel("SUPPORT", {"h"}),
    }
    qrels = {"1": {"a": 1}, "2": {"a": 1}, "3": {"b": 1, "b2": 1}, "4": {"c": 1}, "5": {"e": 1},
             "6": {"f": 1}, "7": {"g": 1}, "8": {"h": 1}}
    top = {q: [*sorted(qrels[q]), "x1", "a", "x2", "x3", "x4"][:5] for q in queries}
    docs = {d: {"title": d.upper(), "text": f"abstract {d}"}
            for d in ["a", "b", "b2", "c", "e", "f", "g", "h", "x1", "x2", "x3", "x4", "r1", "r2"]}
    return queries, labels, qrels, top, docs


def test_pairs_labels_and_sources():
    queries, labels, qrels, top, docs = world()
    pairs = td.build_pairs(["3", "5"], queries, labels, qrels, top, docs, n_hard=2, random_frac=0.0)
    got = {(p.qid, p.doc_id): (p.label, p.source) for p in pairs}
    assert got[("3", "b")] == ("SUPPORT", "rationale")
    assert got[("3", "b2")] == ("NEI", "cited_no_rationale")  # cited, no rationale
    assert got[("5", "e")] == ("NEI", "cited_no_rationale")  # an NEI claim's cited doc
    assert all(p.label == "NEI" for p in pairs if p.source == "hard_negative")
    assert pairs[0].title == "B" and pairs[0].claim == "claim 3"


def test_hard_negatives_exclude_rationale_cited_and_forbidden_docs():
    queries, labels, qrels, top, docs = world()
    pairs = td.build_pairs(list(queries), queries, labels, qrels, top, docs, n_hard=2,
                           random_frac=1.0, forbidden_docs={"x1"})
    for p in pairs:
        if p.source in ("hard_negative", "random"):
            assert p.doc_id not in td.cited_docs(p.qid, labels, qrels)
            assert p.doc_id not in labels[p.qid].rationale_doc_ids
            assert p.doc_id != "x1"
    per = {}
    for p in pairs:
        if p.source == "hard_negative":
            per.setdefault(p.qid, []).append(p.doc_id)
    assert per["3"] == ["a", "x2"]  # rank order, cited b/b2 and forbidden x1 skipped
    assert all(len(v) <= 2 for v in per.values())
    assert sum(p.source == "random" for p in pairs) == len(queries)  # random_frac=1


def test_build_is_deterministic():
    w = world()
    assert td.build_pairs(list(w[0]), *w, random_frac=0.5) == td.build_pairs(list(w[0]), *w, random_frac=0.5)


def test_tuning_claims_and_doc_overlapping_claims_are_excluded():
    queries, labels, qrels, top, docs = world()
    s = td.build_splits(queries, labels, qrels, top, docs, tuning={"1"}, val_frac=0.3)
    used = {p.qid for p in s["train"] + s["val"]}
    assert "1" not in used
    assert "2" not in used and s["excluded_qids"] == ["2"]  # shares doc a with tuning claim 1
    assert s["report"]["excluded_doc_overlap"] == 1
    # The tuning claim's evidence is never a negative for anyone either.
    assert all(p.doc_id != "a" for p in s["train"] + s["val"])
    # Validation is grouped by doc: no cited doc on both sides.
    tr_docs = {d for q in s["train_qids"] for d in td.cited_docs(q, labels, qrels)}
    va_docs = {d for q in s["val_qids"] for d in td.cited_docs(q, labels, qrels)}
    assert s["val_qids"] and not tr_docs & va_docs
    assert set(s["report"]["class_weights"]) == set(td.LABELS)


def test_leakage_exclusions_by_shared_doc():
    queries, labels, qrels, *_ = world()
    assert td.leakage_exclusions(queries, {"2"}, labels, qrels) == {"1"}
    assert td.leakage_exclusions(queries, {"8"}, labels, qrels) == set()


def test_group_split_keeps_components_whole():
    docs_of = {"a": {"d1"}, "b": {"d1", "d2"}, "c": {"d2"}, "d": {"d3"}, "e": {"d4"}}
    comps = td.doc_components(sorted(docs_of), docs_of)
    assert ["a", "b", "c"] in comps and ["d"] in comps
    for seed in range(5):
        train, val = td.group_split(sorted(docs_of), docs_of, frac=0.2, seed=seed)
        assert set(train) | set(val) == set(docs_of) and not set(train) & set(val)
        assert {"a", "b", "c"} <= set(train) or {"a", "b", "c"} <= set(val)


def test_build_splits_refuses_other_datasets_and_a_bad_tuning_set():
    w = world()
    with pytest.raises(td.ContaminationError):
        td.build_splits(*w, tuning={"1"}, dataset="beir/scifact/test")
    with pytest.raises(td.ContaminationError):
        td.build_splits(*w, tuning={"999"})
    with pytest.raises(td.ContaminationError):
        td.build_splits(*w, tuning=set())


def test_tuning_ids_reads_rows_and_adds_the_sample(tmp_path):
    p = tmp_path / "rag.json"
    p.write_text(json.dumps({"run": {"dataset": "beir/scifact/train"}, "rows": [{"query_id": "1"}]}))
    assert td.tuning_ids(p, sample=["2"]) == {"1", "2"}
    p.write_text(json.dumps({"run": {"dataset": "beir/scifact/test"}, "rows": []}))
    with pytest.raises(td.ContaminationError):
        td.tuning_ids(p)


def test_class_weights_balance_the_labels():
    mk = lambda lab: td.Pair("q", "d", "c", "t", "a", lab, "rationale")  # noqa: E731
    w = td.class_weights([mk("SUPPORT")] + [mk("NEI")] * 3 + [mk("CONTRADICT")] * 2)
    assert w["NEI"] * 3 == pytest.approx(w["SUPPORT"] * 1) == pytest.approx(w["CONTRADICT"] * 2)


# --- training helpers ---------------------------------------------------------------------------


def test_epoch_batches_cover_every_example_once_and_are_reproducible():
    lengths = [5, 50, 7, 300, 12, 9, 100, 8, 64, 3, 20]
    b1 = tr.epoch_batches(lengths, micro_batch=2, chunk=4, seed=13, epoch=0)
    assert sorted(i for b in b1 for i in b) == list(range(len(lengths)))
    assert b1 == tr.epoch_batches(lengths, 2, 4, 13, 0)
    assert b1 != tr.epoch_batches(lengths, 2, 4, 13, 1)
    for b in b1:
        assert len(b) <= 2


def test_config_signature_ignores_save_cadence_only():
    base = tr.TrainConfig()
    assert base.signature() == tr.TrainConfig(save_every=999).signature()
    assert base.signature() != tr.TrainConfig(max_length=384).signature()


# --- TrainedVerifier ------------------------------------------------------------------------------

TRAINED_ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


class ClaimFirstModel:
    """Reads the SECOND segment as the evidence (claim-first input)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.config = SimpleNamespace(id2label=TRAINED_ID2LABEL)

    def eval(self):
        return self

    def __call__(self, row):
        out = []
        for _claim, premise in self.tok.batches[-1]:
            k = 0 if "SUPPORTS" in premise else 1 if "REFUTES" in premise else 2
            out.append([4.0 if i == k else 0.0 for i in range(3)])
        return SimpleNamespace(logits=torch.tensor(out))


def make_trained(sha="a" * 64, windowing="truncate"):
    tok = FakeTokenizer()
    return TrainedVerifier(checkpoint="unused", windowing=windowing, threads=0, batch_size=2,
                           max_length=64, tokenizer=tok, model=ClaimFirstModel(tok),
                           checkpoint_hash=sha, base_model="fake/base")


def test_trained_verifier_feeds_claim_first_and_maps_labels():
    v = make_trained()
    s_sup, s_ref, s_nei = v.score([("T", "This SUPPORTS it.", "the claim"),
                                   ("T", "This REFUTES it.", "c"), ("T", "Other.", "c")])
    assert s_sup.probs["SUPPORT"] > 0.9 and s_ref.probs["CONTRADICT"] > 0.9 and s_nei.probs["NEI"] > 0.9
    first, second = v.tokenizer.batches[0][0]
    assert first in ("the claim", "c") and second.startswith("T. ")


def test_trained_label_order_refuses_nli_or_partial_heads():
    assert trained_label_order({"0": "nei", "1": "support", "2": "contradict"}) == ["NEI", "SUPPORT", "CONTRADICT"]
    for bad in ({0: "entailment", 1: "neutral", 2: "contradiction"}, {0: "SUPPORT", 1: "NEI"}):
        with pytest.raises(LabelMappingError):
            trained_label_order(bad)


def test_cache_key_includes_the_checkpoint_hash(tmp_path, monkeypatch):
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    va, vb = make_trained("a" * 64), make_trained("b" * 64)
    assert va.model_name == model_id("fake/base", "a" * 64) != vb.model_name
    assert nli.cache_path_for(va.model_name, va.dtype) != nli.cache_path_for(vb.model_name, vb.dtype)
    cache = PairCache(tmp_path / "shared.jsonl")  # even one shared file cannot mix them
    pairs = [("d1", "T", "This SUPPORTS it.", "c")]
    score_pairs_cached(va, cache, pairs)
    score_pairs_cached(va, cache, pairs)
    assert va.pairs_scored == 1  # second call was a hit
    score_pairs_cached(vb, cache, pairs)
    assert vb.pairs_scored == 1 and cache.misses == 2


def test_checkpoint_sha_tracks_weights_and_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        checkpoint_sha(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"w1")
    h1 = checkpoint_sha(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"w2")
    h2 = checkpoint_sha(tmp_path)
    (tmp_path / "config.json").write_text('{"x": 1}')
    assert len({h1, h2, checkpoint_sha(tmp_path)}) == 3


def _ckpt(path, weights=("model.safetensors",)):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "meta.json").write_text(json.dumps({"base_model": "fake/base", "max_length": 64}))
    for w in weights:
        (path / w).write_bytes(b"weights")
    return path


def test_trained_verifier_loads_safetensors_locally_without_remote_code(tmp_path, monkeypatch):
    import transformers

    ckpt = _ckpt(tmp_path / "model")
    calls = {}
    tok = FakeTokenizer()

    def fake_tokenizer(path, **kw):
        calls["tokenizer"] = (path, kw)
        return tok

    def fake_model(path, **kw):
        calls["model"] = (path, kw)
        return ClaimFirstModel(tok)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", fake_model)
    v = TrainedVerifier(checkpoint=tmp_path / "model", threads=0, batch_size=2)
    assert calls["tokenizer"] == (ckpt, {"trust_remote_code": False, "local_files_only": True})
    path, kw = calls["model"]
    assert path == ckpt and kw == {"dtype": torch.float32, "use_safetensors": True,
                                   "trust_remote_code": False, "local_files_only": True}
    assert v.checkpoint_hash == checkpoint_sha(ckpt)


def test_a_pickle_only_checkpoint_is_refused_by_name(tmp_path, monkeypatch):
    import transformers

    def boom(*a, **kw):
        raise AssertionError("must refuse before any from_pretrained call")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", boom)
    assert WEIGHT_FILES == ("model.safetensors",) and "pytorch_model.bin" not in HASHED_FILES
    flat = _ckpt(tmp_path / "flat", weights=("pytorch_model.bin",))
    nested = _ckpt(tmp_path / "zip" / "verifier_model", weights=("pytorch_model.bin",))
    for where, named in ((flat, flat), (tmp_path / "zip", nested)):
        with pytest.raises(FileNotFoundError, match="pytorch_model.bin.*model.safetensors is accepted") as e:
            resolve_checkpoint(where)
        assert str(named) in str(e.value)
        with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
            TrainedVerifier(checkpoint=where)
    with pytest.raises(FileNotFoundError, match="pytorch_model.bin is not accepted"):
        checkpoint_sha(flat)


def test_a_bin_next_to_safetensors_is_ignored_not_hashed(tmp_path):
    a = _ckpt(tmp_path / "a")
    b = _ckpt(tmp_path / "b", weights=("model.safetensors", "pytorch_model.bin"))
    assert resolve_checkpoint(b) == b
    assert checkpoint_sha(a) == checkpoint_sha(b)  # a safetensors-only hash is unchanged


# --- combination rules ----------------------------------------------------------------------------

STRONG_S = Signal(0.9, "SUPPORT", 0)
WEAK_C = Signal(0.2, "CONTRADICT", 1)


def test_r1_is_the_verifier_alone():
    assert rule_r1("NEI", STRONG_S, 0.5) == "SUPPORT"
    assert rule_r1("SUPPORT", WEAK_C, 0.5) == "NEI"
    assert rule_r1("NEI", WEAK_C, 0.2) == "CONTRADICT"  # >= tau


def test_r2_only_demotes_confident_ling_verdicts_the_verifier_cannot_back():
    assert rule_r2("SUPPORT", WEAK_C, 0.3) == "NEI"
    assert rule_r2("CONTRADICT", STRONG_S, 0.3) == "CONTRADICT"  # verifier strong: keep Ling
    assert rule_r2("NEI", WEAK_C, 0.3) == "NEI"
    assert rule_r2("NONE", WEAK_C, 0.3) == "NONE"  # no verdict is not R2's case


def test_r3_only_promotes_ling_abstentions_or_missing_verdicts():
    assert rule_r3("NEI", STRONG_S, 0.8) == "SUPPORT"
    assert rule_r3("NONE", STRONG_S, 0.8) == "SUPPORT"
    assert rule_r3("NEI", STRONG_S, 0.9) == "NEI"  # strictly greater than tau_high
    assert rule_r3("CONTRADICT", STRONG_S, 0.1) == "CONTRADICT"  # never overrides a verdict


def test_r4_is_r2_plus_r3():
    for ling in ("SUPPORT", "CONTRADICT", "NEI", "NONE"):
        for sig in (STRONG_S, WEAK_C):
            want = rule_r2(ling, sig, 0.3) if ling in ("SUPPORT", "CONTRADICT") else rule_r3(ling, sig, 0.8)
            assert rule_r4(ling, sig, 0.3, 0.8) == want
    assert apply_rule("R0", "NONE", STRONG_S) == "NONE"
    with pytest.raises(ValueError):
        apply_rule("R2", "SUPPORT", STRONG_S, {"tau": 0.3})


def test_verifier_signal_takes_the_strongest_non_nei_over_top_k():
    probs = [{"SUPPORT": 0.1, "CONTRADICT": 0.2, "NEI": 0.7}, {"SUPPORT": 0.6, "CONTRADICT": 0.1, "NEI": 0.3}]
    assert verifier_signal(probs) == Signal(0.6, "SUPPORT", 1)
    assert verifier_signal(probs, k=1) == Signal(0.2, "CONTRADICT", 0)


def test_tuning_prefers_ling_on_ties_and_plateaus_over_spikes():
    # Ling already right everywhere: nothing can beat R0, and R0 wins the tie.
    items = {str(i): ("SUPPORT", Signal(0.9, "SUPPORT", 0)) for i in range(6)}
    gold = dict.fromkeys(items, "SUPPORT")
    assert tune_rules(items, gold)["chosen"]["rule"] == "R0"
    # Ling over-eager on two claims the verifier rejects: R2 fixes them.
    items |= {"a": ("SUPPORT", Signal(0.1, "SUPPORT", 0)), "b": ("CONTRADICT", Signal(0.15, "CONTRADICT", 0))}
    gold |= {"a": "NEI", "b": "NEI"}
    t = tune_rules(items, gold)
    assert t["chosen"]["rule"] == "R2" and t["chosen"]["accuracy"] == 1.0
    assert 0.15 < t["chosen"]["params"]["tau_low"] <= 0.9
    assert set(t["best_per_rule"]) == {"R0", "R1", "R2", "R3", "R4"}


def ling_row(qid, gold, pred, top=("d1", "d2")):
    answered = pred != "NEI"
    return {"query_id": qid, "gold_label": gold, "predicted_label": pred, "verdict": "x",
            "answered": answered, "evidence": gold != "NEI", "evidence_qrels": True,
            "abstention_class": "a", "abstention_class_qrels": "a",
            "cited_doc_ids": ["d1"] if answered else [], "retrieved_doc_ids": list(top), "answer": "text"}


def test_combined_rows_are_rag_compare_compatible():
    weak = [PairScore({"SUPPORT": 0.1, "CONTRADICT": 0.1, "NEI": 0.8})] * 2
    strong = [PairScore({"SUPPORT": 0.05, "CONTRADICT": 0.9, "NEI": 0.05}), weak[0]]
    rows = [
        combined_row(ling_row("1", "NEI", "SUPPORT"), weak, "R4", {"tau_low": 0.3, "tau_high": 0.8}),
        combined_row(ling_row("2", "CONTRADICT", "NEI"), strong, "R4", {"tau_low": 0.3, "tau_high": 0.8}),
        combined_row(ling_row("3", "SUPPORT", "SUPPORT"), strong, "R4", {"tau_low": 0.3, "tau_high": 0.8}),
    ]
    r1, r2, r3 = rows
    assert (r1["predicted_label"], r1["answered"], r1["cited_doc_ids"], r1["changed"]) == ("NEI", False, [], True)
    assert r1["abstention_class"] == "correct_abstention"
    assert (r2["predicted_label"], r2["verdict"], r2["cited_doc_ids"]) == ("CONTRADICT", "REFUTED", ["d1"])
    assert r3["changed"] is False and r3["answer"] == "text" and r3["ling_predicted_label"] == "SUPPORT"
    blob = {"run": {"dataset": "beir/scifact/test"}, "rows": rows}
    base = {"run": {"dataset": "beir/scifact/test"},
            "rows": [ling_row("1", "NEI", "SUPPORT"), ling_row("2", "CONTRADICT", "NEI"), ling_row("3", "SUPPORT", "SUPPORT")]}
    res = rag_compare.compare(base, blob)
    assert res["outcomes"]["verdict_correct"]["c"] == 2


# --- output and contamination guards ----------------------------------------------------------------


def test_write_run_refuses_the_committed_artifact(tmp_path, monkeypatch):
    with pytest.raises(UnsafeOutputError):
        verify_combine.write_run(Path("eval/results"), [], {}, "x")
    with pytest.raises(UnsafeOutputError):
        verify_combine.write_run(Path("data/eval_runs/../../eval/results/x"), [], {}, "x")


def test_test_stage_refuses_an_untuned_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_combine, "FROZEN_PATH", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="not tuned"):
        verify_combine.stage_test()
    (tmp_path / "partial.json").write_text(json.dumps({"rule": "R2", "windowing": "truncate"}))
    with pytest.raises(SystemExit, match="not tuned"):
        verify_combine.load_frozen(tmp_path / "partial.json")


FROZEN_OK = {"windowing": "truncate", "rule": "R0", "params": {}, "r1_tau": 0.5,
             "checkpoint_sha": "b" * 64, "tuned_on": "beir/scifact/train"}


def test_test_stage_refuses_a_different_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "frozen.json").write_text(json.dumps(FROZEN_OK))
    monkeypatch.setattr(verify_combine, "FROZEN_PATH", tmp_path / "frozen.json")
    monkeypatch.setattr(verify_combine, "load_verifier", lambda w="truncate": make_trained("a" * 64))
    with pytest.raises(SystemExit, match="not the one the rule was tuned"):
        verify_combine.stage_test()


def test_windowing_choice_is_tied_to_the_checkpoint(tmp_path):
    p = tmp_path / "validation.json"
    p.write_text(json.dumps({"checkpoint_sha": "a" * 64, "chosen_windowing": "window"}))
    assert verify_combine.load_windowing("a" * 64, p) == "window"
    with pytest.raises(SystemExit, match="re-run the validate stage"):
        verify_combine.load_windowing("b" * 64, p)
    with pytest.raises(SystemExit, match="validate stage first"):
        verify_combine.load_windowing("a" * 64, tmp_path / "nope.json")


def test_tuning_refuses_once_the_test_split_was_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_combine, "R1_TEST_DIR", tmp_path / "r1")
    monkeypatch.setattr(verify_combine, "test_dir", lambda rule: tmp_path / rule)
    assert verify_combine.test_outputs_exist() == []
    (tmp_path / "R2").mkdir()
    (tmp_path / "R2" / "rag.json").write_text("{}")
    with pytest.raises(SystemExit, match="re-tuning after the test split"):
        verify_combine.stage_tune()
    with pytest.raises(SystemExit, match="already scored"):
        verify_combine.stage_all()


def test_the_default_stage_is_the_whole_protocol(monkeypatch):
    calls = []
    monkeypatch.setattr(verify_combine, "stage_all", lambda rerun=False: calls.append(("all", rerun)))
    monkeypatch.setattr(verify_combine, "stage_test", lambda rerun=False: calls.append(("test", rerun)))
    verify_combine.main([])
    verify_combine.main(["test", "--rerun"])
    assert calls == [("all", False), ("test", True)]


def test_compare_to_ling_writes_the_paired_report(tmp_path):
    base = {"run": {"dataset": "beir/scifact/test"},
            "rows": [ling_row("1", "NEI", "SUPPORT"), ling_row("2", "SUPPORT", "SUPPORT")]}
    weak = [PairScore({"SUPPORT": 0.1, "CONTRADICT": 0.1, "NEI": 0.8})] * 2
    rows = [combined_row(r, weak, "R2", {"tau_low": 0.3}) for r in base["rows"]]
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "rag.json").write_text(json.dumps({"run": {"dataset": "beir/scifact/test"}, "rows": rows}))
    md = verify_combine.compare_to_ling(tmp_path, baseline=tmp_path / "base.json")
    assert "McNemar" in md and (tmp_path / "compare.md").exists()
    res = json.loads((tmp_path / "compare.json").read_text())
    assert res["outcomes"]["verdict_correct"]["c"] == 1 and res["outcomes"]["verdict_correct"]["b"] == 1
