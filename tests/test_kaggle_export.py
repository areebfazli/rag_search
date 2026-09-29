"""Kaggle packaging: the export's leakage guard, the notebook's logic, and the import path
(app.verify.trained) for the model it returns. Fakes and a random-weight toy BERT built
on the fly — no downloads, no network, no real training."""
import json
import zipfile
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from app.eval.verify_eval import class_scores
from app.ingest.corpus import ClaimLabel
from app.verify import kaggle_export as ke
from app.verify import kaggle_notebook as kn
from app.verify import train as tr
from app.verify import train_data as td
from app.verify import trained
from tests.test_verify_trained import world

# --- the leakage guard --------------------------------------------------------------------------


def rows_for(qid_docs, queries):
    return [{"qid": q, "doc_id": d, "claim": queries[q], "premise": "p", "label": "NEI", "source": "x"}
            for q, d in qid_docs]


def guard(rows, labels, qrels, **kw):
    args = dict(test_ids={"t1"}, test_texts=["a test claim"], test_docs={"tdoc"}, tuning_ids={"1"},
                labels=labels, qrels=qrels)
    return ke.check_leakage(rows, **{**args, **kw})


def test_leakage_guard_passes_clean_data_and_counts_it():
    queries, labels, qrels, *_ = world()
    res = guard({"train": rows_for([("3", "b"), ("4", "x1")], queries), "val": rows_for([("6", "f")], queries)},
                labels, qrels)
    assert res["passed"] and res["test_ids_in_data"] == 0 and res["held_out"]["tuning_doc_overlap_claims"] == 1


@pytest.mark.parametrize("bad, match", [
    ({"train": [("t1", "b")]}, "test claim ids"),
    ({"train": [("1", "b")]}, "held-out tuning claims"),
    ({"train": [("2", "b")]}, "sharing a doc with a tuning claim"),  # 2 shares doc a with tuning claim 1
    ({"train": [("3", "a")]}, "tuning claim's doc"),  # the tuning evidence used as a negative
    ({"train": [("3", "tdoc")]}, "test claim's doc"),
    ({"train": [("3", "b")], "val": [("3", "x1")]}, "both train and val"),
])
def test_leakage_guard_refuses_each_kind_of_leak(bad, match):
    queries, labels, qrels, *_ = world()
    queries = {**queries, "t1": "a test claim"}
    labels = {**labels, "t1": ClaimLabel("NEI")}
    rows = {s: rows_for(v, queries) for s, v in bad.items()}
    with pytest.raises(td.ContaminationError, match=match):
        guard(rows, labels, qrels)


def test_leakage_guard_matches_test_claims_by_text_too():
    queries, labels, qrels, *_ = world()
    rows = {"train": rows_for([("3", "b")], {**queries, "3": "  A TEST  claim"})}
    with pytest.raises(td.ContaminationError, match="identical to a test claim"):
        guard(rows, labels, qrels)


def test_test_doc_overlap_is_refused_in_strict_mode_and_reported_otherwise():
    queries, labels, qrels, *_ = world()
    rows = {"train": rows_for([("3", "b")], queries)}  # claim 3 cites b, b2
    with pytest.raises(td.ContaminationError, match="sharing a doc with a test claim"):
        guard(rows, labels, qrels, test_docs={"b2"})
    res = guard(rows, labels, qrels, test_docs={"b2"}, strict_test_docs=False)
    assert res["test_doc_overlap_claims_in_data"] == 1 and res["test_docs_in_data"] == 0


def test_build_splits_drops_test_duplicates_and_test_doc_overlap():
    queries, labels, qrels, top, docs = world()
    s = td.build_splits(queries, labels, qrels, top, docs, tuning={"1"}, val_frac=0.3,
                        test_texts=["CLAIM 8"], test_docs={"f"})
    used = {p.qid for p in s["train"] + s["val"]}
    assert "8" not in used and "6" not in used  # duplicate text; cites test doc f
    assert s["excluded_test_qids"] == ["6", "8"]
    assert s["report"]["excluded_test_duplicate"] == 1 and s["report"]["excluded_test_doc_overlap"] == 1
    assert all(p.doc_id != "f" for p in s["train"] + s["val"])  # never a negative either


def test_tuning_ids_from_run_holds_out_the_whole_sample(tmp_path):
    p = tmp_path / "rag.json"
    sample = [str(i) for i in range(100)]
    p.write_text(json.dumps({"run": {"dataset": td.TRAIN_DATASET}, "rows": [{"query_id": q} for q in sample[:99]]}))
    assert ke.tuning_ids_from_run(sample, p) == set(sample)
    p.write_text(json.dumps({"run": {"dataset": td.TRAIN_DATASET}, "rows": [{"query_id": "x"}]}))
    with pytest.raises(td.ContaminationError):
        ke.tuning_ids_from_run(sample, p)
    p.write_text(json.dumps({"run": {"dataset": "beir/scifact/test"}, "rows": []}))
    with pytest.raises(td.ContaminationError):
        ke.tuning_ids_from_run(sample, p)


def test_secret_scan():
    assert ke.find_secrets(b'{"claim": "risk-benefit of sk-like enzymes"}') == []
    assert ke.find_secrets(b"key sk-or-v1-" + b"0123456789abcdef" * 4)
    assert ke.find_secrets(b"gsk_" + b"A" * 40)


# --- the notebook ---------------------------------------------------------------------------------


def test_notebook_conversion_and_sha_baking():
    src = ke.render_notebook({"train.jsonl": "a" * 64, "val.jsonl": "b" * 64})
    assert '"train.jsonl": "' + "a" * 64 in src and ke.SHA_PLACEHOLDER not in src
    nb = json.loads(ke.notebook_json(src))
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[0] == "markdown" and "code" in kinds and nb["nbformat"] == 4
    assert nb["cells"][0]["source"][0].startswith("# Train the SciFact")
    code = "".join("".join(c["source"]) + "\n" for c in nb["cells"] if c["cell_type"] == "code")
    compile(code, "train_verifier.ipynb", "exec")
    assert 'if __name__ == "__main__":' in code


def test_notebook_logic_matches_train_py():
    lengths = [5, 50, 7, 300, 12, 9, 100, 8, 64, 3, 20] * 7
    assert kn.epoch_batches(lengths, 8, 64, 13, 1) == tr.epoch_batches(lengths, 8, 64, 13, 1)
    gold = ["SUPPORT", "NEI", "CONTRADICT", "NEI", "NEI"]
    pred = ["SUPPORT", "SUPPORT", "NEI", "NEI", "CONTRADICT"]
    assert kn.class_scores(gold, pred) == class_scores(gold, pred)
    pairs = [td.Pair("q", "d", "c", "t", "a", lab, "rationale") for lab in ("SUPPORT", "NEI", "NEI", "CONTRADICT")]
    assert kn.class_weights([{"label": p.label} for p in pairs]) == td.class_weights(pairs)
    assert kn.LABELS == td.LABELS and kn.KIND == ke.KIND


def test_gpu_config_keeps_the_hyperparameters_and_the_step_composition():
    base = asdict(tr.TrainConfig())
    cfg = kn.make_config({"train_config": base})
    gpu, nb = kn.notebook_keys({"train_config": base}, cfg)
    assert set(gpu) == {f.name for f in fields(tr.TrainConfig)} and set(gpu) | set(nb) == set(cfg)
    # v2 changes only the GPU micro-batching and the epoch budget (early stopping ends it);
    # lr / warmup None = the exported values
    assert {k for k in gpu if gpu[k] != base[k]} == {"micro_batch", "accum", "epochs"}
    assert gpu["epochs"] == 10 and gpu["lr"] == base["lr"] and gpu["warmup_frac"] == base["warmup_frac"]
    assert nb == {k: v for k, v in kn.CONFIG.items() if k not in base}
    assert nb["patience"] == 3 and nb["schedule"] == "cosine" and nb["class_weighting"] == "inverse"
    assert nb["oversample"] == {"CONTRADICT": 2} and nb["stage1"]["enabled"] is True
    assert cfg["micro_batch"] * cfg["accum"] == base["micro_batch"] * base["accum"]
    # the same 16 examples per optimiser step: chunk = micro_batch * accum * 4 is unchanged
    lengths = list(range(200, 0, -1))
    cpu = tr.epoch_batches(lengths, base["micro_batch"], 64, 13, 0)
    gpu_b = kn.epoch_batches(lengths, cfg["micro_batch"], 64, 13, 0)
    step = lambda bs, a: [sum(bs[i : i + a], []) for i in range(0, len(bs), a)]  # noqa: E731
    assert step(cpu, base["accum"]) == step(gpu_b, cfg["accum"])


def test_make_config_refuses_bad_settings():
    base = asdict(tr.TrainConfig())
    for bad in ({"accum": 3}, {"schedule": "step"}, {"class_weighting": "focal"}, {"oversample": {"X": 2}},
                {"oversample": {"CONTRADICT": 0.5}}):
        with pytest.raises(AssertionError):
            kn.make_config({"train_config": base}, {**kn.CONFIG, **bad})
    assert kn.make_config({"train_config": base}, {**kn.CONFIG, "lr": 3e-5})["lr"] == 3e-5


def test_v2_data_defaults_three_hard_negatives_and_a_20pct_validation_split():
    c = tr.TrainConfig()
    assert (c.n_hard, c.val_frac) == (3, 0.20)
    queries, labels, qrels, top, docs = world()
    pairs = td.build_pairs(["3"], queries, labels, qrels, top, docs, random_frac=0.0)  # default n_hard
    assert [p.doc_id for p in pairs if p.source == "hard_negative"] == ["x1", "a", "x2"]
    # the tuning claim's evidence (doc a) is still never a negative, with 3 hard negatives too
    s = td.build_splits(queries, labels, qrels, top, docs, tuning={"1"}, val_frac=0.3)
    assert s["report"]["n_hard"] == 3
    assert all(p.doc_id != "a" for p in s["train"] + s["val"])
    hard = {}
    for p in s["train"] + s["val"]:
        if p.source == "hard_negative":
            hard.setdefault(p.qid, []).append(p.doc_id)
    assert max(len(v) for v in hard.values()) == 3
    rows = {k: [ke.row(p) for p in s[k]] for k in ("train", "val")}
    assert ke.check_leakage(rows, test_ids={"t1"}, test_texts=["a test claim"], test_docs={"tdoc"},
                            tuning_ids={"1"}, labels=labels, qrels=qrels)["passed"]
    # ... and still refuses a leak planted among the extra negatives
    rows["train"].append({**rows["train"][0], "doc_id": "a"})
    with pytest.raises(td.ContaminationError, match="tuning claim's doc"):
        ke.check_leakage(rows, test_ids={"t1"}, test_texts=["a test claim"], test_docs={"tdoc"},
                         tuning_ids={"1"}, labels=labels, qrels=qrels)


def test_hash_file_uses_the_notebook_normalisation():
    claims = ["Aspirin reduces   RISK.", "Café–au–lait spots"]
    docs = [{"title": "A Title", "text": "Some abstract. " * 30}, {"title": "", "text": ""}]
    h = ke.build_hashes(claims, docs, blocklist_texts=["an external near-duplicate"])
    assert h["claims"] == sorted({kn.text_hash("aspirin reduces risk"), kn.text_hash("cafe au lait spots")})
    assert h["titles"] == [kn.text_hash("a title")] and len(h["abstract_prefixes"]) == 1
    assert h["abstract_prefixes"] == [kn.prefix_hash("SOME abstract " * 30)]
    assert h["blocklist"] == [kn.text_hash("An external near-duplicate!")]
    assert kn.norm_text("  Héllo,\tWORLD–2 ") == "hello world 2" and len(kn.text_hash("x")) == 16
    assert ke.find_secrets(json.dumps(h).encode()) == []
    assert ke.EXTERNAL_BLOCKLIST_TEXTS == tuple(ke.EXTERNAL_BLOCKLIST_TEXTS)


def write_dataset(d: Path, rows: dict, cfg: dict, tamper: bool = False, hashes: dict | None = None) -> dict:
    d.mkdir(parents=True)
    blobs = {f"{k}.jsonl": ke.jsonl_bytes(v) for k, v in rows.items()}
    aux = {} if hashes is None else {ke.HASH_FILE: json.dumps(hashes).encode()}
    meta = {"kind": ke.KIND, "version": 2, "git_sha": "test", "train_config": cfg, "report": {"fake": True},
            "class_weights": kn.class_weights(rows["train"]), "leakage_check": {"passed": True},
            "splits": {"val_qids": sorted({r["qid"] for r in rows["val"]})},
            "files": {n: {"sha256": ke.sha256(b), "rows": len(rows[n.split(".")[0]])} for n, b in blobs.items()},
            "aux_files": {n: {"sha256": ke.sha256(b)} for n, b in aux.items()}}
    for n, b in {**blobs, **aux}.items():
        (d / n).write_bytes(b + (b"x" if tamper and n == "val.jsonl" else b""))
    (d / "meta.json").write_text(json.dumps(meta))
    return {n: ke.sha256(b) for n, b in {**blobs, **aux}.items()}


def toy_rows():
    words = ["aspirin", "reduces", "risk", "increases", "no", "effect", "trial"]
    labs = ["SUPPORT", "CONTRADICT", "NEI"]
    mk = lambda i: {"qid": str(i), "doc_id": f"d{i}", "claim": f"{words[i % 7]} {words[(i + 2) % 7]}",  # noqa: E731
                    "premise": " ".join(words[(i + j) % 7] for j in range(5)), "label": labs[i % 3], "source": "x"}
    return {"train": [mk(i) for i in range(9)], "val": [mk(i) for i in range(9, 15)]}


def test_notebook_finds_and_checks_the_dataset(tmp_path):
    rows = toy_rows()
    shas = write_dataset(tmp_path / "in" / "scifact-verifier-data" / "verifier_data", rows, asdict(tr.TrainConfig()))
    d = kn.find_data_dir(str(tmp_path / "in"))
    meta, got = kn.load_data(d, shas)
    assert got == rows and meta["kind"] == ke.KIND
    with pytest.raises(AssertionError, match="notebook"):
        kn.load_data(d, {**shas, "val.jsonl": "0" * 64})
    write_dataset(tmp_path / "bad", rows, asdict(tr.TrainConfig()), tamper=True)
    with pytest.raises(AssertionError, match="meta.json"):
        kn.load_data(tmp_path / "bad", {})
    with pytest.raises(SystemExit, match="Add Input"):
        kn.find_data_dir(str(tmp_path / "empty"))
    with pytest.raises(SystemExit, match="new dataset version"):  # a v1 upload has no hash file
        kn.load_hashes(d, meta)


def test_notebook_checks_the_hash_file_too(tmp_path):
    rows = toy_rows()
    hashes = ke.build_hashes(["claim x"], [{"title": "T", "text": "abstract"}])
    shas = write_dataset(tmp_path / "d", rows, asdict(tr.TrainConfig()), hashes=hashes)
    meta, _ = kn.load_data(tmp_path / "d", shas)
    assert kn.load_hashes(tmp_path / "d", meta)["claims"] == set(hashes["claims"])
    with pytest.raises(AssertionError, match="notebook"):
        kn.load_data(tmp_path / "d", {**shas, ke.HASH_FILE: "0" * 64})
    (tmp_path / "d" / ke.HASH_FILE).write_text("{}")
    with pytest.raises(AssertionError, match="meta.json"):
        kn.load_data(tmp_path / "d", shas)


# --- notebook -> zip -> import, end to end on a toy model ------------------------------------------


def fake_spec(**kw) -> dict:
    return {"name": "fakever", "repo_id": "someone/fake-claims", "revision": "0123456789abcdef" * 2 + "01234567",
            "files": ["train.jsonl"], "format": "jsonl", "split": "train",
            "columns": {"claim": "c", "evidence": "e", "label": "y"},
            "label_map": {"SUPPORTED": "SUPPORT", "REFUTED": "CONTRADICT", "NEUTRAL": "NEI"},
            "cap": 100, "license": "test", "url": "https://example.org/fake", **kw}


def toy_base_model(path: Path) -> Path:
    """A 1-layer, 16-dim BERT with random weights + a 20-token WordPiece vocab."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    path.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "aspirin", "reduces", "risk", "increases",
             "no", "effect", "trial", "t", ".", "a"]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    BertModel(cfg).save_pretrained(path)  # like the real base: no classification head
    return path


def test_notebook_output_imports_as_a_trained_verifier(tmp_path, monkeypatch):
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    base = toy_base_model(tmp_path / "base")
    cfg = {**asdict(tr.TrainConfig()), "max_length": 32}
    rows = toy_rows()
    hashes = ke.build_hashes([rows["train"][0]["claim"]], [{"title": "T", "text": "abstract"}])
    shas = write_dataset(tmp_path / "data", rows, cfg, hashes=hashes)
    # stage 1 on a fake Hub dataset; one row repeats a SciFact claim and must be dropped
    ext = tmp_path / "ext.jsonl"
    ext.write_text("".join(json.dumps(r) + "\n" for r in [
        {"c": "aspirin reduces risk", "e": "aspirin reduces risk trial", "y": "SUPPORTED"},
        {"c": "aspirin increases risk", "e": "aspirin reduces risk", "y": "REFUTED"},
        {"c": "no effect", "e": "trial of aspirin", "y": "NEUTRAL"},
        {"c": rows["train"][0]["claim"].upper(), "e": "risk", "y": "SUPPORTED"},
        {"c": "effect trial", "e": "no", "y": "???"}]))
    calls = []
    monkeypatch.setattr(kn, "hf_download", lambda repo, f, rev: calls.append((repo, f, rev)) or str(ext))
    spec = fake_spec(format="jsonl")
    config = {**kn.CONFIG, "epochs": 4, "patience": 1, "min_delta": 1.0}  # epoch 2 cannot beat epoch 1 by 1.0
    out = tmp_path / "working" / "verifier_model"
    metrics = kn.train(tmp_path / "data", out, base_model=str(base), allow_cpu=True, expected=shas,
                       config=config, external=[spec], metrics_copy=tmp_path / "working" / "metrics.json")
    assert calls == [(spec["repo_id"], "train.jsonl", spec["revision"])]
    assert [(h["stage"], h["epoch"]) for h in metrics["history"]] == [
        ("stage1", 1), ("stage1", 2), ("stage2", 1), ("stage2", 2)]
    assert [h["candidate"] for h in metrics["history"]] == [False, False, True, True]
    assert metrics["best"] == {**metrics["best"], "stage": "stage2", "epoch": 1}
    assert metrics["early_stopping"]["stopped_epoch"] == 2 and metrics["early_stopping"]["epochs_run"] == 2
    (ds,) = metrics["stage1"]["datasets"]
    assert ds["revision"] == spec["revision"] and ds["license"] == "test" and ds["used"] == 3
    assert ds["contamination_dropped"]["claim_matches_scifact"] == 1 and ds["unknown_labels"] == {"???": 1}
    assert metrics["class_counts"]["stage1"] == {"SUPPORT": 1, "CONTRADICT": 1, "NEI": 1}
    raw = metrics["class_counts"]["stage2_raw"]
    assert metrics["class_counts"]["stage2_effective"] == {**raw, "CONTRADICT": 2 * raw["CONTRADICT"]}
    assert metrics["class_weights"]["stage2"] == kn.compute_class_weights(metrics["class_counts"]["stage2_effective"])
    h = metrics["history"][2]
    assert len(h["val_confusion"]) == 3 and sum(map(sum, h["val_confusion"])) == len(rows["val"])
    assert set(h["val_per_label"]) == set(td.LABELS)
    assert json.loads((tmp_path / "working" / "metrics.json").read_text()) == metrics
    meta = json.loads((out / "meta.json").read_text())
    assert meta["config"] == cfg and meta["gpu_config"]["micro_batch"] == 8 and meta["pair_order"] == "claim_first"
    assert tr.TrainConfig(**meta["config"]) == tr.TrainConfig(max_length=32)  # verify_combine re-derives splits
    assert meta["notebook_config"]["patience"] == 1 and meta["gpu_config"]["epochs"] == 4
    assert meta["val_qids"] == sorted(str(i) for i in range(9, 15)) and ke.HASH_FILE in meta["data_export"]["sha256"]
    assert meta["stage1"]["datasets"][0]["revision"] == spec["revision"]
    for f in ("metrics.json", "train_config.json", "config.json", "model.safetensors", "tokenizer.json"):
        assert (out / f).is_file(), f

    zp = kn.zip_model(out, tmp_path / "working" / "verifier_model.zip")
    names = zipfile.ZipFile(zp).namelist()
    assert "config.json" in names and all("/" not in n for n in names)  # files at the archive root
    dest = tmp_path / "models" / "verifier" / "model"
    zipfile.ZipFile(zp).extractall(dest)

    v = trained.TrainedVerifier(checkpoint=dest, threads=0, batch_size=2)
    sha = trained.checkpoint_sha(dest)
    assert v.checkpoint == str(dest) and v.checkpoint_hash == sha and v.max_length == 32
    assert v.model_name == trained.model_id(str(base), sha) and sha[:12] in v.model_name
    assert v.labels == ["SUPPORT", "CONTRADICT", "NEI"]
    (s,) = v.score([("T", "aspirin reduces risk", "aspirin reduces risk")])
    assert sum(s.probs.values()) == pytest.approx(1.0)

    # the per-pair cache key follows the weights: new weights -> new model id -> new cache
    from app.verify import nli

    import safetensors.torch as st
    w = st.load_file(dest / "model.safetensors")
    w = {k: t + 0.01 if k.startswith("classifier") else t for k, t in w.items()}
    st.save_file(w, dest / "model.safetensors", metadata={"format": "pt"})
    v2 = trained.TrainedVerifier(checkpoint=dest, threads=0, batch_size=2)
    assert v2.checkpoint_hash != sha and v2.model_name != v.model_name
    pair = ("T", "aspirin reduces risk", "aspirin")
    key = lambda vv: nli.pair_key(vv.model_name, vv.dtype, vv.window_tag(*pair), vv.max_length, "d1",  # noqa: E731
                                  nli.premise_text(pair[0], pair[1]), pair[2])
    assert key(v) != key(v2)


# --- importer path handling -------------------------------------------------------------------------


def fake_ckpt(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"w")
    return path


def test_resolve_checkpoint_direct_nested_and_refusals(tmp_path):
    assert trained.resolve_checkpoint(fake_ckpt(tmp_path / "a")) == tmp_path / "a"
    fake_ckpt(tmp_path / "b" / "verifier_model")  # zip unpacked into a subfolder
    assert trained.resolve_checkpoint(tmp_path / "b") == tmp_path / "b" / "verifier_model"
    with pytest.raises(FileNotFoundError, match="unzip"):
        trained.resolve_checkpoint(tmp_path / "missing")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "config.json").write_text("{}")  # no weights
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        trained.resolve_checkpoint(tmp_path / "c")
    fake_ckpt(tmp_path / "d" / "x")
    fake_ckpt(tmp_path / "d" / "y")
    with pytest.raises(FileNotFoundError, match="2 checkpoint folders"):
        trained.resolve_checkpoint(tmp_path / "d")
    assert trained.CHECKPOINT_DIR == Path("data/models/verifier/model") == tr.OUT_DIR / tr.MODEL_SUBDIR


def test_checkpoint_hash_covers_the_tokenizer(tmp_path):
    d = fake_ckpt(tmp_path / "m")
    h1 = trained.checkpoint_sha(d)
    (d / "vocab.txt").write_text("a\nb\n")
    assert trained.checkpoint_sha(d) != h1


def test_refuses_a_checkpoint_trained_in_another_pair_order(tmp_path):
    d = fake_ckpt(tmp_path / "m")
    (d / "meta.json").write_text(json.dumps({"pair_order": "premise_first"}))
    with pytest.raises(trained.LabelMappingError, match="pair order"):
        trained.TrainedVerifier(checkpoint=d, tokenizer=object(), model=object())
