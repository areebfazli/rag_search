# %% [markdown]
# # Train the SciFact evidence verifier on a Kaggle GPU (version 2)
#
# Fine-tunes `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT)
# into a 3-way SUPPORT / CONTRADICT / NEI classifier over (claim, abstract) pairs.
# The SciFact pairs come from `python -m app.verify.kaggle_export` (beir/scifact/train only;
# the test claims, the held-out tuning claims and every claim sharing a doc with either
# were removed and re-checked before export).
#
# **What changed from version 1** (every item is a `CONFIG` flag in the next cells):
#
# - up to 10 epochs with early stopping on SciFact validation macro-F1 (patience 3),
#   warmup then cosine decay, the best epoch kept (v1: 3 epochs, linear decay);
# - CONTRADICT pairs duplicated 2x in SciFact train, plus inverse-frequency class weights
#   computed on the oversampled set;
# - 3 hard negatives per claim instead of 2, and a 20% validation split (v1: 15%),
#   still grouped by doc and still strict (claims sharing a paper with a test claim excluded);
# - stage 1 on external claim-verification data (`EXTERNAL_DATASETS`: HealthVer, ~8.8k
#   pairs, and PubMedQA-L as claims, 1,000 pairs; 2 epochs), loaded from the Hugging Face
#   Hub at pinned commits and re-checked here against hashes of every SciFact claim and
#   corpus title/abstract; then stage 2 fine-tunes on SciFact train. Model selection uses
#   SciFact validation only, never external data;
# - each epoch prints per-class F1 and a confusion matrix; `metrics.json` records the
#   per-epoch history, the datasets and revisions used, class counts and weights.
#
# **Settings needed:** Accelerator = GPU T4 x2, Internet = On (the base model and the
# stage-1 datasets download from Hugging Face), the dataset attached (Add Input).
# Then Run All: about 10-15 minutes on a T4 (downloads ~2 min, stage 1 ~3-5 min, stage 2
# at most 10 epochs of ~30 s, usually fewer). More than 45 minutes means something is wrong.
#
# **Output:** `/kaggle/working/verifier_model.zip` and `/kaggle/working/metrics.json`
# (Output tab / file browser, right side).

# %%
import copy
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

EXPECTED_SHA256: dict = {}  # filled in by app.verify.kaggle_export
KIND = "scifact-verifier-data"
LABELS = ("SUPPORT", "CONTRADICT", "NEI")
LABEL_IDS = {lab: i for i, lab in enumerate(LABELS)}
HASH_FILE = "scifact_hashes.json"
PREFIX_CHARS = 200  # evidence prefix compared against SciFact abstract prefixes
OUT_DIR = Path("/kaggle/working/verifier_model")
ZIP_PATH = Path("/kaggle/working/verifier_model.zip")
METRICS_PATH = Path("/kaggle/working/metrics.json")


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


# %% [markdown]
# ## 0. Configuration
#
# Defaults are version 2. `None` keeps the value exported in `meta.json`'s `train_config`
# (the same hyperparameters as `app/verify/train.py`: lr 2e-5, weight decay 0.01, 10% warmup,
# seed 13, max length 512). Set `CONFIG["stage1"]["enabled"] = False` to train on SciFact only.

# %%
CONFIG: dict = {
    # GPU micro-batching: 8 x 2 = the same effective batch (16) as the CPU run's 2 x 8.
    "micro_batch": 8,
    "accum": 2,
    # Stage 2 (SciFact train): maximum epochs; early stopping usually ends it sooner.
    "epochs": 10,
    "lr": None,  # None = train_config's (2e-5)
    "warmup_frac": None,  # None = train_config's (0.1)
    "schedule": "cosine",  # "cosine" | "linear" (decay after the warmup)
    "patience": 3,  # stop after this many epochs without a new best val macro-F1; 0 = never
    "min_delta": 0.0,  # an epoch must beat the best by more than this to count
    "class_weighting": "inverse",  # "inverse" | "sqrt_inverse" | "none"
    "oversample": {"CONTRADICT": 2},  # duplication factor per label (SciFact train only)
    # Stage 1 (external claim-verification data, EXTERNAL_DATASETS below), run before stage 2.
    "stage1": {
        "enabled": True,
        "epochs": 2,
        "lr": 2e-5,
        "class_weighting": "inverse",
        "max_pairs": 12000,  # total cap over all external datasets (seeded sample); ~9.8k exist
    },
}

# External datasets for stage 1, each pinned to a Hub commit. One entry:
#   {"name", "repo_id", "revision" (40-hex commit), "files" [paths in the repo],
#    "format" ("parquet" | "jsonl" | "json" | "csv" | "tsv"), "split",
#    "columns": {"claim", "evidence" (a column or a list of columns, joined), "title"?, "label", "id"?},
#    "label_map": {raw label -> SUPPORT/CONTRADICT/NEI}, "drop_labels"?: [raw labels],
#    "where"?: {column: [allowed values]}, "records_key"? (json only), "cap", "license", "url"}
#
# Both below were downloaded at these commits and checked against all 1,109 SciFact claims and
# the 5,183 corpus docs before being listed (0 exact or near-duplicate claims, 0 title or
# abstract matches); this notebook re-checks every row against scifact_hashes.json anyway.
EXTERNAL_DATASETS: list = [
    {
        # HealthVer (Sarrouti et al., Findings of EMNLP 2021): COVID-19 health claims vs.
        # evidence snippets from CORD-19 abstracts. A faithful subset of the original CSVs.
        "name": "healthver",
        "repo_id": "jpd459/healthver_resplit_larger",
        "revision": "48fadcc0bb737318989ad7d5e63f400127308cef",
        "files": ["train.jsonl"],
        "format": "jsonl",
        "split": "train",
        "columns": {"claim": "Claim", "evidence": "Evidence", "label": "Label"},
        "label_map": {"Supports": "SUPPORT", "Refutes": "CONTRADICT", "Neutral": "NEI"},
        "cap": None,  # all ~8.8k pairs
        "license": "unclear: original repo has no LICENSE; mirror card says apache-2.0, "
                   "dwadden/healthver_entailment says cc-by-nc-2.0 -> treat as research / non-commercial",
        "url": "https://github.com/sarrouti/HealthVer",
    },
    {
        # PubMedQA-L (Jin et al., EMNLP 2019), the 1,000 expert-labelled questions rewritten
        # as claims; evidence = the abstract without its conclusion. long_answer (the
        # conclusion) is never read: it states the answer.
        "name": "pubmedqa_l",
        "repo_id": "umbc-scify/PubMedClaim",
        "revision": "d04924b16fb940e12658ef2fd9e7bde91029f39c",
        "files": ["pqa_labeled/val-00000-of-00001.parquet", "pqa_labeled/test-00000-of-00001.parquet"],
        "format": "parquet",
        "split": "pqa_labeled (val + test = all 1,000)",
        "columns": {"claim": "claim", "evidence": "context.contexts", "label": "final_decision", "id": "pubid"},
        "label_map": {"yes": "SUPPORT", "no": "CONTRADICT", "maybe": "NEI"},
        "cap": None,  # all 1,000
        "license": "PubMedQA: MIT; the umbc claim rewrite states no license",
        "url": "https://huggingface.co/datasets/umbc-scify/PubMedClaim",
    },
]

WEIGHTINGS = ("inverse", "sqrt_inverse", "none")
SCHEDULES = ("cosine", "linear")
FORMATS = ("parquet", "jsonl", "json", "csv", "tsv")


# %% [markdown]
# ## 1. Find and check the data

# %%
def find_data_dir(root: str = "/kaggle/input") -> Path:
    """The attached dataset, wherever Kaggle mounted it (found by its meta.json marker)."""
    for m in sorted(glob.glob(f"{root}/**/meta.json", recursive=True)):
        try:
            if json.loads(Path(m).read_text()).get("kind") == KIND:
                return Path(m).parent
        except (OSError, ValueError):
            continue
    raise SystemExit(f"No {KIND} dataset under {root}: attach it with 'Add Input' (right panel).")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_data(data_dir: Path, expected: dict | None = None) -> tuple[dict, dict]:
    """(meta, {"train": rows, "val": rows}); every file's sha256 (the .jsonl files and the
    aux files such as scifact_hashes.json) must match meta.json and the hashes baked into
    this notebook at export time."""
    meta = json.loads((data_dir / "meta.json").read_text())
    rows = {}
    for name, info in meta["files"].items():
        got = file_sha256(data_dir / name)
        assert got == info["sha256"], f"{name}: sha256 {got} != meta.json {info['sha256']}"
        if expected:
            assert got == expected[name], f"{name}: sha256 {got} != notebook {expected[name]}"
        rows[name.split(".")[0]] = [json.loads(ln) for ln in (data_dir / name).read_text().splitlines() if ln]
        assert len(rows[name.split(".")[0]]) == info["rows"], f"{name}: row count differs from meta.json"
    aux = meta.get("aux_files") or {}
    for name, info in aux.items():
        got = file_sha256(data_dir / name)
        assert got == info["sha256"], f"{name}: sha256 {got} != meta.json {info['sha256']}"
        if expected:
            assert got == expected[name], f"{name}: sha256 {got} != notebook {expected[name]}"
    if expected:
        assert set(expected) == set(meta["files"]) | set(aux), "data files differ from the ones exported"
    assert meta["leakage_check"]["passed"], "the export's leakage check did not pass"
    return meta, rows


def load_hashes(data_dir: Path, meta: dict) -> dict:
    """The SciFact contamination hashes (verified by load_data), as sets."""
    if HASH_FILE not in (meta.get("aux_files") or {}):
        raise SystemExit(f"The attached dataset has no {HASH_FILE} (a version-1 upload?). Upload the new "
                         "verifier_data folder as a new dataset version, or set CONFIG['stage1']['enabled'] = False.")
    blob = json.loads((data_dir / HASH_FILE).read_text())
    return {k: set(blob.get(k, ())) for k in ("claims", "titles", "abstract_prefixes", "blocklist")}


def make_config(meta: dict, config: dict | None = None) -> dict:
    """train_config with CONFIG's overrides for keys it shares (None keeps the exported
    value), plus CONFIG's notebook-only keys. Refuses a changed effective batch."""
    config = CONFIG if config is None else config
    base = meta["train_config"]
    cfg = dict(base)
    for k, v in config.items():
        if k in base:
            if v is not None:
                cfg[k] = v
        else:
            cfg[k] = copy.deepcopy(v)
    assert cfg["micro_batch"] * cfg["accum"] == base["micro_batch"] * base["accum"], "effective batch changed"
    assert cfg["schedule"] in SCHEDULES, f"schedule must be one of {SCHEDULES}"
    assert cfg["class_weighting"] in WEIGHTINGS, f"class_weighting must be one of {WEIGHTINGS}"
    assert cfg["stage1"]["class_weighting"] in WEIGHTINGS, f"stage1 class_weighting must be one of {WEIGHTINGS}"
    assert cfg["patience"] >= 0 and cfg["epochs"] >= 1 and cfg["stage1"]["epochs"] >= 0
    for lab, f in cfg["oversample"].items():
        assert lab in LABELS and f >= 1, f"oversample: {lab} x {f}"
    return cfg


def notebook_keys(meta: dict, cfg: dict) -> tuple[dict, dict]:
    """(TrainConfig-shaped effective config, notebook-only settings)."""
    base = meta["train_config"]
    return {k: cfg[k] for k in base}, {k: v for k, v in cfg.items() if k not in base}


# %% [markdown]
# ## 2. Helpers (pure Python; the batching / scoring logic is app/verify/train.py's)

# %%
def label_counts(rows) -> dict:
    c = Counter(r["label"] for r in rows)
    return {lab: c.get(lab, 0) for lab in LABELS}


def compute_class_weights(counts: dict, mode: str = "inverse") -> dict:
    """inverse: n / (3 * n_label) (a balanced loss; app.verify.train_data.class_weights);
    sqrt_inverse: its square root (a softer correction); none: 1.0. A label with no
    examples gets 0.0 under the inverse modes."""
    assert mode in WEIGHTINGS, mode
    n = sum(counts.get(lab, 0) for lab in LABELS)
    out = {}
    for lab in LABELS:
        c = counts.get(lab, 0)
        if mode == "none":
            out[lab] = 1.0
        elif not c:
            out[lab] = 0.0
        else:
            w = n / (len(LABELS) * c)
            out[lab] = round(math.sqrt(w) if mode == "sqrt_inverse" else w, 6)
    return out


def class_weights(rows) -> dict:
    """n / (3 * n_label) — app.verify.train_data.class_weights."""
    return compute_class_weights(label_counts(rows), "inverse")


def oversample(rows, factors: dict, seed: int = 13) -> list:
    """rows + extra copies per label: factor 2 = every row of that label twice; a
    fractional part adds a seeded sample of that share. Deterministic."""
    rng = random.Random(seed)
    out = list(rows)
    for lab in LABELS:
        f = float(factors.get(lab, 1))
        assert f >= 1, f"oversample factor {lab} x {f}"
        mine = [r for r in rows if r["label"] == lab]
        whole, frac = int(f), f - int(f)
        out.extend(mine * (whole - 1))
        if frac and mine:
            out.extend(rng.sample(mine, round(frac * len(mine))))
    return out


class EarlyStopping:
    """Tracks the best score; `step` -> (improved, stop). An epoch improves when it beats
    the best by more than `min_delta`; `stop` once `patience` epochs in a row did not
    (patience 0 never stops)."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad_epochs, self.stopped_epoch = None, None, 0, None

    def step(self, score: float, epoch: int) -> tuple[bool, bool]:
        improved = self.best is None or score > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        stop = self.patience > 0 and self.bad_epochs >= self.patience
        if stop:
            self.stopped_epoch = epoch
        return improved, stop


def lr_lambda(step: int, warmup: int, total: int, schedule: str = "cosine") -> float:
    """LR multiplier: linear warmup to 1 over `warmup` steps, then cosine or linear decay
    to 0 at `total` (the linear one is transformers.get_linear_schedule_with_warmup)."""
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if schedule == "linear":
        return max(0.0, 1.0 - progress)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}")


def epoch_batches(lengths, micro_batch: int, chunk: int, seed: int, epoch: int) -> list[list[int]]:
    """Seeded shuffle, chunks of `chunk` sorted by length, sliced into micro-batches."""
    order = list(range(len(lengths)))
    random.Random(seed * 1000 + epoch).shuffle(order)
    batches = []
    for c in range(0, len(order), chunk):
        block = sorted(order[c : c + chunk], key=lambda i: lengths[i])
        batches.extend(block[b : b + micro_batch] for b in range(0, len(block), micro_batch))
    return batches


def class_scores(gold, pred) -> dict:
    """Accuracy, per-label P/R/F1, macro-F1 — app.eval.verify_eval.class_scores."""
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    per = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        ng = sum(g == lab for g in gold)
        npred = sum(p == lab for p in pred)
        prec = tp / npred if npred else 0.0
        rec = tp / ng if ng else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"n": ng, "predicted": npred, "precision": round(prec, 4),
                    "recall": round(rec, 4), "f1": round(f1, 4)}
    return {"accuracy": round(correct / n, 4) if n else None, "correct": correct,
            "macro_f1": round(sum(per[lab]["f1"] for lab in LABELS) / len(LABELS), 4), "per_label": per}


def confusion_matrix(gold, pred) -> list[list[int]]:
    """3x3 counts in LABELS order: rows = gold, columns = predicted."""
    m = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, pred, strict=True):
        m[LABEL_IDS[g]][LABEL_IDS[p]] += 1
    return m


def format_confusion(m) -> str:
    w = max(10, *(len(str(v)) for row in m for v in row))
    head = " " * 12 + "".join(f"{'pred ' + lab[:4]:>{w + 1}}" for lab in LABELS)
    lines = [head] + [f"  {'gold ' + lab[:4]:<10}" + "".join(f"{v:>{w + 1}}" for v in row)
                      for lab, row in zip(LABELS, m, strict=True)]
    return "\n".join(lines)


# %% [markdown]
# ## 3. External data (stage 1) and the contamination re-check

# %%
def norm_text(text) -> str:
    """Text for duplicate detection: NFKD with combining marks removed (é -> e), lowercase,
    [a-z0-9]+ tokens joined by one space (app.verify.kaggle_export hashes SciFact with this
    same function)."""
    plain = "".join(ch for ch in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", plain.lower()))


def text_hash(text) -> str:
    """16-hex sha256 prefix of norm_text(text)."""
    return hashlib.sha256(norm_text(text).encode()).hexdigest()[:16]


def prefix_hash(text, n: int = PREFIX_CHARS) -> str:
    """text_hash of the first `n` characters of norm_text(text)."""
    return text_hash(norm_text(text)[:n])


def validate_spec(spec: dict) -> None:
    for k in ("name", "repo_id", "revision", "files", "format", "columns", "label_map", "cap", "license", "url"):
        assert k in spec, f"external dataset spec {spec.get('name')!r} lacks {k!r}"
    assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), f"{spec['name']}: revision must be a pinned 40-hex commit"
    assert spec["format"] in FORMATS, f"{spec['name']}: format {spec['format']!r}"
    assert spec["files"], f"{spec['name']}: no files"
    for k in ("claim", "evidence", "label"):
        assert spec["columns"].get(k), f"{spec['name']}: columns lack {k!r}"
    bad = {v for v in spec["label_map"].values() if v not in LABELS}
    assert not bad, f"{spec['name']}: label_map targets {bad} are not {LABELS}"


def hf_download(repo_id: str, filename: str, revision: str) -> str:
    """One file of a Hub dataset repo at a pinned commit -> local path (no token needed)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)


def read_records(path: str, fmt: str, records_key: str | None = None) -> list[dict]:
    if fmt in ("csv", "tsv"):
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if fmt == "tsv" else ","))
    if fmt == "jsonl":
        with open(path, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]
    if fmt == "json":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if records_key:
            blob = blob[records_key]
        if isinstance(blob, dict):  # columnar {"col": [values]}
            cols = list(blob)
            return [dict(zip(cols, vals, strict=True)) for vals in zip(*(blob[c] for c in cols), strict=True)]
        return list(blob)
    if fmt == "parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"unknown format {fmt!r}")


def field(rec, path: str):
    """rec[path], or a dotted path into nested dicts ("context.contexts"; parquet struct
    columns arrive from pandas as dicts). Missing -> None."""
    if isinstance(rec, dict) and path in rec:
        return rec[path]
    v = rec
    for part in path.split("."):
        if isinstance(v, dict):
            v = v.get(part)
        else:
            v = getattr(v, part, None)
        if v is None:
            return None
    return v


def _text(v) -> str:
    if v is None:
        return ""
    if hasattr(v, "tolist"):  # numpy arrays / scalars from parquet
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return " ".join(_text(x) for x in v).strip()
    if isinstance(v, float) and math.isnan(v):
        return ""
    return " ".join(str(v).split())


def map_label(spec: dict, raw) -> str | None:
    """The spec's label for a raw value (compared as a stripped string), or None when it
    is listed in drop_labels or unknown."""
    key = _text(raw)
    if key in {str(x) for x in spec.get("drop_labels", ())}:
        return None
    return {str(k): v for k, v in spec["label_map"].items()}.get(key)


def premise_text(title: str, evidence: str) -> str:
    """app.verify.nli.premise_text: "title. evidence", or the evidence alone."""
    title, evidence = title.strip(), evidence.strip()
    return f"{title}. {evidence}".strip() if title else evidence


def build_external_rows(records, spec: dict) -> tuple[list[dict], dict]:
    """Spec-mapped rows {qid, doc_id, claim, premise, label, source, title, evidence} and a
    report of what was dropped. Fails loudly if nothing maps."""
    cols = spec["columns"]
    ev_cols = cols["evidence"] if isinstance(cols["evidence"], (list, tuple)) else [cols["evidence"]]
    where = spec.get("where") or {}
    rows, unknown, dropped = [], Counter(), Counter()
    rep = {"rows_raw": 0, "rows_outside_where": 0, "dropped_empty": 0}
    for i, rec in enumerate(records):
        rep["rows_raw"] += 1
        if any(_text(field(rec, c)) not in {str(v) for v in allowed} for c, allowed in where.items()):
            rep["rows_outside_where"] += 1
            continue
        raw = _text(field(rec, cols["label"]))
        label = map_label(spec, raw)
        if label is None:
            (dropped if raw in {str(x) for x in spec.get("drop_labels", ())} else unknown)[raw] += 1
            continue
        claim = _text(field(rec, cols["claim"]))
        evidence = " ".join(t for t in (_text(field(rec, c)) for c in ev_cols) if t)
        title = _text(field(rec, cols["title"])) if cols.get("title") else ""
        if not claim or not evidence:
            rep["dropped_empty"] += 1
            continue
        rid = _text(field(rec, cols["id"])) if cols.get("id") else str(i)
        rows.append({"qid": f"{spec['name']}:{rid}", "doc_id": f"{spec['name']}:{i}", "claim": claim,
                     "premise": premise_text(title, evidence), "label": label,
                     "source": f"ext:{spec['name']}", "title": title, "evidence": evidence})
    rep["dropped_by_label"] = dict(sorted(dropped.items()))
    rep["unknown_labels"] = dict(sorted(unknown.items()))
    rep["mapped"] = len(rows)
    rep["mapped_by_label"] = label_counts(rows)
    if not rows:
        raise SystemExit(f"{spec['name']}: no rows mapped to {LABELS} (columns {cols}, labels seen "
                         f"{dict(unknown) or dict(dropped)}); the spec does not match the data")
    return rows, rep


def filter_contaminated(rows, hashes: dict) -> tuple[list[dict], dict]:
    """Drops rows whose claim matches a SciFact claim (or the blocklist of near-duplicates
    found at export) or whose evidence matches a SciFact corpus title or abstract prefix."""
    claims, block = hashes.get("claims", set()), hashes.get("blocklist", set())
    titles, prefixes = hashes.get("titles", set()), hashes.get("abstract_prefixes", set())
    rep = Counter()
    kept = []
    for r in rows:
        ch = text_hash(r["claim"])
        why = ("claim_matches_scifact" if ch in claims else
               "claim_in_blocklist" if ch in block else
               "title_matches_scifact" if r.get("title") and text_hash(r["title"]) in titles else
               "evidence_matches_scifact_abstract" if prefix_hash(r.get("evidence") or r["premise"]) in prefixes
               else None)
        if why:
            rep[why] += 1
        else:
            kept.append(r)
    keys = ("claim_matches_scifact", "claim_in_blocklist", "title_matches_scifact", "evidence_matches_scifact_abstract")
    return kept, {k: rep.get(k, 0) for k in keys}


def dedup_rows(rows) -> tuple[list[dict], int]:
    """Exact duplicates of (claim, premise) after norm_text: the first is kept when the
    copies agree on the label; a pair whose copies disagree is dropped entirely (label
    noise). Returns (rows, number dropped)."""
    key = lambda r: (norm_text(r["claim"]), norm_text(r["premise"]))  # noqa: E731
    labels: dict = {}
    for r in rows:
        labels.setdefault(key(r), set()).add(r.get("label"))
    seen, out = set(), []
    for r in rows:
        k = key(r)
        if k not in seen and len(labels[k]) == 1:
            seen.add(k)
            out.append(r)
    return out, len(rows) - len(out)


def cap_rows(rows, cap: int | None, seed: int) -> list[dict]:
    """A seeded sample of at most `cap` rows, in their original order."""
    if cap is None or len(rows) <= cap:
        return list(rows)
    keep = sorted(random.Random(seed).sample(range(len(rows)), cap))
    return [rows[i] for i in keep]


def prepare_external(specs, hashes: dict, stage1: dict, seed: int) -> tuple[list[dict], list[dict]]:
    """(stage-1 rows, one report per dataset): download at the pinned revision, map
    labels, drop SciFact-contaminated rows and duplicates, cap per dataset, then cap the
    total at stage1["max_pairs"]."""
    all_rows, reports = [], []
    for k, spec in enumerate(specs):
        validate_spec(spec)
        records = []
        for f in spec["files"]:
            try:
                path = hf_download(spec["repo_id"], f, spec["revision"])
            except Exception as e:  # noqa: BLE001 — any download failure means the same thing here
                raise SystemExit(
                    f"Could not download {spec['repo_id']}/{f}@{spec['revision'][:12]} ({type(e).__name__}: {e}). "
                    "Turn Settings > Internet on, or set CONFIG['stage1']['enabled'] = False to train on "
                    "SciFact only.") from None
            records.extend(read_records(path, spec["format"], spec.get("records_key")))
        rows, rep = build_external_rows(records, spec)
        rows, contam = filter_contaminated(rows, hashes)
        rows, dups = dedup_rows(rows)
        before_cap = len(rows)
        rows = cap_rows(rows, spec["cap"], seed + k)
        rep.update({"contamination_dropped": contam, "duplicates_dropped": dups,
                    "before_cap": before_cap, "used": len(rows), "used_by_label": label_counts(rows)})
        reports.append({"name": spec["name"], "repo_id": spec["repo_id"], "revision": spec["revision"],
                        "files": list(spec["files"]), "split": spec.get("split"), "license": spec["license"],
                        "url": spec["url"], "label_map": dict(spec["label_map"]), **rep})
        log(f"stage 1 data {spec['name']} ({spec['repo_id']}@{spec['revision'][:12]}): {rep['rows_raw']} raw, "
            f"{rep['mapped']} mapped {rep['mapped_by_label']}, contamination dropped {contam}, "
            f"{dups} duplicates, used {len(rows)} {rep['used_by_label']}")
        all_rows.extend(rows)
    all_rows, cross_dups = dedup_rows(all_rows)
    all_rows = cap_rows(all_rows, stage1.get("max_pairs"), seed)
    if reports:
        reports[-1]["cross_dataset_duplicates_dropped"] = cross_dups
    return all_rows, reports


# %% [markdown]
# ## 4. Torch helpers

# %%
def encode(tokenizer, rows, max_length: int) -> list[dict]:
    """[CLS] claim [SEP] premise [SEP], premise truncated, with segment ids — the pair
    encoding app.verify.nli.NLIVerifier builds at inference."""
    enc = tokenizer([r["claim"] for r in rows], [r["premise"] for r in rows],
                    truncation="only_second", max_length=max_length)
    keys = [k for k in ("input_ids", "token_type_ids") if k in enc]
    return [{k: list(enc[k][i]) for k in keys} for i in range(len(rows))]


def collate(feats, pad_id: int, torch, device):
    """Right-pad to the longest in the batch (what tokenizer.pad does for BERT)."""
    n = max(len(f["input_ids"]) for f in feats)
    out = {"input_ids": [f["input_ids"] + [pad_id] * (n - len(f["input_ids"])) for f in feats],
           "attention_mask": [[1] * len(f["input_ids"]) + [0] * (n - len(f["input_ids"])) for f in feats]}
    if "token_type_ids" in feats[0]:
        out["token_type_ids"] = [f["token_type_ids"] + [0] * (n - len(f["token_type_ids"])) for f in feats]
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in out.items()}


def grad_scaler(torch, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def pick_device(torch, allow_cpu: bool = False):
    if not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise SystemExit("No GPU. Settings > Accelerator > GPU T4 x2, then run again.")
    try:
        (torch.ones(8, device="cuda") * 2).sum().item()  # "no kernel image" shows up here
    except RuntimeError as e:
        raise SystemExit(f"This GPU does not run Kaggle's PyTorch build ({e}). "
                         "Switch Accelerator to GPU T4 x2.") from None
    return torch.device("cuda:0")


def evaluate(model, feats, gold, pad_id, torch, device, use_amp: bool, batch: int = 32) -> dict:
    """class_scores + the confusion matrix on a held-out set."""
    model.eval()
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    pred = [None] * len(feats)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for b in range(0, len(order), batch):
            idx = order[b : b + batch]
            logits = model(**collate([feats[i] for i in idx], pad_id, torch, device)).logits
            for i, k in zip(idx, logits.float().argmax(-1).tolist(), strict=True):
                pred[i] = LABELS[k]
    model.train()
    gold = list(gold)
    return {**class_scores(gold, pred), "confusion": confusion_matrix(gold, pred)}


def save_model(model, tokenizer, out: Path, meta: dict) -> None:
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        model.save_pretrained(tmp, safe_serialization=True)
    except TypeError:  # transformers versions without the flag save safetensors anyway
        model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)


# %% [markdown]
# ## 5. Train (stage 1 on external data, then stage 2 on SciFact train)

# %%
class Stage:
    """One training stage: its data, optimiser, warmup + decay schedule and weighted loss."""

    def __init__(self, name, model, feats, labels, cfg, lr, epochs, weights, seed, torch, device, use_amp, pad_id):
        self.name, self.model, self.feats, self.cfg = name, model, feats, cfg
        self.torch, self.device, self.use_amp, self.pad_id, self.seed = torch, device, use_amp, pad_id, seed
        self.y = torch.tensor([LABEL_IDS[lab] for lab in labels], device=device)
        self.lengths = [len(f["input_ids"]) for f in feats]
        self.chunk = cfg["micro_batch"] * cfg["accum"] * 4
        n_micro = math.ceil(len(feats) / cfg["micro_batch"])
        self.total_steps = math.ceil(n_micro / cfg["accum"]) * epochs
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = [p for n, p in named if n.endswith("bias") or "LayerNorm" in n]
        decay = [p for n, p in named if not (n.endswith("bias") or "LayerNorm" in n)]
        self.params = [p for _, p in named]
        self.optim = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                        {"params": no_decay, "weight_decay": 0.0}], lr=lr)
        warmup = int(cfg["warmup_frac"] * self.total_steps)
        sched = cfg["schedule"]
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lambda s: lr_lambda(s, warmup, self.total_steps, sched))
        self.scaler = grad_scaler(torch, use_amp)
        w = torch.tensor([weights[lab] for lab in LABELS], dtype=torch.float32, device=device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    def run_epoch(self, epoch: int) -> float:
        torch, cfg = self.torch, self.cfg
        batches = epoch_batches(self.lengths, cfg["micro_batch"], self.chunk, self.seed, epoch)
        running, seen = 0.0, 0
        for m, idx in enumerate(batches):
            enc = collate([self.feats[i] for i in idx], self.pad_id, torch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(**enc).logits
            loss = self.loss_fn(logits.float(), self.y[idx])
            self.scaler.scale(loss / cfg["accum"]).backward()
            running += loss.item() * len(idx)
            seen += len(idx)
            if (m + 1) % cfg["accum"] == 0 or m == len(batches) - 1:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)
            if (m + 1) % 100 == 0:
                log(f"{self.name} epoch {epoch + 1} micro {m + 1}/{len(batches)} loss {running / seen:.4f} "
                    f"lr {self.sched.get_last_lr()[0]:.2e}")
        return running / max(seen, 1)


def epoch_entry(stage: str, epoch: int, loss: float, lr: float, vs: dict, seconds: float, candidate: bool) -> dict:
    return {"stage": stage, "epoch": epoch, "train_loss": round(loss, 5), "lr_end": lr,
            "val_accuracy": vs["accuracy"], "val_macro_f1": vs["macro_f1"], "val_per_label": vs["per_label"],
            "val_confusion": vs["confusion"], "epoch_s": round(seconds, 1), "candidate": candidate}


def log_epoch(e: dict) -> None:
    log(f"{e['stage']} epoch {e['epoch']}: SciFact val macro-F1 {e['val_macro_f1']} acc {e['val_accuracy']} | "
        + " ".join(f"{lab} F1 {e['val_per_label'][lab]['f1']}" for lab in LABELS)
        + f" | train loss {e['train_loss']} ({e['epoch_s']:.0f} s)")
    print(format_confusion(e["val_confusion"]), flush=True)


def train(data_dir: Path, out: Path = OUT_DIR, base_model: str | None = None, allow_cpu: bool = False,
          expected: dict | None = None, config: dict | None = None, external: list | None = None,
          metrics_copy: Path | None = METRICS_PATH) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t_start = time.time()
    meta, rows = load_data(data_dir, EXPECTED_SHA256 if expected is None else expected)
    cfg = make_config(meta, config)
    gpu_cfg, nb_cfg = notebook_keys(meta, cfg)
    specs = EXTERNAL_DATASETS if external is None else external
    base = base_model or cfg["base_model"]
    assert class_weights(rows["train"]) == meta["class_weights"], "class weights differ from the export"
    log(f"data {data_dir} (export v{meta.get('version', 1)}): train {len(rows['train'])} pairs, "
        f"val {len(rows['val'])} pairs; leakage check passed at export ({meta['git_sha']})")
    log(f"config {json.dumps(cfg)}")

    # Stage-1 data first: a download or contamination problem stops the run before the GPU work.
    s1 = cfg["stage1"]
    ext_rows, ext_reports = [], []
    if s1["enabled"] and s1["epochs"] and specs:
        ext_rows, ext_reports = prepare_external(specs, load_hashes(data_dir, meta), s1, cfg["seed"])
        log(f"stage 1: {len(ext_rows)} external pairs {label_counts(ext_rows)} from {len(specs)} datasets")
    elif s1["enabled"] and s1["epochs"]:
        log("stage 1 enabled but EXTERNAL_DATASETS is empty: training on SciFact only")
    use_stage1 = bool(ext_rows)

    s2_rows = oversample(rows["train"], cfg["oversample"], cfg["seed"])
    counts = {"stage2_raw": label_counts(rows["train"]), "stage2_effective": label_counts(s2_rows),
              "val": label_counts(rows["val"])}
    weights = {"stage2": compute_class_weights(counts["stage2_effective"], cfg["class_weighting"])}
    if use_stage1:
        counts["stage1"] = label_counts(ext_rows)
        weights["stage1"] = compute_class_weights(counts["stage1"], s1["class_weighting"])
    log(f"class counts {json.dumps(counts)}")
    log(f"class weights {json.dumps(weights)}")

    device = pick_device(torch, allow_cpu)
    use_amp = device.type == "cuda"
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log(f"device {gpu}, torch {torch.__version__}, transformers {transformers.__version__}, amp {use_amp}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label={i: lab for lab, i in LABEL_IDS.items()}, label2id=dict(LABEL_IDS))
    model = model.float().to(device)  # fp32 master weights; autocast does the fp16
    enc_base = getattr(model, model.base_model_prefix)
    if cfg["freeze_embeddings"] or cfg["freeze_layers"]:
        for p in enc_base.embeddings.parameters():
            p.requires_grad = False
    for layer in enc_base.encoder.layer[: cfg["freeze_layers"]]:
        for p in layer.parameters():
            p.requires_grad = False
    pad_id = tok.pad_token_id or 0
    val_f = encode(tok, rows["val"], cfg["max_length"])
    val_gold = [r["label"] for r in rows["val"]]
    common = dict(model=model, cfg=cfg, torch=torch, device=device, use_amp=use_amp, pad_id=pad_id)

    history = []
    stage1_info = {"enabled": bool(s1["enabled"]), "used": use_stage1, "datasets": ext_reports,
                   "pairs": len(ext_rows), "by_label": label_counts(ext_rows)}
    if use_stage1:
        t1 = time.time()
        feats = encode(tok, ext_rows, cfg["max_length"])
        st = Stage("stage1", feats=feats, labels=[r["label"] for r in ext_rows], lr=s1["lr"], epochs=s1["epochs"],
                   weights=weights["stage1"], seed=cfg["seed"] + 1, **common)
        for epoch in range(s1["epochs"]):
            t_ep = time.time()
            loss = st.run_epoch(epoch)
            vs = evaluate(model, val_f, val_gold, pad_id, torch, device, use_amp)
            e = epoch_entry("stage1", epoch + 1, loss, st.sched.get_last_lr()[0], vs, time.time() - t_ep, False)
            history.append(e)
            log_epoch(e)
        stage1_info["minutes"] = round((time.time() - t1) / 60, 2)
        log(f"stage 1 done in {stage1_info['minutes']} min (SciFact val shown for information; not a candidate)")
        del st, feats

    train_f = encode(tok, s2_rows, cfg["max_length"])
    lengths = [len(f["input_ids"]) for f in train_f]
    log(f"stage 2 tokens/pair: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}, "
        f"truncated {sum(n >= cfg['max_length'] for n in lengths)}/{len(lengths)}")
    st = Stage("stage2", feats=train_f, labels=[r["label"] for r in s2_rows], lr=cfg["lr"], epochs=cfg["epochs"],
               weights=weights["stage2"], seed=cfg["seed"], **common)
    stopper = EarlyStopping(cfg["patience"], cfg["min_delta"])
    best, reason = None, "max_epochs"
    for epoch in range(cfg["epochs"]):
        t_ep = time.time()
        loss = st.run_epoch(epoch)
        vs = evaluate(model, val_f, val_gold, pad_id, torch, device, use_amp)
        improved, stop = stopper.step(vs["macro_f1"], epoch + 1)
        e = epoch_entry("stage2", epoch + 1, loss, st.sched.get_last_lr()[0], vs, time.time() - t_ep, True)
        e["improved"] = improved
        history.append(e)
        log_epoch(e)
        if improved:
            best = {"stage": "stage2", "epoch": epoch + 1, "val_macro_f1": vs["macro_f1"],
                    "val_accuracy": vs["accuracy"], "val_per_label_f1": {lab: vs["per_label"][lab]["f1"] for lab in LABELS},
                    "val_confusion": vs["confusion"]}
            save_model(model, tok, out, {
                "base_model": base, "max_length": cfg["max_length"], "pair_order": "claim_first",
                "labels": list(LABELS), "config": meta["train_config"], "gpu_config": gpu_cfg,
                "notebook_config": nb_cfg, "notebook_version": 2,
                "best": best, "history": history, "data": meta["report"],
                "val_qids": meta["splits"]["val_qids"],
                "data_export": {"sha256": {n: i["sha256"] for n, i in {**meta["files"], **(meta.get("aux_files") or {})}.items()},
                                "git_sha": meta["git_sha"], "version": meta.get("version", 1)},
                "stage1": {k: v for k, v in stage1_info.items() if k != "datasets"}
                | {"datasets": [{k: d[k] for k in ("name", "repo_id", "revision", "files", "license", "used")}
                                for d in ext_reports]},
                "class_counts": counts, "class_weights": weights,
                "trained_on": {"device": gpu, "torch": torch.__version__,
                               "transformers": transformers.__version__, "amp_fp16": use_amp},
            })
            log(f"  new best (stage 2 epoch {epoch + 1}) -> {out}")
        if stop:
            reason = f"no val macro-F1 gain > {cfg['min_delta']} for {cfg['patience']} epochs"
            log(f"early stop after epoch {epoch + 1}: {reason}; best epoch {stopper.best_epoch}")
            break

    minutes = round((time.time() - t_start) / 60, 1)
    m = json.loads((out / "meta.json").read_text())
    m["final_history"] = history
    m["train_minutes"] = minutes
    (out / "meta.json").write_text(json.dumps(m, indent=2))
    metrics = {
        "notebook_version": 2, "best": best, "history": history, "train_minutes": minutes, "device": gpu,
        "early_stopping": {"patience": cfg["patience"], "min_delta": cfg["min_delta"],
                           "epochs_run": sum(h["stage"] == "stage2" for h in history),
                           "max_epochs": cfg["epochs"], "stopped_epoch": stopper.stopped_epoch, "reason": reason},
        "stage1": stage1_info, "class_counts": counts, "class_weights": weights,
        "config": {"train_config": meta["train_config"], "gpu_config": gpu_cfg, "notebook_config": nb_cfg},
        "data_export": m["data_export"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if metrics_copy is not None:
        Path(metrics_copy).write_text(json.dumps(metrics, indent=2))
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2))
    id2label = json.loads((out / "config.json").read_text())["id2label"]
    assert sorted(id2label.values()) == sorted(LABELS), f"saved head labels {id2label}"
    per = " ".join(f"{lab} {best['val_per_label_f1'][lab]}" for lab in LABELS)
    log(f"done in {minutes} min; best stage-2 epoch {best['epoch']} val macro-F1 {best['val_macro_f1']} ({per})")
    return metrics


def zip_model(out: Path = OUT_DIR, zip_path: Path = ZIP_PATH) -> Path:
    """Files at the archive root: `unzip verifier_model.zip -d data/models/verifier/model`."""
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file():
                z.write(f, arcname=f.name)
    log(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB): download it from the Output tab")
    return zip_path


# %% [markdown]
# ## 6. Run

# %%
if __name__ == "__main__":
    train(find_data_dir())
    zip_model()
