"""The fine-tuned evidence verifier (PubMedBERT + 3-way head) behind the Verifier protocol.

Trained on beir/scifact/train only (see app.verify.train_data for the held-out tuning
set and the leakage exclusions) — on a free Kaggle GPU by the notebook that
app.verify.kaggle_export writes (data/kaggle/README.md), or on CPU by app.verify.train.

**Where it loads from:** data/models/verifier/model/ — a plain HF save_pretrained
directory (config.json, model.safetensors, tokenizer files, meta.json). Unzip the Kaggle
output there:

    unzip verifier_model.zip -d data/models/verifier/model

A zip that unpacks into one subfolder (model/verifier_model/...) is found too; anything
else (no weights, several candidate folders) is refused with a message saying so.

**Safetensors only, local only, no remote code.** The weights must be
``model.safetensors``: a ``pytorch_model.bin`` is a pickle, and unpickling a file can
execute code, so a directory with only that file is refused (re-save it with
``save_pretrained(..., safe_serialization=True)``, which train.py and the Kaggle notebook
already do). A ``.bin`` next to the safetensors file is ignored — never loaded, never
hashed. Model and tokenizer load with ``trust_remote_code=False`` and
``local_files_only=True`` (the model also with ``use_safetensors=True``), so a
checkpoint's config can neither pull code from the Hub nor trigger a download.

Differences from the zero-shot NLIVerifier, all read from the checkpoint rather than
assumed:

* **Pair order** — claim first, abstract second ([CLS] claim [SEP] title. abstract [SEP]),
  the order it was trained on; the abstract is what gets truncated or windowed.
* **Labels** — the head's id2label names SciFact labels directly (SUPPORT / CONTRADICT /
  NEI), checked by `trained_label_order`.
* **Cache identity** — `model_name` carries the checkpoint's content hash
  (``ft:<base>@<sha12>``), so the per-pair cache (app.verify.nli.PairCache) can never
  serve one checkpoint's probabilities for another's, even at the same path.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from app.verify.nli import LABELS, LabelMappingError, NLIVerifier

CHECKPOINT_DIR = Path("data/models/verifier/model")
WEIGHT_FILES = ("model.safetensors",)  # the only weights accepted (and hashed)
PICKLE_WEIGHT_FILES = ("pytorch_model.bin",)  # recognised only to refuse them by name
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")
HASHED_FILES = ("config.json", *WEIGHT_FILES, *TOKENIZER_FILES)


def is_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and any((path / w).is_file() for w in WEIGHT_FILES)


def _pickle_only(path: Path) -> bool:
    """config.json + pickle weights but no model.safetensors: a checkpoint we refuse."""
    return (path / "config.json").is_file() and not is_checkpoint(path) and any(
        (path / w).is_file() for w in PICKLE_WEIGHT_FILES)


def resolve_checkpoint(path: Path | str = CHECKPOINT_DIR) -> Path:
    """The checkpoint directory: `path` itself, or its single checkpoint subfolder. A
    folder whose only weights are pytorch_model.bin is refused by name (FileNotFoundError,
    like every other "no usable checkpoint" case, but saying why)."""
    path = Path(path)
    if is_checkpoint(path):
        return path
    dirs = [d for d in sorted(path.iterdir()) if d.is_dir()] if path.is_dir() else []
    subs = [d for d in dirs if is_checkpoint(d)]
    if len(subs) == 1:
        return subs[0]
    if not subs and (legacy := [d for d in (path, *dirs) if _pickle_only(d)]):
        raise FileNotFoundError(
            f"{legacy[0]} has only {' / '.join(PICKLE_WEIGHT_FILES)} (pickle weights, which can "
            f"execute code when loaded); only {' / '.join(WEIGHT_FILES)} is accepted. Re-save the "
            "checkpoint with save_pretrained(..., safe_serialization=True)."
        )
    what = f"{len(subs)} checkpoint folders" if subs else "no checkpoint (config.json + weights)"
    raise FileNotFoundError(
        f"{what} in {path}: unzip the Kaggle verifier_model.zip there "
        f"(unzip verifier_model.zip -d {CHECKPOINT_DIR}; see data/kaggle/README.md)"
    )


def trained_label_order(id2label: Mapping) -> list[str]:
    """Our label per logit index; refuses anything but exactly SUPPORT/CONTRADICT/NEI."""
    try:
        items = sorted((int(k), str(v).strip().upper()) for k, v in id2label.items())
    except (TypeError, ValueError) as e:
        raise LabelMappingError(f"unreadable id2label {dict(id2label)!r}") from e
    if [i for i, _ in items] != list(range(len(items))) or sorted(n for _, n in items) != sorted(LABELS):
        raise LabelMappingError(f"expected exactly {list(LABELS)} in id2label, got {dict(id2label)!r}")
    return [n for _, n in items]


def checkpoint_sha(path: Path | str) -> str:
    """sha256 over the checkpoint's config, weight and tokenizer files (name + bytes, fixed
    order). Anything that changes what the model computes changes the hash."""
    path = Path(path)
    h = hashlib.sha256()
    found = False
    for name in HASHED_FILES:
        f = path / name
        if not f.exists():
            continue
        found = found or name in WEIGHT_FILES
        h.update(name.encode() + b"\0")
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    if not found:
        raise FileNotFoundError(f"no weights ({' / '.join(WEIGHT_FILES)}; "
                                f"{' / '.join(PICKLE_WEIGHT_FILES)} is not accepted) in {path}")
    return h.hexdigest()


def read_meta(path: Path | str) -> dict:
    try:
        return json.loads((Path(path) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def model_id(base_model: str, sha: str) -> str:
    return f"ft:{base_model}@{sha[:12]}"


class TrainedVerifier(NLIVerifier):
    """A fine-tuned checkpoint as a Verifier. `tokenizer` / `model` / `checkpoint_hash`
    may be injected (tests); otherwise everything loads from `checkpoint`, offline:
    model.safetensors only, local files only, no remote code (see the module docstring)."""

    claim_first = True

    def __init__(
        self,
        checkpoint: Path | str = CHECKPOINT_DIR,
        windowing: str = "truncate",
        threads: int | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        tokenizer=None,
        model=None,
        checkpoint_hash: str | None = None,
        base_model: str | None = None,
    ):
        checkpoint = Path(checkpoint)
        if model is None or checkpoint_hash is None:
            checkpoint = resolve_checkpoint(checkpoint)
            meta = read_meta(checkpoint)
        else:
            meta = {}
        if meta.get("pair_order", "claim_first") != "claim_first":
            raise LabelMappingError(f"{checkpoint} was trained on pair order {meta['pair_order']!r}")
        if tokenizer is None or model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                checkpoint, trust_remote_code=False, local_files_only=True
            )
            model = model or AutoModelForSequenceClassification.from_pretrained(
                checkpoint, dtype=torch.float32, use_safetensors=True,
                trust_remote_code=False, local_files_only=True,
            )
        sha = checkpoint_hash or checkpoint_sha(checkpoint)
        base = base_model or meta.get("base_model") or "unknown"
        super().__init__(
            model_name=model_id(base, sha),
            windowing=windowing,
            int8=False,
            threads=threads,
            batch_size=batch_size,
            max_length=max_length or int(meta.get("max_length", 512)),
            tokenizer=tokenizer,
            model=model,
        )
        self.checkpoint = str(checkpoint)
        self.checkpoint_hash = sha
        self.meta = meta

    def _label_names(self, id2label: Mapping) -> list[str]:
        return trained_label_order(id2label)
