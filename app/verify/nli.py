"""Claim verification with a dedicated NLI cross-encoder — no LLM, no network.

An MNLI/FEVER/ANLI-trained DeBERTa-v3-base scores each (abstract, claim) pair as
entailment / neutral / contradiction, which map to SciFact's SUPPORT / NEI / CONTRADICT.
The claim-level verdict aggregates the per-passage probabilities over the top-k
retrieved passages (`aggregate_claim`).

Conventions, checked rather than assumed:

* **Input order** — premise (evidence) first, hypothesis (claim) second: the MNLI
  convention this model was trained on ([CLS] premise [SEP] claim [SEP]).
* **Label order** — read from ``model.config.id2label`` (`label_order`); this model's is
  0 entailment, 1 neutral, 2 contradiction, but another checkpoint may differ.
* **Premise** — ``"<title>. <abstract>"``, the same joined form as ``hit_passage``.
* **Long abstracts** — 305 of 5,183 SciFact abstracts exceed 512 DeBERTa tokens. A pair
  that fits is scored whole (window tag ``full``), so the two strategies only differ on
  the pairs that do not fit: ``truncate`` cuts the premise at the budget, ``window``
  scores overlapping sentence windows (title + consecutive sentences, packed to the
  budget, `OVERLAP_SENTENCES` shared between neighbours) and keeps the window whose
  strongest non-neutral probability is highest (max over windows).

The frozen claim-level settings (`FROZEN`) were chosen on beir/scifact/train ONLY: the
300 "test" claims are SciFact's public dev set, so they are touched once, at the end, with
these values fixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.config import settings

LABELS = ("SUPPORT", "CONTRADICT", "NEI")
NON_NEI = ("SUPPORT", "CONTRADICT")
# NLI label name (lower-cased, as in id2label) -> SciFact claim label.
NLI_TO_LABEL = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "NEI"}
WINDOWINGS = ("truncate", "window")
RULES = ("max", "mean")
MAX_LENGTH = 512  # DeBERTa-v3's position budget
OVERLAP_SENTENCES = 1  # sentences shared by neighbouring windows
# Safety margin on the per-window token budget: sentences are counted one at a time, and
# joining them can shift a SentencePiece boundary by a token. The tokenizer still
# truncates `only_first` as a backstop, so the margin only keeps that from biting.
WINDOW_MARGIN = 4

# The claim-level decision, frozen after tuning on beir/scifact/train (all 809 claims;
# `SSR_RAG_DATASET=beir/scifact/train SSR_RAG_N=all make verify-eval ARGS=--tune`, 2026-09-29).
# The evaluator refuses to score a non-train split without these (or an explicit,
# recorded override). Train evidence (3-class accuracy, each at its own best τ):
# * aggregation: max@1 0.707 > max@2 0.687 > max@3 0.669 > max@5 0.650 > mean@3/5 <= 0.648
#   (windowed). Looking past the top-ranked passage only adds spurious entail/contradict
#   calls on topically related abstracts that are not the evidence.
# * windowing: sentence windows beat truncation under every aggregation (max@1 0.707 vs
#   0.685; paired exact McNemar b=7 c=25, p=0.002).
# * τ: the max@1 curve is flat from 0.25 to 0.65 (0.693-0.707); the peak is 0.44.
# * int8: dynamic int8 collapsed the model (accuracy 0.72 -> 0.37 on 300 train claims,
#   almost everything NEI; max |Δp| 0.86) and was not faster, so it stays off.
FROZEN: dict = {
    "windowing": "window",
    "rule": "max",
    "k": 1,
    "tau": 0.44,
    "int8": False,
    "tuned_on": "beir/scifact/train (all 809 claims, seed 13)",
}


class LabelMappingError(ValueError):
    """The checkpoint's id2label is not the 3-way entailment/neutral/contradiction set."""


def label_order(id2label: Mapping) -> list[str]:
    """Our label for each logit index, from the model's own id2label.

    Keys may be ints or strings ("0"), names any case. Refuses anything that is not
    exactly the three NLI classes, so a 2-way (entailment / not_entailment) or relabelled
    head can never be silently read as SUPPORT/CONTRADICT/NEI.
    """
    try:
        items = sorted((int(k), str(v).strip().lower()) for k, v in id2label.items())
    except (TypeError, ValueError) as e:
        raise LabelMappingError(f"unreadable id2label {dict(id2label)!r}") from e
    if [i for i, _ in items] != list(range(len(items))):
        raise LabelMappingError(f"id2label indices are not 0..n-1: {dict(id2label)!r}")
    names = [n for _, n in items]
    if sorted(names) != sorted(NLI_TO_LABEL):
        raise LabelMappingError(
            f"expected exactly {sorted(NLI_TO_LABEL)} in id2label, got {names}"
        )
    return [NLI_TO_LABEL[n] for n in names]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    """Abstract -> sentences. SciFact abstracts are sentences joined by single spaces
    (with section headers like "BACKGROUND" run into the next sentence), so a split on
    terminal punctuation followed by an upper-case/digit/bracket start is enough."""
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text.strip())) if s]


def window_spans(lengths: Sequence[int], budget: int, overlap: int = OVERLAP_SENTENCES) -> list[tuple[int, int]]:
    """[(start, end)) sentence spans whose summed token `lengths` fit `budget`.

    Greedy packing; each next window starts `overlap` sentences before the previous one
    ended, but always at least one sentence later, so it terminates. A single sentence
    longer than the budget gets a window of its own (the tokenizer truncates it).
    Every sentence is covered by at least one window.
    """
    n = len(lengths)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end, used = start, 0
        while end < n and (end == start or used + lengths[end] <= budget):
            used += lengths[end]
            end += 1
        spans.append((start, end))
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return spans


@dataclass(frozen=True)
class PairScore:
    """One (passage, claim) pair: label probabilities in our label space, how many
    windows were scored, and which one was kept (0 when the pair fit whole)."""

    probs: dict[str, float]
    n_windows: int = 1
    window: int = 0

    def strength(self) -> float:
        return max(self.probs["SUPPORT"], self.probs["CONTRADICT"])


@runtime_checkable
class Verifier(Protocol):
    """Scores (title, abstract, claim) triples. `window_tag` names how a triple will be
    fed to the model (``full`` when it fits), which is part of its cache key."""

    model_name: str
    dtype: str

    def window_tag(self, title: str, abstract: str, claim: str) -> str: ...

    def score(self, items: Sequence[tuple[str, str, str]]) -> list[PairScore]: ...


def premise_text(title: str, abstract: str) -> str:
    title, abstract = title.strip(), abstract.strip()
    return f"{title}. {abstract}".strip() if title else abstract


class NLIVerifier:
    """HF sequence-classification NLI model behind the `Verifier` protocol.

    `tokenizer` / `model` may be injected (tests); otherwise both load from the HF cache
    by `model_name`. Nothing here downloads unless the model is missing locally.
    """

    def __init__(
        self,
        model_name: str | None = None,
        windowing: str = FROZEN["windowing"],
        int8: bool | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
        max_length: int = MAX_LENGTH,
        tokenizer=None,
        model=None,
    ):
        if windowing not in WINDOWINGS:
            raise ValueError(f"windowing must be one of {WINDOWINGS}, got {windowing!r}")
        import torch

        self.torch = torch
        self.model_name = model_name or settings.nli_model
        self.windowing = windowing
        self.int8 = settings.nli_int8 if int8 is None else int8
        self.batch_size = batch_size or settings.nli_batch_size
        self.max_length = max_length
        threads = settings.nli_threads if threads is None else threads
        if threads > 0:
            torch.set_num_threads(threads)
        if tokenizer is None or model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.model_name)
            # dtype pinned to fp32: transformers 5 defaults to the checkpoint's stored
            # dtype, and this one is saved as float16 — which CPU matmul runs ~10x slower
            # (7.0 s vs 0.70 s for one 306-token pair on the reference laptop).
            model = model or AutoModelForSequenceClassification.from_pretrained(
                self.model_name, dtype=torch.float32
            )
        model.eval()
        if self.int8:
            from torch.ao.quantization import quantize_dynamic

            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.tokenizer = tokenizer
        self.model = model
        self.labels = label_order(model.config.id2label)
        self._specials = tokenizer.num_special_tokens_to_add(pair=True)
        # Throughput accounting: model forwards only (tokenisation included, cache
        # lookups excluded), so ms/pair is what a cold pair costs.
        self.forward_seconds = 0.0
        self.windows_scored = 0
        self.pairs_scored = 0

    @property
    def dtype(self) -> str:
        return "int8-dynamic" if self.int8 else "fp32"

    # --- inputs ---------------------------------------------------------------------

    def _ntok(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _fits(self, premise: str, claim: str) -> bool:
        return self._ntok(premise) + self._ntok(claim) + self._specials <= self.max_length

    def window_tag(self, title: str, abstract: str, claim: str) -> str:
        if self._fits(premise_text(title, abstract), claim):
            return "full"
        if self.windowing == "truncate":
            return f"truncate{self.max_length}"
        return f"window{self.max_length}o{OVERLAP_SENTENCES}m{WINDOW_MARGIN}"

    def premises(self, title: str, abstract: str, claim: str) -> list[str]:
        """The premise text(s) the model sees for one pair."""
        whole = premise_text(title, abstract)
        if self.windowing == "truncate" or self._fits(whole, claim):
            return [whole]
        sentences = split_sentences(abstract)
        head = f"{title.strip()}. " if title.strip() else ""
        budget = (
            self.max_length - self._specials - self._ntok(claim) - self._ntok(head) - WINDOW_MARGIN
        )
        spans = window_spans([self._ntok(s) for s in sentences], max(budget, 1))
        return [(head + " ".join(sentences[a:b])).strip() for a, b in spans]

    # --- scoring --------------------------------------------------------------------

    def _forward(self, premises: Sequence[str], claims: Sequence[str]) -> list[list[float]]:
        enc = self.tokenizer(
            list(premises),
            list(claims),
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        return logits.float().softmax(dim=-1).tolist()

    def score(self, items: Sequence[tuple[str, str, str]]) -> list[PairScore]:
        """PairScore per (title, abstract, claim), batched across all their windows.

        Windows are sorted by length before batching, so a batch pads to similar lengths.
        """
        t0 = time.perf_counter()
        flat: list[tuple[int, str, str]] = []  # (item index, premise, claim)
        for i, (title, abstract, claim) in enumerate(items):
            flat.extend((i, p, claim) for p in self.premises(title, abstract, claim))
        order = sorted(range(len(flat)), key=lambda j: len(flat[j][1]) + len(flat[j][2]))
        probs: list[list[float] | None] = [None] * len(flat)
        for b in range(0, len(order), self.batch_size):
            idx = order[b : b + self.batch_size]
            out = self._forward([flat[j][1] for j in idx], [flat[j][2] for j in idx])
            for j, p in zip(idx, out, strict=True):
                probs[j] = p
        per_item: list[list[dict[str, float]]] = [[] for _ in items]
        for (i, _, _), p in zip(flat, probs, strict=True):
            per_item[i].append({lab: float(v) for lab, v in zip(self.labels, p, strict=True)})
        results = []
        for windows in per_item:
            best = max(range(len(windows)), key=lambda w: (_strength(windows[w]), -w))
            results.append(PairScore(windows[best], n_windows=len(windows), window=best))
        self.forward_seconds += time.perf_counter() - t0
        self.windows_scored += len(flat)
        self.pairs_scored += len(items)
        return results


def _strength(p: Mapping[str, float]) -> float:
    return max(p["SUPPORT"], p["CONTRADICT"])


# --- claim-level aggregation ------------------------------------------------------------


@dataclass(frozen=True)
class ClaimDecision:
    label: str  # SUPPORT | CONTRADICT | NEI
    confidence: float  # the non-NEI probability the decision rests on
    candidate: str  # the non-NEI label that was considered (SUPPORT or CONTRADICT)
    index: int | None  # passage index (0-based, in retrieval order) behind `candidate`


def aggregate_claim(
    passage_probs: Sequence[Mapping[str, float]], tau: float, rule: str = "max", k: int = 5
) -> ClaimDecision:
    """Claim verdict from the first `k` passages' label probabilities.

    ``max``: the (passage, non-NEI label) with the highest probability — so a conflict
    (one passage supports, another contradicts) resolves to the more confident one.
    ``mean``: average each label over the k passages, take the higher non-NEI mean; its
    passage is the one with the highest probability for that label.
    Either way the label is returned only if its confidence >= `tau`, else NEI. Exact
    ties go to the earlier (higher-ranked) passage, then SUPPORT over CONTRADICT, so the
    decision is deterministic. No passages at all is NEI.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if k < 1:
        raise ValueError("k must be >= 1")
    ps = list(passage_probs)[:k]
    if not ps:
        return ClaimDecision("NEI", 0.0, "SUPPORT", None)
    if rule == "max":
        conf, neg_i, neg_lab = max(
            (p[lab], -i, -NON_NEI.index(lab)) for i, p in enumerate(ps) for lab in NON_NEI
        )
        cand, index = NON_NEI[-neg_lab], -neg_i
    else:
        means = {lab: sum(p[lab] for p in ps) / len(ps) for lab in NON_NEI}
        cand = max(NON_NEI, key=lambda lab: (means[lab], -NON_NEI.index(lab)))
        conf = means[cand]
        index = max(range(len(ps)), key=lambda i: (ps[i][cand], -i))
    return ClaimDecision(cand if conf >= tau else "NEI", float(conf), cand, index)


# --- per-pair probability cache ---------------------------------------------------------

CACHE_DIR = Path("data/eval_cache/nli")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def pair_key(
    model: str, dtype: str, window_tag: str, max_length: int, doc_id: str, premise: str, claim: str
) -> str:
    """Cache key of one pair. The premise text hash rides along with the doc id so an
    edited corpus can never serve a stale score under the same id."""
    payload = json.dumps(
        [model, dtype, window_tag, max_length, doc_id, _sha(premise), _sha(claim)]
    )
    return _sha(payload)[:32]


@dataclass
class PairCache:
    """Append-only JSONL of {key: PairScore}. Each scored chunk is appended and flushed
    at once, so a killed run resumes where it stopped; a torn last line (a kill
    mid-write) or any unreadable line is skipped, never fatal."""

    path: Path
    entries: dict[str, PairScore] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists() and not os.environ.get("SSR_EVAL_REFRESH"):
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    probs = {lab: float(rec["probs"][lab]) for lab in LABELS}
                    self.entries[rec["key"]] = PairScore(
                        probs, int(rec.get("n_windows", 1)), int(rec.get("window", 0))
                    )
                except (ValueError, KeyError, TypeError):
                    continue

    def get(self, key: str) -> PairScore | None:
        return self.entries.get(key)

    def put_many(self, items: Iterable[tuple[str, PairScore]]) -> None:
        items = list(items)
        if not items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            for key, s in items:
                self.entries[key] = s
                fh.write(
                    json.dumps(
                        {"key": key, "probs": s.probs, "n_windows": s.n_windows, "window": s.window}
                    )
                    + "\n"
                )
            fh.flush()
            os.fsync(fh.fileno())


def cache_path_for(model: str, dtype: str) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in model).strip("-")
    return CACHE_DIR / f"{slug}_{dtype}.jsonl"


def score_pairs_cached(
    verifier: Verifier,
    cache: PairCache,
    pairs: Sequence[tuple[str, str, str, str]],
    chunk: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> list[PairScore]:
    """PairScore per (doc_id, title, abstract, claim): cache hits are free; misses are
    scored `chunk` at a time and appended to the cache after every chunk."""
    keys = []
    for doc_id, title, abstract, claim in pairs:
        tag = verifier.window_tag(title, abstract, claim)
        keys.append(
            pair_key(
                verifier.model_name, verifier.dtype, tag,
                getattr(verifier, "max_length", MAX_LENGTH), doc_id,
                premise_text(title, abstract), claim,
            )
        )
    out: list[PairScore | None] = [cache.get(k) for k in keys]
    todo = [i for i, s in enumerate(out) if s is None]
    # Deduplicate: the same (doc, claim) pair twice in one call is scored once.
    first: dict[str, int] = {}
    for i in todo:
        first.setdefault(keys[i], i)
    unique = list(first.values())
    cache.hits += len(pairs) - len(todo)
    cache.misses += len(unique)
    for c in range(0, len(unique), chunk):
        idx = unique[c : c + chunk]
        scored = verifier.score([pairs[i][1:] for i in idx])
        cache.put_many((keys[i], s) for i, s in zip(idx, scored, strict=True))
        if progress:
            progress(min(c + chunk, len(unique)), len(unique))
    return [cache.get(k) if s is None else s for k, s in zip(keys, out, strict=True)]
