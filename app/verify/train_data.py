"""Training pairs for the fine-tuned evidence verifier — beir/scifact/train ONLY.

Each example is one (claim, abstract) pair with a SciFact label:

* **positives** — every rationale doc of a claim, labelled with the claim's label
  (SUPPORT or CONTRADICT; BEIR SciFact has one label per claim across its docs);
* **NEI, cited** — docs in the claim's qrels that carry no rationale (every NEI claim's
  cited docs, and the few rationale-free cited docs of labelled claims);
* **NEI, hard** — the hybrid retriever's top-5 for the claim (the same top-5 rag_eval
  feeds the LLM) minus its rationale and cited docs, the first `n_hard` in rank order.
  These are the abstracts the verifier must learn to reject at inference time;
* **NEI, random** — a uniformly random abstract for a `random_frac` share of claims.

Contamination and leakage (the reasons this module exists as code, not a notebook):

* the 300 "test" claims are SciFact's public dev set. They are read only to be kept
  out: a train claim whose text is identical to a test claim's is dropped (SciFact has
  two: train 871 = test 870, train 1291 = test 1292, same doc and label), and by default
  (`test_docs`) so is every train claim sharing a rationale or cited doc with a test
  claim, and no test claim's doc is used as a negative — the same rule as for the tuning
  set, so the thresholds tuned on clean tuning claims meet equally clean test claims;
* the train claims Ling already answered (`TUNING_RUN`, the 100-claim train run) are the
  COMBINATION-TUNING set and are never trained on;
* any other train claim that shares a rationale or cited doc with a tuning claim is
  dropped too (SciFact's negated claim pairs share their docs, so claim X in training
  would leak the verdict of not-X in tuning), and no tuning claim's rationale or cited
  doc is used as a negative for a training claim — so the tuning set's evidence abstracts
  are never seen in training at all;
* the validation split (early stopping / model selection) is grouped by cited doc: the
  connected components of "shares a rationale or cited doc" are split whole, so no doc's
  label is learnt in training and then scored in validation.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingest.corpus import ClaimLabel

TRAIN_DATASET = "beir/scifact/train"
TEST_DATASET = "beir/scifact/test"  # read only to exclude it
TUNING_RUN = Path("data/eval_runs/rag_beir-scifact-train_100_d0921f4e/rag.json")
LABELS = ("SUPPORT", "CONTRADICT", "NEI")
SOURCES = ("rationale", "cited_no_rationale", "hard_negative", "random")
SEED = 13


class ContaminationError(RuntimeError):
    """A split other than beir/scifact/train, or a tuning claim among the training data."""


@dataclass(frozen=True)
class Pair:
    qid: str
    doc_id: str
    claim: str
    title: str
    abstract: str
    label: str  # SUPPORT | CONTRADICT | NEI
    source: str  # one of SOURCES


def tuning_ids(path: Path = TUNING_RUN, sample: Collection[str] = ()) -> set[str]:
    """Query ids of the combination-tuning set: every row of the Ling train run, plus
    `sample` (the run's full seeded sample, so a claim the run skipped is held out too)."""
    blob = json.loads(Path(path).read_text())
    run = blob.get("run") or {}
    if run.get("dataset") not in (None, TRAIN_DATASET):
        raise ContaminationError(f"{path} is a {run.get('dataset')} run, not {TRAIN_DATASET}")
    return {str(r["query_id"]) for r in blob["rows"]} | set(sample)


def normalize_claim(text: str) -> str:
    """Claim text for duplicate detection: case- and whitespace-insensitive."""
    return " ".join(text.lower().split())


def cited_docs(qid: str, labels: Mapping[str, ClaimLabel], qrels: Mapping[str, Mapping[str, int]]) -> set[str]:
    """Rationale docs ∪ qrels-relevant docs of one claim."""
    rel = {d for d, r in (qrels.get(qid) or {}).items() if r > 0}
    return set(labels[qid].rationale_doc_ids) | rel


def leakage_exclusions(
    candidates: Iterable[str],
    tuning: Collection[str],
    labels: Mapping[str, ClaimLabel],
    qrels: Mapping[str, Mapping[str, int]],
) -> set[str]:
    """Candidate claims (not themselves tuning claims) that share any rationale or cited
    doc with a tuning claim."""
    held_docs = set().union(*(cited_docs(q, labels, qrels) for q in tuning)) if tuning else set()
    return {q for q in candidates if q not in tuning and cited_docs(q, labels, qrels) & held_docs}


def doc_components(qids: Sequence[str], docs_of: Mapping[str, set[str]]) -> list[list[str]]:
    """Connected components of claims linked by a shared doc (union-find), each sorted,
    in order of their first claim in `qids` — deterministic for a given input."""
    parent = {q: q for q in qids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for q in qids:
        for d in sorted(docs_of.get(q, ())):
            if d in owner:
                a, b = find(q), find(owner[d])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[d] = q
    groups: dict[str, list[str]] = {}
    for q in qids:
        groups.setdefault(find(q), []).append(q)
    return [sorted(g) for g in groups.values()]


def group_split(
    qids: Sequence[str], docs_of: Mapping[str, set[str]], frac: float = 0.20, seed: int = SEED
) -> tuple[list[str], list[str]]:
    """(train, validation) claim ids: whole doc components go to validation, in a seeded
    shuffle, until it holds >= `frac` of the claims."""
    comps = doc_components(sorted(qids), docs_of)
    random.Random(seed).shuffle(comps)
    target = frac * len(qids)
    val: list[str] = []
    train: list[str] = []
    for c in comps:
        (val if len(val) < target else train).extend(c)
    return sorted(train), sorted(val)


def build_pairs(
    qids: Sequence[str],
    queries: Mapping[str, str],
    labels: Mapping[str, ClaimLabel],
    qrels: Mapping[str, Mapping[str, int]],
    top: Mapping[str, Sequence[str]],
    docs: Mapping[str, Mapping[str, str]],
    n_hard: int = 3,
    random_frac: float = 0.25,
    forbidden_docs: Collection[str] = (),
    seed: int = SEED,
) -> list[Pair]:
    """All pairs for `qids` (see the module docstring). `forbidden_docs` are never used
    as a hard or random negative (the tuning set's evidence). Deterministic."""
    rng = random.Random(seed)
    all_docs = sorted(docs)
    forbidden = set(forbidden_docs)
    out: list[Pair] = []

    def pair(q: str, d: str, label: str, source: str) -> Pair:
        doc = docs[d]
        return Pair(q, d, queries[q], doc.get("title", ""), doc.get("text", ""), label, source)

    for q in sorted(qids):
        lab = labels[q]
        rationale = set(lab.rationale_doc_ids)
        cited = cited_docs(q, labels, qrels)
        for d in sorted(rationale):
            out.append(pair(q, d, lab.label, "rationale"))
        for d in sorted(cited - rationale):
            out.append(pair(q, d, "NEI", "cited_no_rationale"))
        hard = [d for d in top.get(q, ()) if d not in cited and d not in forbidden and d in docs]
        for d in hard[:n_hard]:
            out.append(pair(q, d, "NEI", "hard_negative"))
        if rng.random() < random_frac:
            used = cited | set(hard[:n_hard]) | forbidden
            while (d := rng.choice(all_docs)) in used:
                pass
            out.append(pair(q, d, "NEI", "random"))
    return out


def counts(pairs: Iterable[Pair]) -> dict:
    pairs = list(pairs)
    return {
        "pairs": len(pairs),
        "claims": len({p.qid for p in pairs}),
        "by_label": {lab: sum(p.label == lab for p in pairs) for lab in LABELS},
        "by_source": dict(sorted(Counter(p.source for p in pairs).items())),
    }


def class_weights(pairs: Iterable[Pair]) -> dict[str, float]:
    """Inverse-frequency weights, n / (3 * n_label): a balanced loss over the three labels."""
    c = Counter(p.label for p in pairs)
    n = sum(c.values())
    return {lab: round(n / (len(LABELS) * c[lab]), 6) if c[lab] else 0.0 for lab in LABELS}


def build_splits(
    queries: Mapping[str, str],
    labels: Mapping[str, ClaimLabel],
    qrels: Mapping[str, Mapping[str, int]],
    top: Mapping[str, Sequence[str]],
    docs: Mapping[str, Mapping[str, str]],
    tuning: Collection[str],
    dataset: str = TRAIN_DATASET,
    val_frac: float = 0.20,
    n_hard: int = 3,
    random_frac: float = 0.25,
    seed: int = SEED,
    test_texts: Collection[str] = (),
    test_docs: Collection[str] = (),
) -> dict:
    """{"train": [Pair], "val": [Pair], "val_qids", "train_qids", "report"}.

    `test_texts`: the test claims' texts — a train claim with the same text is dropped.
    `test_docs`: the test claims' rationale/cited docs — a train claim citing any is
    dropped and none is used as a negative (pass () to keep SciFact's official overlap).

    Refuses any dataset but beir/scifact/train, and a tuning set that is not a subset of
    the split (a typo'd path would otherwise silently hold out nothing).
    """
    if dataset != TRAIN_DATASET:
        raise ContaminationError(f"training pairs come from {TRAIN_DATASET} only, not {dataset}")
    tuning = set(tuning)
    if not tuning or not tuning <= set(queries):
        raise ContaminationError("the tuning set must be a non-empty subset of the train claims")
    leaked = leakage_exclusions(queries, tuning, labels, qrels)
    rest = set(queries) - tuning - leaked
    texts = {normalize_claim(t) for t in test_texts}
    dup = {q for q in rest if normalize_claim(queries[q]) in texts}
    tdocs = set(test_docs)
    test_overlap = {q for q in rest - dup if cited_docs(q, labels, qrels) & tdocs}
    usable = sorted(rest - dup - test_overlap)
    docs_of = {q: cited_docs(q, labels, qrels) for q in usable}
    train_q, val_q = group_split(usable, docs_of, val_frac, seed)
    forbidden = set().union(*(cited_docs(q, labels, qrels) for q in tuning)) | tdocs
    kw = dict(queries=queries, labels=labels, qrels=qrels, top=top, docs=docs,
              n_hard=n_hard, random_frac=random_frac, forbidden_docs=forbidden)
    train = build_pairs(train_q, seed=seed, **kw)
    val = build_pairs(val_q, seed=seed + 1, **kw)
    assert not ({p.qid for p in train} | {p.qid for p in val}) & (tuning | dup | test_overlap)
    assert not {d for q in train_q for d in docs_of[q]} & {d for q in val_q for d in docs_of[q]}
    report = {
        "dataset": dataset,
        "train_claims_total": len(queries),
        "tuning_claims": len(tuning),
        "excluded_doc_overlap": len(leaked),
        "excluded_test_duplicate": len(dup),
        "excluded_test_doc_overlap": len(test_overlap),
        "test_docs_held_out": len(tdocs),
        "usable_claims": len(usable),
        "train_claims": len(train_q),
        "val_claims": len(val_q),
        "forbidden_negative_docs": len(forbidden),
        "n_hard": n_hard,
        "random_frac": random_frac,
        "seed": seed,
        "train": counts(train),
        "val": counts(val),
        "class_weights": class_weights(train),
    }
    return {"train": train, "val": val, "train_qids": train_q, "val_qids": val_q,
            "tuning_qids": sorted(tuning), "excluded_qids": sorted(leaked),
            "excluded_test_qids": sorted(dup | test_overlap), "report": report}


def pair_dict(p: Pair) -> dict:
    return asdict(p)
