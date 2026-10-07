"""Select the worked examples of the opt-in ``fewshot`` prompt variant — offline, no LLM.

``SSR_LLM_PROMPT_VARIANT=fewshot`` (app.generate.prompts) appends a few short worked claim
checks to the product system prompt, to show where SciFact draws the line between a
stance and NOT ENOUGH EVIDENCE: a SUPPORT/CONTRADICT label needs an explicit rationale
sentence in the cited abstract; an on-topic abstract that does not state the finding, or
states it for another population, is NEI. The examples are chosen here by a FIXED rule
over SciFact's own labels — never by looking at model replies or model errors — and the
result is committed as ``app/generate/fewshot_examples.json`` (the prompt reads that file;
this module is only needed to regenerate or check it).

Pool (the leakage guards):

* ``beir/scifact/train`` claims only, minus the first ``PROTECTED_PREFIX`` (300) of the
  seeded (SEED=13) shuffle rag_eval samples from — the 100-claim dev sample and the
  101-300 replication pool stay clean — and never a test claim (train and test ids are
  disjoint; checked anyway).
* No example may use a document that any protected claim (those 300 train claims, every
  test claim) cites in its qrels or rationale: SciFact writes several claims (and their
  negations) per abstract, so an excerpt from such a document would show evidence for a
  claim the examples are later scored on.
* Exactly one cited document per claim (the excerpt then has one unambiguous source).

Sentences come from the original SciFact release (``corpus.jsonl``: each abstract as the
annotators' sentence list, which rationale indices refer to), downloaded once from the
pinned URL into ``data/scifact_release/`` and sha256-checked (a mismatch stops). BEIR's
abstract text is exactly those sentences joined by single spaces (checked: all 5,183
docs), so every excerpt is a verbatim substring of the passage the retriever serves.

Rule (all thresholds fixed in this module; candidates are visited in ``order_key``
order — sha256 of ``SALT:qid`` — so the pick does not favour low ids):

* ``support_paraphrase`` (2): label SUPPORT; one rationale sentence; the claim and that
  sentence share at most ``PARAPHRASE_MAX_COVERAGE`` of the claim's content words (the
  support is stated in different words) and at least ``MIN_COVERAGE``.
* ``contradict`` (1): label CONTRADICT; one rationale sentence; coverage >= MIN_COVERAGE.
* ``nei_on_topic`` (3): label NEI; the cited abstract's sentence with the highest
  coverage covers at least ``ON_TOPIC_MIN_COVERAGE`` of the claim's content words (the
  abstract is on topic and reads close to the claim) yet SciFact found no rationale in it.

A fourth category was planned and dropped before any example was rendered or any model
call: "NEI, the abstract is about another population/species". Over the whole clean pool
(86 NEI claims with one cited, unbanned abstract) no claim names a population group that
its abstract lacks while the abstract names another one, so that rule selects nothing; its
slot went to a third ``nei_on_topic`` example.

Every excerpt is one complete sentence (``complete_sentence``) of ``EXCERPT_WORDS``
words; claims are at most ``MAX_CLAIM_WORDS`` words; no two examples share a document.
The one-line reason is a fixed template per category (``REASONS``). Order in the prompt:
``PROMPT_ORDER``.

Run:
    uv run python -m app.eval.fewshot_select            # check the committed JSON
    uv run python -m app.eval.fewshot_select --write    # regenerate it
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.generate.prompts import FEWSHOT_PATH

ROOT = Path(__file__).resolve().parents[2]
RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
RELEASE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
RELEASE_DIR = ROOT / "data" / "scifact_release"
RELEASE_MEMBER = "data/corpus.jsonl"

TRAIN_DATASET = "beir/scifact/train"
PROTECTED_PREFIX = 300  # first N of rag_eval's seeded train shuffle: dev 1-100, replication 101-300
SALT = "ssr-fewshot-v1"

MIN_COVERAGE = 0.2
PARAPHRASE_MAX_COVERAGE = 0.5
ON_TOPIC_MIN_COVERAGE = 0.5
EXCERPT_WORDS = (8, 40)
MAX_CLAIM_WORDS = 25

QUOTA = {"support_paraphrase": 2, "contradict": 1, "nei_on_topic": 3}
# Interleaved so neither label opens or closes the block in a run.
PROMPT_ORDER = (
    ("nei_on_topic", 0), ("support_paraphrase", 0), ("nei_on_topic", 1),
    ("contradict", 0), ("nei_on_topic", 2), ("support_paraphrase", 1),
)
VERDICT = {
    "support_paraphrase": "SUPPORTED",
    "contradict": "REFUTED",
    "nei_on_topic": "NOT ENOUGH EVIDENCE",
}
REASONS = {
    "support_paraphrase": "The passage reports this finding in different words, so it supports the claim [1].",
    "contradict": "The passage reports a result that conflicts with the claim, so it refutes the claim [1].",
    "nei_on_topic": (
        "The passage is on the same topic but does not state the claimed result or its "
        "opposite, so it neither supports nor refutes the claim [1]."
    ),
}

_STOP = frozenset(
    "a an the of in on at to for from by with without and or not no is are was were be been "
    "being has have had do does did can could may might will would shall should this that "
    "these those it its their there than then as into onto via per vs versus which who whom "
    "whose what when where while also more less most least very such both either neither "
    "between among after before during over under up down out about against".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    """Lower-cased alphanumeric tokens minus stopwords, with a crude plural fold."""
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def coverage(claim: str, sentence: str) -> float:
    """Share of the claim's content words that also occur in `sentence` (0 for an empty claim)."""
    c = content_words(claim)
    return len(c & content_words(sentence)) / len(c) if c else 0.0


_COMPLETE = re.compile(r"^[A-Z0-9(\[].*[.!?][)\]\"']?$", re.S)


def complete_sentence(s: str) -> bool:
    """One whole sentence of acceptable length: starts upper-case/digit/bracket, ends with
    terminal punctuation (the original segmentation sometimes cuts mid-sentence, e.g. at
    "+/-"), and has EXCERPT_WORDS words."""
    s = s.strip()
    lo, hi = EXCERPT_WORDS
    return bool(_COMPLETE.match(s)) and lo <= len(s.split()) <= hi


def order_key(qid: str) -> str:
    return hashlib.sha256(f"{SALT}:{qid}".encode()).hexdigest()


@dataclass(frozen=True)
class Claim:
    qid: str
    text: str
    label: str  # SUPPORT | CONTRADICT | NEI
    cited: frozenset[str]  # qrels docs
    # {doc_id: [[sentence indices], ...]} — one list per evidence set (empty for NEI).
    rationales: Mapping[str, Sequence[Sequence[int]]]


def _candidate(claim: Claim, abstracts: Mapping[str, Sequence[str]], banned_docs: Collection[str]):
    """(category, doc_id, excerpt) or None — the rule above, per claim."""
    if len(claim.text.split()) > MAX_CLAIM_WORDS or len(claim.cited) != 1:
        return None
    (doc,) = claim.cited
    if doc in banned_docs or doc not in abstracts:
        return None
    sents = [s.strip() for s in abstracts[doc]]
    if claim.label in ("SUPPORT", "CONTRADICT"):
        if set(claim.rationales) != {doc}:
            return None
        sets = claim.rationales[doc]
        if len(sets) != 1 or len(sets[0]) != 1:
            return None
        sent = sents[sets[0][0]]
        if not complete_sentence(sent):
            return None
        cov = coverage(claim.text, sent)
        if cov < MIN_COVERAGE:
            return None
        if claim.label == "CONTRADICT":
            return "contradict", doc, sent
        return ("support_paraphrase", doc, sent) if cov <= PARAPHRASE_MAX_COVERAGE else None
    if claim.label != "NEI" or claim.rationales:
        return None
    pool = [s for s in sents if complete_sentence(s)]
    if not pool:
        return None
    best = max(pool, key=lambda s: (coverage(claim.text, s), -sents.index(s)))
    if coverage(claim.text, best) < ON_TOPIC_MIN_COVERAGE:
        return None
    return "nei_on_topic", doc, best


def select_examples(
    claims: Iterable[Claim],
    abstracts: Mapping[str, Sequence[str]],
    protected_qids: Collection[str],
    banned_docs: Collection[str],
    quota: Mapping[str, int] = QUOTA,
) -> list[dict]:
    """The worked examples, in PROMPT_ORDER. Pure: the same inputs give the same list.

    `protected_qids` are never examples; `banned_docs` are never excerpted. Raises if a
    category cannot be filled (the rule, not a fallback, decides)."""
    protected = set(protected_qids)
    picked: dict[str, list[dict]] = {c: [] for c in quota}
    used_docs: set[str] = set()
    for claim in sorted(claims, key=lambda c: order_key(c.qid)):
        if claim.qid in protected:
            continue
        cand = _candidate(claim, abstracts, banned_docs)
        if cand is None:
            continue
        cat, doc, excerpt = cand
        if len(picked[cat]) >= quota[cat] or doc in used_docs:
            continue
        used_docs.add(doc)
        picked[cat].append({
            "query_id": claim.qid,
            "category": cat,
            "gold_label": claim.label,
            "doc_id": doc,
            "claim": claim.text,
            "excerpt": excerpt,
            "reason": REASONS[cat],
            "verdict": VERDICT[cat],
        })
    if short := {c: quota[c] - len(v) for c, v in picked.items() if len(v) < quota[c]}:
        raise ValueError(f"the rule cannot fill every category: missing {short}")
    return [picked[cat][i] for cat, i in PROMPT_ORDER if i < quota.get(cat, 0)]


# --- data loading (the real SciFact files) ---------------------------------------------------

def release_corpus(fetch: bool = True, opener: Callable[[str], bytes] | None = None) -> dict[str, list[str]]:
    """{doc_id: [abstract sentences]} from the pinned original SciFact release. The tarball
    is read in memory (members are never extracted to disk)."""
    path = RELEASE_DIR / "data.tar.gz"
    if path.exists():
        data = path.read_bytes()
    elif fetch:
        get = opener or (lambda url: urllib.request.urlopen(url, timeout=60).read())  # noqa: S310
        data = get(RELEASE_URL)
    else:
        raise FileNotFoundError(f"{path} missing; run without --no-fetch to download it")
    if (got := hashlib.sha256(data).hexdigest()) != RELEASE_SHA256:
        raise ValueError(f"SciFact release sha256 {got[:12]}… != pinned {RELEASE_SHA256[:12]}…")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        fh = tf.extractfile(RELEASE_MEMBER)
        if fh is None:
            raise ValueError(f"{RELEASE_MEMBER} is not a regular file in the release")
        lines = fh.read().decode().splitlines()
    return {str(j["doc_id"]): [str(s) for s in j["abstract"]] for j in map(json.loads, lines) if j}


def _split_ids(zf: zipfile.ZipFile, split: str) -> dict[str, set[str]]:
    rows = zf.read(f"scifact/qrels/{split}.tsv").decode().splitlines()[1:]
    out: dict[str, set[str]] = {}
    for line in rows:
        if line.strip():
            q, d, *_ = line.split("\t")
            out.setdefault(q, set()).add(d)
    return out


def load_inputs(fetch: bool = True) -> dict:
    """Everything select_examples needs, from the BEIR source zip + the pinned release.
    The train shuffle is rag_eval.sample_claims over the train qrels' query ids (the same
    809 ids ir_datasets' beir/scifact/train yields)."""
    from app.eval.rag_eval import sample_claims
    from app.ingest.corpus import scifact_source_zip

    with zipfile.ZipFile(scifact_source_zip()) as zf:
        train_qrels, test_qrels = _split_ids(zf, "train"), _split_ids(zf, "test")
        queries = {
            str(j["_id"]): j
            for j in map(json.loads, zf.read("scifact/queries.jsonl").decode().splitlines())
            if j
        }
    if set(train_qrels) & set(test_qrels):
        raise ValueError("train and test claim ids overlap")
    protected_train = sample_claims(train_qrels, None)[:PROTECTED_PREFIX]
    claims = []
    for qid, cited in train_qrels.items():
        meta = queries[qid].get("metadata") or {}
        labels = {ev["label"] for evs in meta.values() for ev in evs}
        label = "NEI" if not meta else labels.pop() if len(labels) == 1 else "MIXED"
        claims.append(Claim(
            qid, queries[qid]["text"], label, frozenset(cited),
            {d: [list(ev["sentences"]) for ev in evs] for d, evs in meta.items()},
        ))
    banned: set[str] = set()
    for qid in protected_train:
        banned |= train_qrels[qid] | set((queries[qid].get("metadata") or {}))
    for qid, docs in test_qrels.items():
        banned |= docs | set((queries[qid].get("metadata") or {}))
    return {
        "claims": claims,
        "abstracts": release_corpus(fetch),
        "protected_qids": set(protected_train) | set(test_qrels),
        "banned_docs": banned,
        "protected_train": protected_train,
        "test_qids": set(test_qrels),
    }


def build_record(examples: Sequence[Mapping]) -> dict:
    """The committed JSON: the examples plus the rule's provenance."""
    return {
        "version": 1,
        "generated_by": "app.eval.fewshot_select",
        "source": {
            "claims_labels": f"BEIR SciFact source.zip ({TRAIN_DATASET} claims, queries.jsonl metadata)",
            "sentences": f"{RELEASE_URL} ({RELEASE_MEMBER}), sha256 {RELEASE_SHA256}",
            "license": "SciFact data: CC BY-NC 2.0 (allenai/scifact)",
        },
        "rule": {
            "salt": SALT,
            "protected_train_prefix": PROTECTED_PREFIX,
            "quota": dict(QUOTA),
            "min_coverage": MIN_COVERAGE,
            "paraphrase_max_coverage": PARAPHRASE_MAX_COVERAGE,
            "on_topic_min_coverage": ON_TOPIC_MIN_COVERAGE,
            "excerpt_words": list(EXCERPT_WORDS),
            "max_claim_words": MAX_CLAIM_WORDS,
        },
        "examples": [dict(e) for e in examples],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"(re)write {FEWSHOT_PATH}")
    ap.add_argument("--no-fetch", action="store_true", help="never download the SciFact release")
    a = ap.parse_args(argv)
    inp = load_inputs(fetch=not a.no_fetch)
    rec = build_record(select_examples(inp["claims"], inp["abstracts"], inp["protected_qids"],
                                       inp["banned_docs"]))
    for e in rec["examples"]:
        print(f"{e['query_id']:>5} {e['category']:<18} {e['gold_label']:<10} doc {e['doc_id']}: {e['claim']}")
    if a.write:
        FEWSHOT_PATH.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {FEWSHOT_PATH}")
        return 0
    committed = json.loads(FEWSHOT_PATH.read_text())
    if committed != rec:
        print(f"{FEWSHOT_PATH} differs from what the rule selects; re-run with --write", file=sys.stderr)
        return 1
    print(f"{FEWSHOT_PATH} matches the rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
