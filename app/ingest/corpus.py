"""Load the BEIR/SciFact corpus, queries, and gold qrels via ir_datasets.

SciFact documents are short scientific abstracts and BEIR relevance judgments are
at the document level, so we index whole documents (title + abstract) as single
passages — no chunking. A chunker would slot in here for longer corpora.
"""
from __future__ import annotations

import json
import zipfile
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ir_datasets

from app.core.config import settings


def load_documents(dataset: str | None = None) -> list[dict]:
    ds = ir_datasets.load(dataset or settings.corpus_dataset)
    docs: list[dict] = []
    for d in ds.docs_iter():
        docs.append(
            {
                "doc_id": d.doc_id,
                "title": getattr(d, "title", "") or "",
                "text": getattr(d, "text", "") or "",
            }
        )
    return docs


def document_passage(doc: dict) -> str:
    """The text we embed and BM25-index for a document (title + abstract)."""
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    return f"{title}\n\n{text}".strip() if title else text


def load_queries_qrels(
    dataset: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries, qrels). qrels is {query_id: {doc_id: relevance}} — the exact
    nested-dict shape ranx.Qrels ingests."""
    ds = ir_datasets.load(dataset or settings.eval_dataset)
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = int(qr.relevance)
    return queries, qrels


# --- SciFact claim labels ------------------------------------------------------------
#
# SciFact is a claim-verification dataset: each "query" is a scientific claim, and the
# original annotation says, per cited abstract, whether it SUPPORTs or CONTRADICTs the
# claim (with rationale sentence indices). Claims with no such annotation are NEI (not
# enough info). BEIR keeps all of this in the `metadata` field of queries.jsonl, but
# ir_datasets' GenericQuery drops it — and BEIR's qrels still mark a doc relevant for
# every claim, NEI included. So "relevant" means two different things on this corpus,
# and the labels are needed to tell them apart.

ClaimLabelName = Literal["SUPPORT", "CONTRADICT", "NEI"]


@dataclass(frozen=True)
class ClaimLabel:
    label: ClaimLabelName
    # Docs the annotators cited with rationale sentences. Empty for NEI claims. Not
    # always equal to the qrels set: BEIR's qrels also include cited docs that carry no
    # rationale (13 of the 188 labelled test claims).
    rationale_doc_ids: set[str] = field(default_factory=set)


def parse_claim_labels(
    lines: Iterable[str | bytes], query_ids: Collection[str] | None = None
) -> dict[str, ClaimLabel]:
    """Parse BEIR SciFact queries.jsonl lines into {query_id: ClaimLabel}.

    `query_ids` restricts the result to one split (queries.jsonl holds train + test).

    Claim-level label: NEI if `metadata` is empty, else the single label shared by all
    evidence on all cited docs. A claim whose evidence mixes SUPPORT and CONTRADICT
    has no honest three-way label, so it raises rather than picking one — silently
    choosing would put the claim in the wrong stratum. This does not occur in BEIR
    SciFact (0 of 1,109 train+test claims; every cited doc carries one label and every
    labelled claim one label across its docs), so the check is a guard against a
    different or future release, not a code path the shipped data exercises.
    """
    wanted = set(query_ids) if query_ids is not None else None
    labels: dict[str, ClaimLabel] = {}
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["_id"])
        if wanted is not None and qid not in wanted:
            continue
        metadata = rec.get("metadata") or {}
        found = {ev["label"] for evidence in metadata.values() for ev in evidence}
        if not metadata:
            labels[qid] = ClaimLabel("NEI")
            continue
        if len(found) != 1 or not found <= {"SUPPORT", "CONTRADICT"}:
            raise ValueError(
                f"claim {qid}: cannot assign one claim-level label from evidence labels "
                f"{sorted(found)} (mixed SUPPORT/CONTRADICT, empty, or unknown)"
            )
        labels[qid] = ClaimLabel(found.pop(), set(metadata))
    if wanted is not None and (missing := wanted - labels.keys()):
        raise ValueError(f"{len(missing)} query ids have no entry in queries.jsonl")
    return labels


def scifact_source_zip() -> Path:
    """Where ir_datasets keeps the raw BEIR SciFact download.

    ir_datasets exposes no public accessor for a BEIR source archive, so this follows
    its storage convention: <home_path()>/beir/scifact/source.zip, where home_path()
    honours IR_DATASETS_HOME (default ~/.ir_datasets).
    """
    return ir_datasets.util.home_path() / "beir" / "scifact" / "source.zip"


def load_claim_labels(
    dataset: str | None = None,
    source_zip: str | Path | None = None,
    query_ids: Collection[str] | None = None,
) -> dict[str, ClaimLabel]:
    """{query_id: ClaimLabel} for a SciFact split (default: settings.eval_dataset).

    Works for beir/scifact/test and beir/scifact/train; the split's query ids come from
    ir_datasets unless passed explicitly as `query_ids`.
    """
    path = Path(source_zip) if source_zip is not None else scifact_source_zip()
    if not path.exists():
        raise FileNotFoundError(
            f"SciFact source archive not found at {path}. It is downloaded by ir_datasets "
            "on first use of beir/scifact (e.g. `make index`); set IR_DATASETS_HOME if "
            "your ir_datasets cache lives elsewhere."
        )
    if query_ids is None:
        ds = ir_datasets.load(dataset or settings.eval_dataset)
        query_ids = {q.query_id for q in ds.queries_iter()}
    with zipfile.ZipFile(path) as zf, zf.open("scifact/queries.jsonl") as fh:
        return parse_claim_labels(fh, query_ids)
