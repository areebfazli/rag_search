"""Re-score a rag.json under the SciFact label audit (Sylvestre 2026) — offline, no LLM calls.

Source: J. Sylvestre, "Gold Label Errors in the SciFact Benchmark: An LLM-Assisted
Annotation Audit", BioNLP 2026 (https://aclanthology.org/2026.bionlp-1.9/). Companion
data: github.com/Kefez/scifact-audit-bionlp2026, pinned at ``AUDIT_COMMIT``; the files
used are ``AUDIT_FILES`` (fetched from raw.githubusercontent.com at that commit into
``data/label_audit/<commit>/``, sha256-verified — a hash mismatch stops, never
overwrites). Corrections and review CSVs are CC BY 4.0 (derived from SciFact); the
repo's code is MIT.

The audit re-read all 209 evidence-bearing SciFact dev claim–doc pairs (BEIR
``beir/scifact/test`` is SciFact dev, same claim ids) and reports, with ONE annotator:

* 11 confirmed errors (``corrections/dev_corrections.json``, every entry whose
  error_type is not ``per_document_outcome_mismatch``). All 11 have ``doc_id: null``
  ("applies to all evidence docs of this claim") and each of those claims has exactly
  one rationale doc here, so the pair-to-claim step is unambiguous.
* 1 per-document error, claim 597 (listed in the same file, but the paper excludes it
  from the 11 and describes it as ONE of the claim's three evidence docs, 12779444,
  reporting mortality not incidence). Not applied in any tier; see ``PER_DOCUMENT_DOCS``.
* 8 "debatable but defensible" claims, deliberately left at their gold label: the
  ``my_verdict == "DEBATABLE"`` rows of the two stage-2 false-alarm review CSVs, minus
  the 2 later upgraded to corrections — cross-checked against the id list in the
  paper's Appendix A, and refused if the two disagree.

Pair -> claim mapping, the same rule as ``app.ingest.corpus.parse_claim_labels``: every
rationale doc of a claim carries the claim's label (that parser guarantees it). A
correction sets the label of its doc (or, with ``doc_id: null``, of every rationale doc
of the claim); a doc corrected to NEI has no rationale for the claim and leaves the
rationale set, exactly like a cited-but-unannotated doc in BEIR's metadata. The claim's
label is then re-derived: no rationale doc left -> NEI; one label across the remaining
docs -> that label; SUPPORT and CONTRADICT mixed -> error (``parse_claim_labels``
refuses the same case). A correction whose ``original_label`` disagrees with our label,
whose doc is not a rationale doc of the claim, or whose claim is not in the split is
refused rather than guessed at. Qrels are untouched (the audit judges the label, not
relevance), so the legacy qrels oracle and retrieval metrics do not change.

Per changed claim, the rationale oracle is recomputed from the row's own
``retrieved_doc_ids``: a claim relabelled NEI has no rationale doc, so its ``evidence``
becomes False and abstaining becomes the correct action for it.

Tiers: ``original`` (the run as scored — must reproduce its stored verdict accuracy),
``corrected_strict`` (the 11 corrections), ``corrected_excl_debatable`` (strict, with
the 8 debatable claims dropped from scoring; n is reported). This is a secondary,
sensitivity artifact: the headline stays the original labels, and the input rag.json
is never modified or replaced.

Run:
    uv run python -m app.eval.label_audit [RAG_JSON ...] [--no-fetch]
    (default: eval/results/rag.json -> eval/results/rag_label_audit.{md,json}; any other
    rag.json writes rag_label_audit.{md,json} next to itself)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.paths import display_path, is_within
from app.eval import rag_eval
from app.eval.rag_eval import LABELS, _abstention, _abstention_class, _rate, verdict_scores
from app.eval.verify_eval import wilson
from app.ingest.corpus import ClaimLabel, load_claim_labels

AUDIT_REPO = "Kefez/scifact-audit-bionlp2026"
AUDIT_COMMIT = "5711f3fdf2305552df2b6e700dca2ebd7c8806e1"  # 2026-05-08, repo HEAD when fetched
AUDIT_PAPER = (
    "Sylvestre, J. (2026). Gold Label Errors in the SciFact Benchmark: An LLM-Assisted "
    "Annotation Audit. BioNLP 2026. https://aclanthology.org/2026.bionlp-1.9/"
)
AUDIT_LICENSE = "corrections + review CSVs: CC BY 4.0 (derived from SciFact); code: MIT"
AUDIT_DATASET = "beir/scifact/test"  # = SciFact dev, the split the audit's dev corrections cover
AUDIT_DIR = Path("data/label_audit")  # gitignored (under data/)

CORRECTIONS_FILE = "corrections/dev_corrections.json"
DEBATABLE_CSVS = (
    "results/stage2_manual_review/scifact_false_alarms_25_review.csv",
    "results/stage2_manual_review/scifact_false_alarms_remaining_24_review.csv",
)
PAPER_TEX = "paper/scifact_audit_bionlp2026.tex"
# Every file read, with its sha256 at AUDIT_COMMIT (LICENSE/README are provenance only).
AUDIT_FILES = {
    "LICENSE": "0548d6f5882e0c3fea922f19208f6228da22d7a9d22d6f91536cc032234330a0",
    "corrections/README.md": "bdd98405b6c5e188b10f72b115b23d6cf4116bebc1ece51035b110a3a63c8eb6",
    CORRECTIONS_FILE: "c45ff40e3a8906192595462596dd56146e476aafa5b39fb3cf5a367b80a62b23",
    DEBATABLE_CSVS[0]: "9704c80788db4eaee051eef6c60b188df71e7555a3f7c64c821c5fef2974eb85",
    DEBATABLE_CSVS[1]: "c84af6ee62148f927a3712f37f290e45cac524f425169ba64dd81d2a1f5674dd",
    PAPER_TEX: "eebcc98f78c89b66cb045e09303ef3865e2326355f4b9a3df02885624e056841",
}

PER_DOCUMENT_ERROR = "per_document_outcome_mismatch"
# The one per-document error, which the corrections file records as doc_id null / NEI but
# the paper (sec. "Per-Document Error") and corrections/README.md describe as a single doc
# of three. The doc is the one the stage-2 review flags for mortality (CSV row 597, paper
# "Effect of screening on cervical cancer mortality in England and Wales") and the one the
# stage-1 audit JSON marks as an error (doc 12779444). Applied per-document it would
# leave claim 597 SUPPORT (its other two docs still support it), so it is reported, not
# applied.
PER_DOCUMENT_DOCS = {"597": "12779444"}

AUDIT_LABELS = frozenset({"SUPPORT", "CONTRADICT", "NEI"})
TIERS = ("original", "corrected_strict", "corrected_excl_debatable")
TIER_TITLES = {
    "original": "Original SciFact labels (headline)",
    "corrected_strict": "Corrected labels — strict (the 11 confirmed errors only)",
    "corrected_excl_debatable": "Corrected labels, 8 debatable claims excluded from scoring",
}
REPORT_STEM = "rag_label_audit"

_CORRECTION_KEYS = ("claim_id", "doc_id", "original_label", "corrected_label", "error_type")
_DEBATABLE_TEX = re.compile(r"final debatable claim IDs are ([0-9,\sand]+?)\.")


class AuditError(ValueError):
    """The audit data, or a run, can't be mapped onto our labels without guessing."""


@dataclass(frozen=True)
class Correction:
    claim_id: str
    doc_id: str | None  # None: every rationale doc of the claim
    original_label: str
    corrected_label: str
    error_type: str
    justification: str = ""


@dataclass(frozen=True)
class Audit:
    corrections: tuple[Correction, ...]  # strict: the confirmed claim-level errors
    per_document: tuple[Correction, ...]  # reported, never applied
    debatable: frozenset[str]


# --- fetching the pinned files ---------------------------------------------------------------


def audit_path(rel: str, root: Path = AUDIT_DIR) -> Path:
    return root / AUDIT_COMMIT / rel


def raw_url(rel: str) -> str:
    return f"https://raw.githubusercontent.com/{AUDIT_REPO}/{AUDIT_COMMIT}/{rel}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_audit_files(
    root: Path = AUDIT_DIR,
    fetch: bool = True,
    opener: Callable[[str], bytes] | None = None,
) -> None:
    """Every AUDIT_FILES entry present under root/<commit>/ with its pinned sha256. A missing
    file is downloaded from the pinned commit (if `fetch`); a present file with the wrong
    hash is an error, never silently replaced."""
    get = opener or (lambda url: urllib.request.urlopen(url, timeout=30).read())  # noqa: S310
    for rel, want in AUDIT_FILES.items():
        p = audit_path(rel, root)
        if p.exists():
            if (got := _sha256(p.read_bytes())) != want:
                raise AuditError(f"{p}: sha256 {got[:12]}… != pinned {want[:12]}… (delete it to re-fetch)")
            continue
        if not fetch:
            raise AuditError(f"{p} missing; re-run without --no-fetch to download it from {raw_url(rel)}")
        data = get(raw_url(rel))
        if (got := _sha256(data)) != want:
            raise AuditError(f"{raw_url(rel)}: sha256 {got[:12]}… != pinned {want[:12]}…")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


# --- parsing ------------------------------------------------------------------------------


def parse_corrections(text: str) -> tuple[list[Correction], list[Correction]]:
    """(strict claim-level corrections, per-document corrections) from dev_corrections.json.
    Anything outside the documented schema is refused."""
    try:
        blob = json.loads(text)
        entries = blob["corrections"]
    except (ValueError, KeyError, TypeError) as e:
        raise AuditError(f"corrections: not the documented schema ({type(e).__name__})") from None
    if not isinstance(entries, list):
        raise AuditError("corrections: `corrections` is not a list")
    strict, per_doc, seen = [], [], set()
    for e in entries:
        if not isinstance(e, dict) or (missing := [k for k in _CORRECTION_KEYS if k not in e]):
            raise AuditError(f"corrections: entry lacks {missing if isinstance(e, dict) else 'shape'}: {e!r}")
        c = Correction(
            claim_id=str(e["claim_id"]),
            doc_id=None if e["doc_id"] is None else str(e["doc_id"]),
            original_label=e["original_label"],
            corrected_label=e["corrected_label"],
            error_type=str(e["error_type"]),
            justification=str(e.get("justification", "")),
        )
        if c.original_label not in {"SUPPORT", "CONTRADICT"} or c.corrected_label not in AUDIT_LABELS:
            raise AuditError(f"corrections: claim {c.claim_id}: labels {c.original_label}->{c.corrected_label}")
        if c.original_label == c.corrected_label:
            raise AuditError(f"corrections: claim {c.claim_id}: corrected label equals the original")
        if (key := (c.claim_id, c.doc_id)) in seen:
            raise AuditError(f"corrections: duplicate entry for claim {c.claim_id} doc {c.doc_id}")
        seen.add(key)
        (per_doc if c.error_type == PER_DOCUMENT_ERROR else strict).append(c)
    if len({c.claim_id for c in strict}) != len(strict):
        raise AuditError("corrections: a claim has more than one claim-level correction")
    return strict, per_doc


def parse_debatable(csv_texts: Sequence[str], tex: str, corrected_ids: Collection[str]) -> frozenset[str]:
    """The final debatable claim ids: DEBATABLE rows of the review CSVs minus claims that
    were later upgraded to corrections, which must equal the paper's Appendix A list."""
    from_csv: set[str] = set()
    for text in csv_texts:
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or not {"id", "my_verdict"} <= rows[0].keys():
            raise AuditError("review CSV lacks the `id` / `my_verdict` columns")
        from_csv |= {str(r["id"]).strip() for r in rows if r["my_verdict"].strip() == "DEBATABLE"}
    from_csv -= set(corrected_ids)
    m = _DEBATABLE_TEX.search(" ".join(tex.split()))
    if not m:
        raise AuditError("paper: no 'final debatable claim IDs are …' sentence found")
    from_paper = set(re.findall(r"\d+", m.group(1)))
    if from_csv != from_paper:
        raise AuditError(f"debatable ids disagree: CSVs {sorted(from_csv)} vs paper {sorted(from_paper)}")
    return frozenset(from_csv)


def load_audit(root: Path = AUDIT_DIR, fetch: bool = True) -> Audit:
    ensure_audit_files(root, fetch=fetch)
    strict, per_doc = parse_corrections(audit_path(CORRECTIONS_FILE, root).read_text())
    debatable = parse_debatable(
        [audit_path(p, root).read_text() for p in DEBATABLE_CSVS],
        audit_path(PAPER_TEX, root).read_text(),
        {c.claim_id for c in (*strict, *per_doc)},
    )
    if overlap := debatable & {c.claim_id for c in strict}:
        raise AuditError(f"claims both corrected and debatable: {sorted(overlap)}")
    return Audit(tuple(strict), tuple(per_doc), debatable)


# --- pair corrections -> claim labels ---------------------------------------------------------


def derive_claim_label(claim_id: str, doc_labels: Mapping[str, str]) -> ClaimLabel:
    """parse_claim_labels' rule over per-doc labels: NEI docs carry no rationale; none left
    is NEI; one shared label is the claim's; mixed SUPPORT/CONTRADICT is refused."""
    rationale = {d: lab for d, lab in doc_labels.items() if lab != "NEI"}
    if not rationale:
        return ClaimLabel("NEI")
    found = set(rationale.values())
    if len(found) != 1:
        raise AuditError(
            f"claim {claim_id}: corrected docs mix {sorted(found)} — no single claim-level label "
            f"(parse_claim_labels refuses the same case)"
        )
    return ClaimLabel(found.pop(), set(rationale))


def apply_corrections(
    labels: Mapping[str, ClaimLabel], corrections: Sequence[Correction]
) -> dict[str, ClaimLabel]:
    """A new {query_id: ClaimLabel} with the corrections applied at the doc level, then the
    claim label re-derived. Unaffected claims keep their ClaimLabel object."""
    doc_labels: dict[str, dict[str, str]] = {}
    for c in corrections:
        if c.claim_id not in labels:
            raise AuditError(f"claim {c.claim_id} is not in this split's labels")
        lab = labels[c.claim_id]
        docs = doc_labels.setdefault(c.claim_id, {d: lab.label for d in lab.rationale_doc_ids})
        if lab.label == "NEI":
            raise AuditError(f"claim {c.claim_id}: NEI here, so it has no evidence pair to correct")
        targets = sorted(lab.rationale_doc_ids) if c.doc_id is None else [c.doc_id]
        for d in targets:
            if d not in lab.rationale_doc_ids:
                raise AuditError(f"claim {c.claim_id}: doc {d} is not one of its rationale docs")
            if docs[d] != c.original_label:
                raise AuditError(
                    f"claim {c.claim_id} doc {d}: audit says original {c.original_label}, "
                    f"ours is {docs[d]}"
                )
            docs[d] = c.corrected_label
    out = dict(labels)
    for qid, docs in doc_labels.items():
        out[qid] = derive_claim_label(qid, docs)
    return out


def label_changed(a: ClaimLabel, b: ClaimLabel) -> bool:
    return a.label != b.label or a.rationale_doc_ids != b.rationale_doc_ids


# --- re-scoring a run -----------------------------------------------------------------------


def _check_row(row: Mapping, lab: ClaimLabel) -> None:
    if row["gold_label"] != lab.label or set(row.get("rationale_doc_ids", ())) != lab.rationale_doc_ids:
        raise AuditError(
            f"claim {row['query_id']}: run was scored against {row['gold_label']} "
            f"{sorted(row.get('rationale_doc_ids', ()))}, labels here say {lab.label} "
            f"{sorted(lab.rationale_doc_ids)} — a different label source"
        )
    if bool(row["evidence"]) != bool(lab.rationale_doc_ids & set(row["retrieved_doc_ids"])):
        raise AuditError(f"claim {row['query_id']}: stored `evidence` disagrees with its retrieved ids")


def relabel_row(row: Mapping, lab: ClaimLabel) -> dict:
    """The row scored against `lab`: gold label, rationale docs, and the rationale-oracle
    evidence flag + quadrant recomputed from the row's own retrieved ids. The prediction is
    untouched."""
    ev = bool(lab.rationale_doc_ids & set(row["retrieved_doc_ids"]))
    return {
        **row,
        "gold_label": lab.label,
        "rationale_doc_ids": sorted(lab.rationale_doc_ids),
        "evidence": ev,
        "abstention_class": _abstention_class(bool(row["answered"]), ev),
    }


def relabel_rows(
    rows: Sequence[Mapping], original: Mapping[str, ClaimLabel], corrected: Mapping[str, ClaimLabel]
) -> list[dict]:
    """Copies of `rows`, with every claim whose label changed re-scored (after checking the
    row really was scored against `original`)."""
    out = []
    for r in rows:
        qid = r["query_id"]
        if qid not in original:
            raise AuditError(f"claim {qid} is not in this split's labels")
        if label_changed(original[qid], corrected[qid]):
            _check_row(r, original[qid])
            out.append(relabel_row(r, corrected[qid]))
        else:
            out.append(dict(r))
    return out


def score(rows: Sequence[Mapping]) -> dict:
    """Verdict accuracy (+ Wilson 95% CI), per-gold-label accuracy, the confusion matrix and
    the rationale-oracle abstention metrics — rag_eval's own definitions."""
    rows = list(rows)
    correct = sum(r["predicted_label"] == r["gold_label"] for r in rows)
    ab = _abstention(rows, "evidence")
    return {
        "n": len(rows),
        "correct": correct,
        "verdict_accuracy": _rate(correct, len(rows)),
        "verdict_accuracy_ci95": wilson(correct, len(rows)),
        "per_label": {
            g: {
                "n": (n_g := sum(r["gold_label"] == g for r in rows)),
                "correct": (k := sum(r["gold_label"] == g and r["predicted_label"] == g for r in rows)),
                "accuracy": _rate(k, n_g),
            }
            for g in LABELS
        },
        "confusion": verdict_scores(rows)["confusion"],
        "answered_rate": _rate(sum(bool(r["answered"]) for r in rows), len(rows)),
        **ab,
    }


def tier_rows(
    rows: Sequence[Mapping], labels: Mapping[str, ClaimLabel], audit: Audit, tier: str
) -> list[dict]:
    """The run's rows as scored under one tier."""
    if tier == "original":
        return [dict(r) for r in rows]
    corrected = relabel_rows(rows, labels, apply_corrections(labels, audit.corrections))
    if tier == "corrected_strict":
        return corrected
    if tier == "corrected_excl_debatable":
        return [r for r in corrected if r["query_id"] not in audit.debatable]
    raise ValueError(f"unknown tier {tier!r} (one of {TIERS})")


def audit_run(blob: Mapping, labels: Mapping[str, ClaimLabel], audit: Audit, source: str = "") -> dict:
    """The label-audit report for one rag.json. The original tier must reproduce the run's
    stored verdict accuracy, so this can never report a different headline."""
    rows = blob["rows"]
    dataset = (blob.get("run") or {}).get("dataset")
    if dataset != AUDIT_DATASET:
        raise AuditError(f"run dataset is {dataset!r}; the dev-set audit only covers {AUDIT_DATASET}")
    corrected = apply_corrections(labels, audit.corrections)
    tiers = {t: score(tier_rows(rows, labels, audit, t)) for t in TIERS}
    stored = blob.get("verdict_accuracy")
    if stored is not None and tiers["original"]["verdict_accuracy"] != stored:
        raise AuditError(
            f"original tier gives {tiers['original']['verdict_accuracy']}, the run stores {stored}"
        )

    by_id = {r["query_id"]: r for r in rows}
    reasons = {c.claim_id: c for c in audit.corrections}
    changed = []
    for qid in sorted((q for q in by_id if label_changed(labels[q], corrected[q])), key=int):
        r, after = by_id[qid], relabel_row(by_id[qid], corrected[qid])
        changed.append({
            "query_id": qid,
            "error_type": reasons[qid].error_type,
            "justification": reasons[qid].justification,
            "gold_before": labels[qid].label,
            "gold_after": corrected[qid].label,
            "rationale_docs_before": len(labels[qid].rationale_doc_ids),
            "rationale_docs_after": len(corrected[qid].rationale_doc_ids),
            "predicted_label": r["predicted_label"],
            "correct_before": r["predicted_label"] == labels[qid].label,
            "correct_after": r["predicted_label"] == corrected[qid].label,
            "answered": bool(r["answered"]),
            "evidence_before": bool(r["evidence"]),
            "evidence_after": after["evidence"],
            "abstention_class_before": r["abstention_class"] if "abstention_class" in r
            else _abstention_class(bool(r["answered"]), bool(r["evidence"])),
            "abstention_class_after": after["abstention_class"],
        })
    debatable = [
        {
            "query_id": q,
            "gold_label": by_id[q]["gold_label"],
            "predicted_label": by_id[q]["predicted_label"],
            "correct": by_id[q]["predicted_label"] == by_id[q]["gold_label"],
        }
        for q in sorted(audit.debatable & by_id.keys(), key=int)
    ]
    per_document = []
    for c in audit.per_document:
        doc = PER_DOCUMENT_DOCS.get(c.claim_id)
        entry = {"query_id": c.claim_id, "doc_id": doc, "error_type": c.error_type,
                 "file_says": f"doc_id {c.doc_id} -> {c.corrected_label}", "applied": False,
                 "in_run": c.claim_id in by_id}
        if doc and c.claim_id in labels:
            lab = labels[c.claim_id]
            per_doc = derive_claim_label(c.claim_id, {d: ("NEI" if d == doc else lab.label)
                                                      for d in lab.rationale_doc_ids})
            entry["claim_label_if_applied_per_document"] = per_doc.label
            entry["rationale_docs"] = f"{len(lab.rationale_doc_ids)} -> {len(per_doc.rationale_doc_ids)}"
            if c.claim_id in by_id:
                entry["evidence_flag_would_change"] = bool(by_id[c.claim_id]["evidence"]) != bool(
                    per_doc.rationale_doc_ids & set(by_id[c.claim_id]["retrieved_doc_ids"]))
        per_document.append(entry)
    run = blob.get("run") or {}
    return {
        "secondary_artifact": True,
        "headline": "original",
        "source_run": source,
        "run": {k: run.get(k) for k in ("git_sha", "dataset", "generator_model", "judge_model",
                                        "prompt_hash", "n_requested", "sample_seed", "canonical")},
        "audit": {
            "paper": AUDIT_PAPER,
            "repo": f"https://github.com/{AUDIT_REPO}",
            "commit": AUDIT_COMMIT,
            "files": AUDIT_FILES,
            "license": AUDIT_LICENSE,
            "annotators": 1,
            "n_corrections_strict": len(audit.corrections),
            "n_per_document": len(audit.per_document),
            "n_debatable": len(audit.debatable),
            "debatable_ids": sorted(audit.debatable, key=int),
        },
        "n_claims": len(rows),
        "n_label_changed": len(changed),
        "n_debatable_in_run": len(debatable),
        "tiers": tiers,
        "changed_claims": changed,
        "debatable_claims": debatable,
        "per_document": per_document,
    }


# --- report ---------------------------------------------------------------------------------


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _tier_md(t: Mapping) -> list[str]:
    ci = t["verdict_accuracy_ci95"]
    cols = (*LABELS, rag_eval.NO_VERDICT)
    lines = [
        f"n = {t['n']} · verdict accuracy **{_f(t['verdict_accuracy'])}** "
        f"({t['correct']}/{t['n']}; 95% Wilson CI {ci[0]:.3f}–{ci[1]:.3f})" if ci else f"n = {t['n']}",
        "",
        "| Gold \\ predicted | " + " | ".join(cols) + " | n | Accuracy |",
        "|---|" + "---|" * (len(cols) + 2),
    ]
    for g in LABELS:
        pl = t["per_label"][g]
        lines.append(f"| **{g}** | " + " | ".join(str(t["confusion"][g][p]) for p in cols)
                     + f" | {pl['n']} | {_f(pl['accuracy'])} |")
    q = t["quadrants"]
    lines += [
        "",
        f"Abstention (rationale oracle): answered {_f(t['answered_rate'])} · evidence retrieved "
        f"{_f(t['evidence_rate'])} · abstention precision {_f(t['abstention_precision'])} · "
        f"recall {_f(t['abstention_recall'])} · false abstention {_f(t['false_abstention_rate'])} · "
        f"answered without evidence {_f(t['answered_without_evidence_rate'])} · quadrants "
        f"(ans+ev / ans−ev / false abst. / correct abst.) {q['answered_with_evidence']} / "
        f"{q['answered_without_evidence']} / {q['false_abstention']} / {q['correct_abstention']}",
        "",
    ]
    return lines


def to_markdown(rep: Mapping) -> str:
    a, t = rep["audit"], rep["tiers"]
    o, s, x = t["original"], t["corrected_strict"], t["corrected_excl_debatable"]
    fixed = sum(not c["correct_before"] and c["correct_after"] for c in rep["changed_claims"])
    broken = sum(c["correct_before"] and not c["correct_after"] for c in rep["changed_claims"])
    lines = [
        "# RAG verdicts under the SciFact label audit (secondary artifact)",
        "",
        f"Source run: `{rep['source_run']}` (git {str(rep['run'].get('git_sha'))[:7]}, "
        f"generator `{rep['run'].get('generator_model')}`). Same stored answers, re-scored "
        f"offline — no LLM calls.",
        "",
        "**The headline stays the original SciFact labels** (`eval/results/rag.md`). This page is "
        "a sensitivity check against one published label audit, whose corrections come from a "
        "**single annotator** and have not been independently ratified; the audit's own "
        "authors recommend a multi-annotator re-audit before any corrected release.",
        "",
        f"Audit: {a['paper']} Data: {a['repo']} @ `{a['commit']}` — `{CORRECTIONS_FILE}` "
        f"(corrections), `{DEBATABLE_CSVS[0]}` + `{DEBATABLE_CSVS[1]}` (debatable ids, "
        f"cross-checked against `{PAPER_TEX}` Appendix A). License: {a['license']}.",
        "",
        "| Labels | n | Verdict accuracy | SUPPORT | CONTRADICT | NEI |",
        "|---|---|---|---|---|---|",
    ]
    for key in TIERS:
        tt = t[key]
        lines.append(
            f"| {TIER_TITLES[key]} | {tt['n']} | {_f(tt['verdict_accuracy'])} | "
            + " | ".join(f"{_f(tt['per_label'][g]['accuracy'])} (n={tt['per_label'][g]['n']})" for g in LABELS)
            + " |"
        )
    lines += [
        "",
        f"{rep['n_label_changed']} of {rep['n_claims']} claims change label under the strict "
        f"corrections; on those, the stored prediction goes wrong→right on {fixed} and "
        f"right→wrong on {broken}. Accuracy moves {_f(o['verdict_accuracy'])} → "
        f"{_f(s['verdict_accuracy'])} (strict) and → {_f(x['verdict_accuracy'])} with the "
        f"{rep['n_debatable_in_run']} debatable claims excluded (n={x['n']}). The predictions "
        f"are identical in every row; only the answer key differs, so this is not a paired "
        f"test of a model change (to compare two runs under the corrected key: "
        f"`make rag-compare A=… B=… ARGS=--labels=audit`).",
        "",
        "## Caveats — read before quoting the corrected numbers",
        "",
        "- **Single annotator.** Every correction and every debatable call is one person's "
        "judgement (with an LLM second opinion in chat), not an adjudicated re-annotation.",
        "- **LLM-assisted.** 8 of the 11 errors come from the 57 pairs an LLM screen "
        "(GPT-5.4-mini) flagged, adjudicated with a frontier-LLM (GPT-5.4) second opinion; the "
        "other 3 from the same annotator's review of the 152 unflagged pairs (paper, Stage 2). The "
        "corrections therefore lean toward how an LLM reads the evidence, so an LLM generator "
        "agreeing with them is expected in part — read the gain as a one-sided "
        "sensitivity check, not as hidden accuracy.",
        "- **One direction only.** Only the 188 evidence-bearing claims (209 pairs) were "
        "audited; the 112 NEI claims were not, so a label can move to NEI or flip, but an NEI "
        "claim can never be corrected to SUPPORT/CONTRADICT.",
        "- **Not a model comparison.** Same predictions, different key: the delta measures the "
        "labels, not the pipeline. Compare pipelines under one key (`rag_compare --labels`).",
        "",
        "## How corrections map onto claim labels",
        "",
        "The audit corrects claim–doc pairs; our label is per claim (`load_claim_labels`: NEI "
        "if no doc has rationale, else the one label all rationale docs share). A correction "
        "relabels its doc (`doc_id: null` = every rationale doc of the claim); a doc corrected "
        "to NEI drops out of the claim's rationale set; the claim label is then re-derived "
        "(none left → NEI; mixed SUPPORT/CONTRADICT → refused). All 11 strict corrections are "
        "`doc_id: null` on single-rationale-doc claims, so each maps 1:1. A claim relabelled NEI "
        "has no rationale doc, so its `evidence` flag becomes false and abstaining becomes "
        "correct. Qrels, retrieval and the qrels oracle are unchanged.",
        "",
        "## Claims whose label changed (strict)",
        "",
        "| Claim | Error type | Gold before → after | Predicted | Correct before → after | "
        "Evidence before → after | Abstention class before → after |",
        "|---|---|---|---|---|---|---|",
    ]
    yn = {True: "yes", False: "no"}
    for c in rep["changed_claims"]:
        lines.append(
            f"| {c['query_id']} | {c['error_type']} | {c['gold_before']} → {c['gold_after']} | "
            f"{c['predicted_label']} | {yn[c['correct_before']]} → {yn[c['correct_after']]} | "
            f"{yn[c['evidence_before']]} → {yn[c['evidence_after']]} | "
            f"{c['abstention_class_before']} → {c['abstention_class_after']} |"
        )
    lines += [
        "",
        "## Debatable claims (kept at their gold label; excluded in the last tier)",
        "",
        f"{len(rep['debatable_claims'])} of the audit's {a['n_debatable']} debatable claims "
        f"({', '.join(a['debatable_ids'])}) are in this run.",
        "",
        "| Claim | Gold | Predicted | Correct |",
        "|---|---|---|---|",
    ]
    lines += [f"| {d['query_id']} | {d['gold_label']} | {d['predicted_label']} | {yn[d['correct']]} |"
              for d in rep["debatable_claims"]]
    lines += ["", "## Per-document error (reported, not applied)", ""]
    for p in rep["per_document"]:
        lines.append(
            f"Claim {p['query_id']}: the corrections file records `{p['file_says']}`, but the paper "
            f"excludes it from the 11 and describes one of the claim's evidence docs "
            f"({p['doc_id']}, mortality not incidence) as mismatched. Applied per document, the "
            f"claim would stay {p.get('claim_label_if_applied_per_document', 'n/a')} "
            f"(rationale docs {p.get('rationale_docs', 'n/a')})"
            + (f"; this run's evidence flag would "
               f"{'change' if p.get('evidence_flag_would_change') else 'not change'}." if p["in_run"]
               else "; the claim is not in this run.")
        )
    lines += [
        "",
        "## Tier detail",
        "",
    ]
    for key in TIERS:
        lines += [f"### {TIER_TITLES[key]}", "", *_tier_md(t[key])]
    return "\n".join(lines)


def output_paths(src: Path) -> tuple[Path, Path]:
    """rag_label_audit.{json,md} next to the input rag.json — never the rag.* files."""
    return src.parent / f"{REPORT_STEM}.json", src.parent / f"{REPORT_STEM}.md"


def relabel_blob(blob: Mapping, labels: Mapping[str, ClaimLabel], audit: Audit, tier: str) -> dict:
    """A copy of a rag.json whose rows are scored under `tier` (for rag_compare). Only the
    rows change; the stored aggregates are dropped so nothing can mistake them for the
    relabelled numbers."""
    if (blob.get("run") or {}).get("dataset") != AUDIT_DATASET:
        raise AuditError(f"the dev-set audit only covers {AUDIT_DATASET}")
    return {"n": None, "run": dict(blob["run"]), "rows": tier_rows(blob["rows"], labels, audit, tier),
            "labels": tier}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.eval.label_audit", description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", default=[str(rag_eval.OUT / "rag.json")],
                    help="rag.json files (default: the committed eval/results/rag.json)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="never download the pinned audit files; fail if they are missing")
    args = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        audit = load_audit(fetch=not args.no_fetch)
        labels = load_claim_labels(AUDIT_DATASET)
        for src in map(Path, args.runs):
            blob = json.loads(src.read_text())
            out_json, out_md = output_paths(src)
            # rag_eval.OUT is repo-anchored; is_within resolves symlinks / `..` / aliases,
            # so neither the source nor a report path can lead a non-canonical run in.
            if not (blob.get("run") or {}).get("canonical") and any(
                is_within(p, rag_eval.OUT) for p in (src, out_json, out_md)
            ):
                raise AuditError(f"{src}: only the canonical run is reported into {rag_eval.OUT}/")
            rep = audit_run(blob, labels, audit, source=display_path(src))
            out_json.write_text(json.dumps(rep, indent=2) + "\n")
            out_md.write_text(to_markdown(rep))
            t = rep["tiers"]
            print(
                f"{src}: {rep['n_label_changed']}/{rep['n_claims']} labels changed; accuracy "
                f"{t['original']['verdict_accuracy']} (original) -> "
                f"{t['corrected_strict']['verdict_accuracy']} (strict) -> "
                f"{t['corrected_excl_debatable']['verdict_accuracy']} "
                f"(excl. debatable, n={t['corrected_excl_debatable']['n']})\nWrote {out_json}, {out_md}"
            )
    except AuditError as e:
        raise SystemExit(f"label_audit: {e}") from None


if __name__ == "__main__":
    main()
