"""Export the verifier's training data + a self-contained notebook for a free Kaggle GPU.

    uv run --locked python -m app.verify.kaggle_export      (make verify-export)

Writes, under data/kaggle/ (gitignored):

  verifier_data/train.jsonl  one row per (claim, abstract) pair: qid, doc_id, claim,
  verifier_data/val.jsonl    premise (= "title. abstract", the text the model reads), label, source
  verifier_data/meta.json    counts, split claim ids, the training config, the leakage
                             check's result and a sha256 of each .jsonl
  verifier_data/scifact_hashes.json  16-hex sha256 prefixes of every BEIR SciFact claim
                             (train + test), corpus title and abstract prefix (+ a blocklist
                             of external claims found near-duplicate at research time): the
                             notebook re-checks its stage-1 external data against them
  train_verifier.py          the notebook as a script (source: app/verify/kaggle_notebook.py,
  train_verifier.ipynb       with the data sha256s filled in), and the same as a notebook

The pairs are exactly what app.verify.train would train on: both call
app.verify.train.load_train_splits with the same TrainConfig. Retrieval (the hard
negatives) comes from verify_eval's cached top-5, or SearchService when it is missing.

Before anything is written, `check_leakage` re-derives the held-out sets from their
sources (not from the splits it is checking) and refuses the export if any row belongs
to a test claim (by id or by text), a tuning claim, a claim sharing a doc with a tuning
claim, or — with the default config — a claim sharing a doc with a test claim; or uses a
held-out claim's doc at all. Nothing in the export is read from .env or settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from app.verify import train_data as td

OUT = Path("data/kaggle")
DATA_SUBDIR = "verifier_data"
KIND = "scifact-verifier-data"  # the notebook finds its dataset by this marker
NOTEBOOK_SRC = Path(__file__).with_name("kaggle_notebook.py")
SHA_PLACEHOLDER = "EXPECTED_SHA256: dict = {}  # filled in by app.verify.kaggle_export"
FILES = ("train.jsonl", "val.jsonl")
HASH_FILE = "scifact_hashes.json"
VERSION = 2
# Raw claim texts of external (stage-1) rows found near-duplicate of a SciFact claim by the
# local check; their hashes ship in scifact_hashes.json["blocklist"] and the notebook drops them.
EXTERNAL_BLOCKLIST_TEXTS: tuple = (
    "It is Crohn's disease.",  # pubmedqa_l; token-set 92 vs SciFact train 1231/1232 (a false positive, dropped anyway)
)

# Shapes of API keys that must never ride along in an upload (OpenRouter, Groq, OpenAI,
# Hugging Face, S2-style long tokens are covered by the generic pattern).
SECRET_PATTERNS = (
    re.compile(rb"sk-or-v1-[0-9a-f]{16,}"),
    re.compile(rb"\bgsk_[A-Za-z0-9]{20,}"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{32,}"),
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(rb"(?i)(api[_-]?key|secret|token)\"?\s*[:=]\s*\"[A-Za-z0-9_-]{20,}"),
)


def row(p: td.Pair) -> dict:
    from app.verify.nli import premise_text

    return {"qid": p.qid, "doc_id": p.doc_id, "claim": p.claim,
            "premise": premise_text(p.title, p.abstract), "label": p.label, "source": p.source}


def jsonl_bytes(rows: Iterable[Mapping]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_secrets(data: bytes) -> list[str]:
    return [p.pattern.decode() for p in SECRET_PATTERNS if p.search(data)]


def build_hashes(claims: Iterable[str], docs: Iterable[Mapping], blocklist_texts: Iterable[str] = ()) -> dict:
    """The notebook's contamination reference: sorted unique hashes (kaggle_notebook's
    norm_text -> sha256[:16]) of SciFact claims, corpus titles, corpus abstract prefixes,
    and the external-claim blocklist."""
    from app.verify import kaggle_notebook as kn

    docs = list(docs)
    return {
        "normalization": "NFKD minus combining marks, lowercase, [a-z0-9]+ tokens joined by one space; sha256 hex[:16]; "
                         f"abstract prefixes = first {kn.PREFIX_CHARS} chars of the normalized abstract",
        "claims": sorted({kn.text_hash(c) for c in claims}),
        "titles": sorted({kn.text_hash(d.get("title", "")) for d in docs if kn.norm_text(d.get("title", ""))}),
        "abstract_prefixes": sorted({kn.prefix_hash(d.get("text", "")) for d in docs
                                     if kn.norm_text(d.get("text", ""))}),
        "blocklist": sorted({kn.text_hash(t) for t in blocklist_texts}),
    }


def check_leakage(
    rows: Mapping[str, Sequence[Mapping]],
    *,
    test_ids: Collection[str],
    test_texts: Collection[str],
    test_docs: Collection[str],
    tuning_ids: Collection[str],
    labels: Mapping[str, td.ClaimLabel],
    qrels: Mapping[str, Mapping[str, int]],
    strict_test_docs: bool = True,
) -> dict:
    """{"passed": True, ...counts} or ContaminationError naming the first violations.

    `rows` = {"train": [...], "val": [...]} export rows; `labels`/`qrels` are the TRAIN
    split's; the held-out docs are recomputed here from `tuning_ids` and `test_docs`."""
    tuning = set(tuning_ids)
    tuning_docs = set().union(*(td.cited_docs(q, labels, qrels) for q in tuning)) if tuning else set()
    overlap_tuning = td.leakage_exclusions(labels, tuning, labels, qrels)
    texts = {td.normalize_claim(t) for t in test_texts}
    tdocs = set(test_docs)
    problems: list[str] = []
    used_q = {s: {r["qid"] for r in rs} for s, rs in rows.items()}
    all_rows = [r for rs in rows.values() for r in rs]
    all_q = set().union(*used_q.values()) if used_q else set()

    def flag(what: str, bad: Collection) -> int:
        if bad:
            problems.append(f"{len(bad)} {what}: {sorted(bad)[:5]}")
        return len(bad)

    res = {
        "test_ids_in_data": flag("test claim ids", all_q & set(test_ids)),
        "test_texts_in_data": flag("claims identical to a test claim",
                                   {r["qid"] for r in all_rows if td.normalize_claim(r["claim"]) in texts}),
        "tuning_ids_in_data": flag("held-out tuning claims", all_q & tuning),
        "tuning_doc_overlap_claims_in_data": flag("claims sharing a doc with a tuning claim",
                                                  all_q & overlap_tuning),
        "tuning_docs_in_data": flag("pairs on a tuning claim's doc",
                                    {r["doc_id"] for r in all_rows if r["doc_id"] in tuning_docs}),
        "claims_in_both_train_and_val": flag("claims in both train and val",
                                             used_q.get("train", set()) & used_q.get("val", set())),
    }
    tr_docs = {d for q in used_q.get("train", ()) for d in td.cited_docs(q, labels, qrels)}
    va_docs = {d for q in used_q.get("val", ()) for d in td.cited_docs(q, labels, qrels)}
    res["cited_docs_in_both_train_and_val"] = flag("cited docs in both train and val", tr_docs & va_docs)
    test_overlap_q = {q for q in all_q if td.cited_docs(q, labels, qrels) & tdocs}
    test_doc_rows = {r["doc_id"] for r in all_rows if r["doc_id"] in tdocs}
    if strict_test_docs:
        res["test_doc_overlap_claims_in_data"] = flag("claims sharing a doc with a test claim", test_overlap_q)
        res["test_docs_in_data"] = flag("pairs on a test claim's doc", test_doc_rows)
    else:  # SciFact's official train/dev overlap, kept on purpose: report it, don't refuse it
        res["test_doc_overlap_claims_in_data"] = len(test_overlap_q)
        res["test_docs_in_data"] = len(test_doc_rows)
    if problems:
        raise td.ContaminationError("leakage check failed — " + "; ".join(problems))
    return {"passed": True, "strict_test_docs": strict_test_docs, **res,
            "held_out": {"test_claims": len(set(test_ids)), "tuning_claims": len(tuning),
                         "tuning_doc_overlap_claims": len(overlap_tuning),
                         "tuning_docs": len(tuning_docs), "test_docs": len(tdocs)}}


def tuning_ids_from_run(sample: Collection[str], path: Path = td.TUNING_RUN) -> set[str]:
    """The held-out claims, re-derived independently of the splits being checked: the
    100-claim seeded sample Ling's train run drew (`sample`) plus every row in the run
    file. Refuses a run that is not on the train split or has rows outside the sample
    (the run answered 99 of its 100; the skipped one is held out all the same)."""
    blob = json.loads(Path(path).read_text())
    rows = {str(r["query_id"]) for r in blob["rows"]}
    if (blob.get("run") or {}).get("dataset") != td.TRAIN_DATASET or len(set(sample)) != 100 or rows - set(sample):
        raise td.ContaminationError(f"{path} is not a run over the 100-claim {td.TRAIN_DATASET} sample")
    return rows | set(sample)


def notebook_cells(source: str) -> list[dict]:
    """`# %%` cells -> ipynb cells; `# %% [markdown]` cells lose their leading '# '."""
    cells: list[dict] = []
    for block in re.split(r"(?m)^# %%", source):
        if not block.strip():
            continue
        head, _, body = block.partition("\n")
        if head.strip() == "[markdown]":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip("\n").splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": _lines(body.strip("\n").splitlines())})
    return cells


def _lines(lines: Sequence[str]) -> list[str]:
    return [ln + "\n" for ln in lines[:-1]] + list(lines[-1:])


def notebook_json(source: str) -> str:
    nb = {"cells": notebook_cells(source), "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def render_notebook(shas: Mapping[str, str]) -> str:
    src = NOTEBOOK_SRC.read_text()
    if src.count(SHA_PLACEHOLDER) != 1:
        raise RuntimeError(f"{NOTEBOOK_SRC} lost its sha256 placeholder line")
    return src.replace(SHA_PLACEHOLDER, f"EXPECTED_SHA256: dict = {json.dumps(dict(shas), sort_keys=True)}")


def export(out: Path = OUT, cfg=None, log=print) -> dict:
    from app.eval.rag_eval import _git_sha
    from app.ingest.corpus import load_claim_labels, load_documents, load_queries_qrels
    from app.verify import train as tr

    cfg = cfg or tr.TrainConfig()
    splits = tr.load_train_splits(cfg)
    rows = {"train": [row(p) for p in splits["train"]], "val": [row(p) for p in splits["val"]]}

    queries, qrels = load_queries_qrels(td.TRAIN_DATASET)
    labels = load_claim_labels(td.TRAIN_DATASET, query_ids=set(queries))
    t_queries, t_qrels = load_queries_qrels(td.TEST_DATASET)
    t_labels = load_claim_labels(td.TEST_DATASET, query_ids=set(t_queries))
    t_docs = set().union(*(td.cited_docs(q, t_labels, t_qrels) for q in t_queries))
    from app.eval.rag_eval import SEED, sample_claims

    tuning = tuning_ids_from_run(sample_claims(queries, 100, seed=SEED))
    leak = check_leakage(rows, test_ids=set(t_queries), test_texts=list(t_queries.values()),
                         test_docs=t_docs, tuning_ids=tuning, labels=labels, qrels=qrels,
                         strict_test_docs=cfg.exclude_test_doc_overlap)
    if len(t_queries) != 300:
        raise td.ContaminationError(f"expected the 300 test claims, got {len(t_queries)}")

    all_claims = [*queries.values(), *t_queries.values()]
    if len(all_claims) != 1109:
        raise td.ContaminationError(f"expected the 1,109 BEIR SciFact claims, got {len(all_claims)}")
    hashes = build_hashes(all_claims, load_documents(), EXTERNAL_BLOCKLIST_TEXTS)
    hash_bytes = (json.dumps(hashes, indent=0) + "\n").encode()

    blobs = {name: jsonl_bytes(rows[name.split(".")[0]]) for name in FILES}
    for name, data in {**blobs, HASH_FILE: hash_bytes}.items():
        if hits := find_secrets(data):
            raise RuntimeError(f"{name} matches a secret pattern ({hits}); refusing to export")
    shas = {name: sha256(data) for name, data in blobs.items()}
    aux_shas = {HASH_FILE: sha256(hash_bytes)}
    meta = {
        "kind": KIND,
        "version": VERSION,
        "dataset": td.TRAIN_DATASET,
        "git_sha": _git_sha(),
        "base_model": cfg.base_model,
        "train_config": asdict(cfg),
        "config_signature": cfg.signature(),
        "labels": list(td.LABELS),
        "label_ids": dict(tr.LABEL_IDS),
        "pair_order": "claim_first",
        "class_weights": splits["report"]["class_weights"],
        "files": {n: {"sha256": shas[n], "rows": len(rows[n.split(".")[0]]), "bytes": len(blobs[n])}
                  for n in FILES},
        "aux_files": {HASH_FILE: {"sha256": aux_shas[HASH_FILE], "bytes": len(hash_bytes),
                                  **{k: len(hashes[k]) for k in ("claims", "titles", "abstract_prefixes", "blocklist")},
                                  "source_claims": len(all_claims)}},
        "report": splits["report"],
        "leakage_check": leak,
        "splits": {k: splits[k] for k in tr.SPLIT_KEYS if k in splits},
    }
    meta_bytes = (json.dumps(meta, indent=1) + "\n").encode()
    if hits := find_secrets(meta_bytes):
        raise RuntimeError(f"meta.json matches a secret pattern ({hits}); refusing to export")

    data_dir = out / DATA_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        (data_dir / name).write_bytes(data)
    (data_dir / HASH_FILE).write_bytes(hash_bytes)
    (data_dir / "meta.json").write_bytes(meta_bytes)
    script = render_notebook({**shas, **aux_shas})
    (out / "train_verifier.py").write_text(script)
    (out / "train_verifier.ipynb").write_text(notebook_json(script))
    for name, want in {**shas, **aux_shas}.items():  # re-read what landed on disk
        if sha256((data_dir / name).read_bytes()) != want:
            raise RuntimeError(f"{data_dir / name} changed while writing")
    r = splits["report"]
    log(f"train: {r['train']['pairs']} pairs / {r['train']['claims']} claims {r['train']['by_label']}")
    log(f"val:   {r['val']['pairs']} pairs / {r['val']['claims']} claims {r['val']['by_label']}")
    log(f"excluded: {r['tuning_claims']} tuning + {r['excluded_doc_overlap']} tuning-doc-overlap + "
        f"{r['excluded_test_duplicate']} test-duplicate + {r['excluded_test_doc_overlap']} test-doc-overlap claims")
    log(f"leakage check passed: {json.dumps({k: v for k, v in leak.items() if k != 'held_out'})}")
    log(f"contamination hashes: {len(hashes['claims'])} claims (of {len(all_claims)}), {len(hashes['titles'])} titles, "
        f"{len(hashes['abstract_prefixes'])} abstract prefixes, {len(hashes['blocklist'])} blocklisted external claims")
    for name, h in {**shas, **aux_shas}.items():
        log(f"{data_dir / name}  sha256 {h}")
    log(f"wrote {data_dir}/meta.json, {out}/train_verifier.py, {out}/train_verifier.ipynb")
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.verify.kaggle_export", description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep-test-doc-overlap", action="store_true",
                    help="keep train claims that share a doc with a test claim (SciFact's official split)")
    a = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    from app.verify import train as tr

    export(Path(a.out), tr.TrainConfig(exclude_test_doc_overlap=not a.keep_test_doc_overlap))


if __name__ == "__main__":
    main()
