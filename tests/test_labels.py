import json
import zipfile

import pytest

from app.ingest.corpus import ClaimLabel, load_claim_labels, parse_claim_labels

# Shape of BEIR SciFact queries.jsonl: metadata maps cited doc id -> evidence sets.
ROWS = [
    {"_id": "1", "text": "NEI claim", "metadata": {}},
    {
        "_id": "2",
        "text": "supported claim, two cited docs",
        "metadata": {
            "10": [{"sentences": [0], "label": "SUPPORT"}],
            "11": [{"sentences": [2, 3], "label": "SUPPORT"}, {"sentences": [5], "label": "SUPPORT"}],
        },
    },
    {"_id": "3", "text": "contradicted", "metadata": {"12": [{"sentences": [4], "label": "CONTRADICT"}]}},
    {"_id": "4", "text": "train-only claim", "metadata": {}},
]


def _lines(rows):
    return [json.dumps(r) + "\n" for r in rows]


def test_labels_and_rationale_docs():
    labels = parse_claim_labels(_lines(ROWS))
    assert labels["1"] == ClaimLabel("NEI", set())
    assert labels["2"] == ClaimLabel("SUPPORT", {"10", "11"})
    assert labels["3"] == ClaimLabel("CONTRADICT", {"12"})


def test_query_ids_restrict_to_one_split():
    labels = parse_claim_labels(_lines(ROWS), query_ids={"1", "3"})
    assert set(labels) == {"1", "3"}


def test_split_id_missing_from_source_fails_loudly():
    with pytest.raises(ValueError, match="no entry"):
        parse_claim_labels(_lines(ROWS), query_ids={"1", "999"})


def test_mixed_support_contradict_is_rejected():
    mixed = {
        "_id": "5",
        "text": "mixed",
        "metadata": {
            "20": [{"sentences": [0], "label": "SUPPORT"}],
            "21": [{"sentences": [1], "label": "CONTRADICT"}],
        },
    }
    with pytest.raises(ValueError, match="claim 5"):
        parse_claim_labels(_lines([mixed]))


def test_blank_lines_and_bytes_are_accepted():
    lines = [line.encode() for line in _lines(ROWS[:1])] + [b"\n"]
    assert parse_claim_labels(lines) == {"1": ClaimLabel("NEI", set())}


def test_load_from_synthetic_zip(tmp_path):
    src = tmp_path / "source.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("scifact/queries.jsonl", "".join(_lines(ROWS)))
    labels = load_claim_labels(source_zip=src, query_ids={"2", "4"})
    assert labels == {"2": ClaimLabel("SUPPORT", {"10", "11"}), "4": ClaimLabel("NEI", set())}


def test_missing_zip_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="IR_DATASETS_HOME"):
        load_claim_labels(source_zip=tmp_path / "nope.zip", query_ids={"1"})
