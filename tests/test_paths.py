"""The repo-anchored eval/results/ guard (app.core.paths) and the harness guards built on
it, exercised from a working directory OUTSIDE the repo. Pure filesystem, no network."""
import json
import os

import pytest

from app.core import paths
from app.core.paths import REPO_ROOT, RESULTS, assert_outside, display_path, is_within
from app.eval import judge_agreement, rag_eval, rag_rescore, rag_secondlook, retrieval_eval, verify_combine, web_eval
from app.eval.verify_eval import UnsafeOutputError


def test_results_is_anchored_to_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert REPO_ROOT == paths.Path(paths.__file__).resolve().parents[2]
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert RESULTS == REPO_ROOT / "eval" / "results" and RESULTS.is_absolute()
    for mod in (rag_eval, judge_agreement, retrieval_eval, web_eval):
        assert mod.OUT == RESULTS
    assert judge_agreement.SOURCE == RESULTS / "rag.json"


@pytest.fixture
def links(tmp_path, monkeypatch):
    """cwd = tmp_path (outside the repo), with symlinks leading into the repo's results."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "resdir").symlink_to(RESULTS, target_is_directory=True)
    (tmp_path / "evaldir").symlink_to(RESULTS.parent, target_is_directory=True)
    (tmp_path / "ragfile").symlink_to(RESULTS / "rag.json")
    (tmp_path / "dangling").symlink_to(RESULTS / "never_created_by_tests.json")
    return tmp_path


def refused_paths(tmp_path):
    up = os.path.relpath(RESULTS, tmp_path)  # ../../…/eval/results from the cwd
    return [
        f"{RESULTS}/rag.json", f"{RESULTS}/", str(RESULTS), f"{REPO_ROOT}/./eval/results/rag.json",
        f"{RESULTS}/sub/dir/rag.json", f"{REPO_ROOT}/data/../eval/results/x.json",
        f"{up}/rag.json", f"{up}/../results/rag.json",
        "resdir/rag.json", "resdir", "ragfile", "dangling", "evaldir/results/x.json",
        f"{tmp_path}/resdir/deep/x.json",
    ]


def allowed_paths():
    # Relative to the cwd (tmp_path): these are NOT the repo's eval/results.
    return ["eval/results/rag.json", "eval/results", "./eval/results/x.json",
            "data/eval_runs/x/rag.json", "eval/results_other/rag.json",
            f"{REPO_ROOT}/eval/results_other/rag.json", f"{REPO_ROOT}/data/eval_runs/x/rag.json"]


def test_is_within_catches_every_route_into_the_repo_results(links):
    for p in refused_paths(links):
        assert is_within(p, RESULTS), p
    for p in allowed_paths():
        assert not is_within(p, RESULTS), p


def test_is_within_refuses_a_hard_link_to_a_committed_file(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "rag.json").write_text("{}")
    os.link(root / "rag.json", tmp_path / "alias.json")
    assert is_within(tmp_path / "alias.json", root)
    (tmp_path / "plain.json").write_text("{}")
    assert not is_within(tmp_path / "plain.json", root)
    assert not is_within(tmp_path / "x.json", tmp_path / "missing_root")


def test_rag_rescore_refuses_the_repo_results_from_another_cwd(links):
    for dst in refused_paths(links):
        with pytest.raises(SystemExit, match="refusing"):
            rag_rescore.main(["/nonexistent_src.json", dst])
    for dst in allowed_paths():  # passes the guard, then fails reading the missing source
        with pytest.raises(FileNotFoundError):
            rag_rescore.main(["/nonexistent_src.json", dst])


@pytest.mark.parametrize("mod", [rag_secondlook, verify_combine])
def test_eval_runs_outputs_refuse_the_repo_results_from_another_cwd(links, mod, monkeypatch):
    for p in refused_paths(links):
        with pytest.raises(UnsafeOutputError):
            mod.assert_safe_output(paths.Path(p))
    # data/eval_runs symlinked into the committed results: a runs path that resolves there.
    runs = links / "data" / "eval_runs"
    runs.parent.mkdir()
    runs.symlink_to(RESULTS, target_is_directory=True)
    with pytest.raises(UnsafeOutputError):
        mod.assert_safe_output(runs / "x")
    real = links / "real_runs"
    real.mkdir()
    monkeypatch.setattr("app.eval.verify_eval.RUNS", real)
    assert mod.assert_safe_output(real / "x") == real / "x"


def test_non_canonical_output_dirs_refuse_a_runs_dir_aliasing_the_results(links, monkeypatch):
    runs = links / "runs_alias"
    runs.symlink_to(RESULTS, target_is_directory=True)
    with pytest.raises(SystemExit, match="non-canonical"):
        assert_outside(runs / "rag_x")
    monkeypatch.setattr(rag_eval, "RUNS", runs)
    with pytest.raises(SystemExit, match="non-canonical"):
        rag_eval.output_dir("beir/scifact/train", 0, 5, "d0921f4e" * 8)
    monkeypatch.setattr(judge_agreement, "RUNS", runs)
    with pytest.raises(SystemExit, match="non-canonical"):
        judge_agreement.output_dir(3)
    # The canonical branch and an ordinary runs dir are untouched.
    monkeypatch.setattr(rag_eval, "RUNS", links / "runs")
    assert rag_eval.output_dir("beir/scifact/test", 0, 300, "d0921f4e" * 8, split_size=300) == (
        RESULTS, True)
    out, canonical = rag_eval.output_dir("beir/scifact/train", 0, 5, "d0921f4e" * 8)
    assert not canonical and out.parent == links / "runs"


def test_label_audit_refuses_non_canonical_runs_through_a_symlink(tmp_path, monkeypatch):
    from app.eval import label_audit

    other = tmp_path / "cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    out = tmp_path / "eval" / "results"
    out.mkdir(parents=True)
    monkeypatch.setattr(label_audit.rag_eval, "OUT", out)
    monkeypatch.setattr(label_audit, "load_audit", lambda fetch=True: None)
    monkeypatch.setattr(label_audit, "load_claim_labels", lambda ds: {})
    (out / "rag.json").write_text(json.dumps({"rows": [], "run": {"dataset": label_audit.AUDIT_DATASET}}))
    (other / "alias").symlink_to(out, target_is_directory=True)
    (other / "file_link.json").symlink_to(out / "rag.json")
    for src in ("alias/rag.json", "file_link.json", str(out / "rag.json")):
        with pytest.raises(SystemExit, match="only the canonical run"):
            label_audit.main([src])
    assert sorted(p.name for p in out.iterdir()) == ["rag.json"]


def test_display_path_is_repo_relative_inside_the_repo_only(tmp_path):
    assert display_path(RESULTS / "rag.json") == "eval/results/rag.json"
    assert display_path("data/eval_runs/x/rag.json") == "data/eval_runs/x/rag.json"
    assert display_path(tmp_path / "x.json") == str(tmp_path / "x.json")
