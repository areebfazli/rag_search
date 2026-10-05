"""Repo-anchored paths for the committed eval artifact, and the guard that keeps every
other output out of it.

``eval/results/`` is the committed headline artifact. Its path is anchored to the repo
root (this file's location), not the working directory, so a harness started from
another directory still writes its canonical output to — and its guards still protect —
the repo's eval/results/, never a stray ``./eval/results`` under the cwd.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "eval" / "results"


def _stat(p: Path) -> os.stat_result | None:
    try:
        return p.stat()
    except OSError:
        return None


def is_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    """True if writing to `path` would land at `root` or anywhere below it.

    `path` is resolved first (relative to the cwd, every existing symlink followed —
    including a dangling one at `path` itself, which resolve() follows to its target —
    and every ``..`` collapsed), then compared two ways:

    * lexically, against the resolved `root`;
    * by identity: if any existing ancestor of the resolved path is the same directory
      as `root` (same device and inode), so a case-insensitive filesystem
      (``Eval/Results``), a bind mount or any other alias of the same directory is
      caught too, not only its canonical spelling.

    A regular file at `path` with more than one hard link is also refused when one of
    its links is under `root`: writing through it would change the committed file.
    """
    root_r = Path(root).resolve()
    target = Path(path).resolve()
    if target == root_r or root_r in target.parents:
        return True
    root_st = _stat(root_r)
    if root_st is None:  # root does not exist: nothing else can alias it
        return False
    for p in (target, *target.parents):
        if (st := _stat(p)) is not None and os.path.samestat(st, root_st):
            return True
    st = _stat(target)
    if st is not None and stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
        for f in root_r.rglob("*"):
            if (fst := _stat(f)) is not None and os.path.samestat(st, fst):
                return True
    return False


def is_under_results(path: str | os.PathLike[str], results: str | os.PathLike[str] | None = None) -> bool:
    """`is_within(path, results)`, `results` defaulting to the repo's eval/results/."""
    return is_within(path, RESULTS if results is None else results)


def display_path(path: str | os.PathLike[str]) -> str:
    """`path` as recorded in a report: an absolute path inside the repo becomes
    repo-relative (so a committed artifact never carries the local checkout's absolute
    location); anything else is kept as given."""
    p = Path(path)
    if p.is_absolute():
        try:
            return p.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            pass
    return str(p)


def assert_outside(path: Path, root: str | os.PathLike[str] | None = None) -> Path:
    """`path`, unless it lands at or below `root` (default: the repo's eval/results/) —
    for a NON-canonical output directory, which must never reach the committed artifact
    (e.g. through a data/eval_runs symlinked into it)."""
    root = RESULTS if root is None else root
    if is_within(path, root):
        raise SystemExit(f"refusing to write a non-canonical run to {path}: it is inside {root}")
    return path
