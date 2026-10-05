"""Re-score a finished rag.json with the CURRENT verdict parser — no LLM calls.

Every stored row keeps the answer text generate() served (a well-formed verdict line
stripped, everything else whole), so rows with no verdict can be parsed again
(rag_eval.reparse_row) and the aggregates recomputed. Generation, retrieval and the
judge's scores are untouched: the only thing that changes is how the same replies are
read. The output never goes to eval/results/ — it is refused — and its ``run`` block says
what it was re-scored from, so rag_compare can pair it against the original.

Run:
    uv run python -m app.eval.rag_rescore eval/results/rag.json data/eval_runs/<name>/rag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.core.paths import is_within
from app.eval import rag_eval
from app.ingest.corpus import load_queries_qrels


def rescore(blob: dict, queries: dict[str, str]) -> dict:
    """The blob with every row re-parsed and the aggregates rebuilt from them."""
    rows = [rag_eval.reparse_row(r, queries[r["query_id"]]) for r in blob["rows"]]
    run = blob.get("run") or {}
    agg = rag_eval.aggregate(
        rows,
        run.get("generator_model"),
        run.get("judge_model"),
        run.get("generator_provider"),
        run.get("judge_provider"),
    )
    kept = {k: blob[k] for k in ("skipped", "skip_reasons", "judge_parse_failures",
                                 "judge_calls", "cost") if k in blob}
    return {
        **agg,
        **kept,
        "run": {**run, "canonical": False, "rescored": True},
        "rows": rows,
    }


def is_under_results(dst: Path) -> bool:
    """True if `dst` lands anywhere under eval/results/ (rag_eval.OUT, anchored to the repo
    root — not the cwd) — directly or in a nested subdir — once resolved, so absolute
    paths, `..` segments, symlinked directories or files (dangling ones too) and other
    aliases of the same directory are all caught (app.core.paths.is_within)."""
    return is_within(dst, rag_eval.OUT)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.eval.rag_rescore",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("src", help="a rag_eval rag.json")
    ap.add_argument("dst", help="where to write the re-scored rag.json (never eval/results/)")
    args = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    src, dst = Path(args.src), Path(args.dst)
    if is_under_results(dst):
        raise SystemExit(f"rag_rescore: refusing to write into {rag_eval.OUT}/ (the committed artifact)")
    blob = json.loads(src.read_text())
    queries, _ = load_queries_qrels(blob["run"]["dataset"])
    out = rescore(blob, queries)
    out["run"]["rescored_from"] = str(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    before = sum(r["predicted_label"] == r["gold_label"] for r in blob["rows"])
    print(
        f"{len(out['rows'])} rows; verdict_accuracy {blob.get('verdict_accuracy')} -> "
        f"{out['verdict_accuracy']} ({before} -> "
        f"{sum(r['predicted_label'] == r['gold_label'] for r in out['rows'])} correct); "
        f"verdict sources {out['verdict_sources']}\nWrote {dst}"
    )


if __name__ == "__main__":
    main()
