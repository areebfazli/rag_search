"""Retrieval evaluation on BEIR/SciFact gold qrels with ranx.

Runs each mode (bm25 / dense / hybrid / hybrid_rerank) over the test queries and
reports nDCG@10, Recall@100, MRR@10, MAP@100 — the headline artifact of the
project. Writes a Markdown table + JSON to eval/results/.

Run:
    uv run python -m app.eval.retrieval_eval
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.ingest.corpus import load_queries_qrels
from app.retrieve.service import SearchService

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
METRICS = ["ndcg@10", "recall@100", "mrr@10", "map@100"]
PRETTY = {"ndcg@10": "nDCG@10", "recall@100": "Recall@100", "mrr@10": "MRR@10", "map@100": "MAP@100"}
DEPTH = 100  # first-stage candidate depth = result depth (enough for Recall@100)

OUT = Path("eval/results")


def build_run(service: SearchService, queries: dict[str, str], mode: str) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for qid, qtext in queries.items():
        hits = service.retrieve(qtext, mode=mode, top_k=DEPTH, candidate_k=DEPTH)
        # ranx needs strictly-decreasing distinct scores to preserve rank order;
        # rank-derived scores guarantee that regardless of the raw score scale.
        run[qid] = {h.doc_id: float(len(hits) - rank) for rank, h in enumerate(hits)}
    return run


def to_markdown(rows: dict[str, dict[str, float]]) -> str:
    header = "| Config | " + " | ".join(PRETTY[m] for m in METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    lines = [header, sep]
    for mode in MODES:
        cells = " | ".join(f"{rows[mode][m]:.4f}" for m in METRICS)
        lines.append(f"| {mode} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    from ranx import Qrels, Run, evaluate

    queries, qrels_dict = load_queries_qrels()
    n_judg = sum(len(v) for v in qrels_dict.values())
    print(f"Eval set: {len(queries)} queries, {n_judg} relevance judgments")

    service = SearchService()
    qrels = Qrels(qrels_dict)

    rows: dict[str, dict[str, float]] = {}
    for mode in MODES:
        t0 = time.time()
        run = Run(build_run(service, queries, mode), name=mode)
        rows[mode] = evaluate(qrels, run, METRICS)
        line = "  ".join(f"{PRETTY[m]}={rows[mode][m]:.4f}" for m in METRICS)
        print(f"  {mode:16s} {line}   ({time.time() - t0:.1f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    table = to_markdown(rows)
    (OUT / "retrieval.md").write_text(
        f"# Retrieval evaluation — BEIR/SciFact ({len(queries)} test queries)\n\n{table}\n"
    )
    (OUT / "retrieval.json").write_text(json.dumps(rows, indent=2))
    print("\n" + table)
    print(f"\nWrote {OUT/'retrieval.md'} and {OUT/'retrieval.json'}")


if __name__ == "__main__":
    main()
