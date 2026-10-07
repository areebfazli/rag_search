"""Paired comparison of two rag_eval runs: exact McNemar tests on per-claim outcomes.

Two rag.json files scored on the SAME claims are paired by query_id, and for each binary
per-claim outcome the discordant pairs decide the test:

* **verdict correct** — predicted_label == gold_label (the 3-class verdict accuracy);
* **abstention correct** (rationale oracle) — answered exactly when a rationale doc was
  retrieved: answered_with_evidence or correct_abstention (each run's own ``evidence``
  flag, so a retrieval change between the runs is part of what is compared);
* **has a citation** — at least one cited doc id.

b = A right / B wrong, c = A wrong / B right. The exact two-sided McNemar p-value is the
binomial tail on the b + c discordant pairs at p = 1/2: min(1, 2 * P[X <= min(b, c)]),
hand-rolled with integer binomial coefficients (no scipy); b = c = 0 gives p = 1.0.
Δ is B minus A.

A paired test is only meaningful on one claim set, so differing sets are refused unless
``--intersect`` is passed — then only the common claims are scored and the report says
how many were dropped from each side. The settings that differ between the runs
(models, prompt hashes, dataset, sample size, ...) are listed, so a delta can be read
against what actually changed.

``--labels audit`` (or ``audit-excl-debatable``) scores both runs against the SciFact
label-audit corrections instead of the original gold labels (app.eval.label_audit: same
predictions, the 11 corrected claims relabelled — and, for the second, the 8 debatable
claims dropped); the report says which answer key was used.

Run:
    uv run python -m app.eval.rag_compare A.json B.json [--intersect] [--labels audit] [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

MAX_P = 0.05
# --labels choice -> label_audit tier (None: the rows' own gold labels).
LABEL_TIERS = {"original": None, "audit": "corrected_strict",
               "audit-excl-debatable": "corrected_excl_debatable"}

# (key, label, outcome). Each outcome is True when the run got that claim "right".
OUTCOMES: tuple[tuple[str, str, Callable[[Mapping], bool]], ...] = (
    ("verdict_correct", "Verdict correct (predicted == gold)",
     lambda r: r["predicted_label"] == r["gold_label"]),
    ("abstention_correct", "Abstention correct (rationale oracle)",
     lambda r: bool(r["answered"]) == bool(r["evidence"])),
    ("has_citation", "Has >= 1 citation", lambda r: bool(r.get("cited_doc_ids"))),
)
REQUIRED_ROW_KEYS = ("query_id", "gold_label", "predicted_label", "answered", "evidence")



def _run(key: str) -> Callable[[Mapping], object]:
    """run.<key>, falling back to the top-level copy rag_eval also writes (top_k,
    sample_seed, the models and providers) for a file whose run block lacks it."""
    return lambda b: b["run"][key] if key in b["run"] else b.get(key)


# Run settings reported side by side: (label, getter over the rag.json blob). Every
# run-block key that can change a scored outcome is here. Deliberately NOT compared:
# prose that restates a listed setting (oracle_definition, legacy_oracle[_definition],
# answered_source, label_source.loader/file, reask.judged — the legacy qrels oracle and
# the judging note do not enter any outcome below), values derived from listed ones
# (default_models, canonical), and bookkeeping that cannot move a result (throttle_s,
# reask_cache — a path — and reask_replies, the cached/fetched/failed counts).
SETTINGS: tuple[tuple[str, Callable[[Mapping], object]], ...] = (
    ("dataset", lambda b: b["run"].get("dataset")
     or (b["run"].get("label_source") or {}).get("dataset")),
    ("n (scored claims)", lambda b: b.get("n")),
    ("n_requested", lambda b: b["run"].get("n_requested")),
    ("n_sample", lambda b: b["run"].get("n_sample")),
    ("eval_limit", lambda b: b["run"].get("eval_limit")),
    ("sample_seed", _run("sample_seed")),
    # The SciFact label file the gold labels were read from.
    ("label_source.source_zip_sha256",
     lambda b: (b["run"].get("label_source") or {}).get("source_zip_sha256")),
    ("oracle", lambda b: b["run"].get("oracle")),
    ("generator_provider", _run("generator_provider")),
    ("generator_model", _run("generator_model")),
    ("generator_base_url", lambda b: b["run"].get("generator_base_url")),
    ("judge_provider", _run("judge_provider")),
    ("judge_model", _run("judge_model")),
    ("judge_base_url", lambda b: b["run"].get("judge_base_url")),
    ("prompt_hash", lambda b: b["run"].get("prompt_hash")),
    ("judge_prompt_hash", lambda b: b["run"].get("judge_prompt_hash")),
    ("mode", lambda b: b["run"].get("mode")),
    ("top_k", _run("top_k")),
    ("reranker_model", lambda b: b["run"].get("reranker_model")),
    ("generator_max_completion_tokens", lambda b: b["run"].get("generator_max_completion_tokens")),
    ("generator_reasoning_effort", lambda b: b["run"].get("generator_reasoning_effort")),
    # What was actually sent: the reasoning parameter, the temperature (None = not sent)
    # and the provider routing / extra request body of each role.
    ("generator_reasoning_param", lambda b: b["run"].get("generator_reasoning_param")),
    ("generator_temperature", lambda b: b["run"].get("generator_temperature")),
    ("generator_extra_body", lambda b: b["run"].get("generator_extra_body")),
    ("judge_extra_body", lambda b: b["run"].get("judge_extra_body")),
    # The verdict-only re-ask changes predicted labels: whether it ran, its prompt and
    # budget. (A run from before the re-ask has no run.reask block: all three are None.)
    ("reask.enabled", lambda b: (b["run"].get("reask") or {}).get("enabled")),
    ("reask.prompt_hash", lambda b: (b["run"].get("reask") or {}).get("prompt_hash")),
    ("reask.max_tokens", lambda b: (b["run"].get("reask") or {}).get("max_tokens")),
    ("reask.trigger", lambda b: (b["run"].get("reask") or {}).get("trigger")),
    # Opt-in experiment switches (rag_eval): the product prompt variant, the
    # self-consistency vote's k, and a run with no judge (its no-verdict rows score
    # NO_VERDICT). A file written before they existed ran the defaults: "default", 1, False.
    ("prompt_variant", lambda b: b["run"].get("prompt_variant") or "default"),
    ("votes.k", lambda b: (b["run"].get("votes") or {}).get("k", 1)),
    ("judge_skipped", lambda b: bool(b["run"].get("judge_skipped"))),
    # True for a rag_rescore output (same replies, re-read by a newer parser).
    ("rescored", lambda b: b["run"].get("rescored")),
    ("git_sha", lambda b: b["run"].get("git_sha")),
    # rag_eval's resume-checkpoint key: a hash over the settings that define a run.
    ("checkpoint_signature", lambda b: b["run"].get("checkpoint_signature")),
)


class CompareError(ValueError):
    """The two runs can't be paired (bad file, or differing claim sets)."""


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: 2 * P[Binom(b + c, 1/2) <= min(b, c)], capped
    at 1. Integer arithmetic until the final division, so it is exact to float precision
    for any n; b = c = 0 (no discordant pairs) is 1.0."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def load_run(path: str | Path) -> dict:
    """A rag.json with provenance and per-claim rows; CompareError otherwise."""
    try:
        blob = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise CompareError(f"{path}: cannot read as JSON ({type(e).__name__})") from None
    if not (isinstance(blob, dict) and isinstance(blob.get("rows"), list)):
        raise CompareError(f"{path}: not a rag_eval rag.json (no `rows` list)")
    if not isinstance(blob.get("run"), dict):
        raise CompareError(f"{path}: no `run` provenance block")
    seen: set[str] = set()
    for r in blob["rows"]:
        if not isinstance(r, dict) or (missing := [k for k in REQUIRED_ROW_KEYS if k not in r]):
            raise CompareError(f"{path}: a row lacks {missing if isinstance(r, dict) else 'shape'}")
        if r["query_id"] in seen:
            raise CompareError(f"{path}: duplicate query_id {r['query_id']!r}")
        seen.add(r["query_id"])
    return blob


def pair_rows(a: Mapping, b: Mapping, intersect: bool = False) -> tuple[list[tuple[dict, dict]], dict]:
    """[(row_a, row_b)] by query_id, in A's order, plus a description of the pairing.
    Refuses differing claim sets unless `intersect`; refuses a claim whose gold label
    differs between the files (they cannot be the same claim)."""
    ra = {r["query_id"]: r for r in a["rows"]}
    rb = {r["query_id"]: r for r in b["rows"]}
    only_a, only_b = sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))
    if (only_a or only_b) and not intersect:
        raise CompareError(
            f"claim sets differ: {len(only_a)} only in A, {len(only_b)} only in B "
            f"({len(set(ra) & set(rb))} common). A paired test needs the same claims; pass "
            f"--intersect to score only the common ones."
        )
    common = [q for q in ra if q in rb]
    if not common:
        raise CompareError("the two runs share no claims")
    if clash := [q for q in common if ra[q]["gold_label"] != rb[q]["gold_label"]]:
        raise CompareError(f"gold labels differ for {len(clash)} shared query ids, e.g. {clash[:3]}")
    return [(ra[q], rb[q]) for q in common], {
        "n_paired": len(common),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "intersected": bool(only_a or only_b),
    }


def compare_outcome(pairs: Sequence[tuple[Mapping, Mapping]], outcome: Callable[[Mapping], bool]) -> dict:
    """The 2x2 table, accuracies, Δ (B - A) and the exact McNemar p for one outcome."""
    both = only_a = only_b = neither = 0
    for x, y in pairs:
        ox, oy = outcome(x), outcome(y)
        if ox and oy:
            both += 1
        elif ox:
            only_a += 1
        elif oy:
            only_b += 1
        else:
            neither += 1
    n = len(pairs)
    acc_a, acc_b = (both + only_a) / n, (both + only_b) / n
    return {
        "n": n,
        "both": both,
        "b": only_a,  # A right, B wrong
        "c": only_b,  # A wrong, B right
        "neither": neither,
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "delta": round(acc_b - acc_a, 4),
        "p": mcnemar_exact(only_a, only_b),
    }


def settings_diff(a: Mapping, b: Mapping) -> tuple[list[tuple[str, object, object]], list[str]]:
    """([(setting, A, B)] that differ, [settings that are identical])."""
    differ, same = [], []
    for label, get in SETTINGS:
        va, vb = get(a), get(b)
        if va != vb:
            differ.append((label, va, vb))
        else:
            same.append(label)
    return differ, same


def compare(a: Mapping, b: Mapping, intersect: bool = False) -> dict:
    pairs, pairing = pair_rows(a, b, intersect)
    differ, same = settings_diff(a, b)
    return {
        "pairing": pairing,
        "outcomes": {key: compare_outcome(pairs, fn) for key, _, fn in OUTCOMES},
        "settings_differ": differ,
        "settings_same": same,
    }


def _v(v: object) -> str:
    return "—" if v is None else f"`{v}`"


def to_markdown(result: Mapping, name_a: str, name_b: str) -> str:
    p = result["pairing"]
    pairing = f"{p['n_paired']} claims paired by query_id"
    if p["intersected"]:
        pairing += (
            f" — INTERSECTION only: {p['only_a']} claims only in A and {p['only_b']} only in B "
            f"were dropped"
        )
    lines = [
        "# RAG run comparison — exact McNemar on paired claims",
        "",
        f"A = `{name_a}`  ",
        f"B = `{name_b}`",
        "",
        f"{pairing}. Δ = B − A; b = A right & B wrong, c = A wrong & B right; exact two-sided "
        f"McNemar (binomial on b + c), significant at p < {MAX_P}.",
        "",
        *([f"Answer key: **{result['labels']}** — both runs re-scored against the SciFact "
           f"label-audit corrections (app.eval.label_audit), not the original gold labels.", ""]
          if result.get("labels", "original") != "original" else []),
        "| Outcome | A | B | Δ | b | c | both | neither | p | Significant |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, label, _ in OUTCOMES:
        o = result["outcomes"][key]
        lines.append(
            f"| {label} | {o['acc_a']:.4f} | {o['acc_b']:.4f} | {o['delta']:+.4f} | {o['b']} | "
            f"{o['c']} | {o['both']} | {o['neither']} | {o['p']:.4f} | "
            f"{'yes' if o['p'] < MAX_P else 'no'} |"
        )
    lines += ["", "## Settings that differ", ""]
    if result["settings_differ"]:
        lines += ["| Setting | A | B |", "|---|---|---|"]
        lines += [f"| {k} | {_v(va)} | {_v(vb)} |" for k, va, vb in result["settings_differ"]]
    else:
        lines.append("None of the recorded settings differ.")
    lines += ["", f"Identical: {', '.join(result['settings_same']) or 'none'}.", ""]
    return "\n".join(lines)


def relabel_runs(a: Mapping, b: Mapping, tier: str) -> tuple[dict, dict]:
    """Both runs with rows re-scored under a label_audit tier. Imported lazily: it loads the
    SciFact labels and the pinned audit files, which the default comparison never needs."""
    from app.eval import label_audit
    from app.ingest.corpus import load_claim_labels

    audit = label_audit.load_audit()
    labels = load_claim_labels(label_audit.AUDIT_DATASET)
    return (label_audit.relabel_blob(a, labels, audit, tier),
            label_audit.relabel_blob(b, labels, audit, tier))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.eval.rag_compare", description=__doc__.split("\n")[0])
    ap.add_argument("a", help="baseline rag.json (A)")
    ap.add_argument("b", help="candidate rag.json (B)")
    ap.add_argument("--intersect", action="store_true",
                    help="score only the common claims when the claim sets differ")
    ap.add_argument("--labels", choices=tuple(LABEL_TIERS), default="original",
                    help="answer key: the runs' own gold labels (default), or the SciFact label "
                         "audit's corrections (strict, or with debatable claims excluded)")
    ap.add_argument("--out", help="also write the markdown report to this path")
    args = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        a, b = load_run(args.a), load_run(args.b)
        if (tier := LABEL_TIERS[args.labels]) is not None:
            a, b = relabel_runs(a, b, tier)
        result = compare(a, b, intersect=args.intersect)
    except (CompareError, ValueError) as e:
        raise SystemExit(f"rag_compare: {e}") from None
    result["labels"] = args.labels
    md = to_markdown(result, args.a, args.b)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
