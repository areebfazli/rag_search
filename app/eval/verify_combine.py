"""The fine-tuned verifier alone and combined with Ling's verdicts — no LLM calls at all.

Ling's answers are already stored (rag.json rows); this module only re-reads them and
scores the SAME top-5 passages Ling saw (each row's `retrieved_doc_ids`) with
app.verify.trained.TrainedVerifier, then applies one of five pre-declared rules to the
two signals. The verifier signal is its strongest non-NEI probability over the top-5
passages (``s``) and that label (``cand``):

  R0  Ling alone
  R1  verifier alone:  cand if s >= τ else NEI
  R2  Ling, but a SUPPORT/CONTRADICT verdict with s < τ_low becomes NEI   (over-eager fix)
  R3  Ling, but an NEI / no-verdict reply with s > τ_high becomes cand     (too-cautious fix)
  R4  R2 + R3 (their conditions are disjoint: they act on different Ling verdicts)

Stages, in the only order the contamination rule allows (`all`, the default, runs the
three in order; each writes the file the next one reads, tied to the checkpoint's hash):

  validate  the verifier's own validation split (train claims, grouped by doc):
            truncation vs sentence windows for abstracts over the length budget, by
            pair-level macro-F1 -> data/eval_runs/verify_trained_validation/validation.json.
  tune      the 100 held-out train claims Ling answered (rescored with the current
            verdict parser): every rule on a coarse τ grid (0.05 steps); the choice is
            frozen in data/eval_runs/verify_combined_tuning_train_100/frozen.json. Never
            touches test, and refuses to re-tune once a test output exists.
  test      ONCE, with frozen.json: the 300 test claims of the committed
            eval/results/rag.json (read only) -> data/eval_runs/verify_combined_<rule>_test_300/
            and, for reference, data/eval_runs/verify_trained_R1_test_300/ (rag.json,
            rag.md, and compare.md/json: rag_compare's paired McNemar vs eval/results/rag.json).
            Refuses to run untuned, with a different checkpoint than the one tuned on, or
            when the output already exists (pass --rerun to reproduce it on purpose).

Ling's rows are always read as the FIRST PASS left them (``ling_first_pass``, i.e.
rag_eval.first_pass_row on every row): the verdict-only re-ask that rag_eval now applies
as a post-step (and that the committed rag.json rows carry) is undone at load, for the
test run, the train tuning run and the rag_compare baseline alike. The combination rules
were tuned on, and the committed verify_combined_* / verify_trained_R1_* outputs were
scored against, Ling's pre-re-ask verdicts; on the committed rag.json the mapping
reproduces those rows exactly, so a --rerun still measures "verifier vs Ling's first
pass", not "verifier vs Ling + re-ask".

The model comes from data/models/verifier/model/ (the unzipped Kaggle output; see
app.verify.trained).

Selection on the tuning set (pre-declared): each grid point is scored by its 3-class
accuracy averaged over itself and its grid neighbours (±1 step on each τ), so a lone
spike loses to a plateau; ties go to the higher raw accuracy, then the rule order
R0 < R2 < R3 < R4 < R1 (Ling-anchored and fewer parameters first), then the point that
changes fewer of Ling's verdicts.

Run:
    make verify-combine                 # validate -> tune -> test, once
    make verify-combine STAGE=validate  # or one stage at a time
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.paths import RESULTS, is_within
from app.eval.rag_eval import LABELS, TOP_K, _abstention_class, first_pass_row
from app.eval.verify_eval import (
    LABEL_TO_VERDICT,
    RUNS,
    UnsafeOutputError,
    class_scores,
    summarize,
)
from app.eval.verify_eval import assert_safe_output as _assert_safe_output
from app.verify import nli
from app.verify.nli import NON_NEI

def assert_safe_output(path: Path) -> Path:
    """verify_eval.assert_safe_output (strictly inside data/eval_runs/), checked against
    the repo-anchored eval/results/ rather than a cwd-relative one, and refusing any alias
    of it that a symlink, `..` or absolute path could reach (app.core.paths.is_within)."""
    if is_within(path, RESULTS):
        raise UnsafeOutputError(f"{path} is inside the committed {RESULTS}")
    return _assert_safe_output(path, forbidden=RESULTS)


TRAIN_DATASET = "beir/scifact/train"
TEST_DATASET = "beir/scifact/test"
TEST_RUN = Path("eval/results/rag.json")  # read only: Ling's committed test answers
RULE_ORDER = ("R0", "R2", "R3", "R4", "R1")  # tie-break preference
PARAMS: dict[str, tuple[str, ...]] = {
    "R0": (), "R1": ("tau",), "R2": ("tau_low",), "R3": ("tau_high",), "R4": ("tau_low", "tau_high"),
}
GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95
K = TOP_K  # the verifier reads the same top-5 Ling was given

N_TUNING = 100
N_TEST = 300
VALIDATION_DIR = RUNS / "verify_trained_validation"
TUNING_DIR = RUNS / f"verify_combined_tuning_train_{N_TUNING}"
FROZEN_PATH = TUNING_DIR / "frozen.json"  # written by `tune`, the only input `test` takes
R1_TEST_DIR = RUNS / f"verify_trained_R1_test_{N_TEST}"
FROZEN_KEYS = ("windowing", "rule", "params", "r1_tau", "checkpoint_sha", "tuned_on")


def ling_first_pass(blob: Mapping) -> dict:
    """A Ling rag.json with the verdict re-ask undone: every row through
    rag_eval.first_pass_row (a no-op on rows that never carried it), and the run's re-ask
    settings dropped so a combined run block copied from it does not claim a re-ask its
    rows no longer have. Only rows and run are rebuilt; nothing here reads the file's
    top-level aggregates."""
    rows = [first_pass_row(r) for r in blob["rows"]]
    run = dict(blob.get("run") or {})
    if any(k.startswith("reask") for k in run) or any(
            "first_pass" in r or "reask_attempted" in r for r in blob["rows"]):
        run = {k: v for k, v in run.items() if not k.startswith("reask")}
        run["ling_rows"] = "first_pass: verdict re-ask undone (rag_eval.first_pass_row)"
    return {**blob, "run": run, "rows": rows}


def test_dir(rule: str) -> Path:
    return RUNS / f"verify_combined_{rule}_test_{N_TEST}"


def test_outputs_exist() -> list[Path]:
    return [p for p in (*(test_dir(r) for r in RULE_ORDER), R1_TEST_DIR) if (p / "rag.json").exists()]


def load_frozen(path: Path | None = None) -> dict:
    """The tuned choice; SystemExit when `tune` has not run (or wrote something else)."""
    path = path or FROZEN_PATH
    try:
        f = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise SystemExit(f"not tuned: no {path} — run the validate and tune stages on train first") from None
    if not isinstance(f, dict) or any(f.get(k) is None for k in FROZEN_KEYS) or f["rule"] not in PARAMS:
        raise SystemExit(f"not tuned: {path} lacks {[k for k in FROZEN_KEYS if f.get(k) is None]}")
    return f


def load_windowing(sha: str, path: Path | None = None) -> str:
    """The validate stage's choice, for this checkpoint only."""
    path = path or VALIDATION_DIR / "validation.json"
    try:
        v = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise SystemExit(f"no {path}: run the validate stage first") from None
    if v.get("checkpoint_sha") != sha:
        raise SystemExit(f"{path} was chosen for checkpoint {str(v.get('checkpoint_sha'))[:12]}, "
                         f"not {sha[:12]}: re-run the validate stage")
    return v["chosen_windowing"]


# --- the two signals and the rules ------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    strength: float  # max non-NEI probability over the top-k passages
    candidate: str  # its label (SUPPORT or CONTRADICT)
    index: int | None  # passage index behind it (None when there are no passages)


def verifier_signal(passage_probs: Sequence[Mapping[str, float]], k: int = K) -> Signal:
    d = nli.aggregate_claim(passage_probs, tau=0.0, rule="max", k=k)
    return Signal(d.confidence, d.candidate, d.index)


def rule_r1(ling: str, sig: Signal, tau: float) -> str:
    return sig.candidate if sig.strength >= tau else "NEI"


def rule_r2(ling: str, sig: Signal, tau_low: float) -> str:
    return "NEI" if ling in NON_NEI and sig.strength < tau_low else ling


def rule_r3(ling: str, sig: Signal, tau_high: float) -> str:
    return sig.candidate if ling not in NON_NEI and sig.strength > tau_high else ling


def rule_r4(ling: str, sig: Signal, tau_low: float, tau_high: float) -> str:
    return rule_r2(ling, sig, tau_low) if ling in NON_NEI else rule_r3(ling, sig, tau_high)


def apply_rule(rule: str, ling: str, sig: Signal, params: Mapping[str, float] | None = None) -> str:
    """The final label for one claim. `ling` is Ling's predicted label (SUPPORT /
    CONTRADICT / NEI, or NONE for an answer with no parseable verdict)."""
    p = dict(params or {})
    if set(p) != set(PARAMS[rule]):
        raise ValueError(f"{rule} takes {PARAMS[rule]}, got {sorted(p)}")
    if rule == "R0":
        return ling
    if rule == "R1":
        return rule_r1(ling, sig, p["tau"])
    if rule == "R2":
        return rule_r2(ling, sig, p["tau_low"])
    if rule == "R3":
        return rule_r3(ling, sig, p["tau_high"])
    if rule == "R4":
        return rule_r4(ling, sig, p["tau_low"], p["tau_high"])
    raise ValueError(f"unknown rule {rule!r}")


# --- tuning ------------------------------------------------------------------------------------


def grid_points(rule: str, grid: Sequence[float] = GRID) -> list[dict[str, float]]:
    names = PARAMS[rule]
    return [dict(zip(names, vals, strict=True)) for vals in itertools.product(grid, repeat=len(names))]


def score_point(rule, params, items, gold) -> dict:
    """items: {qid: (ling_label, Signal)}."""
    pred = {q: apply_rule(rule, ling, sig, params) for q, (ling, sig) in items.items()}
    qids = list(items)
    cs = class_scores([gold[q] for q in qids], [pred[q] for q in qids])
    return {"rule": rule, "params": dict(params), "accuracy": cs["accuracy"], "macro_f1": cs["macro_f1"],
            "correct": cs["correct"], "changed": sum(pred[q] != items[q][0] for q in qids),
            "per_label_accuracy": {lab: cs["per_label"][lab]["recall"] for lab in LABELS}}


def _neighbours(params: Mapping[str, float], grid: Sequence[float]) -> list[dict[str, float]]:
    idx = {k: grid.index(v) for k, v in params.items()}
    out = []
    for deltas in itertools.product((-1, 0, 1), repeat=len(idx)):
        pos = {k: i + d for (k, i), d in zip(idx.items(), deltas, strict=True)}
        if all(0 <= i < len(grid) for i in pos.values()):
            out.append({k: grid[i] for k, i in pos.items()})
    return out


def tune_rules(items, gold, grid: Sequence[float] = GRID) -> dict:
    """Every rule at every grid point; the plateau-smoothed choice per rule and overall."""
    table: dict[str, list[dict]] = {}
    best_per_rule: dict[str, dict] = {}
    for rule in RULE_ORDER:
        pts = [score_point(rule, p, items, gold) for p in grid_points(rule, grid)]
        acc = {json.dumps(p["params"], sort_keys=True): p["accuracy"] for p in pts}
        for p in pts:
            nb = [acc[json.dumps(n, sort_keys=True)] for n in _neighbours(p["params"], list(grid))] or [p["accuracy"]]
            p["smoothed_accuracy"] = round(sum(nb) / len(nb), 4)
            p["neighbour_accuracy_range"] = [min(nb), max(nb)]
        table[rule] = pts
        best_per_rule[rule] = max(pts, key=lambda p: (p["smoothed_accuracy"], p["accuracy"], -p["changed"]))
    chosen = max(
        best_per_rule.values(),
        key=lambda p: (p["smoothed_accuracy"], p["accuracy"], -RULE_ORDER.index(p["rule"]), -p["changed"]),
    )
    return {"chosen": chosen, "best_per_rule": best_per_rule, "table": table, "grid": list(grid), "n": len(items)}


# --- rows ----------------------------------------------------------------------------------------


def passage_probs_block(doc_ids, scores) -> list[dict]:
    return [
        {"doc_id": d, "rank": i + 1, **{lab: round(s.probs[lab], 6) for lab in LABELS},
         "n_windows": s.n_windows, "window": s.window}
        for i, (d, s) in enumerate(zip(doc_ids, scores, strict=True))
    ]


def combined_row(ling_row: Mapping, scores: Sequence[nli.PairScore], rule: str, params: Mapping[str, float]) -> dict:
    """Ling's row with the rule's final label. Unchanged verdicts keep Ling's row as is
    (answer, citations); a changed one gets its verdict, answered flag and abstention
    classes recomputed, and cites the verifier's passage when it now asserts a label."""
    top = list(ling_row["retrieved_doc_ids"])
    sig = verifier_signal([s.probs for s in scores])
    ling = ling_row["predicted_label"]
    final = apply_rule(rule, ling, sig, params)
    row = dict(ling_row)
    chosen = top[sig.index] if sig.index is not None and sig.index < len(top) else None
    if final != ling:
        answered = final != "NEI"
        row.update(
            verdict=LABEL_TO_VERDICT[final],
            predicted_label=final,
            answered=answered,
            answered_source=f"combined:{rule}",
            abstention_class=_abstention_class(answered, row["evidence"]),
            abstention_class_qrels=_abstention_class(answered, row["evidence_qrels"]),
            cited_doc_ids=[chosen] if answered and chosen else [],
        )
    row.update(
        ling_predicted_label=ling,
        combination={"rule": rule, "params": dict(params)},
        changed=final != ling,
        verifier_candidate=sig.candidate,
        verifier_strength=round(sig.strength, 6),
        verifier_doc_id=chosen,
        passage_probs=passage_probs_block(top, scores),
    )
    return row


# --- scoring helpers ---------------------------------------------------------------------------


def load_verifier(windowing: str = "truncate"):
    from app.verify.trained import TrainedVerifier

    return TrainedVerifier(windowing=windowing, batch_size=1)


def score_rows(verifier, cache, rows: Sequence[Mapping], queries: Mapping[str, str], docs, log=print) -> dict[str, list[nli.PairScore]]:
    pairs, owner = [], []
    for r in rows:
        for d in r["retrieved_doc_ids"]:
            pairs.append((d, docs[d]["title"], docs[d]["text"], queries[r["query_id"]]))
            owner.append(r["query_id"])
    t0 = time.time()

    def progress(done, total):
        el = time.time() - t0
        log(f"    {done}/{total} new pairs ({el / 60:.1f} min, ~{el / done * (total - done) / 60:.0f} min left)")

    scores = nli.score_pairs_cached(verifier, cache, pairs, chunk=16, progress=progress)
    out: dict[str, list[nli.PairScore]] = {r["query_id"]: [] for r in rows}
    for q, s in zip(owner, scores, strict=True):
        out[q].append(s)
    return out


def run_block(ling_run: Mapping, verifier, rule: str, params: Mapping, source: str, dataset: str,
              frozen: Mapping | None = None) -> dict:
    from app.eval.rag_eval import _git_sha

    return {
        **{k: v for k, v in ling_run.items() if k not in ("canonical", "checkpoint_signature")},
        "git_sha": _git_sha(),
        "dataset": dataset,
        "generator_model": f"{ling_run.get('generator_model')} + {verifier.model_name} [{rule}]",
        "canonical": False,
        "combination": {"rule": rule, "params": dict(params), "k": K, "source": source,
                        "frozen": dict(frozen or {})},
        "verifier": {"model": verifier.model_name, "checkpoint": getattr(verifier, "checkpoint", None),
                     "checkpoint_sha": getattr(verifier, "checkpoint_hash", None),
                     "windowing": verifier.windowing, "max_length": verifier.max_length,
                     "pair_order": "claim_first"},
        "ling_run_git_sha": ling_run.get("git_sha"),
    }


def write_run(out: Path, rows: list[dict], run: dict, title: str) -> dict:
    assert_safe_output(out)
    agg = summarize(rows)
    agg["changed_vs_ling"] = sum(r.get("changed", False) for r in rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.json").write_text(json.dumps({**agg, "run": run, "rows": rows}, indent=2))
    lines = [
        f"# {title}", "",
        f"Rule `{run['combination']['rule']}` {run['combination']['params']} — {run['combination']['source']}.",
        f"Verifier `{run['verifier']['model']}` ({run['verifier']['windowing']}), top-{K} of Ling's own context.", "",
        "| Metric | Value |", "|---|---|",
        f"| Verdict accuracy | {agg['verdict_accuracy']} (95% CI {agg['verdict_accuracy_ci95']}) |",
        f"| Macro-F1 | {agg['macro_f1']} |",
        *[f"| Accuracy on gold {lab} | {agg['per_label_accuracy'][lab]} |" for lab in LABELS],
        f"| Verdicts changed vs Ling | {agg['changed_vs_ling']} |", "",
    ]
    (out / "rag.md").write_text("\n".join(lines))
    return agg


# --- stages --------------------------------------------------------------------------------------


def stage_validate(log=print) -> Path:
    """Windowing choice on the verifier's validation split (train claims only)."""
    from app.verify import train as tr
    from app.verify.trained import read_meta, resolve_checkpoint

    meta = read_meta(resolve_checkpoint())
    cfg = tr.TrainConfig(**meta["config"])
    splits = tr.load_train_splits(cfg)
    want = meta.get("val_qids")
    if want is None:  # a CPU checkpoint from before val_qids rode along in meta.json
        want = json.loads((tr.OUT_DIR / "splits.json").read_text())["val_qids"]
    if want != splits["val_qids"]:
        raise SystemExit("validation split differs from the one the model was trained with")
    val = splits["val"]
    out = assert_safe_output(VALIDATION_DIR)
    res: dict = {"n_pairs": len(val), "by_windowing": {}}
    verifier = load_verifier("truncate")
    cache = nli.PairCache(nli.cache_path_for(verifier.model_name, verifier.dtype))
    for w in ("truncate", "window"):
        verifier.windowing = w
        scores = nli.score_pairs_cached(verifier, cache, [(p.doc_id, p.title, p.abstract, p.claim) for p in val])
        pred = [max(LABELS, key=lambda lab, s=s: s.probs[lab]) for s in scores]
        cs = class_scores([p.label for p in val], pred)
        res["by_windowing"][w] = {"accuracy": cs["accuracy"], "macro_f1": cs["macro_f1"],
                                  "per_label": cs["per_label"],
                                  "long_pairs": sum(s.n_windows > 1 for s in scores)}
        log(f"  validation pairs, {w}: acc {cs['accuracy']} macro-F1 {cs['macro_f1']}")
    t, wv = res["by_windowing"]["truncate"], res["by_windowing"]["window"]
    res["chosen_windowing"] = "window" if (wv["macro_f1"], wv["accuracy"]) > (t["macro_f1"], t["accuracy"]) else "truncate"
    res["checkpoint_sha"] = verifier.checkpoint_hash
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(res, indent=2))
    log(f"chosen windowing: {res['chosen_windowing']} -> {out}/validation.json")
    return out


def _ling_train_rows() -> tuple[list[dict], dict, dict]:
    from app.eval.rag_rescore import rescore
    from app.ingest.corpus import load_queries_qrels
    from app.verify.train_data import TUNING_RUN

    blob = ling_first_pass(json.loads(TUNING_RUN.read_text()))  # before re-parsing
    if blob["run"]["dataset"] != TRAIN_DATASET:
        raise SystemExit(f"{TUNING_RUN} is not a {TRAIN_DATASET} run")
    queries, _ = load_queries_qrels(TRAIN_DATASET)
    re = rescore(blob, queries)  # the CURRENT verdict parser, no LLM
    return re["rows"], re["run"], queries


def stage_tune(rerun: bool = False, log=print) -> Path:
    from app.ingest.corpus import load_documents

    if (done := test_outputs_exist()) and not rerun:
        raise SystemExit(f"{done[0]} exists: re-tuning after the test split was scored would let "
                         "test results steer the choice (--rerun only to reproduce the same tuning)")
    rows, ling_run, queries = _ling_train_rows()
    if not N_TUNING - 5 <= len(rows) <= N_TUNING:  # the run answered 99 of its 100-claim sample
        raise SystemExit(f"expected Ling's answers to the {N_TUNING} held-out train claims, got {len(rows)}")
    docs = {d["doc_id"]: d for d in load_documents()}
    verifier = load_verifier()
    verifier.windowing = load_windowing(verifier.checkpoint_hash)
    cache = nli.PairCache(nli.cache_path_for(verifier.model_name, verifier.dtype))
    scored = score_rows(verifier, cache, rows, queries, docs, log)
    items = {r["query_id"]: (r["predicted_label"], verifier_signal([s.probs for s in scored[r["query_id"]]])) for r in rows}
    gold = {r["query_id"]: r["gold_label"] for r in rows}
    t = tune_rules(items, gold)
    t["checkpoint_sha"] = verifier.checkpoint_hash
    t["windowing"] = verifier.windowing
    out = assert_safe_output(TUNING_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning.json").write_text(json.dumps(t, indent=2))
    (out / "tuning.md").write_text(tuning_markdown(t))
    c = t["chosen"]
    frozen = {"windowing": verifier.windowing, "rule": c["rule"], "params": c["params"],
              "r1_tau": t["best_per_rule"]["R1"]["params"]["tau"],
              "checkpoint_sha": verifier.checkpoint_hash,
              "tuned_on": f"{TRAIN_DATASET}: the {len(rows)} held-out claims of {_tuning_run()}"}
    rrows = [combined_row(r, scored[r["query_id"]], c["rule"], c["params"]) for r in rows]
    write_run(out, rrows, run_block(ling_run, verifier, c["rule"], c["params"], "tuned in-sample on train",
                                    TRAIN_DATASET, frozen),
              f"Ling + verifier on {len(rows)} train tuning claims (in-sample)")
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2))
    log(f"frozen -> {FROZEN_PATH}: {json.dumps(frozen)}")
    log(tuning_markdown(t))
    return out


def tuning_markdown(t: Mapping) -> str:
    c = t["chosen"]
    lines = [
        f"# Combination rules on {t['n']} held-out train claims", "",
        f"Chosen: **{c['rule']}** {c['params']} — accuracy {c['accuracy']:.4f} "
        f"(plateau mean {c['smoothed_accuracy']:.4f}), {c['changed']} verdicts changed vs Ling.", "",
        "| Rule | Params | Accuracy | Plateau mean | Neighbour range | Macro-F1 | Changed | SUPPORT | CONTRADICT | NEI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule in ("R0", "R1", "R2", "R3", "R4"):
        p = t["best_per_rule"][rule]
        pl = p["per_label_accuracy"]
        lines.append(f"| {rule} | {p['params'] or '—'} | {p['accuracy']:.4f} | {p['smoothed_accuracy']:.4f} | "
                     f"{p['neighbour_accuracy_range']} | {p['macro_f1']:.4f} | {p['changed']} | "
                     f"{pl['SUPPORT']:.3f} | {pl['CONTRADICT']:.3f} | {pl['NEI']:.3f} |")
    for rule, key in (("R1", "tau"), ("R2", "tau_low"), ("R3", "tau_high")):
        lines += ["", f"{rule} accuracy by {key}: " + ", ".join(
            f"{p['params'][key]:.2f}:{p['accuracy']:.3f}" for p in t["table"][rule])]
    return "\n".join(lines) + "\n"


def stage_test(rerun: bool = False, log=print) -> Path:
    from app.ingest.corpus import load_documents, load_queries_qrels

    frozen = load_frozen()
    verifier = load_verifier(frozen["windowing"])
    if verifier.checkpoint_hash != frozen["checkpoint_sha"]:
        raise SystemExit(f"checkpoint {verifier.checkpoint_hash[:12]} is not the one the rule was tuned "
                         f"for ({str(frozen['checkpoint_sha'])[:12]})")
    rule, params = frozen["rule"], frozen["params"]
    out = assert_safe_output(test_dir(rule))
    out_r1 = assert_safe_output(R1_TEST_DIR)
    if (out / "rag.json").exists() and not rerun:
        raise SystemExit(f"{out}/rag.json exists: the test split is touched once (--rerun to reproduce it)")
    blob = ling_first_pass(json.loads(TEST_RUN.read_text()))
    if blob["run"]["dataset"] != TEST_DATASET or len(blob["rows"]) != N_TEST:
        raise SystemExit(f"{TEST_RUN} is not the 300-claim test run")
    queries, _ = load_queries_qrels(TEST_DATASET)
    docs = {d["doc_id"]: d for d in load_documents()}
    cache = nli.PairCache(nli.cache_path_for(verifier.model_name, verifier.dtype))
    scored = score_rows(verifier, cache, blob["rows"], queries, docs, log)
    src = f"frozen, tuned on {frozen['tuned_on']}"
    rows = [combined_row(r, scored[r["query_id"]], rule, params) for r in blob["rows"]]
    agg = write_run(out, rows, run_block(blob["run"], verifier, rule, params, src, TEST_DATASET, frozen),
                    f"Ling + verifier ({rule}) on the {N_TEST} test claims")
    r1p = {"tau": frozen["r1_tau"]}
    r1 = [combined_row(r, scored[r["query_id"]], "R1", r1p) for r in blob["rows"]]
    agg1 = write_run(out_r1, r1, run_block(blob["run"], verifier, "R1", r1p, src, TEST_DATASET, frozen),
                     f"Verifier alone (R1) on the {N_TEST} test claims")
    log(f"{rule}: acc {agg['verdict_accuracy']} {agg['per_label_accuracy']} changed {agg['changed_vs_ling']}\n"
        f"R1: acc {agg1['verdict_accuracy']} {agg1['per_label_accuracy']}\nWrote {out}/rag.json, {out_r1}/rag.json")
    for d in dict.fromkeys((out, out_r1)):  # R1 chosen -> one dir
        log(compare_to_ling(d))
    return out


def compare_to_ling(run_dir: Path, baseline: Path | None = None) -> str:
    """rag_compare (paired exact McNemar) of a run against Ling's committed test run (its
    first pass, like the rows the run was built from) -> run_dir/compare.{md,json};
    returns the markdown."""
    from app.eval import rag_compare

    baseline = TEST_RUN if baseline is None else baseline
    base = ling_first_pass(rag_compare.load_run(baseline))
    res = rag_compare.compare(base, rag_compare.load_run(run_dir / "rag.json"))
    md = rag_compare.to_markdown(res, str(baseline), str(run_dir / "rag.json"))
    (run_dir / "compare.json").write_text(json.dumps(res, indent=2, default=list))
    (run_dir / "compare.md").write_text(md)
    return md


def _tuning_run() -> str:
    from app.verify.train_data import TUNING_RUN

    return str(TUNING_RUN)


def stage_all(rerun: bool = False, log=print) -> Path:
    """validate -> tune -> test: the whole pre-declared protocol, test touched once."""
    if (done := test_outputs_exist()) and not rerun:
        raise SystemExit(f"{done[0]} exists: the test split was already scored (--rerun to reproduce it)")
    stage_validate(log=log)
    stage_tune(rerun=rerun, log=log)
    return stage_test(rerun=rerun, log=log)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.eval.verify_combine", description=__doc__.split("\n")[0])
    ap.add_argument("stage", nargs="?", default="all", choices=("all", "validate", "tune", "test"))
    ap.add_argument("--rerun", action="store_true", help="reproduce an existing test output on purpose")
    a = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    if a.stage == "all":
        stage_all(rerun=a.rerun)
    elif a.stage == "validate":
        stage_validate()
    elif a.stage == "tune":
        stage_tune(rerun=a.rerun)
    else:
        stage_test(rerun=a.rerun)


if __name__ == "__main__":
    main()
