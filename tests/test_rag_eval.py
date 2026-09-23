"""Tests for the RAG eval's scoring helpers.

These are pure functions over the judge's reply, but they decide the published
numbers — a coercion bug here silently corrupts a headline metric rather than
raising. That is precisely what happened with ``bool("false")``.
"""
import json
import subprocess

import pytest

from app.core.config import settings
from app.core.interfaces import Answer, SearchHit
from app.eval import rag_eval
from app.eval.rag_eval import (
    JudgeParseError,
    _as_bool,
    _as_score,
    _parse,
    aggregate,
    evidence_flags,
    predicted_label,
    resolve_answered,
)
from app.generate.generator import GeneratedAnswer, split_verdict
from app.ingest.corpus import ClaimLabel


def row(
    answered: bool,
    evidence: bool,
    faith: float = 1.0,
    ctx: float = 1.0,
    evidence_qrels: bool | None = None,
    gold: str = "SUPPORT",
    pred: str = "SUPPORT",
    verdict: str | None = "SUPPORTED",
) -> dict:
    return {
        "answered": answered,
        "evidence": evidence,
        "evidence_qrels": evidence if evidence_qrels is None else evidence_qrels,
        "faithfulness": faith,
        "context_relevance": ctx,
        "gold_label": gold,
        "predicted_label": pred,
        "verdict": verdict,
        "answered_source": "verdict" if verdict is not None else "judge",
        "judge_answered": answered,
    }


# --- judge reply coercion -------------------------------------------------


def test_as_bool_handles_json_string_false():
    # A small judge model routinely emits "false" as a STRING. bool("false") is True,
    # which would count a correct abstention as an answer and move both headline
    # metrics in opposite directions at once.
    assert _as_bool("false") is False
    assert _as_bool("no") is False
    assert _as_bool("False") is False
    assert _as_bool(False) is False


def test_as_bool_handles_truthy_forms():
    assert _as_bool(True) is True
    assert _as_bool("true") is True
    assert _as_bool("yes") is True
    assert _as_bool("1") is True


def test_as_score_survives_null_and_out_of_range():
    assert _as_score(None) == 0.0  # `"faithfulness": null` used to raise TypeError
    assert _as_score("not a number") == 0.0
    assert _as_score(1.7) == 1.0  # clamped
    assert _as_score(-3) == 0.0
    assert _as_score("0.75") == 0.75


def test_as_score_rejects_booleans():
    # bool subclasses int and float(True) is 1.0. The judge already emits a boolean for
    # the adjacent `answered` field, so `"faithfulness": true` is a plausible slip — and
    # it is the one remaining way malformed output could INFLATE a published metric.
    assert _as_score(True) == 0.0
    assert _as_score(False) == 0.0


def test_as_score_rejects_all_non_finite_values():
    # min(1.0, nan) is nan and max(0.0, nan) is 1.0, so a naive clamp turns NaN into a
    # PERFECT score and inflates the headline metric. json.loads accepts bare NaN, so
    # the judge really can emit one.
    #
    # inf is deliberately 0.0 rather than clamped to 1.0: a judge emitting Infinity is
    # malfunctioning, and the rule this whole function exists to enforce is that
    # malformed output must never *inflate* a published number.
    assert _as_score(float("nan")) == 0.0
    assert _as_score(float("inf")) == 0.0
    assert _as_score(float("-inf")) == 0.0


def test_parse_skips_a_wrapper_object():
    # raw_decode at the first '{' returns the OUTER object, whose .get("answered") is
    # None — silently scoring a real answer as a zero-faithfulness abstention, without
    # incrementing the parse-failure counter. It must find the inner verdict instead.
    reply = '{"result": {"answered": true, "faithfulness": 0.9, "context_relevance": 0.8}}'
    parsed = _parse(reply)
    assert parsed["answered"] is True
    assert parsed["faithfulness"] == 0.9


def test_parse_stops_at_first_object():
    # A greedy {.*} span runs to the LAST brace, so trailing prose containing another
    # brace made the whole parse fail and silently dropped the query.
    reply = 'Here is my rating {"answered": true, "faithfulness": 0.5} and notes {see below}'
    assert _parse(reply)["answered"] is True


def test_parse_raises_on_unparseable_reply():
    # Must be distinguishable from a confident abstention, not folded into it.
    with pytest.raises(JudgeParseError):
        _parse("I cannot produce JSON for this one.")


# --- abstention scored against an evidence oracle -------------------------


def test_abstention_is_scored_not_assumed_correct():
    # Two abstentions: one with no evidence retrieved (correct) and one where the
    # evidence WAS retrieved (a false abstention — over-conservative). Abstention
    # rate alone cannot tell these apart, which is why it was the overstated claim.
    agg = aggregate([row(answered=False, evidence=False), row(answered=False, evidence=True)])
    assert agg["answered_rate"] == 0.0
    assert agg["abstention_precision"] == 0.5
    assert agg["false_abstention_rate"] == 1.0  # 1 of the 1 case that had evidence


def test_perfect_behaviour_scores_one():
    agg = aggregate([row(answered=True, evidence=True), row(answered=False, evidence=False)])
    assert agg["abstention_precision"] == 1.0
    assert agg["abstention_recall"] == 1.0
    assert agg["false_abstention_rate"] == 0.0
    assert agg["answered_without_evidence_rate"] == 0.0


def test_answering_without_evidence_is_surfaced():
    # The hallucination-risk quadrant: answered although nothing relevant was found.
    agg = aggregate([row(answered=True, evidence=False), row(answered=True, evidence=True)])
    assert agg["answered_without_evidence_rate"] == 0.5


def test_faithfulness_averages_only_over_answered():
    # Scoring a correct "not in context" as unfaithful would be wrong, so abstentions
    # must not drag the faithfulness mean down.
    agg = aggregate(
        [row(answered=True, evidence=True, faith=0.8), row(answered=False, evidence=False, faith=0.0)]
    )
    assert agg["faithfulness_answered"] == 0.8
    assert agg["context_relevance"] == 1.0  # this one IS over all rows


def test_empty_rows_do_not_crash():
    agg = aggregate([])
    assert agg["n"] == 0
    assert agg["faithfulness_answered"] is None
    assert agg["abstention_precision"] is None
    assert agg["verdict_accuracy"] is None
    assert agg["qrels_oracle"]["abstention_precision"] is None


def test_nei_abstention_is_correct_under_rationale_but_false_under_qrels():
    # The labelling artifact the rationale oracle exists to remove: an NEI claim whose
    # cited (rationale-free) abstract was retrieved, and the model rightly abstained.
    nei = row(answered=False, evidence=False, evidence_qrels=True, gold="NEI", pred="NEI")
    agg = aggregate([nei])
    assert agg["oracle"] == "rationale"
    assert agg["abstention_precision"] == 1.0 and agg["quadrants"]["correct_abstention"] == 1
    assert agg["qrels_oracle"]["abstention_precision"] == 0.0
    assert agg["qrels_oracle"]["quadrants"]["false_abstention"] == 1


# --- the rationale oracle -------------------------------------------------


def test_evidence_flags_support_claim():
    lab = ClaimLabel("SUPPORT", {"d1"})
    assert evidence_flags(["d1", "d2"], lab, {"d1"}) == {"evidence": True, "evidence_qrels": True}
    assert evidence_flags(["d2"], lab, {"d1"}) == {"evidence": False, "evidence_qrels": False}


def test_evidence_flags_contradict_claim_is_evidence():
    # A rebuttal needs evidence just as much as support does — CONTRADICT rationale
    # docs count, and answering ("the context refutes this") is then the right action.
    lab = ClaimLabel("CONTRADICT", {"d3"})
    assert evidence_flags(["d3"], lab, {"d3"})["evidence"] is True


def test_evidence_flags_nei_claim_never_has_rationale_evidence():
    # BEIR's qrels still mark the cited abstract relevant, so the two oracles disagree.
    lab = ClaimLabel("NEI")
    assert evidence_flags(["d1", "d2"], lab, {"d1"}) == {"evidence": False, "evidence_qrels": True}


def test_evidence_flags_qrels_superset_of_rationale():
    # 13 test claims have a qrels doc WITHOUT rationale: retrieving only that one is
    # evidence under qrels but not under rationale.
    lab = ClaimLabel("SUPPORT", {"d1"})
    assert evidence_flags(["d7"], lab, {"d1", "d7"}) == {"evidence": False, "evidence_qrels": True}


# --- verdict -> answered / label ------------------------------------------


def test_resolve_answered_prefers_the_verdict():
    assert resolve_answered("SUPPORTED", False) == (True, "verdict")
    assert resolve_answered("REFUTED", False) == (True, "verdict")  # a rebuttal answers
    assert resolve_answered("NOT ENOUGH EVIDENCE", True) == (False, "verdict")


def test_resolve_answered_falls_back_to_judge_without_verdict():
    assert resolve_answered(None, True) == (True, "judge")
    assert resolve_answered(None, False) == (False, "judge")


def test_predicted_label_mapping():
    assert predicted_label("SUPPORTED", True) == "SUPPORT"
    assert predicted_label("REFUTED", True) == "CONTRADICT"
    assert predicted_label("NOT ENOUGH EVIDENCE", False) == "NEI"
    assert predicted_label(None, False) == "NEI"  # abstained without a verdict line
    assert predicted_label(None, True) == rag_eval.NO_VERDICT  # can never be "correct"


def test_verdict_accuracy_and_confusion():
    rows = [
        row(True, True, gold="SUPPORT", pred="SUPPORT"),
        row(True, True, gold="SUPPORT", pred="CONTRADICT"),
        row(True, True, gold="CONTRADICT", pred="CONTRADICT"),
        row(True, True, gold="CONTRADICT", pred=rag_eval.NO_VERDICT, verdict=None),
        row(False, False, gold="NEI", pred="NEI"),
    ]
    agg = aggregate(rows)
    assert agg["verdict_accuracy"] == 0.6
    assert agg["verdict_parsed_rate"] == 0.8
    c = agg["confusion"]
    assert c["SUPPORT"] == {"SUPPORT": 1, "CONTRADICT": 1, "NEI": 0, "NONE": 0}
    assert c["CONTRADICT"] == {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 0, "NONE": 1}
    assert c["NEI"]["NEI"] == 1
    assert sum(sum(r.values()) for r in c.values()) == len(rows)


# --- end-to-end main() over fakes: rows, provenance, judge scope -----------

QUERIES = {
    "1": "claim one",
    "2": "claim two",
    "3": "claim three",
    "4": "claim four",
    "5": "claim five",
}
# Every claim has a qrels doc (as in BEIR, NEI included); retrieval always returns d1, d2.
QRELS = {"1": {"d1": 1}, "2": {"d9": 1}, "3": {"d1": 1}, "4": {"d2": 1, "d5": 1}, "5": {"d1": 1}}
CLAIM_LABELS = {
    "1": ClaimLabel("SUPPORT", {"d1"}),  # evidence retrieved, SUPPORTED verdict
    "2": ClaimLabel("CONTRADICT", {"d9"}),  # evidence missed, still says REFUTED
    "3": ClaimLabel("NEI"),  # qrels doc retrieved, rightly NOT ENOUGH EVIDENCE
    "4": ClaimLabel("SUPPORT", {"d2"}),  # no verdict line; judge: abstained
    "5": ClaimLabel("CONTRADICT", {"d1"}),  # no verdict line; judge: answered
}
REPLIES = {
    "claim one": "Supported [1].\nVerdict: SUPPORTED",
    "claim two": "Refuted [2].\nVerdict: REFUTED",
    "claim three": "The context does not say.\nVerdict: NOT ENOUGH EVIDENCE",
    "claim four": "The context does not contain enough information.",
    "claim five": "Passage [1] contradicts this.",
}


class FakeService:
    def retrieve(self, query, mode="hybrid", top_k=None):
        return [SearchHit("d1", 1.0, "t1"), SearchHit("d2", 0.5, "t2")]


class FakeGenerator:
    def __init__(self, model=None):
        pass

    def generate(self, query, hits):
        text, verdict = split_verdict(REPLIES[query])  # the real parser, fake model
        if query == "claim five":  # a plain Answer: another Generator, no side channel
            return Answer(text=text, citations=["d1"], hits=hits, verdict=verdict)
        # q4 is the reply still cut off after the retry (no verdict line, as observed).
        truncated = query == "claim four"
        return GeneratedAnswer(
            text=text,
            citations=["d1"],
            hits=hits,
            verdict=verdict,
            finish_reason="length" if truncated else "stop",
            completion_tokens=50,
            reasoning_tokens=10,
            attempts=2 if truncated else 1,
            truncated=truncated,
        )


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def __call__(self, client, model, q, contexts, answer):
        self.calls += 1
        # The judge calls q3 "answered" — the verdict must override it (the
        # ambiguity this change removes). Only q4/q5 should use its answer.
        return {"answered": q != "claim four", "faithfulness": 0.9, "context_relevance": 0.5}


def _run_main(tmp_path, monkeypatch, mode="hybrid"):
    fake_judge = FakeJudge()
    monkeypatch.setattr(rag_eval, "OUT", tmp_path)
    monkeypatch.setattr(rag_eval, "MODE", mode)
    monkeypatch.setattr(rag_eval, "THROTTLE_S", 0.0)
    monkeypatch.setattr(rag_eval, "load_queries_qrels", lambda: (dict(QUERIES), QRELS))
    monkeypatch.setattr(rag_eval, "load_claim_labels", lambda query_ids=None: CLAIM_LABELS)
    monkeypatch.setattr(rag_eval, "scifact_source_zip", lambda: tmp_path / "missing.zip")
    monkeypatch.setattr(rag_eval, "SearchService", FakeService)
    monkeypatch.setattr(rag_eval, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(rag_eval, "OpenAI", lambda **kw: None)
    monkeypatch.setattr(rag_eval, "judge", fake_judge)
    rag_eval.main()
    return json.loads((tmp_path / "rag.json").read_text()), fake_judge


@pytest.fixture()
def rag_run(tmp_path, monkeypatch):
    blob, fake_judge = _run_main(tmp_path, monkeypatch)
    return tmp_path, blob, fake_judge


def test_rag_json_keeps_every_scored_query(rag_run):
    _, blob, _ = rag_run
    rows = {r["query_id"]: r for r in blob["rows"]}
    assert set(rows) == set(QUERIES)
    assert rows["1"] == {
        "query_id": "1",
        "mode": rag_eval.MODE,
        "gold_label": "SUPPORT",
        "rationale_doc_ids": ["d1"],
        "qrels_doc_ids": ["d1"],
        "verdict": "SUPPORTED",
        "predicted_label": "SUPPORT",
        "answered": True,
        "answered_source": "verdict",
        "judge_answered": True,
        "faithfulness": 0.9,
        "context_relevance": 0.5,
        "evidence": True,
        "evidence_qrels": True,
        "abstention_class": "answered_with_evidence",
        "abstention_class_qrels": "answered_with_evidence",
        "answer": "Supported [1].",  # verdict line stripped from the display text
        "cited_doc_ids": ["d1"],
        "retrieved_doc_ids": ["d1", "d2"],
        "finish_reason": "stop",
        "truncated": False,
        "generation_attempts": 1,
        "completion_tokens": 50,
        "reasoning_tokens": 10,
    }
    assert rows["2"]["abstention_class"] == "answered_without_evidence"
    # NEI + rationale-free qrels doc retrieved: the two oracles disagree on this one.
    assert rows["3"]["abstention_class"] == "correct_abstention"
    assert rows["3"]["abstention_class_qrels"] == "false_abstention"
    assert rows["3"]["answered"] is False and rows["3"]["judge_answered"] is True
    assert rows["4"]["answered_source"] == "judge" and rows["4"]["predicted_label"] == "NEI"
    assert rows["4"]["abstention_class"] == "false_abstention"
    assert rows["5"]["predicted_label"] == rag_eval.NO_VERDICT
    assert rows["4"]["truncated"] is True and rows["4"]["finish_reason"] == "length"
    assert rows["4"]["generation_attempts"] == 2
    # A Generator without the side channel still yields a well-formed row.
    assert rows["5"]["finish_reason"] is None and rows["5"]["truncated"] is False
    assert rows["5"]["generation_attempts"] == 1 and rows["5"]["completion_tokens"] is None


def test_truncated_answers_are_counted_in_aggregates_and_markdown(rag_run):
    path, blob, _ = rag_run
    assert blob["truncated_answers"] == 1 and blob["retried_answers"] == 1
    md = (path / "rag.md").read_text()
    assert "1 answer truncated at the token budget" in md
    assert "| Truncated answers (hit the token budget after 1 retry) | 1 of 5 |" in md
    assert blob["run"]["generator_max_completion_tokens"] == settings.llm_max_completion_tokens
    assert "generator_reasoning_effort" in blob["run"]


def test_aggregate_tolerates_rows_without_generation_fields():
    agg = aggregate([row(True, True)])  # rows predating the fields
    assert agg["truncated_answers"] == 0 and agg["retried_answers"] == 0


def test_rag_json_aggregates_both_oracles_and_verdicts(rag_run):
    _, blob, _ = rag_run
    # The headline fields keep their names and position; they now mean the rationale
    # oracle, which `oracle` says explicitly.
    assert list(blob)[:3] == ["n", "evidence_rate", "answered_rate"]
    assert blob["oracle"] == "rationale"
    assert blob["n"] == 5 and blob["evidence_rate"] == 0.6  # q1, q4, q5
    assert blob["quadrants"] == {
        "answered_with_evidence": 2,
        "answered_without_evidence": 1,
        "false_abstention": 1,
        "correct_abstention": 1,
    }
    assert blob["abstention_precision"] == 0.5
    assert blob["qrels_oracle"]["evidence_rate"] == 0.8  # q3's NEI qrels doc counts here
    assert blob["qrels_oracle"]["abstention_precision"] == 0.0
    assert blob["verdict_accuracy"] == 0.6  # q1, q2, q3 right; q4 (NEI) and q5 (NONE) wrong
    assert blob["confusion"]["CONTRADICT"]["NONE"] == 1
    assert blob["by_label"]["NEI"] == {"n": 1, "answered": 0}


def test_judge_answered_is_consulted_only_without_a_verdict(rag_run):
    _, blob, fake_judge = rag_run
    # One judge call per query (faithfulness + context relevance have no gold label),
    # but its `answered` call decides only the 2 verdict-less replies, not all 5.
    assert fake_judge.calls == blob["judge_calls"] == 5
    assert blob["judge_answered_fallbacks"] == 2
    assert sum(r["answered_source"] == "judge" for r in blob["rows"]) == 2
    # Where the verdict decided, the judge agreed on q1 and q2 but not q3.
    assert blob["judge_verdict_agreement"] == round(2 / 3, 4)


def test_oracle_split_is_printed_before_any_llm_call(tmp_path, monkeypatch, capsys):
    _run_main(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    head = out.split("[1/5]")[0]  # everything before the first generated row
    assert "3 have a rationale doc in top-5 -> 2 should abstain" in head
    assert "4 have a qrels doc in top-5 -> 1 should abstain" in head
    assert "SUPPORT 2, CONTRADICT 2, NEI 1" in head


def test_rag_json_run_provenance(rag_run):
    path, blob, _ = rag_run
    run = blob["run"]
    assert run["prompt_hash"] == rag_eval.prompt_hash()
    assert run["generator_model"] == rag_eval.GEN_MODEL
    assert run["judge_model"] == rag_eval.JUDGE_MODEL
    assert run["mode"] == rag_eval.MODE and run["reranker_model"] is None
    assert run["n_requested"] == rag_eval.N and run["sample_seed"] == rag_eval.SEED
    assert run["oracle"] == "rationale" and run["legacy_oracle"] == "qrels"
    assert "rationale" in run["oracle_definition"]
    assert run["label_source"]["loader"] == "app.ingest.corpus.load_claim_labels"
    assert run["label_source"]["source_zip_sha256"] is None  # archive absent here
    assert "git_sha" in run  # a value in a checkout, None outside one — never missing
    md = (path / "rag.md").read_text()
    assert md.startswith("# RAG answer quality")
    assert "rationale oracle (headline)" in md and "legacy qrels oracle" in md
    assert "3-class verdict accuracy | 0.60" in md
    assert "| **CONTRADICT** | 0 | 1 | 0 | 1 |" in md


def test_rerank_mode_reports_the_reranker_in_effect(tmp_path, monkeypatch, capsys):
    blob, _ = _run_main(tmp_path, monkeypatch, mode="hybrid_rerank")
    assert blob["run"]["reranker_model"] == settings.reranker_model
    out = capsys.readouterr().out
    assert f"reranking with {settings.reranker_model}" in out
    if settings.reranker_model == rag_eval._DEFAULT_RERANKER:
        assert "HURTS" in out


def test_abstention_class_covers_all_quadrants():
    assert rag_eval._abstention_class(True, True) == "answered_with_evidence"
    assert rag_eval._abstention_class(True, False) == "answered_without_evidence"
    assert rag_eval._abstention_class(False, True) == "false_abstention"
    assert rag_eval._abstention_class(False, False) == "correct_abstention"


def test_prompt_hash_tracks_the_prompt(monkeypatch):
    base = rag_eval.prompt_hash()
    assert rag_eval.prompt_hash() == base  # deterministic
    monkeypatch.setattr(rag_eval, "SYSTEM", rag_eval.SYSTEM + " Be brief.")
    assert rag_eval.prompt_hash() != base


def test_git_sha_tolerates_no_git(monkeypatch):
    def no_git(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert rag_eval._git_sha() is None
    assert rag_eval.run_metadata()["git_sha"] is None
