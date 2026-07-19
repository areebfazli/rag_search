"""Tests for the RAG eval's scoring helpers.

These are pure functions over the judge's reply, but they decide the published
numbers — a coercion bug here silently corrupts a headline metric rather than
raising. That is precisely what happened with ``bool("false")``.
"""
import pytest

from app.eval.rag_eval import JudgeParseError, _as_bool, _as_score, _parse, aggregate


def row(answered: bool, evidence: bool, faith: float = 1.0, ctx: float = 1.0) -> dict:
    return {
        "answered": answered,
        "evidence": evidence,
        "faithfulness": faith,
        "context_relevance": ctx,
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


# --- abstention scored against gold qrels ---------------------------------


def test_abstention_is_scored_not_assumed_correct():
    # Two abstentions: one with no evidence retrieved (correct) and one where the
    # gold doc WAS retrieved (a false abstention — over-conservative). Abstention
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
