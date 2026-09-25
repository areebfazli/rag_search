"""Tests for the RAG eval's scoring helpers.

These are pure functions over the judge's reply, but they decide the published
numbers — a coercion bug here silently corrupts a headline metric rather than
raising. That is precisely what happened with ``bool("false")``.
"""
import json
import subprocess
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import Settings, settings
from app.core.llm_endpoints import (
    FREE_ROUTING,
    PAID_ROUTING,
    EmptyCompletionError,
    LLMEndpoint,
)
from app.core.interfaces import Answer, SearchHit
from app.eval import rag_eval
from app.eval.rag_eval import (
    DailyTokenBudgetExhausted,
    JudgeParseError,
    _as_bool,
    _as_score,
    _is_daily_cap,
    _parse,
    aggregate,
    evidence_flags,
    predicted_label,
    resolve_answered,
)
from app.generate.generator import GeneratedAnswer, split_verdict
from app.ingest.corpus import ClaimLabel

FAKE_OPENROUTER_KEY = "sk-or-v1-FAKE-TEST-KEY-never-real-0000"


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
    def __init__(self, model=None, endpoint=None):
        self.endpoint = endpoint

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
            cost_usd=GEN_COST_USD,
            provider="AkashML",
        )


GEN_COST_USD = 0.0003  # what OpenRouter reports for one gpt-oss-120b generation


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def __call__(self, client, model, q, contexts, answer):
        self.calls += 1
        # The judge calls q3 "answered" — the verdict must override it (the
        # ambiguity this change removes). Only q4/q5 should use its answer.
        return {"answered": q != "claim four", "faithfulness": 0.9, "context_relevance": 0.5}


DEFAULT_GEN_MODEL = Settings.model_fields["openrouter_llm_model"].default
# Env the harness reads; cleared in every main() test so a developer's shell can't leak in.
RAG_ENV = (
    "SSR_RAG_N", "SSR_RAG_DATASET", "SSR_EVAL_LIMIT", "SSR_EVAL_REFRESH", "SSR_RAG_CHECK_QUOTA",
)
PAID_GEN_MODEL = "openai/gpt-oss-120b"


def _patch_main(
    tmp_path, monkeypatch, out, mode="hybrid", generator=FakeGenerator, judge=None,
    gen_model=DEFAULT_GEN_MODEL,
):
    fake_judge = judge if judge is not None else FakeJudge()
    # Pin the provider mix and generator model with a fake key, so the run neither depends
    # on the developer's .env nor needs one (CI has none). Nothing here touches the network.
    # gen_model=PAID_GEN_MODEL exercises the paid, spend-capped path.
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "judge_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_llm_model", gen_model)
    monkeypatch.setattr(settings, "openrouter_paid_model_allowlist", (PAID_GEN_MODEL,))
    monkeypatch.setattr(settings, "openrouter_api_key", FAKE_OPENROUTER_KEY)
    monkeypatch.setattr(rag_eval, "OUT", out)
    # Checkpoints and non-canonical runs stay under tmp_path, never the real data/.
    monkeypatch.setattr(rag_eval, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    for var in RAG_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rag_eval, "MODE", mode)
    monkeypatch.setattr(rag_eval, "THROTTLE_S", 0.0)
    monkeypatch.setattr(
        rag_eval, "load_queries_qrels", lambda dataset=None: (dict(QUERIES), QRELS)
    )
    monkeypatch.setattr(
        rag_eval, "load_claim_labels", lambda dataset=None, query_ids=None: CLAIM_LABELS
    )
    monkeypatch.setattr(rag_eval, "scifact_source_zip", lambda: tmp_path / "missing.zip")
    monkeypatch.setattr(rag_eval, "SearchService", FakeService)
    monkeypatch.setattr(rag_eval, "LLMGenerator", generator)
    monkeypatch.setattr(rag_eval, "OpenAI", lambda **kw: None)
    monkeypatch.setattr(rag_eval, "judge", fake_judge)
    return fake_judge


def _run_main(tmp_path, monkeypatch, mode="hybrid", gen_model=DEFAULT_GEN_MODEL):
    fake_judge = _patch_main(tmp_path, monkeypatch, tmp_path, mode=mode, gen_model=gen_model)
    rag_eval.main()
    return json.loads((tmp_path / "rag.json").read_text()), fake_judge


@pytest.fixture()
def rag_run(tmp_path, monkeypatch):
    blob, fake_judge = _run_main(tmp_path, monkeypatch)
    return tmp_path, blob, fake_judge


@pytest.fixture()
def paid_rag_run(tmp_path, monkeypatch):
    blob, fake_judge = _run_main(tmp_path, monkeypatch, gen_model=PAID_GEN_MODEL)
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
        "generation_cost_usd": GEN_COST_USD,
        "generation_provider": "AkashML",
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


# --- rate limits: per-minute 429s are retried, a daily-cap 429 ends the run -----

# Groq's 429 bodies, as the openai client stringifies them. Only the window differs.
TPM_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_test` service tier `on_demand` on tokens "
    "per minute (TPM): Limit 8000, Used 7400, Requested 3100. Please try again in 18.75s.', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
TPD_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_test` service tier `on_demand` on tokens "
    "per day (TPD): Limit 200000, Used 198900, Requested 3100. Please try again in 14m2.4s.', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
WAIT_S = 12.5  # distinct from THROTTLE_S (0.0 in these runs) so the two sleeps can't blur


def rate_limit_error(message: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request, json={"error": {"message": message}})
    return openai.RateLimitError(message=message, response=response, body=None)


class FlakyGenerator(FakeGenerator):
    """FakeGenerator that raises ``errors`` (in order) for one query, then answers."""

    calls: list[tuple[str, list]]
    fail_query: str
    errors: list[Exception]

    def generate(self, query, hits):
        type(self).calls.append((query, hits))
        if query == self.fail_query and self.errors:
            raise self.errors.pop(0)
        return super().generate(query, hits)


class FlakyJudge(FakeJudge):
    def __init__(self, fail_query=None, errors=()):
        super().__init__()
        self.queries: list[str] = []
        self.fail_query, self.errors = fail_query, list(errors)

    def __call__(self, client, model, q, contexts, answer):
        self.queries.append(q)
        if q == self.fail_query and self.errors:
            raise self.errors.pop(0)
        return super().__call__(client, model, q, contexts, answer)


@pytest.fixture()
def rate_limited(tmp_path, monkeypatch):
    """Patch main() for rate-limit tests; returns (setup, sleeps, out).

    ``setup(source, fail_query, errors)`` makes the generator or the judge raise
    ``errors`` for ``fail_query`` and returns (generator_cls, judge). ``out`` is a
    not-yet-existing results dir, so "nothing written" also covers the mkdir.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(rag_eval.time, "sleep", sleeps.append)
    monkeypatch.setattr(rag_eval, "RATE_LIMIT_WAIT_S", WAIT_S)
    out = tmp_path / "results"

    def setup(source, fail_query, errors):
        gen = type("Gen", (FlakyGenerator,), {"calls": [], "fail_query": None, "errors": []})
        fake_judge = FlakyJudge()
        if source == "generate":
            gen.fail_query, gen.errors = fail_query, list(errors)
        else:
            fake_judge.fail_query, fake_judge.errors = fail_query, list(errors)
        _patch_main(tmp_path, monkeypatch, out, generator=gen, judge=fake_judge)
        return gen, fake_judge

    return setup, sleeps, out


def test_is_daily_cap_distinguishes_tpd_from_tpm():
    assert _is_daily_cap(rate_limit_error(TPD_MESSAGE)) is True
    assert _is_daily_cap(rate_limit_error(TPM_MESSAGE)) is False
    # Case-insensitive, and either marker alone is enough.
    assert _is_daily_cap(rate_limit_error("Limit reached on Tokens Per Day")) is True
    assert _is_daily_cap(rate_limit_error("429: limit exceeded (TPD)")) is True
    assert _is_daily_cap(rate_limit_error("requests per day (RPD): Limit 1000")) is True
    # Per-minute and unlabelled 429s stay retryable.
    assert _is_daily_cap(rate_limit_error("requests per minute (RPM): Limit 30")) is False
    assert _is_daily_cap(rate_limit_error("Error code: 429 - Too Many Requests")) is False


def test_daily_token_budget_exhausted_is_not_a_rate_limit_error():
    # main()'s inner `except RateLimitError` must not re-catch it and retry.
    assert issubclass(DailyTokenBudgetExhausted, RuntimeError)
    assert not issubclass(DailyTokenBudgetExhausted, openai.RateLimitError)


@pytest.mark.parametrize("source", ["generate", "judge"])
def test_daily_cap_stops_the_run_without_retry_or_writing(rate_limited, source):
    setup, sleeps, out = rate_limited
    gen, fake_judge = setup(source, "claim three", [rate_limit_error(TPD_MESSAGE)])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    msg = str(exc.value.code)
    assert "daily cap is exhausted" in msg and "Nothing was written" in msg
    assert "tokens per day (TPD)" in msg
    # No retry: no rate-limit wait, and the failing query was attempted exactly once.
    assert WAIT_S not in sleeps
    assert [q for q, _ in gen.calls].count("claim three") == 1
    assert fake_judge.queries.count("claim three") == (1 if source == "judge" else 0)
    # A partial run never overwrites the committed artifact.
    assert not (out / "rag.json").exists()
    assert not (out / "rag.md").exists()
    assert not out.exists()


@pytest.mark.parametrize("source", ["generate", "judge"])
def test_per_minute_rate_limit_retries_the_same_query(rate_limited, source):
    setup, sleeps, out = rate_limited
    fails = rag_eval.RATE_LIMIT_RETRIES - 1  # succeeds on the last allowed attempt
    gen, fake_judge = setup(source, "claim two", [rate_limit_error(TPM_MESSAGE)] * fails)
    rag_eval.main()  # completes: no SystemExit
    assert sleeps.count(WAIT_S) == fails
    # Only the rate-limited step retries, for the SAME query with the same hits: a judge
    # 429 must not re-run (and, on a paid generator, re-pay for) a finished generation.
    tries = [hits for q, hits in gen.calls if q == "claim two"]
    assert len(tries) == (fails + 1 if source == "generate" else 1)
    assert all(h == tries[0] for h in tries)
    judge_tries = fake_judge.queries.count("claim two")
    assert judge_tries == (fails + 1 if source == "judge" else 1)
    # Every other query ran once, and the retried one is scored, not skipped.
    assert len(gen.calls) == len(QUERIES) + (fails if source == "generate" else 0)
    blob = json.loads((out / "rag.json").read_text())
    assert (out / "rag.md").exists()
    assert blob["skipped"] == 0 and blob["skip_reasons"] == {}
    assert blob["n"] == len(QUERIES)
    assert {r["query_id"] for r in blob["rows"]} == set(QUERIES)


def test_per_minute_rate_limit_skips_the_query_once_retries_are_exhausted(rate_limited):
    setup, sleeps, out = rate_limited
    attempts = rag_eval.RATE_LIMIT_RETRIES + 1
    gen, _ = setup("generate", "claim two", [rate_limit_error(TPM_MESSAGE)] * (attempts + 1))
    rag_eval.main()  # the run survives: the query is skipped, not fatal
    assert sleeps.count(WAIT_S) == rag_eval.RATE_LIMIT_RETRIES
    assert [q for q, _ in gen.calls].count("claim two") == attempts
    blob = json.loads((out / "rag.json").read_text())
    assert blob["skipped"] == 1 and blob["skip_reasons"] == {"RateLimitError": 1}
    assert {r["query_id"] for r in blob["rows"]} == set(QUERIES) - {"2"}
    assert blob["n"] == len(QUERIES) - 1


# --- providers, spend ceiling, fatal errors ------------------------------------------------

# OpenRouter's free-tier daily cap, as the openai client stringifies the 429 body.
OPENROUTER_DAILY_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-day. "
    "Add 10 credits to unlock 1000 free model requests per day', 'code': 429}}"
)


def test_is_daily_cap_recognises_openrouter_free_tier():
    assert _is_daily_cap(rate_limit_error(OPENROUTER_DAILY_MESSAGE)) is True
    assert _is_daily_cap(rate_limit_error("Rate limit exceeded: free-models-per-min.")) is False
    # Window only in the headers: an exhausted limit above the 20/min one is the daily cap.
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    def err(limit):
        response = httpx.Response(
            429, request=request, json={"error": {"message": "Rate limit exceeded"}},
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"},
        )
        return openai.RateLimitError(message="Rate limit exceeded", response=response, body=None)

    assert _is_daily_cap(err(1000)) is True
    assert _is_daily_cap(err(20)) is False


def test_default_run_is_all_free(capsys, rag_run):
    path, blob, _ = rag_run
    run = blob["run"]
    assert run["generator_model"] == DEFAULT_GEN_MODEL and DEFAULT_GEN_MODEL.endswith(":free")
    assert run["judge_model"].endswith(":free")
    assert run["generator_extra_body"] == {"provider": FREE_ROUTING}  # no paid pinning
    assert run["generator_reasoning_effort"] is None and run["generator_reasoning_param"] is None
    # A free generator is never counted at the paid worst case, even when unreported (q5).
    assert blob["cost"]["generator_paid"] is False
    assert blob["cost"]["unreported_paid_generations"] == 0
    assert blob["cost"]["counted_usd"] == blob["cost"]["reported_usd"]
    out = capsys.readouterr().out
    assert "both roles are free: $0 run" in out and "worst-case spend $0 (no paid role)" in out
    assert "total: 10-15 free OpenRouter requests" in out  # 5 claims x (1-2 gen + 1 judge)
    assert "paid requests" not in out and "(paid)" not in out


def test_estimate_line_free_vs_paid():
    url = "https://openrouter.ai/api/v1"
    judge_ep = LLMEndpoint("judge", "openrouter", url, "nvidia/nemotron-3-ultra-550b-a55b:free")
    free = rag_eval._estimate_line(
        50, LLMEndpoint("generator", "openrouter", url, DEFAULT_GEN_MODEL), judge_ep, 12.0
    )
    assert f"generator: 50-100 free requests to {DEFAULT_GEN_MODEL} ($0)" in free
    assert "judge: 50 free requests" in free
    assert "total: 100-150 free OpenRouter requests (both roles are free: $0 run" in free
    assert "1,000/day" in free and "~10 min at a 12.0s throttle" in free
    paid = rag_eval._estimate_line(
        50, LLMEndpoint("generator", "openrouter", url, PAID_GEN_MODEL), judge_ep, 5.0
    )
    assert "generator: 50-100 paid requests" in paid and "worst case $" in paid
    assert "total: 50 free OpenRouter requests (free tier" in paid  # the judge only
    assert "both roles are free" not in paid and "no paid role" not in paid


def test_rag_json_records_providers_and_cost_but_never_the_key(paid_rag_run):
    path, blob, _ = paid_rag_run
    run = blob["run"]
    assert run["generator_provider"] == "openrouter" and run["judge_provider"] == "openrouter"
    assert run["generator_base_url"] == run["judge_base_url"] == "https://openrouter.ai/api/v1"
    assert run["generator_model"] == PAID_GEN_MODEL and run["judge_model"].endswith(":free")
    assert run["generator_extra_body"] == {"provider": PAID_ROUTING}
    assert run["generator_reasoning_param"] == "reasoning.effort"
    assert blob["cost"]["generator_paid"] is True
    # q5's generator is a plain Answer (no cost side channel): counted at the worst case.
    assert blob["cost"]["reported_usd"] == pytest.approx((len(QUERIES) - 1) * GEN_COST_USD)
    assert blob["cost"]["unreported_paid_generations"] == 1
    assert blob["cost"]["counted_usd"] > blob["cost"]["reported_usd"]
    for f in ("rag.json", "rag.md"):
        assert FAKE_OPENROUTER_KEY not in (path / f).read_text()
    assert "(openrouter)" in (path / "rag.md").read_text()


def test_throttle_is_provider_aware(monkeypatch):
    monkeypatch.setattr(rag_eval, "THROTTLE_S", None)  # no SSR_RAG_THROTTLE_S override
    url = "https://openrouter.ai/api/v1"
    paid = LLMEndpoint("generator", "openrouter", url, "openai/gpt-oss-120b")
    free_gen = LLMEndpoint("generator", "openrouter", url, DEFAULT_GEN_MODEL)
    judge_ep = LLMEndpoint("judge", "openrouter", url, "nvidia/nemotron-3-ultra-550b-a55b:free")
    groq = LLMEndpoint("generator", "groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b")
    assert rag_eval.default_throttle_s(groq, judge_ep) == 30.0
    # One free request per query (the judge): the 5 s floor, <= 12 free requests/min.
    assert rag_eval.default_throttle_s(paid, judge_ep) == 5.0
    assert rag_eval.free_requests_per_query(paid, judge_ep) == (1, 1)
    # The default: free generator (1, or 2 with the truncation retry) + free judge.
    assert rag_eval.free_requests_per_query(free_gen, judge_ep) == (2, 3)
    throttle = rag_eval.default_throttle_s(free_gen, judge_ep)
    assert throttle == pytest.approx(12.0)
    # Worst case, every query retrying: query starts in any 60 s window x 3 requests stays
    # under the 20/min free cap with margin (sleep alone spaces them; latency only helps).
    worst_per_min = (60 // throttle) * 3
    assert worst_per_min <= 15 < rag_eval.OPENROUTER_FREE_REQUESTS_PER_MIN
    monkeypatch.setattr(rag_eval, "THROTTLE_S", 1.0)
    assert rag_eval.default_throttle_s(paid, judge_ep) == 1.0


def test_spend_ceiling_stops_the_run_before_writing(tmp_path, monkeypatch):
    out = tmp_path / "results"
    _patch_main(tmp_path, monkeypatch, out, gen_model=PAID_GEN_MODEL)
    # Room for exactly two queries at the worst-case bound: the third is refused up front.
    worst = rag_eval.worst_case_generation_cost(rag_eval.resolve_endpoint("generator"))
    monkeypatch.setattr(settings, "rag_max_spend_usd", 2 * GEN_COST_USD + worst)
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    msg = str(exc.value.code)
    assert "spend ceiling reached" in msg and "Nothing was written" in msg
    assert not out.exists()


def test_unreported_paid_cost_counts_at_the_worst_case(tmp_path, monkeypatch):
    class Unreported(FakeGenerator):
        def generate(self, query, hits):
            ans = super().generate(query, hits)
            ans.cost_usd = None
            return ans

    _patch_main(tmp_path, monkeypatch, tmp_path, generator=Unreported, gen_model=PAID_GEN_MODEL)
    rag_eval.main()
    cost = json.loads((tmp_path / "rag.json").read_text())["cost"]
    worst = rag_eval.worst_case_generation_cost(rag_eval.resolve_endpoint("generator"))
    assert cost["reported_usd"] == 0.0 and cost["unreported_paid_generations"] == len(QUERIES)
    assert cost["counted_usd"] == pytest.approx(len(QUERIES) * worst, abs=1e-6)


@pytest.mark.parametrize("status", [401, 402])
def test_auth_or_credit_errors_stop_the_run_instead_of_skipping(rate_limited, status):
    setup, _, out = rate_limited
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "nope"}})
    err = openai.APIStatusError("nope", response=response, body=None)
    setup("generate", "claim two", [err])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    assert "Nothing was written" in str(exc.value.code)
    assert not out.exists()


def test_missing_openrouter_key_fails_before_any_work(tmp_path, monkeypatch):
    _patch_main(tmp_path, monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(
        rag_eval, "load_queries_qrels", lambda dataset=None: pytest.fail("ran retrieval")
    )
    with pytest.raises(Exception, match="SSR_OPENROUTER_API_KEY"):
        rag_eval.main()


# --- sample size, dataset, canonical-output guard ------------------------------------------


def test_samples_nest_so_runs_of_different_n_share_a_prefix():
    ids = [str(i) for i in range(1, 301)]
    s50, s300 = rag_eval.sample_claims(ids, 50), rag_eval.sample_claims(ids, 300)
    assert s50 == s300[:50]
    assert rag_eval.sample_claims(ids, None) == s300  # "all"
    assert rag_eval.sample_claims(list(reversed(ids)), 50) == s50  # input order is irrelevant
    assert s50 != sorted(ids)[:50]  # a real shuffle, not the first ids


def test_committed_sample_is_the_first_50_of_the_full_test_split():
    # The committed 50-claim artifact must be exactly the prefix of the N=all sample, so a
    # 300-claim run nests over it. Test query ids come straight from the source archive
    # (no ir_datasets, no network); skipped where the archive isn't downloaded (CI).
    import zipfile
    from pathlib import Path

    src = rag_eval.scifact_source_zip()
    if not src.exists():
        pytest.skip("SciFact source archive not downloaded")
    with zipfile.ZipFile(src) as zf:
        lines = zf.read("scifact/qrels/test.tsv").decode().splitlines()[1:]
    test_ids = {line.split("\t")[0] for line in lines if line.strip()}
    assert len(test_ids) == 300
    committed = json.loads(Path("eval/results/rag.json").read_text())
    ids = [r["query_id"] for r in committed["rows"]]
    assert rag_eval.sample_claims(test_ids, 50) == ids[:50]
    assert rag_eval.sample_claims(test_ids, None)[: len(ids)] == ids


def test_parse_n():
    assert rag_eval.parse_n(None) == rag_eval.N == 50  # default unchanged
    assert rag_eval.parse_n("") == 50
    assert rag_eval.parse_n("all") is None and rag_eval.parse_n(" ALL ") is None
    assert rag_eval.parse_n("300") == 300
    for bad in ("0", "-3", "fifty"):
        with pytest.raises(ValueError):
            rag_eval.parse_n(bad)


def test_output_dir_only_canonical_test_split_writes_eval_results(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    ph = "abcdef0123456789"
    assert rag_eval.output_dir(rag_eval.CANONICAL_DATASET, 0, 300, ph) == (tmp_path / "results", True)
    train, canonical = rag_eval.output_dir("beir/scifact/train", 0, 100, ph)
    assert not canonical and train == tmp_path / "runs" / "rag_beir-scifact-train_100_abcdef01"
    # A limited (smoke) run on the test split is never canonical either.
    smoke, canonical = rag_eval.output_dir(rag_eval.CANONICAL_DATASET, 5, 5, ph)
    assert not canonical and smoke.parent == tmp_path / "runs"


def test_train_split_run_never_touches_eval_results(tmp_path, monkeypatch, capsys):
    out = tmp_path / "results"
    _patch_main(tmp_path, monkeypatch, out)
    seen = {}

    def load(dataset=None):
        seen["queries"] = dataset
        return dict(QUERIES), QRELS

    def labels(dataset=None, query_ids=None):
        seen["labels"] = dataset
        return CLAIM_LABELS

    monkeypatch.setattr(rag_eval, "load_queries_qrels", load)
    monkeypatch.setattr(rag_eval, "load_claim_labels", labels)
    monkeypatch.setenv("SSR_RAG_DATASET", "beir/scifact/train")
    monkeypatch.setenv("SSR_RAG_N", "all")
    rag_eval.main()
    assert seen == {"queries": "beir/scifact/train", "labels": "beir/scifact/train"}
    assert not out.exists()  # the committed-artifact dir is never created
    runs = tmp_path / "runs"
    (run_dir,) = runs.iterdir()
    assert run_dir.name == f"rag_beir-scifact-train_5_{rag_eval.prompt_hash()[:8]}"
    blob = json.loads((run_dir / "rag.json").read_text())
    assert blob["run"]["dataset"] == "beir/scifact/train" and blob["run"]["canonical"] is False
    assert blob["run"]["n_requested"] == "all" and blob["run"]["n_sample"] == 5
    assert blob["run"]["label_source"]["dataset"] == "beir/scifact/train"
    assert "5 claims from beir/scifact/train" in (run_dir / "rag.md").read_text()
    assert "non-canonical" in capsys.readouterr().out


def test_smoke_limit_on_test_split_is_non_canonical(tmp_path, monkeypatch):
    out = tmp_path / "results"
    _patch_main(tmp_path, monkeypatch, out)
    monkeypatch.setenv("SSR_EVAL_LIMIT", "2")
    rag_eval.main()
    assert not out.exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert json.loads((run_dir / "rag.json").read_text())["n"] == 2


def test_canonical_run_with_a_different_n_prints_a_loud_notice(tmp_path, monkeypatch, capsys):
    _patch_main(tmp_path, monkeypatch, tmp_path)
    (tmp_path / "rag.json").write_text(json.dumps({"n": 3, "run": {"n_requested": 3}}))
    rag_eval.main()
    out = capsys.readouterr().out
    assert "NOTICE: this canonical run REPLACES" in out
    assert "committed sample: N=3 claims -> this run: N=5 claims" in out
    assert out.split("[1/5]")[0].count("NOTICE") == 1  # up front, before any LLM call
    assert out.count("NOTICE") == 2  # and again when it writes
    # Same N as committed (the rewritten file records n_sample=5): no notice.
    rag_eval.main()
    assert "NOTICE" not in capsys.readouterr().out


# --- checkpoint + resume --------------------------------------------------------------------


def _sample_order():
    return rag_eval.sample_claims(QUERIES, rag_eval.N)


def _checkpoint_files(tmp_path):
    return sorted((tmp_path / "cache").glob("*.json"))


def test_daily_cap_stop_checkpoints_done_rows_and_the_rerun_resumes(rate_limited, tmp_path):
    setup, _, out = rate_limited
    order = _sample_order()
    fail_qid = order[2]  # two rows complete before the cap hits
    fail_q = QUERIES[fail_qid]
    gen, _ = setup("generate", fail_q, [rate_limit_error(OPENROUTER_DAILY_MESSAGE)])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    msg = str(exc.value.code)
    assert "resume by re-running the same command after the quota resets" in msg.lower()
    assert "2/5 completed rows are checkpointed" in msg
    assert not out.exists()  # nothing written to the output dir until every row is done
    (ckpt,) = _checkpoint_files(tmp_path)
    assert set(json.loads(ckpt.read_text())["rows"]) == set(order[:2])

    # Quota reset: the same command resumes and runs ONLY the three remaining rows.
    gen2, judge2 = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == [QUERIES[q] for q in order[2:]]
    assert judge2.queries == [QUERIES[q] for q in order[2:]]
    blob = json.loads((out / "rag.json").read_text())
    assert [r["query_id"] for r in blob["rows"]] == order  # sample order, resumed included
    assert blob["n"] == 5 and blob["skipped"] == 0
    assert blob["judge_calls"] == 5  # cumulative across both sessions
    assert blob["run"]["checkpoint_signature"] == ckpt.stem


def test_resumed_output_matches_an_uninterrupted_run(rate_limited, tmp_path, monkeypatch):
    setup, _, out = rate_limited
    setup("generate", QUERIES[_sample_order()[3]], [rate_limit_error(TPD_MESSAGE)])
    with pytest.raises(SystemExit):
        rag_eval.main()
    setup("generate", None, [])
    rag_eval.main()
    resumed = json.loads((out / "rag.json").read_text())
    setup("generate", None, [])
    monkeypatch.setenv("SSR_EVAL_REFRESH", "1")  # the same run, uninterrupted, from scratch
    rag_eval.main()
    fresh = json.loads((out / "rag.json").read_text())
    for k in ("rows", "verdict_accuracy", "quadrants", "confusion", "judge_calls"):
        assert resumed[k] == fresh[k], k


def test_skipped_rows_are_not_checkpointed_and_are_retried(rate_limited, tmp_path):
    setup, _, out = rate_limited
    gen, _ = setup("generate", "claim two", [RuntimeError("upstream 500")])
    rag_eval.main()  # completes with the row skipped
    blob = json.loads((out / "rag.json").read_text())
    assert blob["skipped"] == 1 and "2" not in {r["query_id"] for r in blob["rows"]}
    (ckpt,) = _checkpoint_files(tmp_path)
    assert "2" not in json.loads(ckpt.read_text())["rows"]
    gen2, _ = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == ["claim two"]  # only the skipped row re-runs
    blob = json.loads((out / "rag.json").read_text())
    assert blob["n"] == 5 and blob["skipped"] == 0


def test_unparseable_judge_rows_are_not_checkpointed(rate_limited, tmp_path):
    setup, _, out = rate_limited
    setup("judge", "claim four", [JudgeParseError("garbage")])
    rag_eval.main()
    (ckpt,) = _checkpoint_files(tmp_path)
    assert "4" not in json.loads(ckpt.read_text())["rows"]
    gen2, _ = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == ["claim four"]


@pytest.mark.parametrize(
    "content", ["{not json", "null", "[]", '{"signature": "x", "rows": []}', "ROWS_BAD_SHAPE"]
)
def test_corrupt_checkpoint_degrades_to_recompute(rate_limited, tmp_path, content):
    setup, _, out = rate_limited
    setup("generate", None, [])
    rag_eval.main()
    (ckpt,) = _checkpoint_files(tmp_path)
    if content == "ROWS_BAD_SHAPE":  # right signature, but every row is unusable
        blob = json.loads(ckpt.read_text())
        blob["rows"] = {q: {"query_id": q} for q in blob["rows"]} | {"1": "not a row"}
        content = json.dumps(blob)
    ckpt.write_text(content)
    gen2, _ = setup("generate", None, [])
    rag_eval.main()  # no crash: everything is recomputed
    assert sorted(q for q, _ in gen2.calls) == sorted(QUERIES.values())
    assert json.loads((out / "rag.json").read_text())["n"] == 5


def test_eval_refresh_ignores_the_checkpoint(rate_limited, tmp_path, monkeypatch):
    setup, _, _ = rate_limited
    setup("generate", None, [])
    rag_eval.main()
    gen2, _ = setup("generate", None, [])
    monkeypatch.setenv("SSR_EVAL_REFRESH", "1")
    rag_eval.main()
    assert len(gen2.calls) == len(QUERIES)


def test_complete_checkpoint_rewrites_the_artifact_with_no_llm_calls(rate_limited, tmp_path):
    setup, _, out = rate_limited
    setup("generate", None, [])
    rag_eval.main()
    first = json.loads((out / "rag.json").read_text())
    gen2, judge2 = setup("generate", None, [])
    rag_eval.main()
    assert gen2.calls == [] and judge2.queries == []
    assert json.loads((out / "rag.json").read_text())["rows"] == first["rows"]


def _endpoints(gen_model=DEFAULT_GEN_MODEL):
    url = "https://openrouter.ai/api/v1"
    return (
        LLMEndpoint("generator", "openrouter", url, gen_model),
        LLMEndpoint("judge", "openrouter", url, "nvidia/nemotron-3-ultra-550b-a55b:free"),
    )


def test_signature_tracks_everything_that_changes_a_row(monkeypatch):
    gen, judge_ep = _endpoints()
    base = rag_eval.rag_signature(rag_eval.signature_fields("beir/scifact/test", 50, gen, judge_ep))
    same = rag_eval.rag_signature(rag_eval.signature_fields("beir/scifact/test", 50, gen, judge_ep))
    assert base == same

    def sig(dataset="beir/scifact/test", n=50, g=gen, j=judge_ep):
        return rag_eval.rag_signature(rag_eval.signature_fields(dataset, n, g, j))

    changed = {
        "n": sig(n=300),
        "dataset": sig(dataset="beir/scifact/train"),
        "generator model": sig(g=_endpoints("openai/gpt-oss-120b")[0]),
        "judge model": sig(j=LLMEndpoint("judge", "openrouter", judge_ep.base_url, "other:free")),
        "provider": sig(g=LLMEndpoint("generator", "groq", "https://api.groq.com/openai/v1", "m")),
    }
    with monkeypatch.context() as m:
        m.setattr(rag_eval, "SYSTEM", rag_eval.SYSTEM + " Be brief.")
        changed["prompt"] = sig()
    with monkeypatch.context() as m:
        m.setattr(rag_eval, "JUDGE_SYSTEM", rag_eval.JUDGE_SYSTEM + " Strictly.")
        changed["judge prompt"] = sig()
    for attr, value in (("TOP_K", 8), ("MODE", "bm25"), ("SEED", 7)):
        with monkeypatch.context() as m:
            m.setattr(rag_eval, attr, value)
            changed[attr] = sig()
    for field, value in (("llm_max_completion_tokens", 4096), ("llm_reasoning_effort", "high")):
        with monkeypatch.context() as m:
            m.setattr(settings, field, value)
            changed[field] = sig()
    assert all(s != base for s in changed.values()), [k for k, s in changed.items() if s == base]
    assert len(set(changed.values())) == len(changed)
    # Git-independent: a new commit that changes none of the above keeps the checkpoint.
    monkeypatch.setattr(rag_eval, "_git_sha", lambda: "deadbeef")
    assert sig() == base


# --- up-front estimate + optional quota check -------------------------------------------------


def test_request_estimate_counts_gen_retry_and_judge():
    gen, judge_ep = _endpoints()
    est = rag_eval.request_estimate(300, gen, judge_ep, 12.0, 0.24)
    assert est["requests"] == 672  # 300 x (1 + 0.24 + 1)
    assert est["free_requests"] == 672 and est["free_requests_worst"] == 900
    assert est["seconds"] == 300 * (12.0 + rag_eval.EST_QUERY_LATENCY_S)
    paid = rag_eval.request_estimate(300, _endpoints(PAID_GEN_MODEL)[0], judge_ep, 5.0, 0.24)
    assert paid["requests"] == 672 and paid["free_requests"] == 300  # only the judge is free


def test_estimate_is_printed_for_the_remaining_rows(tmp_path, monkeypatch, capsys):
    _run_main(tmp_path, monkeypatch)
    head = capsys.readouterr().out.split("[1/5]")[0]
    assert "Remaining work: 5 of 5 claims" in head
    assert "~12 LLM requests (per claim: 1 generation + 0.24 expected truncation" in head


def _key_info(remaining, limit=1000):
    return {"data": {"label": "x", "free_model_daily_requests": {
        "used": limit - remaining, "limit": limit, "remaining": remaining}}}


def test_free_requests_remaining_parses_the_key_response():
    assert rag_eval.free_requests_remaining(_key_info(255)) == (255, 1000)
    assert rag_eval.free_requests_remaining({"data": {"label": "x"}}) == (None, None)
    assert rag_eval.free_requests_remaining(None) == (None, None)


def test_check_quota_aborts_when_short_and_prints_both_numbers(capsys):
    gen, judge_ep = _endpoints()
    est = rag_eval.request_estimate(300, gen, judge_ep, 12.0, 0.24)
    with pytest.raises(SystemExit) as exc:
        rag_eval.check_quota(est, gen, judge_ep, fetch=lambda key: _key_info(255))
    assert "255 free requests left today < ~672 needed" in str(exc.value.code)
    assert "255 free OpenRouter requests left today (of 1000); this run needs ~672" in (
        capsys.readouterr().out
    )
    rag_eval.check_quota(est, gen, judge_ep, fetch=lambda key: _key_info(1000))  # enough: no exit


@pytest.mark.parametrize("fetch", [lambda key: {"data": {}}, lambda key: 1 / 0])
def test_check_quota_fails_closed(fetch):
    gen, judge_ep = _endpoints()
    est = rag_eval.request_estimate(10, gen, judge_ep, 12.0, 0.24)
    with pytest.raises(SystemExit, match="aborting before any LLM call"):
        rag_eval.check_quota(est, gen, judge_ep, fetch=fetch)


@pytest.mark.parametrize("how", ["flag", "env"])
def test_check_quota_in_main_aborts_before_any_llm_call(rate_limited, monkeypatch, how):
    setup, _, out = rate_limited
    gen, fake_judge = setup("generate", None, [])
    keys = []
    monkeypatch.setattr(rag_eval, "fetch_key_info", lambda key: keys.append(key) or _key_info(3))
    argv = ["--check-quota"] if how == "flag" else []
    if how == "env":
        monkeypatch.setenv("SSR_RAG_CHECK_QUOTA", "1")
    with pytest.raises(SystemExit, match="3 free requests left today < ~12 needed"):
        rag_eval.main(argv)
    assert gen.calls == [] and fake_judge.queries == [] and not out.exists()
    assert keys == [FAKE_OPENROUTER_KEY]  # sent the OpenRouter key, to the OpenRouter URL only
    assert rag_eval.OPENROUTER_KEY_URL == "https://openrouter.ai/api/v1/key"


def test_quota_check_is_off_by_default(rate_limited, monkeypatch):
    setup, _, _ = rate_limited
    setup("generate", None, [])
    monkeypatch.setattr(rag_eval, "fetch_key_info", lambda key: pytest.fail("called /key"))
    rag_eval.main()


# --- HTTP 200 with no completion: transient, retried like a per-minute 429 -----------------


def _empty(detail="code 502 Provider returned error"):
    return EmptyCompletionError(f"LLM backend returned no completion (choices: null): {detail}")


@pytest.mark.parametrize("choices", [None, []])
def test_judge_raises_empty_completion_on_null_or_empty_choices(choices):
    resp = SimpleNamespace(choices=choices, error={"message": "upstream", "code": 502})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: resp)))
    with pytest.raises(EmptyCompletionError, match="upstream"):
        rag_eval.judge(client, "m", "q", ["ctx"], "answer")


@pytest.mark.parametrize("source", ["generate", "judge"])
def test_empty_completion_retries_the_same_query(rate_limited, source):
    setup, sleeps, out = rate_limited
    fails = rag_eval.RATE_LIMIT_RETRIES - 1
    gen, fake_judge = setup(source, "claim two", [_empty()] * fails)
    rag_eval.main()
    assert sleeps.count(WAIT_S) == fails
    # A judge-side empty completion retries only the judge: the finished generation
    # is not re-run (or re-paid for).
    gen_tries = [q for q, _ in gen.calls].count("claim two")
    assert gen_tries == (fails + 1 if source == "generate" else 1)
    assert fake_judge.queries.count("claim two") == (fails + 1 if source == "judge" else 1)
    blob = json.loads((out / "rag.json").read_text())
    assert blob["skipped"] == 0 and blob["n"] == len(QUERIES)


@pytest.mark.parametrize("source", ["generate", "judge"])
def test_empty_completion_skips_uncheckpointed_once_retries_are_exhausted(
    rate_limited, tmp_path, source
):
    setup, sleeps, out = rate_limited
    attempts = rag_eval.RATE_LIMIT_RETRIES + 1
    gen, fake_judge = setup(source, "claim two", [_empty()] * (attempts + 1))
    rag_eval.main()  # survives: the row is skipped
    assert sleeps.count(WAIT_S) == rag_eval.RATE_LIMIT_RETRIES
    if source == "judge":
        assert [q for q, _ in gen.calls].count("claim two") == 1
        assert fake_judge.queries.count("claim two") == attempts
    else:
        assert [q for q, _ in gen.calls].count("claim two") == attempts
    blob = json.loads((out / "rag.json").read_text())
    assert blob["skipped"] == 1 and blob["skip_reasons"] == {"EmptyCompletionError": 1}
    (ckpt,) = _checkpoint_files(tmp_path)
    assert "2" not in json.loads(ckpt.read_text())["rows"]  # a re-run retries it
    gen2, _ = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == ["claim two"]


def test_empty_completion_naming_the_daily_cap_stops_the_run(rate_limited):
    setup, sleeps, out = rate_limited
    gen, _ = setup("judge", "claim three",
                   [_empty("code 429 Rate limit exceeded: free-models-per-day")])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    assert "daily cap is exhausted" in str(exc.value.code)
    assert WAIT_S not in sleeps and not out.exists()
