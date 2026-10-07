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
    EmptyCompletionError,
    LLMEndpoint,
    paid_routing,
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
from app.generate.generator import GeneratedAnswer, ReaskReply, split_verdict
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
    "1": "One claim holds.",
    "2": "Two claim holds.",
    "3": "Three claim holds.",
    "4": "Four claim holds.",
    "5": "Five claim holds.",
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
    "One claim holds.": "Supported [1].\nVerdict: SUPPORTED",
    "Two claim holds.": "Refuted [2].\nVerdict: REFUTED",
    "Three claim holds.": "The context does not say.\nVerdict: NOT ENOUGH EVIDENCE",
    # q4/q5 carry no verdict in any form (no line, no inline field, no first-sentence
    # stance), so they stay the judge-fallback cases under generator.parse_verdict too.
    "Four claim holds.": "Nothing here settles it.",
    "Five claim holds.": "Passage [1] reports the opposite finding.",
}


class FakeService:
    def retrieve(self, query, mode="hybrid", top_k=None):
        return [SearchHit("d1", 1.0, "t1"), SearchHit("d2", 0.5, "t2")]


# The corpus behind FakeService's hits, for re-asks of resumed rows (rebuilt from it).
CORPUS = ({"doc_id": "d1", "title": "", "text": "t1"}, {"doc_id": "d2", "title": "", "text": "t2"})


class FakeGenerator:
    # The verdict-only re-ask: rag_eval runs it as its own post-step (generate() is built
    # with reask=False). Unparseable by default, so the first-pass rows keep their labels;
    # tests that want a re-ask verdict override REASK_REPLY. Reset per test (_patch_main).
    REASK_REPLY = "I cannot tell from these passages."
    reask_calls: list[tuple[str, list]] = []

    init_kwargs: list[dict] = []

    def __init__(self, model=None, endpoint=None, reask=True, **kw):
        self.endpoint = endpoint
        self.reask_enabled = reask
        type(self).init_kwargs.append(kw)  # the SDK retry policy rag_eval asks for

    def reask_verdict(self, query, hits):
        type(self).reask_calls.append((query, hits))
        return ReaskReply(raw=self.REASK_REPLY, finish_reason="stop", completion_tokens=7,
                          reasoning_tokens=3, cost_usd=0.0, provider="P")

    def generate(self, query, hits):
        text, verdict = split_verdict(REPLIES[query])  # the real parser, fake model
        if query == "Five claim holds.":  # a plain Answer: another Generator, no side channel
            return Answer(text=text, citations=["d1"], hits=hits, verdict=verdict)
        # q4 is the reply still cut off after the retry (no verdict line, as observed).
        truncated = query == "Four claim holds."
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
        return {"answered": q != "Four claim holds.", "faithfulness": 0.9, "context_relevance": 0.5}


DEFAULT_GEN_MODEL = Settings.model_fields["openrouter_llm_model"].default
DEFAULT_JUDGE_MODEL = Settings.model_fields["openrouter_judge_model"].default
# Env the harness reads; cleared in every main() test so a developer's shell can't leak in.
RAG_ENV = (
    "SSR_RAG_N", "SSR_RAG_DATASET", "SSR_EVAL_LIMIT", "SSR_EVAL_REFRESH", "SSR_RAG_CHECK_QUOTA",
    "SSR_RAG_SKIP_JUDGE", "SSR_RAG_CONTEXT", "SSR_RAG_OFFSET",
)
PAID_GEN_MODEL = "openai/gpt-oss-120b"
LUNA = "openai/gpt-6-luna"


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
    monkeypatch.setattr(settings, "openrouter_judge_model", DEFAULT_JUDGE_MODEL)
    monkeypatch.setattr(settings, "openrouter_paid_model_allowlist", (PAID_GEN_MODEL, LUNA))
    monkeypatch.setattr(settings, "openrouter_api_key", FAKE_OPENROUTER_KEY)
    monkeypatch.setattr(rag_eval, "OUT", out)
    # Checkpoints and non-canonical runs stay under tmp_path, never the real data/.
    monkeypatch.setattr(rag_eval, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(rag_eval, "REASK_CACHE", tmp_path / "reask_cache")
    monkeypatch.setattr(rag_eval, "load_documents", lambda: [dict(d) for d in CORPUS])
    monkeypatch.setattr(settings, "llm_reask", True)
    # The opt-in vote and prompt variant off (code defaults), whatever the developer's .env.
    monkeypatch.setattr(settings, "llm_votes", 1)
    monkeypatch.setattr(settings, "llm_prompt_variant", "default")
    # The code default: the re-ask cache's legacy-key migration depends on it.
    monkeypatch.setattr(settings, "llm_reasoning_effort", "auto")
    monkeypatch.setattr(FakeGenerator, "reask_calls", [])
    monkeypatch.setattr(FakeGenerator, "init_kwargs", [])
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


def _written_dir(tmp_path, gen_model=DEFAULT_GEN_MODEL):
    """Where main() wrote: OUT (= tmp_path) for the default models, else the single
    non-canonical run dir — a non-default generator must never write OUT."""
    if gen_model == DEFAULT_GEN_MODEL:
        return tmp_path
    assert not (tmp_path / "rag.json").exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    return run_dir


def _run_main(tmp_path, monkeypatch, mode="hybrid", gen_model=DEFAULT_GEN_MODEL):
    fake_judge = _patch_main(tmp_path, monkeypatch, tmp_path, mode=mode, gen_model=gen_model)
    rag_eval.main()
    return json.loads((_written_dir(tmp_path, gen_model) / "rag.json").read_text()), fake_judge


@pytest.fixture()
def rag_run(tmp_path, monkeypatch):
    blob, fake_judge = _run_main(tmp_path, monkeypatch)
    return tmp_path, blob, fake_judge


@pytest.fixture()
def paid_rag_run(tmp_path, monkeypatch):
    blob, fake_judge = _run_main(tmp_path, monkeypatch, gen_model=PAID_GEN_MODEL)
    return _written_dir(tmp_path, PAID_GEN_MODEL), blob, fake_judge


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
        "verdict_source": "line",  # a well-formed final line
        "evidence_quote": None,  # the fake reply quotes nothing
        "quote_passage": None,
        "quote_found": None,
        "reask_attempted": False,  # it had a verdict: never re-asked
    }
    assert rows["2"]["abstention_class"] == "answered_without_evidence"
    # NEI + rationale-free qrels doc retrieved: the two oracles disagree on this one.
    assert rows["3"]["abstention_class"] == "correct_abstention"
    assert rows["3"]["abstention_class_qrels"] == "false_abstention"
    assert rows["3"]["answered"] is False and rows["3"]["judge_answered"] is True
    # q4 abstained per the judge, but its reply was cut off: no verdict is NONE, never NEI.
    assert rows["4"]["answered_source"] == "judge" and rows["4"]["predicted_label"] == rag_eval.NO_VERDICT
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
    assert blob["verdict_accuracy"] == 0.6  # q1, q2, q3 right; q4 and q5 (NONE) wrong
    assert blob["confusion"]["CONTRADICT"]["NONE"] == 1
    assert blob["confusion"]["SUPPORT"]["NONE"] == 1 and blob["confusion"]["SUPPORT"]["NEI"] == 0
    assert blob["by_label"]["NEI"] == {"n": 1, "answered": 0}


def test_eval_generator_uses_the_batch_retry_policy(rag_run):
    # The API's LLMGenerator() defaults are interactive (few SDK retries); the eval is a
    # batch and keeps quick SDK retries in front of its own 429 / empty-completion layer.
    from app.generate.generator import BATCH_MAX_RETRIES, REASK_MAX_RETRIES

    # votes=1: the eval draws a vote's extra samples itself; the prompt variant is the
    # setting's (the default here).
    assert FakeGenerator.init_kwargs[-1] == {
        "max_retries": BATCH_MAX_RETRIES, "reask_max_retries": REASK_MAX_RETRIES,
        "votes": 1, "prompt_variant": "default",
    }


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
    gen, fake_judge = setup(source, "Three claim holds.", [rate_limit_error(TPD_MESSAGE)])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    msg = str(exc.value.code)
    assert "daily cap is exhausted" in msg and "Nothing was written" in msg
    assert "tokens per day (TPD)" in msg
    # No retry: no rate-limit wait, and the failing query was attempted exactly once.
    assert WAIT_S not in sleeps
    assert [q for q, _ in gen.calls].count("Three claim holds.") == 1
    assert fake_judge.queries.count("Three claim holds.") == (1 if source == "judge" else 0)
    # A partial run never overwrites the committed artifact.
    assert not (out / "rag.json").exists()
    assert not (out / "rag.md").exists()
    assert not out.exists()


@pytest.mark.parametrize("source", ["generate", "judge"])
def test_per_minute_rate_limit_retries_the_same_query(rate_limited, source):
    setup, sleeps, out = rate_limited
    fails = rag_eval.RATE_LIMIT_RETRIES - 1  # succeeds on the last allowed attempt
    gen, fake_judge = setup(source, "Two claim holds.", [rate_limit_error(TPM_MESSAGE)] * fails)
    rag_eval.main()  # completes: no SystemExit
    assert sleeps.count(WAIT_S) == fails
    # Only the rate-limited step retries, for the SAME query with the same hits: a judge
    # 429 must not re-run (and, on a paid generator, re-pay for) a finished generation.
    tries = [hits for q, hits in gen.calls if q == "Two claim holds."]
    assert len(tries) == (fails + 1 if source == "generate" else 1)
    assert all(h == tries[0] for h in tries)
    judge_tries = fake_judge.queries.count("Two claim holds.")
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
    gen, _ = setup("generate", "Two claim holds.", [rate_limit_error(TPM_MESSAGE)] * (attempts + 1))
    # The other claims still run (the query is skipped, not fatal), but a canonical run
    # with a claim missing refuses to write: its denominator would silently shrink.
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    assert "Refusing to write the canonical" in str(exc.value.code)
    assert "1 of 5 sampled claims have no row (1 skipped" in str(exc.value.code)
    assert sleeps.count(WAIT_S) == rag_eval.RATE_LIMIT_RETRIES
    assert [q for q, _ in gen.calls].count("Two claim holds.") == attempts
    assert len(gen.calls) == attempts + len(QUERIES) - 1
    assert not out.exists()


def test_a_non_canonical_run_with_a_missing_claim_writes_with_a_loud_note(rate_limited, tmp_path,
                                                                          monkeypatch, capsys):
    setup, _, out = rate_limited
    attempts = rag_eval.RATE_LIMIT_RETRIES + 1
    setup("generate", "Two claim holds.", [rate_limit_error(TPM_MESSAGE)] * (attempts + 1))
    monkeypatch.setenv("SSR_EVAL_LIMIT", "5")  # all 5 claims, but a limited run: non-canonical
    rag_eval.main()
    assert not out.exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    blob = json.loads((run_dir / "rag.json").read_text())
    assert blob["skipped"] == 1 and blob["skip_reasons"] == {"RateLimitError": 1}
    assert {r["query_id"] for r in blob["rows"]} == set(QUERIES) - {"2"}
    assert blob["n"] == len(QUERIES) - 1 and blob["missing_query_ids"] == ["2"]
    assert "INCOMPLETE: 1 sampled claim missing" in (run_dir / "rag.md").read_text()
    assert "INCOMPLETE: 1 of 5 sampled claims have no row" in capsys.readouterr().out


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
    assert run["generator_extra_body"] == {"provider": paid_routing(PAID_GEN_MODEL)}
    assert run["generator_reasoning_param"] == "reasoning.effort"
    assert blob["cost"]["generator_paid"] is True
    # q5's generator is a plain Answer (no cost side channel): counted at the worst case.
    assert blob["cost"]["reported_usd"] == pytest.approx((len(QUERIES) - 1) * GEN_COST_USD)
    assert blob["cost"]["unreported_paid_generations"] == 1
    assert blob["cost"]["counted_usd"] > blob["cost"]["reported_usd"]
    for f in ("rag.json", "rag.md"):
        assert FAKE_OPENROUTER_KEY not in (path / f).read_text()
    assert "(openrouter)" in (path / "rag.md").read_text()


def test_a_failed_generation_with_a_billed_attempt_is_counted(tmp_path, monkeypatch):
    # generate() attaches the billed cost of an attempt that preceded a failure (e.g. the
    # truncation retry hit a 429): the spend ceiling must count it, not lose it.
    err = rate_limit_error(TPM_MESSAGE)
    err.cost_usd = 0.002
    gen = type("Gen", (FlakyGenerator,), {"calls": [], "fail_query": "Two claim holds.", "errors": [err]})
    monkeypatch.setattr(rag_eval.time, "sleep", lambda s: None)
    _patch_main(tmp_path, monkeypatch, tmp_path, generator=gen, gen_model=PAID_GEN_MODEL)
    rag_eval.main()
    blob = json.loads((_written_dir(tmp_path, PAID_GEN_MODEL) / "rag.json").read_text())
    assert blob["cost"]["reported_usd"] == pytest.approx((len(QUERIES) - 1) * GEN_COST_USD + 0.002)


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
    cost = json.loads((_written_dir(tmp_path, PAID_GEN_MODEL) / "rag.json").read_text())["cost"]
    worst = rag_eval.worst_case_generation_cost(rag_eval.resolve_endpoint("generator"))
    assert cost["reported_usd"] == 0.0 and cost["unreported_paid_generations"] == len(QUERIES)
    assert cost["counted_usd"] == pytest.approx(len(QUERIES) * worst, abs=1e-6)


@pytest.mark.parametrize("status", [401, 402])
def test_auth_or_credit_errors_stop_the_run_instead_of_skipping(rate_limited, status):
    setup, _, out = rate_limited
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "nope"}})
    err = openai.APIStatusError("nope", response=response, body=None)
    setup("generate", "Two claim holds.", [err])
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
    test = rag_eval.CANONICAL_DATASET
    assert rag_eval.output_dir(test, 0, 300, ph, split_size=300) == (tmp_path / "results", True)
    train, canonical = rag_eval.output_dir("beir/scifact/train", 0, 809, ph, split_size=809)
    assert not canonical and train == tmp_path / "runs" / "rag_beir-scifact-train_809_abcdef01"
    # A limited (smoke) run on the test split is never canonical either.
    smoke, canonical = rag_eval.output_dir(test, 5, 5, ph, split_size=5)
    assert not canonical and smoke.parent == tmp_path / "runs"


def test_a_partial_test_sample_is_never_canonical(monkeypatch, tmp_path):
    # The default SSR_RAG_N=50 must never overwrite the committed 300-claim artifact.
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    ph = "abcdef0123456789"
    test = rag_eval.CANONICAL_DATASET
    part, canonical = rag_eval.output_dir(test, 0, rag_eval.N, ph, split_size=300)
    assert not canonical and part == tmp_path / "runs" / f"rag_beir-scifact-test_{rag_eval.N}_abcdef01"
    assert rag_eval.output_dir(test, 0, 299, ph, split_size=300)[1] is False
    # An unknown split size is never assumed to be covered.
    assert rag_eval.output_dir(test, 0, 300, ph)[1] is False


def test_default_n_on_a_larger_test_split_writes_a_run_dir_not_eval_results(tmp_path, monkeypatch, capsys):
    out = tmp_path / "results"
    _patch_main(tmp_path, monkeypatch, out)
    many = {str(i): f"Claim number {i} holds." for i in range(1, 61)}
    monkeypatch.setattr(rag_eval, "load_queries_qrels", lambda dataset=None: (many, {}))
    monkeypatch.setattr(rag_eval, "load_claim_labels",
                        lambda dataset=None, query_ids=None: {q: ClaimLabel("NEI") for q in many})
    gen = type("Gen", (FakeGenerator,), {
        "generate": lambda self, q, hits: GeneratedAnswer(
            text="Nothing.", citations=[], hits=hits, verdict="NOT ENOUGH EVIDENCE"),
    })
    monkeypatch.setattr(rag_eval, "LLMGenerator", gen)
    rag_eval.main()  # SSR_RAG_N unset: the default 50 of 60
    assert not out.exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.startswith(f"rag_beir-scifact-test_{rag_eval.N}_")
    blob = json.loads((run_dir / "rag.json").read_text())
    assert blob["n"] == rag_eval.N and blob["run"]["canonical"] is False
    assert f"only SSR_RAG_N=all writes {out}" in capsys.readouterr().out


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
    gen, _ = setup("generate", "Two claim holds.", [RuntimeError("upstream 500")])
    with pytest.raises(SystemExit) as exc:  # the row is skipped, so the canonical write refuses
        rag_eval.main()
    assert "have no row (1 skipped" in str(exc.value.code) and "Re-run the same command" in str(exc.value.code)
    assert not out.exists()
    (ckpt,) = _checkpoint_files(tmp_path)
    assert "2" not in json.loads(ckpt.read_text())["rows"]
    gen2, _ = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == ["Two claim holds."]  # only the skipped row re-runs
    blob = json.loads((out / "rag.json").read_text())
    assert blob["n"] == 5 and blob["skipped"] == 0


def test_unparseable_judge_rows_are_not_checkpointed(rate_limited, tmp_path):
    setup, _, out = rate_limited
    setup("judge", "Four claim holds.", [JudgeParseError("garbage")])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    assert "0 skipped, 1 unparseable judge replies" in str(exc.value.code)
    assert not out.exists()
    (ckpt,) = _checkpoint_files(tmp_path)
    assert "4" not in json.loads(ckpt.read_text())["rows"]
    gen2, _ = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == ["Four claim holds."]


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
    gen, fake_judge = setup(source, "Two claim holds.", [_empty()] * fails)
    rag_eval.main()
    assert sleeps.count(WAIT_S) == fails
    # A judge-side empty completion retries only the judge: the finished generation
    # is not re-run (or re-paid for).
    gen_tries = [q for q, _ in gen.calls].count("Two claim holds.")
    assert gen_tries == (fails + 1 if source == "generate" else 1)
    assert fake_judge.queries.count("Two claim holds.") == (fails + 1 if source == "judge" else 1)
    blob = json.loads((out / "rag.json").read_text())
    assert blob["skipped"] == 0 and blob["n"] == len(QUERIES)


@pytest.mark.parametrize("source", ["generate", "judge"])
def test_empty_completion_skips_uncheckpointed_once_retries_are_exhausted(
    rate_limited, tmp_path, source
):
    setup, sleeps, out = rate_limited
    attempts = rag_eval.RATE_LIMIT_RETRIES + 1
    gen, fake_judge = setup(source, "Two claim holds.", [_empty()] * (attempts + 1))
    with pytest.raises(SystemExit) as exc:  # survives the row, then refuses the canonical write
        rag_eval.main()
    assert "have no row (1 skipped" in str(exc.value.code) and not out.exists()
    assert sleeps.count(WAIT_S) == rag_eval.RATE_LIMIT_RETRIES
    if source == "judge":
        assert [q for q, _ in gen.calls].count("Two claim holds.") == 1
        assert fake_judge.queries.count("Two claim holds.") == attempts
    else:
        assert [q for q, _ in gen.calls].count("Two claim holds.") == attempts
    (ckpt,) = _checkpoint_files(tmp_path)
    assert "2" not in json.loads(ckpt.read_text())["rows"]  # a re-run retries it
    gen2, _ = setup("generate", None, [])
    rag_eval.main()
    assert [q for q, _ in gen2.calls] == ["Two claim holds."]


def test_empty_completion_naming_the_daily_cap_stops_the_run(rate_limited):
    setup, sleeps, out = rate_limited
    gen, _ = setup("judge", "Three claim holds.",
                   [_empty("code 429 Rate limit exceeded: free-models-per-day")])
    with pytest.raises(SystemExit) as exc:
        rag_eval.main()
    assert "daily cap is exhausted" in str(exc.value.code)
    assert WAIT_S not in sleeps and not out.exists()


# --- non-default models are never canonical; per-model paid caps ----------------------------


def _ep(role, model, provider="openrouter"):
    url = "https://openrouter.ai/api/v1" if provider == "openrouter" else "https://api.groq.com/openai/v1"
    return LLMEndpoint(role, provider, url, model)


DEFAULT_GEN_EP = _ep("generator", DEFAULT_GEN_MODEL)
DEFAULT_JUDGE_EP = _ep("judge", DEFAULT_JUDGE_MODEL)


def test_default_endpoints_are_the_code_defaults_not_the_environment(monkeypatch):
    monkeypatch.setenv("SSR_OPENROUTER_LLM_MODEL", LUNA)  # e.g. a leftover in the shell
    monkeypatch.setattr(settings, "openrouter_llm_model", LUNA)
    assert rag_eval.default_endpoints() == (
        ("openrouter", DEFAULT_GEN_MODEL), ("openrouter", DEFAULT_JUDGE_MODEL),
    )
    assert rag_eval.non_default_roles(DEFAULT_GEN_EP, DEFAULT_JUDGE_EP) == []
    assert rag_eval.non_default_roles(_ep("generator", LUNA), DEFAULT_JUDGE_EP) == ["generator"]
    assert rag_eval.non_default_roles(
        _ep("generator", "openai/gpt-oss-120b", "groq"), _ep("judge", "qwen/qwen3.8-27b", "groq")
    ) == ["generator", "judge"]


def test_output_dir_non_default_models_are_never_canonical(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    ph, test = "abcdef0123456789", rag_eval.CANONICAL_DATASET
    assert rag_eval.output_dir(test, 0, 300, ph, DEFAULT_GEN_EP, DEFAULT_JUDGE_EP, split_size=300) == (
        tmp_path / "results", True
    )
    luna, canonical = rag_eval.output_dir(test, 0, 300, ph, _ep("generator", LUNA), DEFAULT_JUDGE_EP,
                                          split_size=300)
    assert not canonical
    assert luna == tmp_path / "runs" / "rag_beir-scifact-test_300_abcdef01_openai-gpt-6-luna"
    oss, _ = rag_eval.output_dir(test, 0, 300, ph, _ep("generator", PAID_GEN_MODEL), DEFAULT_JUDGE_EP)
    assert oss.name == "rag_beir-scifact-test_300_abcdef01_openai-gpt-oss-120b"
    # Another judge alone is non-canonical too, and names itself.
    judged, canonical = rag_eval.output_dir(
        test, 0, 300, ph, DEFAULT_GEN_EP, _ep("judge", "qwen/qwen3.8-27b:free")
    )
    assert not canonical and judged.parent == tmp_path / "runs"
    assert judged.name.endswith("_judge-qwen-qwen3-8-27b-free")


def test_non_default_generator_on_full_test_split_writes_eval_runs(tmp_path, monkeypatch, capsys):
    out = tmp_path / "results"
    _patch_main(tmp_path, monkeypatch, out, gen_model=LUNA)
    # A committed artifact with a different N: a canonical run would print REPLACES.
    out.mkdir()
    (out / "rag.json").write_text(json.dumps({"n": 50, "run": {"n_requested": 50}}))
    before = (out / "rag.json").read_text()
    monkeypatch.setenv("SSR_RAG_N", "all")
    rag_eval.main()
    assert (out / "rag.json").read_text() == before and sorted(out.iterdir()) == [out / "rag.json"]
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name == f"rag_beir-scifact-test_5_{rag_eval.prompt_hash()[:8]}_openai-gpt-6-luna"
    blob = json.loads((run_dir / "rag.json").read_text())
    run = blob["run"]
    assert run["canonical"] is False and run["default_models"] is False
    assert run["dataset"] == rag_eval.CANONICAL_DATASET and run["n_requested"] == "all"
    assert run["generator_model"] == LUNA and run["generator_temperature"] is None
    assert run["generator_reasoning_effort"] == "medium"
    assert run["generator_extra_body"] == {"provider": paid_routing(LUNA)}
    printed = capsys.readouterr().out
    upfront = printed.split("[1/5]")[0]
    assert "NON-CANONICAL: generator differs from the code defaults" in upfront
    assert f"generator: openrouter {LUNA} (code default: openrouter {DEFAULT_GEN_MODEL})" in upfront
    assert "REPLACES" not in printed and "(non-canonical: never writes" in upfront


def test_default_models_run_records_temperature_and_stays_canonical(rag_run):
    path, blob, _ = rag_run
    assert blob["run"]["canonical"] is True and blob["run"]["default_models"] is True
    assert blob["run"]["generator_temperature"] == 0.1
    assert not (path / "runs").exists()


def test_estimate_and_worst_case_use_the_generators_own_caps():
    luna, oss = _ep("generator", LUNA), _ep("generator", PAID_GEN_MODEL)
    budget = settings.llm_max_completion_tokens
    p = rag_eval.MAX_GEN_PROMPT_TOKENS

    def worst(pin, pout):
        return (2 * p * pin + 3 * budget * pout) / 1e6  # both attempts, the retry at 2x

    # Luna's prompt is bounded at its $0.125/M cache-write rate, above the $0.10 filter cap.
    assert rag_eval.worst_case_generation_cost(luna) == pytest.approx(worst(0.125, 0.50))
    assert rag_eval.worst_case_generation_cost(oss) == pytest.approx(worst(0.03, 0.17))
    typical = (rag_eval.EST_GEN_PROMPT_TOKENS * 0.125 + rag_eval.EST_GEN_COMPLETION_TOKENS * 0.50) / 1e6
    assert rag_eval.typical_generation_cost(luna) == pytest.approx(typical)
    line = rag_eval._estimate_line(300, luna, DEFAULT_JUDGE_EP, 5.0)
    assert f"est. ${300 * typical:.4f}" in line
    assert f"worst case ${300 * worst(0.125, 0.50):.4f}" in line
    assert f"{LUNA}'s billing bound {{'prompt': 0.125, 'completion': 0.5}}" in line
    assert "(max_price caps {'prompt': 0.1, 'completion': 0.5})" in line
    assert "0.03" not in line and "0.17" not in line  # never gpt-oss's caps for Luna


def test_luna_checkpoint_signature_differs_from_the_default_generator(monkeypatch):
    monkeypatch.setattr(rag_eval, "_index_fingerprint", lambda: None)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "auto")
    ds = rag_eval.CANONICAL_DATASET
    ling = rag_eval.signature_fields(ds, 300, DEFAULT_GEN_EP, DEFAULT_JUDGE_EP)
    luna = rag_eval.signature_fields(ds, 300, _ep("generator", LUNA), DEFAULT_JUDGE_EP)
    assert ling["generator"]["model"] == DEFAULT_GEN_MODEL and luna["generator"]["model"] == LUNA
    assert luna["reasoning_effort"] == "medium" and ling["reasoning_effort"] is None
    assert rag_eval.rag_signature(ling) != rag_eval.rag_signature(luna)


# --- verdict recovery on stored rows (reparse_row), resume and offline re-score -------------


def _stored(answer, verdict=None, judge_answered=True, truncated=False, gold="CONTRADICT"):
    return {
        "query_id": "7", "gold_label": gold, "verdict": verdict,
        "predicted_label": predicted_label(verdict, judge_answered)
        if verdict else ("NONE" if judge_answered else "NEI"),
        "answered": judge_answered, "answered_source": "verdict" if verdict else "judge",
        "judge_answered": judge_answered, "faithfulness": 1.0, "context_relevance": 1.0,
        "evidence": True, "evidence_qrels": True, "truncated": truncated,
        "abstention_class": "x", "abstention_class_qrels": "x",
        "answer": answer, "cited_doc_ids": ["a", "b"], "retrieved_doc_ids": ["a", "b", "c"],
    }


def test_reparse_row_recovers_an_inline_verdict_and_rederives_everything():
    r = rag_eval.reparse_row(_stored("Opposite effect [2]. Verdict: REFUTED [3]"), "Statins lower LDL cholesterol.")
    assert (r["verdict"], r["verdict_source"], r["predicted_label"]) == ("REFUTED", "inline", "CONTRADICT")
    assert r["answered"] is True and r["answered_source"] == "verdict"
    assert r["abstention_class"] == "answered_with_evidence"
    # The [3] on the stripped inline verdict is the model's citation too: kept.
    assert r["answer"] == "Opposite effect [2]." and r["cited_doc_ids"] == ["b", "c"]


def test_reparse_row_stance_overrides_the_judge_and_keeps_the_text():
    text = "The provided context does not discuss this. Passage [1] is about mice."
    r = rag_eval.reparse_row(_stored(text, judge_answered=True, gold="NEI"), "Statins lower LDL cholesterol.")
    assert (r["verdict"], r["verdict_source"], r["predicted_label"]) == (
        "NOT ENOUGH EVIDENCE", "stance", "NEI")
    assert r["answered"] is False and r["abstention_class"] == "false_abstention"
    assert r["answer"] == text


def test_reparse_row_leaves_parsed_and_unrecoverable_rows_alone():
    parsed = _stored("Refuted [1].", verdict="REFUTED")
    assert rag_eval.reparse_row(parsed, "c") == {**parsed, "verdict_source": "line"}
    plain = _stored("Nothing here settles it.")
    assert rag_eval.reparse_row(plain, "c") == {**plain, "verdict_source": None}
    # A question never gets a stance verdict, stored or fresh.
    q = _stored("The passages support the use of aspirin [1].")
    assert rag_eval.reparse_row(q, "Does aspirin help?")["verdict"] is None


def test_reparse_row_handles_the_truncation_note():
    note = rag_eval.TRUNCATION_NOTE
    cut = _stored(f"Opposite effect [1]. Verdict: REFUTED\n\n{note}", truncated=True)
    r = rag_eval.reparse_row(cut, "Statins lower LDL cholesterol.")
    assert r["verdict"] == "REFUTED" and r["answer"] == f"Opposite effect [1].\n\n{note}"
    empty = _stored(note, truncated=True, judge_answered=False)
    r = rag_eval.reparse_row(empty, "Statins lower LDL cholesterol.")
    # No verdict in a cut-off reply: NONE however the judge read it, never NEI.
    assert r["verdict"] is None and r["predicted_label"] == rag_eval.NO_VERDICT


def test_reparse_row_reads_no_stance_from_a_truncated_reply():
    note = rag_eval.TRUNCATION_NOTE
    claim = "Statins lower LDL cholesterol."
    cut = _stored(f"The context directly contradicts the claim [1]. It then\n\n{note}",
                  truncated=True, judge_answered=False)
    r = rag_eval.reparse_row(cut, claim)
    assert r["verdict"] is None and r["predicted_label"] == rag_eval.NO_VERDICT
    assert r["answer"] == cut["answer"]
    # A row stored with a stance verdict on a truncated reply (older code) is demoted, as
    # a fresh generation would now record it: the judge decides `answered` again.
    stored = {**cut, "verdict": "REFUTED", "verdict_source": "stance", "predicted_label": "CONTRADICT",
              "answered": True, "answered_source": "verdict"}
    r = rag_eval.reparse_row(stored, claim)
    assert (r["verdict"], r["verdict_source"], r["predicted_label"]) == (None, None, rag_eval.NO_VERDICT)
    assert r["answered"] is False and r["answered_source"] == "judge"
    assert r["abstention_class"] == "false_abstention"
    assert rag_eval.needs_reask(claim, r["verdict"])  # so the re-ask post-step picks it up
    # A complete reply's stance verdict is untouched.
    whole = {**stored, "truncated": False, "answer": "The context directly contradicts the claim [1]."}
    assert rag_eval.reparse_row(whole, claim)["verdict"] == "REFUTED"


def test_predicted_label_no_verdict_is_nei_only_for_a_complete_abstaining_reply():
    assert predicted_label(None, False) == "NEI"
    assert predicted_label(None, False, cut_off=True) == rag_eval.NO_VERDICT
    assert predicted_label(None, True, cut_off=True) == rag_eval.NO_VERDICT
    assert predicted_label("NOT ENOUGH EVIDENCE", False, cut_off=True) == "NEI"  # a verdict wins
    assert rag_eval._broken_reply(False, "") and rag_eval._broken_reply(False, rag_eval.TRUNCATION_NOTE)
    assert rag_eval._broken_reply(True, "Some prose.")
    assert not rag_eval._broken_reply(False, "The passages do not say.")


def test_resume_applies_the_current_parser_with_no_llm_call(rate_limited, tmp_path):
    setup, _, out = rate_limited
    setup("generate", None, [])
    rag_eval.main()
    (ckpt,) = _checkpoint_files(tmp_path)
    blob = json.loads(ckpt.read_text())
    qid = next(q for q, r in blob["rows"].items() if r["answer"] == REPLIES["Five claim holds."])
    # A row checkpointed by an older parser: no verdict, reply ending "... Verdict: X".
    blob["rows"][qid].update(answer="Passage [1] reports the opposite [1]. Verdict: REFUTED",
                             verdict=None, predicted_label="NONE", answered_source="judge")
    ckpt.write_text(json.dumps(blob))
    gen2, judge2 = setup("generate", None, [])
    rag_eval.main()
    assert gen2.calls == [] and judge2.queries == []
    out_row = next(r for r in json.loads((out / "rag.json").read_text())["rows"] if r["query_id"] == qid)
    assert (out_row["verdict"], out_row["verdict_source"], out_row["predicted_label"]) == (
        "REFUTED", "inline", "CONTRADICT")


def test_rag_json_records_verdict_sources_and_quote_stats(rag_run):
    _, blob, _ = rag_run
    assert blob["verdict_sources"] == {"line": 3, "inline": 0, "stance": 0, "reask": 0}
    assert blob["evidence_quotes"]["no_quote"]["n"] == 5  # the fakes quote nothing
    assert all("quote_found" in r and "verdict_source" in r for r in blob["rows"])


def test_rescore_rebuilds_aggregates_and_refuses_eval_results(tmp_path, monkeypatch):
    from app.eval import rag_rescore

    rows = [_stored("Opposite effect [2]. Verdict: REFUTED"), {**_stored("Nothing."), "query_id": "8"}]
    blob = {"verdict_accuracy": 0.0, "run": {"dataset": "d", "canonical": True}, "rows": rows}
    out = rag_rescore.rescore(blob, {"7": "a claim", "8": "another claim"})
    assert out["verdict_accuracy"] == 0.5 and out["verdict_sources"]["inline"] == 1
    assert out["run"]["canonical"] is False and out["run"]["rescored"] is True
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    with pytest.raises(SystemExit, match="refusing"):
        rag_rescore.main(["in.json", str(tmp_path / "results" / "rag.json")])


# --- the verdict-only re-ask post-step -----------------------------------------------------

from app.eval.reply_cache import ReplyCache  # noqa: E402
from app.generate.generator import REASK_MAX_TOKENS, REASK_NOTE, reask_prompt_hash  # noqa: E402
from app.generate.prompts import reask_messages  # noqa: E402

NO_VERDICT_CLAIMS = {"Four claim holds.", "Five claim holds."}  # the two replies with no verdict


def _reask_entries(tmp_path):
    path = tmp_path / "reask_cache" / "beir-scifact-test.json"
    return json.loads(path.read_text())["entries"] if path.exists() else {}


def test_reask_fires_once_per_claim_without_a_verdict_and_sets_it(tmp_path, monkeypatch):
    _patch_main(tmp_path, monkeypatch, tmp_path)
    monkeypatch.setattr(FakeGenerator, "REASK_REPLY", "Verdict: REFUTED [1]")
    rag_eval.main()
    blob = json.loads((tmp_path / "rag.json").read_text())
    rows = {r["query_id"]: r for r in blob["rows"]}
    # Exactly one extra call per claim with no verdict, none for the three that had one.
    assert sorted(q for q, _ in FakeGenerator.reask_calls) == sorted(NO_VERDICT_CLAIMS)
    for qid in ("1", "2", "3"):
        assert rows[qid]["reask_attempted"] is False and "reask" not in rows[qid]
    r4, r5 = rows["4"], rows["5"]
    for r in (r4, r5):
        assert r["reask_attempted"] is True
        assert (r["verdict"], r["verdict_source"], r["predicted_label"]) == ("REFUTED", "reask", "CONTRADICT")
        assert r["answered"] is True and r["answered_source"] == "reask"
        assert r["reask"]["parsed"] == "REFUTED" and r["reask"]["finish_reason"] == "stop"
        assert r["reask"]["completion_tokens"] == 7 and r["reask"]["reasoning_tokens"] == 3
        assert r["first_pass"]["verdict"] is None and r["first_pass"]["verdict_source"] is None
    # q4's first reply was cut off with no verdict: NONE, not NEI (the judge said abstained).
    assert r4["first_pass"]["predicted_label"] == "NONE" and r5["first_pass"]["predicted_label"] == "NONE"
    # Prose answers are kept as served; the judge's scores stay the first reply's.
    assert r4["answer"] == "Nothing here settles it." and r4["faithfulness"] == 0.9
    assert rag_eval.first_pass_row(r4)["predicted_label"] == "NONE"
    assert blob["verdict_sources"]["reask"] == 2 and blob["reasked_answers"] == 2
    assert blob["verdict_accuracy"] == 0.8  # q5 (CONTRADICT) is fixed, q4 (SUPPORT) is not
    assert blob["run"]["reask"]["prompt_hash"] == reask_prompt_hash()
    assert blob["run"]["reask_replies"] == {"cached": 0, "fetched": 2, "failed": 0}
    assert "| Re-asked (claim with no verdict: one verdict-only call) | 2 of 5 (2 gave a verdict) |" in (
        tmp_path / "rag.md").read_text()


def test_reask_replies_are_written_under_v2_keys_only(tmp_path, monkeypatch):
    _patch_main(tmp_path, monkeypatch, tmp_path)
    rag_eval.main()
    gen_ep = rag_eval.resolve_endpoint("generator", require_key=False)
    hits = [SearchHit("d1", 1.0, "t1"), SearchHit("d2", 0.5, "t2")]
    msgs = {qid: reask_messages(QUERIES[qid], hits) for qid in ("4", "5")}
    entries = _reask_entries(tmp_path)
    assert set(entries) == {rag_eval.reask_key_v2(qid, gen_ep, m) for qid, m in msgs.items()}
    assert not {rag_eval.reask_key(qid, gen_ep.model, m) for qid, m in msgs.items()} & set(entries)
    assert all(k.startswith("reask|v2|") for k in entries)
    fp = rag_eval.reask_request_fingerprint(gen_ep)
    assert all(e["kind"] == "reask" and e["max_tokens"] == REASK_MAX_TOKENS
               and e["key_version"] == 2 and e["request"] == fp for e in entries.values())


# --- re-ask cache keys: v2 (endpoint fingerprint) + the legacy-key migration ---------------

CANON_GEN_MODEL = rag_eval.default_endpoints()[0][1]
GROQ_URL = "https://api.groq.com/openai/v1"
REASK_MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "claim"}]


def _canon_ep(model=CANON_GEN_MODEL):
    return LLMEndpoint("generator", "openrouter", "https://openrouter.ai/api/v1", model)


@pytest.fixture()
def auto_effort(monkeypatch):
    monkeypatch.setattr(settings, "llm_reasoning_effort", "auto")


def test_reask_lookup_takes_a_legacy_entry_only_for_the_canonical_endpoint(tmp_path, auto_effort):
    cache = ReplyCache(tmp_path / "c.json")
    cache.put(rag_eval.reask_key("7", CANON_GEN_MODEL, REASK_MSGS), {"raw": "Verdict: SUPPORTED"})
    canon = _canon_ep()
    assert rag_eval.is_legacy_reask_endpoint(canon)
    assert rag_eval.reask_lookup(cache, "7", canon, REASK_MSGS)["raw"] == "Verdict: SUPPORTED"
    # Same model id + messages on another provider: the legacy key would collide, so the
    # migration never reads it there.
    groq = LLMEndpoint("generator", "groq", GROQ_URL, CANON_GEN_MODEL)
    assert not rag_eval.is_legacy_reask_endpoint(groq)
    assert rag_eval.reask_lookup(cache, "7", groq, REASK_MSGS) is None
    assert rag_eval.reask_lookup(cache, "8", canon, REASK_MSGS) is None  # other claim


def test_reask_lookup_v2_hits_and_is_endpoint_specific(tmp_path, auto_effort):
    cache = ReplyCache(tmp_path / "c.json")
    groq = LLMEndpoint("generator", "groq", GROQ_URL, CANON_GEN_MODEL)
    cache.put(rag_eval.reask_key_v2("7", groq, REASK_MSGS), {"raw": "Verdict: REFUTED"})
    assert rag_eval.reask_lookup(cache, "7", groq, REASK_MSGS)["raw"] == "Verdict: REFUTED"
    assert rag_eval.reask_lookup(cache, "7", _canon_ep(), REASK_MSGS) is None
    other_url = LLMEndpoint("generator", "groq", "http://localhost:11434/v1", CANON_GEN_MODEL)
    assert rag_eval.reask_lookup(cache, "7", other_url, REASK_MSGS) is None
    # v2 wins over a legacy entry for the same request on the canonical endpoint.
    canon = _canon_ep()
    cache.put(rag_eval.reask_key("7", CANON_GEN_MODEL, REASK_MSGS), {"raw": "legacy"})
    cache.put(rag_eval.reask_key_v2("7", canon, REASK_MSGS), {"raw": "v2"})
    assert rag_eval.reask_lookup(cache, "7", canon, REASK_MSGS)["raw"] == "v2"


def test_v2_key_covers_reasoning_effort_temperature_and_routing(monkeypatch, auto_effort):
    canon = _canon_ep()
    base = rag_eval.reask_key_v2("7", canon, REASK_MSGS)
    fp = rag_eval.reask_request_fingerprint(canon)
    assert fp == {
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "model": CANON_GEN_MODEL, "reasoning_effort": None, "temperature": 0.1,
        "extra_body": rag_eval.LEGACY_REASK_EXTRA_BODY,
    }
    monkeypatch.setattr(settings, "llm_reasoning_effort", "high")
    assert rag_eval.reask_key_v2("7", canon, REASK_MSGS) != base
    assert not rag_eval.is_legacy_reask_endpoint(canon)  # an explicit effort: no migration
    monkeypatch.setattr(settings, "llm_reasoning_effort", "auto")
    monkeypatch.setattr(rag_eval, "generation_temperature", lambda m: None)
    assert rag_eval.reask_key_v2("7", canon, REASK_MSGS) != base
    assert not rag_eval.is_legacy_reask_endpoint(canon)
    monkeypatch.undo()
    monkeypatch.setattr(settings, "llm_reasoning_effort", "auto")
    monkeypatch.setattr(LLMEndpoint, "extra_body", lambda self: {"provider": {"sort": "price"}})
    assert rag_eval.reask_key_v2("7", canon, REASK_MSGS) != base
    assert not rag_eval.is_legacy_reask_endpoint(canon)


def test_main_off_the_canonical_endpoint_ignores_legacy_entries_and_writes_v2(
    rate_limited, tmp_path, monkeypatch
):
    # Legacy entries the canonical endpoint would hit; with the endpoint treated as
    # non-canonical (as a Groq / re-routed one is) main must fetch fresh, under v2 keys.
    setup, _, out = rate_limited
    setup("generate", None, [])
    settings.llm_reask = False
    rag_eval.main()
    setup("generate", None, [])
    hits = rag_eval.reask_hits(["d1", "d2"], {d["doc_id"]: d for d in CORPUS})
    cache = ReplyCache(tmp_path / "reask_cache" / "beir-scifact-test.json")
    for qid in ("4", "5"):
        msgs = reask_messages(QUERIES[qid], hits)
        cache.put(rag_eval.reask_key(qid, DEFAULT_GEN_MODEL, msgs),
                  {"raw": "Verdict: SUPPORTED [2]", "finish_reason": "stop", "kind": "reask"})
    monkeypatch.setattr(rag_eval, "is_legacy_reask_endpoint", lambda ep: False)
    rag_eval.main()
    assert len(FakeGenerator.reask_calls) == 2  # the legacy entries were not taken
    entries = _reask_entries(tmp_path)
    assert sum(k.startswith("reask|v2|") for k in entries) == 2


# --- faithfulness: re-ask-only answers whose first reply was a placeholder -----------------


def _reask_row(first_answer, faith=0.0, answered=True, flag=None, display=None):
    r = {**row(answered=answered, evidence=True, faith=faith), "verdict_source": "reask",
         "answer": display if display is not None else REASK_NOTE,
         "first_pass": {"answer": first_answer}}
    if flag is not None:
        r["judge_scored_placeholder"] = flag
    return r


def test_faithfulness_leaves_out_reask_answers_judged_on_a_placeholder():
    rows = [
        row(answered=True, evidence=True, faith=0.9),
        _reask_row(TRUNCATED_ANSWER, flag=True),             # flagged (fresh run)
        _reask_row(TRUNCATED_ANSWER),                         # committed rows: no flag
        _reask_row("", display="Verdict-stripped text"),     # detected via first_pass
        _reask_row("Real prose [1].", faith=0.7, display="Real prose [1]."),  # judged
        _reask_row(TRUNCATED_ANSWER, answered=False),         # abstained: not in the mean
    ]
    agg = aggregate(rows)
    assert agg["faithfulness_answered"] == 0.8
    assert agg["faithfulness_n"] == 2 and agg["faithfulness_unjudged_reask"] == 3
    assert agg["context_relevance"] == 1.0  # over ALL rows, placeholders included
    # The cause is read from the rows: 2 first replies hit the token budget, 1 was empty.
    md = rag_eval._markdown(agg, 0, 0, rows=rows)
    assert ("| Faithfulness (over answered, n=2; 3 answered only via the re-ask excluded — the "
            "first reply hit the token budget with no text left (2) or was empty (1), so the "
            "judge saw no answer text) | 0.80 |") in md
    only_cut = rows[:3] + rows[4:]  # the empty-first-reply row dropped
    agg_cut = aggregate(only_cut)
    assert ("2 answered only via the re-ask excluded — the first reply hit the token budget "
            "with no text left, so the judge saw no answer text") in rag_eval._markdown(
                agg_cut, 0, 0, rows=only_cut)
    # Without the rows the writer names no cause it cannot see.
    assert "the first reply had no judgeable text" in rag_eval._markdown(agg, 0, 0)
    clean = aggregate([row(answered=True, evidence=True, faith=0.9)])
    assert clean["faithfulness_unjudged_reask"] == 0 and clean["faithfulness_n"] == 1
    assert "| Faithfulness (over answered) | 0.90 |" in rag_eval._markdown(clean, 0, 0)


def test_apply_reask_flags_a_placeholder_and_recomputes_citations():
    base = {"query_id": "1", "verdict": None, "verdict_source": None, "predicted_label": "NEI",
            "answered": False, "answered_source": "judge", "abstention_class": "false_abstention",
            "abstention_class_qrels": "false_abstention", "evidence": True, "evidence_qrels": True,
            "retrieved_doc_ids": ["d1", "d2"]}
    rec = {"raw": "Verdict: SUPPORTED [1]", "finish_reason": "stop"}
    empty = rag_eval.apply_reask({**base, "answer": TRUNCATED_ANSWER, "cited_doc_ids": []}, rec)
    assert empty["judge_scored_placeholder"] is True and empty["cited_doc_ids"] == []
    assert rag_eval.judge_scored_placeholder(empty)
    prose = rag_eval.apply_reask({**base, "answer": "Passage [2] says so.", "cited_doc_ids": []}, rec)
    assert prose["judge_scored_placeholder"] is False and prose["cited_doc_ids"] == ["d2"]
    assert not rag_eval.judge_scored_placeholder(prose)
    assert "judge_scored_placeholder" not in rag_eval.first_pass_row(prose)


def test_quote_stats_count_rows_without_the_key_as_unknown():
    rows = [{**row(answered=True, evidence=True), "quote_found": True},
            {**row(answered=True, evidence=True), "quote_found": None},
            {**row(answered=True, evidence=True, pred="NEI"), "quote_found": None},
            row(answered=True, evidence=True)]  # legacy / resumed row: no key at all
    q = rag_eval._quote_stats(rows)
    assert {k: v["n"] for k, v in q.items()} == {"found": 1, "not_found": 0, "no_quote": 2, "unknown": 1}
    assert q["no_quote"]["verdict_accuracy"] == 0.5 and q["unknown"]["verdict_accuracy"] == 1.0


def test_rescore_refuses_any_path_under_eval_results(tmp_path, monkeypatch):
    from app.eval import rag_rescore

    results = tmp_path / "results"
    (results / "sub").mkdir(parents=True)
    monkeypatch.setattr(rag_eval, "OUT", results)
    link = tmp_path / "innocent"
    link.symlink_to(results, target_is_directory=True)
    for dst in (results / "rag.json", results / "sub" / "deep" / "rag.json",
                tmp_path / "elsewhere" / ".." / "results" / "x.json", link / "rag.json", results):
        with pytest.raises(SystemExit, match="refusing"):
            rag_rescore.main(["in.json", str(dst)])
    assert not rag_rescore.is_under_results(tmp_path / "results_other" / "rag.json")
    assert not rag_rescore.is_under_results(tmp_path / "runs" / "rag.json")


def test_rerun_from_a_complete_checkpoint_reasks_from_the_cache_with_no_llm_call(rate_limited, tmp_path):
    setup, _, out = rate_limited
    setup("generate", None, [])
    rag_eval.main()
    first = json.loads((out / "rag.json").read_text())
    assert len(FakeGenerator.reask_calls) == 2
    gen2, judge2 = setup("generate", None, [])  # resets reask_calls
    rag_eval.main()
    assert gen2.calls == [] and judge2.queries == [] and FakeGenerator.reask_calls == []
    second = json.loads((out / "rag.json").read_text())
    assert second["rows"] == first["rows"]
    assert second["run"]["reask_replies"] == {"cached": 2, "fetched": 0, "failed": 0}


def test_a_reply_cached_by_rag_secondlook_is_a_hit_for_a_resumed_row(rate_limited, tmp_path, capsys):
    # A complete first-pass checkpoint (re-ask off), then a cache file written the way
    # rag_secondlook writes it, with passages rebuilt from the corpus: the canonical
    # re-run must take the reply from it and make no call at all.
    setup, _, out = rate_limited
    setup("generate", None, [])
    settings.llm_reask = False
    rag_eval.main()
    assert FakeGenerator.reask_calls == []
    gen2, judge2 = setup("generate", None, [])  # llm_reask back on
    hits = rag_eval.reask_hits(["d1", "d2"], {d["doc_id"]: d for d in CORPUS})
    cache = ReplyCache(tmp_path / "reask_cache" / "beir-scifact-test.json")
    for qid in ("4", "5"):
        msgs = reask_messages(QUERIES[qid], hits)
        cache.put(ReplyCache.key("reask", qid, DEFAULT_GEN_MODEL, REASK_MAX_TOKENS, msgs),
                  {"raw": "Verdict: SUPPORTED [2]", "finish_reason": "stop", "kind": "reask"})
    capsys.readouterr()
    rag_eval.main(["--check-quota"])  # nothing needed: no /key request either
    printed = capsys.readouterr().out
    assert "2 resumed claims have no verdict -> 2 replies cached" in printed
    assert "LLM requests needed now: ~0 (0 on OpenRouter's free caps)" in printed
    assert "no free OpenRouter requests needed" in printed
    assert gen2.calls == [] and judge2.queries == [] and FakeGenerator.reask_calls == []
    blob = json.loads((out / "rag.json").read_text())
    assert {r["query_id"]: r["predicted_label"] for r in blob["rows"]}["4"] == "SUPPORT"
    assert blob["run"]["canonical"] is True


def test_estimate_counts_the_reasks_resumed_rows_still_need(rate_limited, tmp_path, capsys):
    setup, _, out = rate_limited
    setup("generate", None, [])
    settings.llm_reask = False
    rag_eval.main()  # first pass only: nothing cached
    setup("generate", None, [])
    capsys.readouterr()
    rag_eval.main()
    printed = capsys.readouterr().out
    assert "2 resumed claims have no verdict -> 0 replies cached" in printed and "2 to fetch" in printed
    assert "LLM requests needed now: ~2 (2 on OpenRouter's free caps)" in printed
    assert len(FakeGenerator.reask_calls) == 2


def test_reask_off_never_calls_and_is_never_canonical(tmp_path, monkeypatch):
    _patch_main(tmp_path, monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "llm_reask", False)
    rag_eval.main()
    assert FakeGenerator.reask_calls == []
    assert not (tmp_path / "rag.json").exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_noreask")
    blob = json.loads((run_dir / "rag.json").read_text())
    assert all(r["reask_attempted"] is False for r in blob["rows"])
    assert blob["run"]["reask"]["enabled"] is False and blob["run"]["canonical"] is False


def test_first_pass_generation_never_reasks_inside_generate(tmp_path, monkeypatch):
    built = []

    class Recording(FakeGenerator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self.reask_enabled)

    _patch_main(tmp_path, monkeypatch, tmp_path, generator=Recording)
    rag_eval.main()
    assert built == [False]  # the post-step owns the re-ask (and its cache)


def test_a_failed_reask_keeps_the_first_pass_and_is_retried_next_run(rate_limited, tmp_path, monkeypatch):
    setup, _, out = rate_limited
    setup("generate", None, [])

    def boom(self, query, hits):
        type(self).reask_calls.append((query, hits))
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(FakeGenerator, "reask_verdict", boom)
    with pytest.raises(SystemExit) as exc:  # canonical: a failed re-ask blocks the write
        rag_eval.main()
    assert "2 re-asks failed: 4, 5" in str(exc.value.code) and not out.exists()
    assert _reask_entries(tmp_path) == {}
    monkeypatch.setenv("SSR_EVAL_LIMIT", "5")  # non-canonical: written, first pass kept
    rag_eval.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    rows = {r["query_id"]: r for r in json.loads((run_dir / "rag.json").read_text())["rows"]}
    assert rows["4"]["reask_attempted"] and rows["4"]["reask_error"] == "RuntimeError"
    assert rows["4"]["predicted_label"] == "NONE" and _reask_entries(tmp_path) == {}
    monkeypatch.delenv("SSR_EVAL_LIMIT")
    monkeypatch.undo()  # back to the working fake (and fresh patches below)
    setup("generate", None, [])
    rag_eval.main()
    assert len(FakeGenerator.reask_calls) == 2 and len(_reask_entries(tmp_path)) == 2


def test_first_pass_row_undoes_the_reask():
    row = {"query_id": "1", "verdict": None, "verdict_source": None, "predicted_label": "NEI",
           "answered": False, "answered_source": "judge", "abstention_class": "false_abstention",
           "abstention_class_qrels": "false_abstention", "answer": TRUNCATED_ANSWER,
           "cited_doc_ids": [], "evidence": True, "evidence_qrels": True}
    out = rag_eval.apply_reask(row, {"raw": "Verdict: SUPPORTED [1]", "finish_reason": "stop"})
    assert out["verdict_source"] == "reask" and out["answer"] == REASK_NOTE
    assert out["abstention_class"] == "answered_with_evidence"
    assert rag_eval.first_pass_row(out) == row
    unparsed = rag_eval.apply_reask(row, {"raw": "unsure", "finish_reason": "length"})
    assert unparsed["reask"]["parsed"] is None and unparsed["predicted_label"] == "NEI"
    assert rag_eval.first_pass_row(unparsed) == row


TRUNCATED_ANSWER = "[Answer truncated: the model ran out of its token budget.]"


def test_request_estimate_adds_reasks():
    gen, judge_ep = _endpoints()
    est = rag_eval.request_estimate(10, gen, judge_ep, 12.0, 0.2, reask_rate=0.1, reask_pending=3,
                                    reask=True)
    assert est["requests"] == 10 * 2 + 2 + 1 + 3  # gen + judge, retries, re-asks, pending
    assert est["free_requests"] == 26 and est["free_requests_worst"] == 10 * 4 + 3
    off = rag_eval.request_estimate(10, gen, judge_ep, 12.0, 0.2, reask_rate=0.1, reask_pending=3)
    assert off["requests"] == 22 and off["reask_pending"] == 0


def test_reparse_row_reads_no_stance_for_a_keyword_query():
    # Same gate as generate(): a keyword query or an instruction is not a claim to judge.
    text = "The passages support the claim [1]."
    assert rag_eval.reparse_row(_stored(text), "Statins lower LDL cholesterol.")["verdict"] == "SUPPORTED"
    for q in ("BRCA1 breast cancer risk", "Explain how statins lower LDL"):
        assert rag_eval.reparse_row(_stored(text), q)["verdict"] is None


# --- rag.md notes derived from the rows, and the offline re-render ---------------------------


def test_truncation_note_breaks_the_count_down_from_the_rows():
    cut = {"truncated": True}
    rows = [
        {**row(True, True), **cut, "verdict_source": "reask"},
        {**row(True, True), **cut, "verdict_source": "reask"},
        {**row(True, True), **cut, "verdict_source": "stance"},
        {**row(False, True, verdict=None, pred="NEI"), **cut, "verdict_source": None},
        row(True, True),  # not truncated: not counted
    ]
    agg = aggregate(rows)
    assert agg["truncated_answers"] == 4
    md = rag_eval._markdown(agg, 0, 0, rows=rows)
    assert ("4 answers truncated at the token budget (of those, 2 got a verdict from the "
            "verdict-only re-ask, 1 kept a verdict parsed from the text before the cut-off, 1 "
            "was scored with no verdict).") in md
    only_none = [rows[3]]
    assert ("1 answer truncated at the token budget (of those, 1 was scored with no verdict)."
            in rag_eval._markdown(aggregate(only_none), 0, 0, rows=only_none))
    # Without the rows: the count only, no claim about verdicts it cannot see.
    assert "4 answers truncated at the token budget.\n" in rag_eval._markdown(agg, 0, 0)
    assert "scored as written" not in md


def test_render_md_rewrites_only_the_markdown_from_the_stored_json(tmp_path, capsys):
    rows = [{**row(True, True), "truncated": True, "verdict_source": "reask"}, row(False, False)]
    agg = aggregate(rows, "gen-m", "judge-m", "openrouter", "openrouter")
    blob = {**agg, "skipped": 0, "judge_parse_failures": 0,
            "run": {"dataset": "beir/scifact/test"}, "rows": rows}
    src = tmp_path / "rag.json"
    src.write_text(json.dumps(blob, indent=2))
    before = src.read_bytes()
    rag_eval.main(["--render-md", str(src)])  # no endpoint, no retrieval, no LLM
    assert src.read_bytes() == before
    md = (tmp_path / "rag.md").read_text()
    # No `no_verdict_scoring` in this blob's run: rendered under the rule it was scored by.
    assert md == rag_eval._markdown(agg, 0, 0, "beir/scifact/test", rows, no_verdict_scoring=None)
    assert "counts as NEI if it abstained and as `NONE`" in md
    assert "1 answer truncated at the token budget (of those, 1 got a verdict" in md
    assert "Wrote" in capsys.readouterr().out


def test_committed_rag_md_is_the_render_of_the_committed_json():
    path = rag_eval.OUT / "rag.json"
    if not path.exists():
        pytest.skip("no committed rag.json")
    md = (rag_eval.OUT / "rag.md").read_text()
    assert rag_eval.markdown_from_json(json.loads(path.read_text())) == md
    assert "empty first reply" not in md and "scored as written" not in md


# --- opt-in experiments: self-consistency vote, prompt variant, no judge ----------------------

# The signature keys of a default run: the vote, the prompt variant (via prompt_hash) and the
# judge switch must add none, so existing checkpoints keep matching.
DEFAULT_SIGNATURE_KEYS = {
    "dataset", "n_sample", "sample_seed", "generator", "judge", "prompt_hash",
    "judge_prompt_hash", "top_k", "mode", "reranker", "rerank_candidates",
    "max_completion_tokens", "reasoning_effort", "rrf_k", "candidate_k", "embedding_model",
    "embedding_query_prefix",
}


def test_default_signature_has_no_new_keys_and_the_default_prompt_hash_is_pinned(monkeypatch):
    monkeypatch.setattr(rag_eval, "_index_fingerprint", lambda: None)
    monkeypatch.setattr(settings, "llm_prompt_variant", "default")
    gen, judge_ep = _endpoints()
    base = rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep)
    assert set(base) == DEFAULT_SIGNATURE_KEYS
    assert rag_eval.prompt_hash().startswith("d0921f4e")  # the committed runs' prompt
    monkeypatch.setattr(settings, "llm_votes", 3)  # the vote is not in the base signature
    assert rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep) == base
    skipped = rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep, judge_skipped=True)
    assert skipped == {**base, "judge_skipped": True}
    monkeypatch.setattr(settings, "llm_prompt_variant", "finding")
    finding = rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep)
    assert set(finding) == DEFAULT_SIGNATURE_KEYS and finding["prompt_hash"] != base["prompt_hash"]
    assert rag_eval.prompt_hash() == rag_eval.prompt_hash("finding") != rag_eval.prompt_hash("default")


def test_output_dir_opt_in_experiments_are_never_canonical(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    ph, test = "abcdef0123456789", rag_eval.CANONICAL_DATASET
    assert rag_eval.output_dir(test, 0, 300, ph, split_size=300)[1] is True
    for kw, suffix in (({"votes": 3}, "_votes3"), ({"judge_skipped": True}, "_nojudge"),
                       ({"prompt_variant": "finding"}, "_abcdef01"),
                       ({"votes": 5, "judge_skipped": True}, "_votes5_nojudge")):
        path, canonical = rag_eval.output_dir(test, 0, 300, ph, split_size=300, **kw)
        assert not canonical and path.name.endswith(suffix), (kw, path)


# Extra samples (2, 3, ...) per claim; sample #1 is the ordinary first pass (REPLIES).
SAMPLE_REPLIES = {
    "One claim holds.": ["Yes [1].\nVerdict: SUPPORTED", "Yes [2].\nVerdict: SUPPORTED"],
    "Two claim holds.": ["Supported [2].\nVerdict: SUPPORTED", "Supported [1].\nVerdict: SUPPORTED"],
    "Three claim holds.": ["Yes [1].\nVerdict: SUPPORTED", "No [1].\nVerdict: REFUTED"],
    "Four claim holds.": ["Yes [2].\nVerdict: SUPPORTED", "Yes [1].\nVerdict: SUPPORTED"],
    "Five claim holds.": ["Nothing here settles it.", "Refuted [1].\nVerdict: REFUTED"],
}


def _voting_generator():
    """A FakeGenerator whose first call per claim (ever, across main() runs in one test) is
    the first pass and every later one the claim's next vote sample. ``fail_at``: raise the
    OpenRouter daily cap on that (0-based) extra call."""

    class Voting(FakeGenerator):
        first_pass: set = set()
        extra: list = []
        fail_at: int | None = None

        def generate(self, query, hits):
            cls = type(self)
            if query not in cls.first_pass:
                cls.first_pass.add(query)
                return super().generate(query, hits)
            if cls.fail_at is not None and len(cls.extra) == cls.fail_at:
                raise rate_limit_error(OPENROUTER_DAILY_MESSAGE)
            j = sum(q == query for q in cls.extra)
            cls.extra.append(query)
            text, verdict = split_verdict(SAMPLE_REPLIES[query][j])
            return GeneratedAnswer(text=text, citations=[], hits=hits, verdict=verdict,
                                   finish_reason="stop", attempts=1, cost_usd=0.0)

    return Voting


def _vote_main(tmp_path, monkeypatch, gen, votes=3, judge=None):
    fake_judge = _patch_main(tmp_path, monkeypatch, tmp_path, generator=gen, judge=judge)
    monkeypatch.setattr(settings, "llm_votes", votes)
    monkeypatch.setattr(rag_eval.time, "sleep", lambda s: None)
    rag_eval.main()
    return fake_judge


def _votes_dir(tmp_path, k=3):
    (run_dir,) = [d for d in (tmp_path / "runs").iterdir() if d.name.endswith(f"_votes{k}")]
    return run_dir, json.loads((run_dir / "rag.json").read_text())


def test_votes3_run_is_non_canonical_with_vote_blocks_and_judges_only_a_new_served_sample(
    tmp_path, monkeypatch,
):
    gen = _voting_generator()
    fake_judge = _vote_main(tmp_path, monkeypatch, gen)
    assert not (tmp_path / "rag.json").exists()  # never OUT
    run_dir, blob = _votes_dir(tmp_path)
    rows = {r["query_id"]: r for r in blob["rows"]}
    assert len(gen.extra) == 10  # samples 2 and 3 of all 5 claims (the eval draws all k)
    chosen = {q: r["vote"]["chosen"] for q, r in rows.items()}
    assert chosen == {"1": 1, "2": 2, "3": 1, "4": 2, "5": 3}
    # 5 first-pass judge calls + exactly one per claim whose served sample is not #1.
    assert fake_judge.calls == blob["judge_calls"] == 5 + 3
    assert rows["2"]["verdict"] == "SUPPORTED" and rows["2"]["vote"]["changed"] is True
    assert rows["2"]["answer"] == "Supported [2]." and rows["2"]["faithfulness"] == 0.9
    assert rows["3"]["vote"]["split"] == "1-1-1" and rows["3"]["vote"]["tie"] is True
    assert rows["3"]["predicted_label"] == "NEI"  # a tie keeps sample #1
    assert rows["5"]["vote"]["verdicts"] == [None, None, "REFUTED"]
    assert rows["5"]["vote"]["abstained"] == 2 and rows["5"]["predicted_label"] == "CONTRADICT"
    # Sample 2 of claim 5 had no verdict: its OWN re-ask (pass 3 re-asked claims 4 and 5).
    assert sorted(q for q, _ in FakeGenerator.reask_calls) == sorted(
        ["Four claim holds.", "Five claim holds.", "Five claim holds."])
    v = blob["votes"]
    assert v["k"] == 3 and v["splits"] == {"1": 1, "1-1-1": 1, "2": 1, "2-1": 1, "3": 1}
    assert v["ties"] == 1 and v["changed_vs_sample1"] == 3
    assert v["sample_accuracy"] == [0.6, 0.4, 0.6] and v["tie_nei_accuracy"] == 0.8
    assert blob["verdict_accuracy"] == 0.8 and blob["missing_vote_samples"] == []
    run = blob["run"]
    assert run["canonical"] is False and run["votes"]["k"] == 3
    assert run["votes"]["vote_signature"] != run["checkpoint_signature"]
    assert run["requests_this_session"]["vote_sample_requests"] == 10 + 1  # 10 samples + 1 re-ask
    samples = json.loads((tmp_path / "cache" / f"{run['checkpoint_signature']}.samples.json").read_text())
    assert samples["signature"] == run["checkpoint_signature"]
    assert set(samples["samples"]["2"]) == {"2", "3"} and "judge" in samples["samples"]["2"]["2"]
    assert samples["samples"]["5"]["2"]["requests"] == 2  # generation + its own re-ask
    assert "## Self-consistency vote (k=3)" in (run_dir / "rag.md").read_text()

    # A re-run draws nothing and judges nothing: every sample, re-ask and judge is cached.
    before = (len(gen.extra), fake_judge.calls, len(FakeGenerator.reask_calls))
    rag_eval.main()
    assert (len(gen.extra), fake_judge.calls, len(FakeGenerator.reask_calls)) == before
    assert json.loads((run_dir / "rag.json").read_text())["rows"] == blob["rows"]


def test_sample_one_reuses_the_unchanged_checkpoint_of_a_k1_run(tmp_path, monkeypatch):
    gen = _voting_generator()
    _vote_main(tmp_path, monkeypatch, gen, votes=1)
    k1 = json.loads((tmp_path / "rag.json").read_text())  # k = 1: canonical as before
    assert "vote" not in k1["rows"][0] and "votes" not in k1 and "votes" not in k1["run"]
    first_pass = len(gen.first_pass)
    _vote_main(tmp_path, monkeypatch, gen, votes=3)
    _, blob = _votes_dir(tmp_path)
    assert len(gen.first_pass) == first_pass == 5 and len(gen.extra) == 10  # no first pass redone
    assert blob["run"]["checkpoint_signature"] == k1["run"]["checkpoint_signature"]


def test_a_daily_cap_in_the_sample_pass_resumes_drawing_only_the_missing_samples(
    tmp_path, monkeypatch,
):
    gen = _voting_generator()
    gen.fail_at = 3  # the 4th extra sample hits the daily cap
    with pytest.raises(SystemExit) as exc:
        _vote_main(tmp_path, monkeypatch, gen)
    msg = str(exc.value.code)
    assert "Stopped in the vote pass" in msg and "only the missing samples are drawn" in msg
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())
    (spath,) = (tmp_path / "cache").glob("*.samples.json")
    cached = json.loads(spath.read_text())["samples"]
    assert sum(len(v) for v in cached.values()) == 3
    gen.fail_at = None
    drawn = len(gen.extra)
    rag_eval.main()
    assert len(gen.extra) - drawn == 10 - 3  # only the missing samples
    _, blob = _votes_dir(tmp_path)
    assert all(r["vote"]["missing"] == [] for r in blob["rows"])


def test_skip_judge_makes_no_judge_call_and_scores_no_verdict_rows_no_verdict(tmp_path, monkeypatch):
    fake_judge = _patch_main(tmp_path, monkeypatch, tmp_path)
    monkeypatch.setenv("SSR_RAG_SKIP_JUDGE", "1")
    rag_eval.main()
    assert fake_judge.calls == 0
    assert not (tmp_path / "rag.json").exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_nojudge")
    blob = json.loads((run_dir / "rag.json").read_text())
    rows = {r["query_id"]: r for r in blob["rows"]}
    assert all(r["judge_answered"] is None and r["faithfulness"] is None
               and r["context_relevance"] is None for r in rows.values())
    for qid in ("4", "5"):  # no verdict even after the re-ask: no judge fallback
        assert rows[qid]["predicted_label"] == "NONE" and rows[qid]["answered_source"] == "no_judge"
    assert blob["faithfulness_answered"] is None and blob["context_relevance"] is None
    assert blob["judge_verdict_agreement"] is None and blob["judge_calls"] == 0
    assert blob["judge_skipped"] is True and blob["no_judge_no_verdict"] == 2
    assert blob["run"]["judge_skipped"] is True and blob["run"]["canonical"] is False
    assert blob["run"]["requests_this_session"]["judge_calls"] == 0
    md = (run_dir / "rag.md").read_text()
    assert "Judge skipped (SSR_RAG_SKIP_JUDGE=1)" in md and "judge=skipped" in md
    # Its own checkpoint: never mixed with judged rows.
    gen, judge_ep = rag_eval.resolve_endpoint("generator"), rag_eval.resolve_endpoint("judge")
    assert blob["run"]["checkpoint_signature"] != rag_eval.rag_signature(
        rag_eval.signature_fields("beir/scifact/test", 5, gen, judge_ep))


def test_aggregate_tolerates_none_judge_fields():
    rows = [row(True, True), {**row(True, True), "faithfulness": None, "context_relevance": None,
                              "judge_answered": None}]
    agg = aggregate(rows)
    assert agg["faithfulness_answered"] == 1.0 and agg["faithfulness_n"] == 1
    assert agg["context_relevance"] == 1.0 and agg["judge_verdict_agreement"] == 1.0
    assert resolve_answered(None, None) == (True, "no_judge")
    assert predicted_label(None, True) == rag_eval.NO_VERDICT


def test_request_estimate_counts_vote_samples_and_rejudges():
    gen, judge_ep = _endpoints()
    est = rag_eval.request_estimate(10, gen, judge_ep, 12.0, 0.2, reask_rate=0.1, reask=True,
                                    vote_samples=20, vote_rejudges=10, vote_throttle=8.0)
    # 10 x (gen + judge + retries + re-asks) + 20 samples x 1.3 + 0.15 x 10 re-judges
    assert est["requests"] == 51 and est["free_requests"] == 51
    assert est["free_requests_worst"] == 10 * 4 + 20 * 3 + 10
    nojudge = rag_eval.request_estimate(10, gen, judge_ep, 12.0, 0.2, reask_rate=0.1, reask=True,
                                        judge=False, vote_samples=20, vote_rejudges=10)
    assert nojudge["requests"] == 39 and nojudge["free_requests_worst"] == 10 * 3 + 20 * 3
    # Without the vote and with the judge on: exactly as before.
    assert rag_eval.request_estimate(300, gen, judge_ep, 12.0, 0.24)["requests"] == 672


def test_a_votes_run_prints_the_samples_it_will_draw(tmp_path, monkeypatch, capsys):
    _vote_main(tmp_path, monkeypatch, _voting_generator())
    head = capsys.readouterr().out.split("[1/5]")[0]
    assert "vote samples: 0 cached" in head and "10 to draw" in head
    assert "self-consistency vote: 10 extra samples still to draw" in head


def test_skip_judge_counts_no_judge_requests(monkeypatch):
    monkeypatch.setattr(rag_eval, "THROTTLE_S", None)
    gen, judge_ep = _endpoints()
    assert rag_eval.free_requests_per_query(gen, judge_ep, judge_skipped=True) == (1, 2)
    assert rag_eval.default_throttle_s(gen, judge_ep, judge_skipped=True) == 8.0
    assert rag_eval.default_throttle_s(gen, judge_ep) == 12.0
    assert "judge: skipped" in rag_eval._estimate_line(5, gen, judge_ep, 8.0, judge_skipped=True)


def test_a_failed_vote_sample_is_not_cached_and_the_claim_is_listed(tmp_path, monkeypatch):
    gen = _voting_generator()
    real = gen.generate

    def flaky(self, query, hits):
        if query in gen.first_pass and query == "Two claim holds." and not getattr(gen, "boom", 0):
            gen.boom = 1
            raise RuntimeError("upstream 500")
        return real(self, query, hits)

    monkeypatch.setattr(gen, "generate", flaky)
    _vote_main(tmp_path, monkeypatch, gen)
    _, blob = _votes_dir(tmp_path)
    rows = {r["query_id"]: r for r in blob["rows"]}
    # Sample 2 of claim 2 failed (the fake then serves sample 3 its first scripted reply).
    assert blob["missing_vote_samples"] == ["2"] and rows["2"]["vote"]["missing"] == [2]
    assert rows["2"]["vote"]["verdicts"] == ["REFUTED", None, "SUPPORTED"]
    assert rows["2"]["vote"]["split"] == "1-1" and rows["2"]["vote"]["chosen"] == 1  # tie: #1
    assert blob["votes"]["claims_missing_samples"] == 1 and blob["run"]["votes"]["samples"]["failed"] == 1
    rag_eval.main()  # the re-run draws exactly the missing sample
    _, blob = _votes_dir(tmp_path)
    assert blob["missing_vote_samples"] == [] and len(gen.extra) == 10


# --- SSR_RAG_CONTEXT=oracle_cited: the closed-corpus diagnostic ---------------------------


def test_rag_context_defaults_to_retrieved_and_rejects_unknown_values():
    assert rag_eval.rag_context({}) == "retrieved"
    assert rag_eval.rag_context({"SSR_RAG_CONTEXT": ""}) == "retrieved"
    assert rag_eval.rag_context({"SSR_RAG_CONTEXT": " Oracle_Cited "}) == "oracle_cited"
    with pytest.raises(ValueError, match="SSR_RAG_CONTEXT"):
        rag_eval.rag_context({"SSR_RAG_CONTEXT": "oracle"})


def test_oracle_cited_ids_are_the_qrels_docs_plus_rationale_docs_in_id_order():
    assert rag_eval.oracle_cited_ids({"120", "9"}, ClaimLabel("NEI")) == ["9", "120"]
    assert rag_eval.oracle_cited_ids({"5"}, ClaimLabel("SUPPORT", {"5", "40"})) == ["5", "40"]


def test_oracle_context_is_in_the_signature_and_never_canonical(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_eval, "_index_fingerprint", lambda: None)
    gen, judge_ep = _endpoints()
    base = rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep)
    assert rag_eval.signature_fields(
        "beir/scifact/train", 100, gen, judge_ep, context="retrieved") == base
    oracle = rag_eval.signature_fields(
        "beir/scifact/train", 100, gen, judge_ep, judge_skipped=True, context="oracle_cited")
    assert oracle == {**base, "judge_skipped": True, "context": "oracle_cited"}
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    ph, test = "abcdef0123456789", rag_eval.CANONICAL_DATASET
    assert rag_eval.output_dir(test, 0, 300, ph, split_size=300, context="retrieved")[1] is True
    path, canonical = rag_eval.output_dir(test, 0, 300, ph, split_size=300, context="oracle_cited")
    assert not canonical and path.name == "rag_beir-scifact-test_300_abcdef01_oracle-cited"
    path, _ = rag_eval.output_dir(test, 0, 300, ph, split_size=300, judge_skipped=True,
                                  context="oracle_cited")
    assert path.name.endswith("_nojudge_oracle-cited")
    assert "context" not in rag_eval.run_metadata()
    assert rag_eval.run_metadata(context="oracle_cited")["context"] == "oracle_cited"


def test_oracle_cited_run_gives_the_generator_only_the_cited_abstracts(tmp_path, monkeypatch):
    seen: dict[str, list[str]] = {}

    class RecordingGenerator(FakeGenerator):
        def generate(self, query, hits):
            seen[query] = [h.doc_id for h in hits]
            return super().generate(query, hits)

    class NoRetrieval:
        def __init__(self, *a, **kw):
            raise AssertionError("the oracle_cited diagnostic must not retrieve")

    fake_judge = _patch_main(tmp_path, monkeypatch, tmp_path, generator=RecordingGenerator)
    corpus = [{"doc_id": d, "title": f"T{d}", "text": f"text {d}"} for d in ("d1", "d2", "d5", "d9")]
    monkeypatch.setattr(rag_eval, "load_documents", lambda: [dict(d) for d in corpus])
    monkeypatch.setattr(rag_eval, "SearchService", NoRetrieval)
    monkeypatch.setenv("SSR_RAG_CONTEXT", "oracle_cited")
    monkeypatch.setenv("SSR_RAG_SKIP_JUDGE", "1")
    rag_eval.main()
    assert fake_judge.calls == 0 and not (tmp_path / "rag.json").exists()  # never canonical
    assert seen == {QUERIES[q]: rag_eval.oracle_cited_ids(QRELS[q], CLAIM_LABELS[q]) for q in QUERIES}
    assert seen["Four claim holds."] == ["d2", "d5"]
    # The re-ask (same policy as the default) sees the same cited passages, same rendering.
    assert FakeGenerator.reask_calls and all(
        [(h.doc_id, h.metadata["title"], h.text) for h in hits]
        == [(d, f"T{d}", f"text {d}") for d in seen[q]]
        for q, hits in FakeGenerator.reask_calls
    )
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_nojudge_oracle-cited")
    blob = json.loads((run_dir / "rag.json").read_text())
    assert blob["run"]["context"] == "oracle_cited" and blob["run"]["canonical"] is False
    assert {r["query_id"]: r["retrieved_doc_ids"] for r in blob["rows"]} == {
        q: seen[QUERIES[q]] for q in QUERIES}
    assert (run_dir / "rag.md").read_text().startswith("> **DIAGNOSTIC")
    assert rag_eval.markdown_from_json(blob).startswith("> **DIAGNOSTIC")


# --- SSR_RAG_OFFSET: a later slice of the same seeded shuffle ---------------------------------


def test_offset_sample_is_a_slice_of_the_same_shuffle():
    ids = [str(i) for i in range(1, 400)]
    full = rag_eval.sample_claims(ids, None)
    assert rag_eval.offset_sample(ids, 100) == rag_eval.sample_claims(ids, 100) == full[:100]
    assert rag_eval.offset_sample(ids, 100, 100) == full[100:200]
    assert rag_eval.offset_sample(ids, None, 300) == full[300:]
    assert rag_eval.offset_sample(ids, 500, 350) == full[350:]
    assert rag_eval.rag_offset({}) == 0 and rag_eval.rag_offset({"SSR_RAG_OFFSET": " "}) == 0
    assert rag_eval.rag_offset({"SSR_RAG_OFFSET": "100"}) == 100
    for bad in ("-1", "x"):
        with pytest.raises(ValueError):
            rag_eval.rag_offset({"SSR_RAG_OFFSET": bad})


def test_offset_changes_signature_and_dir_only_when_set(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_eval, "_index_fingerprint", lambda: None)
    monkeypatch.setattr(rag_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(rag_eval, "RUNS", tmp_path / "runs")
    gen, judge_ep = _endpoints()
    base = rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep)
    assert rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep, offset=0) == base
    off = rag_eval.signature_fields("beir/scifact/train", 100, gen, judge_ep, offset=100)
    assert off == {**base, "sample_offset": 100}
    ph, test = "abcdef0123456789", rag_eval.CANONICAL_DATASET
    path, canonical = rag_eval.output_dir(test, 0, 300, ph, split_size=300, offset=5)
    assert not canonical and path.name.endswith("_offset5")
    path, _ = rag_eval.output_dir("beir/scifact/train", 0, 100, ph, judge_skipped=True, offset=100)
    assert path.name == "rag_beir-scifact-train_100_abcdef01_nojudge_offset100"


def test_offset_run_scores_the_later_slice_and_records_it(tmp_path, monkeypatch, capsys):
    out = tmp_path / "results"
    _patch_main(tmp_path, monkeypatch, out)
    monkeypatch.setenv("SSR_RAG_DATASET", "beir/scifact/train")
    monkeypatch.setenv("SSR_RAG_OFFSET", "2")
    monkeypatch.setenv("SSR_RAG_N", "2")
    rag_eval.main()
    assert not out.exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_offset2") and "_2_" in run_dir.name
    blob = json.loads((run_dir / "rag.json").read_text())
    assert [r["query_id"] for r in blob["rows"]] == rag_eval.sample_claims(QUERIES, None)[2:4]
    assert blob["run"]["sample_offset"] == 2 and blob["run"]["n_sample"] == 2
    assert blob["run"]["canonical"] is False
    assert "SSR_RAG_OFFSET=2: claims 3-4 of the seeded shuffle" in capsys.readouterr().out


def test_offset_past_the_split_stops_before_any_call(tmp_path, monkeypatch):
    _patch_main(tmp_path, monkeypatch, tmp_path / "results")
    monkeypatch.setenv("SSR_RAG_OFFSET", "5")
    with pytest.raises(SystemExit, match="leaves no claims"):
        rag_eval.main()
