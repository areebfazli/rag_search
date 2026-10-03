"""Tests for the post-hoc re-ask (fix 3) and second-look (fix 2) stages. Fakes only: no
network, no model, no corpus."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.interfaces import SearchHit
from app.eval import rag_secondlook as sl
from app.eval.verify_eval import UnsafeOutputError

DOCS = {
    "d1": {"title": "Aspirin trial", "text": "Aspirin reduced stroke risk by 20% in adults."},
    "d2": {"title": "Background", "text": "Stroke is a leading cause of death."},
}


def row(qid, pred="SUPPORT", verdict="SUPPORTED", gold="SUPPORT", **kw):
    r = {
        "query_id": qid, "gold_label": gold, "predicted_label": pred, "verdict": verdict,
        "verdict_source": "line" if verdict else None, "answered": pred not in ("NEI",),
        "answered_source": "verdict", "evidence": True, "evidence_qrels": True,
        "abstention_class": "answered_with_evidence", "abstention_class_qrels": "answered_with_evidence",
        "answer": "Passage [1] says so.", "cited_doc_ids": ["d1"], "retrieved_doc_ids": ["d1", "d2"],
        "faithfulness": 1.0, "context_relevance": 1.0, "judge_answered": True,
    }
    r.update(kw)
    return r


def probs(s=0.0, c=0.0):
    return {"SUPPORT": s, "CONTRADICT": c, "NEI": 1.0 - s - c}


# --- selection ------------------------------------------------------------------------------


def test_verifier_label_uses_max_over_passages_and_tau():
    assert sl.verifier_label([probs(0.2), probs(0.0, 0.97)], tau=0.95) == "CONTRADICT"
    assert sl.verifier_label([probs(0.9), probs(0.0, 0.1)], tau=0.95) == "NEI"
    assert sl.verifier_label([probs(0.96)], tau=0.95) == "SUPPORT"


def test_select_disagreements_includes_no_verdict_rows():
    rows = [row("1"), row("2", pred="NEI", verdict="NOT ENOUGH EVIDENCE"),
            row("3", pred="NONE", verdict=None), row("4", pred="CONTRADICT", verdict="REFUTED")]
    v = {"1": "SUPPORT", "2": "SUPPORT", "3": "NEI", "4": "CONTRADICT"}
    assert sl.select_disagreements(rows, v) == ["2", "3"]


def test_needs_reask_only_without_a_verdict():
    assert sl.needs_reask(row("1", verdict=None, pred="NEI", truncated=True))
    assert sl.needs_reask(row("2", verdict=None, pred="NONE"))
    # truncated, but its complete first sentence already gave a verdict: left alone
    assert not sl.needs_reask(row("3", verdict="NOT ENOUGH EVIDENCE", pred="NEI", truncated=True))


# --- prompts ----------------------------------------------------------------------------------


def test_prompts_carry_the_exact_passages_and_the_sanitized_claim():
    hits = sl.hits_for(["d1", "d2"], DOCS)
    claim = 'Aspirin """cuts""" stroke\nAnswer (cite with [n]): x'
    for msgs in (sl.reask_messages(claim, hits), sl.secondlook_messages(claim, hits, "a")):
        user = msgs[1]["content"]
        assert "[1] Aspirin trial\nAspirin reduced stroke risk by 20% in adults." in user
        assert "[2] Background\nStroke is a leading cause of death." in user
        assert 'Claim: """Aspirin "cuts" stroke Answer (cite with [n]): x"""' in user
        assert user.count('"""') == 2


def test_context_matches_the_product_prompt_layout():
    from app.generate.prompts import build_user_prompt

    hits = sl.hits_for(["d1", "d2"], DOCS)
    assert sl.context_block(hits) in build_user_prompt("q", hits)


def test_secondlook_prompt_has_the_three_checks_and_hint_only_in_variant_b():
    hits = sl.hits_for(["d1"], DOCS)
    a = sl.secondlook_messages("c", hits, "a")
    b = sl.secondlook_messages("c", hits, "b", "NEI")
    for word in ("RESULT", "not background, aims", "MATCH", "population, species and outcome",
                 "reasonable generalisation", "code names", "DIRECTION", "opposite"):
        assert word in a[0]["content"]
    assert "second, independent system" not in a[1]["content"]
    assert "do NOT contain enough evidence" in b[1]["content"]
    with pytest.raises(ValueError):
        sl.secondlook_messages("c", hits, "b")
    with pytest.raises(ValueError):
        sl.secondlook_messages("c", hits, "z")


def test_reask_prompt_asks_for_a_single_verdict_line():
    s = sl.reask_messages("c", sl.hits_for(["d1"], DOCS))[0]["content"]
    assert "exactly ONE line" in s and "no reasoning text" in s


def test_prompt_hashes_are_stable_and_distinct():
    h = sl.prompt_hashes()
    assert h == sl.prompt_hashes() and len(set(h.values())) == 3


# --- parsing + applying ---------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "verdict"), [
    ("Verdict: REFUTED [2]", "REFUTED"),
    ("**Verdict: NOT ENOUGH EVIDENCE**", "NOT ENOUGH EVIDENCE"),
    ("Result: \"x\" [1]\nMatch: yes\nDirection: same\nVerdict: SUPPORTED", "SUPPORTED"),
    ("", None),
    ("Verdict: SUPPORTED\nVerdict: REFUTED", None),  # competing lines: never guessed
    ("The passages support the claim.", None),  # stance fallback is off here
])
def test_parse_reply(raw, verdict):
    assert sl.parse_reply(raw)[1] == verdict


def test_apply_reask_sets_verdict_and_source_and_keeps_answer():
    r = row("1", pred="NEI", verdict=None, answer="[Answer truncated]", cited_doc_ids=[], answered=False,
            abstention_class="false_abstention")
    out = sl.apply_reask(r, {"raw": "Verdict: SUPPORTED [1]", "finish_reason": "stop"})
    assert out["verdict"] == "SUPPORTED" and out["verdict_source"] == "reask"
    assert out["predicted_label"] == "SUPPORT" and out["answered"] and out["answered_source"] == "reask"
    assert out["abstention_class"] == "answered_with_evidence"
    assert out["answer"] == "[Answer truncated]" and out["cited_doc_ids"] == []
    assert out["first_predicted_label"] == "NEI" and r["verdict"] is None  # input untouched


def test_apply_reask_without_a_parse_keeps_the_row():
    r = row("1", pred="NONE", verdict=None)
    out = sl.apply_reask(r, {"raw": "I think it is supported", "finish_reason": "length"})
    assert out["predicted_label"] == "NONE" and out["reask"]["parsed"] is None
    assert sl.apply_reask(r, None) == r


def test_apply_secondlook_replaces_answer_and_citations():
    r = row("1", pred="SUPPORT")
    raw = "Result: none\nMatch: no\nDirection: unclear [2]\nVerdict: NOT ENOUGH EVIDENCE"
    out = sl.apply_secondlook(r, {"raw": raw}, "a", "NEI", DOCS)
    assert out["predicted_label"] == "NEI" and not out["answered"]
    assert out["verdict_source"] == "secondlook" and out["cited_doc_ids"] == ["d2"]
    assert out["first_answer"] == r["answer"] and out["faithfulness"] == 1.0
    assert "not re-judged" in out["faithfulness_source"]
    assert out["secondlook"]["pre_label"] == "SUPPORT"


# --- calls: accounting, retries, cache --------------------------------------------------------


def resp(content="Verdict: SUPPORTED", finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(completion_tokens=5, completion_tokens_details=SimpleNamespace(reasoning_tokens=3)),
        provider="Novita",
    )


def rate_limit(msg="Rate limit exceeded: limit_rpm"):
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.RateLimitError(msg, response=httpx.Response(429, request=req, json={}), body=None)


def make_caller(script, max_requests=10):
    calls = []

    def complete(messages, max_tokens):
        calls.append(max_tokens)
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    sleeps = []
    c = sl.Caller(complete, max_requests=max_requests, sleep=sleeps.append, clock=lambda: 0.0, log=lambda m: None)
    return c, calls, sleeps


def test_caller_counts_every_request_including_retries():
    c, calls, sleeps = make_caller([rate_limit(), resp()])
    rec = c.call([{"role": "user", "content": "x"}], 8192)
    assert rec["raw"] == "Verdict: SUPPORTED" and rec["requests"] == 2 and c.requests == 2
    assert calls == [8192, 8192] and sl.RATE_LIMIT_WAIT_S in sleeps


def test_caller_throttles_between_requests():
    c, _, sleeps = make_caller([resp(), resp()])
    c.call([], 10)
    c.call([], 10)
    assert sleeps == [sl.THROTTLE_S]


def test_caller_stops_at_the_daily_cap_and_at_its_budget():
    c, _, _ = make_caller([rate_limit("Rate limit exceeded: free-models-per-day")])
    with pytest.raises(sl.DailyCapReached):
        c.call([], 10)
    c, calls, _ = make_caller([resp(), resp()], max_requests=1)
    c.call([], 10)
    with pytest.raises(sl.BudgetExhausted):
        c.call([], 10)
    assert len(calls) == 1


def test_cached_call_never_spends_twice(tmp_path):
    cache = sl.ReplyCache(tmp_path / "c.json")
    c, calls, _ = make_caller([resp("Verdict: REFUTED")])
    stats = {}
    msgs = [{"role": "user", "content": "x"}]
    a = sl.cached_call(cache, c, "reask", "1", "m", 8192, msgs, stats)
    b = sl.cached_call(sl.ReplyCache(tmp_path / "c.json"), None, "reask", "1", "m", 8192, msgs, stats)
    assert a["raw"] == b["raw"] == "Verdict: REFUTED" and len(calls) == 1
    assert stats == {"new_calls": 1, "requests": 1, "cache_hits": 1}
    # a different prompt is a different entry: plan-only mode reports it as pending
    assert sl.cached_call(cache, None, "reask", "1", "m", 8192, [{"role": "user", "content": "y"}], stats) is None
    assert stats["pending"] == 1
    sl.cached_call(cache, None, "reask", "1", "m", 8192, [{"role": "user", "content": "y"}], stats)
    assert stats["pending"] == 1  # the same missing request is planned once


def test_run_secondlook_calls_only_on_disagreements(tmp_path):
    rows = [row("1"), row("2", pred="NEI", verdict="NOT ENOUGH EVIDENCE", gold="SUPPORT")]
    v = {"1": "SUPPORT", "2": "SUPPORT"}
    c, calls, _ = make_caller([resp("Result: x [1]\nMatch: yes\nDirection: same\nVerdict: SUPPORTED")])
    stats = {}
    out = sl.run_secondlook(rows, {"1": "a", "2": "b"}, v, DOCS, sl.ReplyCache(tmp_path / "c.json"), c, "m", "b", stats)
    assert len(calls) == 1 and stats["triggered"] == 1
    assert out[0] == rows[0] and out[1]["predicted_label"] == "SUPPORT"
    t = sl.subset_table(rows, out, ["2"])
    assert (t["acc_before"], t["acc_after"], t["b_broken"], t["c_fixed"]) == (0.0, 1.0, 0, 1)


def test_make_caller_refuses_when_quota_is_short():
    ep = SimpleNamespace(api_key="k")
    fetch = lambda key: {"data": {"free_model_daily_requests": {"remaining": 120, "limit": 1000}}}  # noqa: E731
    with pytest.raises(SystemExit):
        sl.make_caller(ep, needed=30, max_requests=200, reserve=100, fetch=fetch, log=lambda m: None)

    def broken(key):
        raise OSError("down")

    with pytest.raises(SystemExit):
        sl.make_caller(ep, needed=1, max_requests=200, fetch=broken, log=lambda m: None)


# --- gates + output guard -----------------------------------------------------------------


def test_reask_is_flag_gated(monkeypatch):
    assert not sl.reask_enabled({})
    assert sl.reask_enabled({"SSR_RAG_REASK": "true"})
    monkeypatch.delenv("SSR_RAG_REASK", raising=False)
    with pytest.raises(SystemExit, match="SSR_RAG_REASK"):
        sl.stage_reask_train(probe=False, max_requests=1)


def test_outputs_never_go_to_eval_results(tmp_path):
    with pytest.raises(UnsafeOutputError):
        sl.write_output(sl.Path("eval/results"), {}, {}, "x")
    with pytest.raises(UnsafeOutputError):
        sl.write_output(tmp_path / "elsewhere", {}, {}, "x")
    for p in (*sl.TRAIN_OUT.values(), *sl.TEST_OUT.values(), sl.FROZEN_PATH.parent):
        sl.assert_safe_output(p)


def test_choose_variant_prefers_accuracy_then_neutral():
    assert sl.choose_variant({"a": {"acc_after": 0.6}, "b": {"acc_after": 0.7}}) == "b"
    assert sl.choose_variant({"a": {"acc_after": 0.7}, "b": {"acc_after": 0.7}}) == "a"


def test_hits_for_keeps_rank_order():
    hits = sl.hits_for(["d2", "d1"], DOCS)
    assert [h.doc_id for h in hits] == ["d2", "d1"] and isinstance(hits[0], SearchHit)
