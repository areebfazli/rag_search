"""Tests for the judge-consistency eval. All offline: fake judge client, fake Qdrant
client, in-memory BM25 — no network, no index on disk, no LLM key."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.core.interfaces import SearchHit, hit_passage
from app.eval import judge_agreement as ja
from app.eval import rag_eval
from app.eval.judge_agreement import (
    Config,
    StaleResultsError,
    check_source,
    fleiss_kappa,
    krippendorff_alpha_interval,
    rebuild_contexts,
    run_repeats,
    select_rows,
    summarize,
)
from app.index.lexical import LexicalIndex
from app.index.vector_store import VectorStore
from app.ingest.corpus import document_passage
from app.retrieve.service import SearchService

# --- Fleiss' kappa ----------------------------------------------------------------------


def test_fleiss_hand_computed_binary():
    # n=3 raters. P_i = 1, 1/3, 1 -> P_bar = 7/9. p_T = 5/9, p_F = 4/9 -> P_e = 41/81.
    # kappa = (63/81 - 41/81) / (40/81) = 22/40.
    r = [[True, True, True], [True, True, False], [False, False, False]]
    assert fleiss_kappa(r) == pytest.approx(0.55)


def test_fleiss_wikipedia_worked_example():
    # The 10-subject, 14-rater, 5-category example from Fleiss' kappa's Wikipedia
    # article: P_bar = 0.378, P_e = 0.213, kappa = 0.210.
    counts = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
    ratings = [[j for j, c in enumerate(row) for _ in range(c)] for row in counts]
    assert fleiss_kappa(ratings) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_perfect_agreement_is_one():
    assert fleiss_kappa([[True] * 3, [False] * 3, [True] * 3]) == pytest.approx(1.0)


def test_fleiss_degenerate_single_category_is_none_not_nan():
    # Every rating identical: P_e = 1, so kappa is 0/0. Must be None, never NaN.
    assert fleiss_kappa([[True] * 3] * 5) is None
    assert fleiss_kappa([]) is None
    assert fleiss_kappa([[True], [False]]) is None  # one rater: no agreement to measure


def test_fleiss_rejects_ragged_ratings():
    with pytest.raises(ValueError):
        fleiss_kappa([[True, True], [True]])


# --- Krippendorff's alpha (interval) ----------------------------------------------------


def test_alpha_hand_computed():
    # units (1,2), (3,3); n = 4. D_o = [2/1 + 0] / 4 = 0.5.
    # D_e: unordered squared diffs 1+4+4+1+1+0 = 11 -> ordered 22, / (4*3) = 11/6.
    # alpha = 1 - 0.5 / (11/6) = 8/11.
    assert krippendorff_alpha_interval([[1, 2], [3, 3]]) == pytest.approx(8 / 11)


def test_alpha_drops_unpairable_units():
    # A unit left with one value (the other repeats unparseable) carries no pair.
    assert krippendorff_alpha_interval([[1, 2], [3, 3], [5]]) == pytest.approx(8 / 11)


def test_alpha_perfect_agreement_is_one():
    assert krippendorff_alpha_interval([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]) == pytest.approx(1.0)


def test_alpha_degenerate_is_none_not_nan():
    assert krippendorff_alpha_interval([[0.8, 0.8], [0.8, 0.8, 0.8]]) is None  # D_e = 0
    assert krippendorff_alpha_interval([]) is None
    assert krippendorff_alpha_interval([[1.0], [0.0]]) is None  # nothing pairable


def test_alpha_can_go_negative_for_systematic_disagreement():
    a = krippendorff_alpha_interval([[0, 1], [1, 0], [0, 1]])
    assert a is not None and a < 0 and not math.isnan(a)


# --- context reconstruction == what rag_eval hands the judge ----------------------------

DOCS = [
    {"doc_id": "d1", "title": "Aspirin and platelets", "text": "Aspirin inhibits platelet aggregation."},
    {"doc_id": "d2", "title": "", "text": "Untitled abstract about platelet counts."},
    {"doc_id": "d3", "title": "  Padded title  ", "text": "Statins lower LDL cholesterol."},
    {"doc_id": "d4", "title": "   ", "text": "Whitespace title, aspirin dosing."},
    {"doc_id": "d5", "title": "Vitamin D", "text": "Vitamin D and bone density in aspirin users."},
]
BY_ID = {d["doc_id"]: d for d in DOCS}


class _FakeQdrant:
    """Stands in for QdrantClient.query_points with the payload VectorStore.upsert
    writes — so VectorStore.search's payload -> SearchHit mapping runs for real."""

    def query_points(self, collection, query, limit, with_payload):
        pts = [
            SimpleNamespace(
                score=1.0 - i / 10,
                payload={"doc_id": d["doc_id"], "title": d.get("title", ""), "text": d["text"]},
            )
            for i, d in enumerate(reversed(DOCS))
        ]
        return SimpleNamespace(points=pts[:limit])


def _vector_store() -> VectorStore:
    vs = object.__new__(VectorStore)  # skip __init__: never opens Qdrant
    vs.client, vs.collection = _FakeQdrant(), "corpus"
    return vs


def _lexical() -> LexicalIndex:
    lex = LexicalIndex()
    lex.build(DOCS, [document_passage(d) for d in DOCS])
    return lex


def test_rebuilt_contexts_match_bm25_hits():
    hits = _lexical().search("aspirin platelet", top_k=5)
    assert len(hits) == 5
    ids = [h.doc_id for h in hits]
    assert rebuild_contexts(ids, BY_ID) == [hit_passage(h) for h in hits]


def test_rebuilt_contexts_match_dense_hits():
    hits = _vector_store().search(np.zeros(3), top_k=5)
    ids = [h.doc_id for h in hits]
    assert rebuild_contexts(ids, BY_ID) == [hit_passage(h) for h in hits]


def test_rebuilt_contexts_match_service_hybrid():
    # The exact path rag_eval takes: SearchService.retrieve(mode="hybrid") -> RRF fusion
    # of dense + BM25 -> [hit_passage(h) for h in hits].
    vs = _vector_store()
    dense = SimpleNamespace(search=lambda q, top_k: vs.search(np.zeros(3), top_k))
    service = SearchService(dense=dense, lexical=_lexical())
    hits = service.retrieve("aspirin platelet", mode=rag_eval.MODE, top_k=rag_eval.TOP_K)
    ids = [h.doc_id for h in hits]
    assert len(ids) == rag_eval.TOP_K
    assert rebuild_contexts(ids, BY_ID) == [hit_passage(h) for h in hits]


def test_rebuild_rejects_unknown_doc():
    with pytest.raises(StaleResultsError):
        rebuild_contexts(["d1", "nope"], BY_ID)


_BM25_DOCS = Path(settings.bm25_path) / "docs.json"


@pytest.mark.skipif(
    not (_BM25_DOCS.exists() and ja.SOURCE.exists()), reason="needs a local index + rag.json"
)
def test_rebuilt_contexts_match_real_index_for_stored_rows():
    # Opportunistic (skipped in CI): the corpus loader and the built BM25 doc store must
    # give identical judge contexts for every doc rag.json actually retrieved. Reads the
    # index's plain-JSON doc list only; never loads bm25s arrays or opens Qdrant.
    from app.ingest.corpus import load_documents

    index_docs = {d["doc_id"]: d for d in json.loads(_BM25_DOCS.read_text())}
    corpus = {d["doc_id"]: d for d in load_documents()}
    rows = json.loads(ja.SOURCE.read_text())["rows"]
    for r in rows:
        ids = r["retrieved_doc_ids"]
        index_hits = [
            SearchHit(d, 0.0, index_docs[d]["text"], {"title": index_docs[d].get("title", "")})
            for d in ids
        ]
        assert rebuild_contexts(ids, corpus) == [hit_passage(h) for h in index_hits]


# --- stale-metadata refusal / row selection ---------------------------------------------


def _row(qid: str, answer: str = "An answer [1].", **kw) -> dict:
    base = {
        "query_id": qid,
        "gold_label": "SUPPORT",
        "answer": answer,
        "retrieved_doc_ids": ["d1", "d2"],
        "verdict": "SUPPORTED",
        "answered": True,
        "answered_source": "verdict",
        "judge_answered": True,
        "faithfulness": 1.0,
        "context_relevance": 0.8,
    }
    return {**base, **kw}


def _blob(**run_overrides) -> dict:
    run = {
        "judge_prompt_hash": rag_eval._sha256(rag_eval.JUDGE_SYSTEM),
        "judge_model": settings.judge_model,
        **run_overrides,
    }
    return {"run": run, "rows": [_row("1")]}


def test_check_source_accepts_current_judge():
    check_source(_blob())


def test_check_source_refuses_changed_prompt():
    with pytest.raises(StaleResultsError, match="judge_prompt_hash"):
        check_source(_blob(judge_prompt_hash=rag_eval._sha256(rag_eval.JUDGE_SYSTEM + " ")))


def test_check_source_refuses_other_judge_model():
    with pytest.raises(StaleResultsError, match="judge_model"):
        check_source(_blob(judge_model="llama-3.1-8b-instant"))


def test_check_source_refuses_missing_metadata_or_rows():
    with pytest.raises(StaleResultsError):
        check_source({"rows": [_row("1")]})
    with pytest.raises(StaleResultsError):
        check_source({**_blob(), "rows": []})
    bad = _blob()
    del bad["rows"][0]["retrieved_doc_ids"]
    with pytest.raises(StaleResultsError, match="retrieved_doc_ids"):
        check_source(bad)


def test_select_rows_skips_empty_answers():
    rows = [_row("1"), _row("2", answer=""), _row("3", answer="  \n"), _row("4")]
    keep, skipped = select_rows(rows, limit=0, seed=13)
    assert [r["query_id"] for r in keep] == ["1", "4"]
    assert skipped == ["2", "3"]


def test_select_rows_limit_is_seeded_subset_in_file_order():
    rows = [_row(str(i)) for i in range(10)]
    a, _ = select_rows(rows, limit=4, seed=7)
    b, _ = select_rows(rows, limit=4, seed=7)
    assert [r["query_id"] for r in a] == [r["query_id"] for r in b]
    assert len(a) == 4
    assert [int(r["query_id"]) for r in a] == sorted(int(r["query_id"]) for r in a)


# --- config / estimate / output location ------------------------------------------------


def test_config_defaults_and_env():
    assert Config.from_env({}) == Config(k=3, temperatures=(0.0, 0.7), limit=0, seed=13)
    cfg = Config.from_env(
        {"SSR_JUDGE_REPEATS": "5", "SSR_JUDGE_TEMPERATURES": "0, 1.0, 0.0", "SSR_JUDGE_LIMIT": "4"}
    )
    assert (cfg.k, cfg.temperatures, cfg.limit) == (5, (0.0, 1.0), 4)
    with pytest.raises(ValueError):
        Config.from_env({"SSR_JUDGE_REPEATS": "1"})
    with pytest.raises(ValueError):
        Config.from_env({"SSR_JUDGE_TEMPERATURES": "3.0"})
    with pytest.raises(ValueError):
        Config.from_env({"SSR_JUDGE_TEMPERATURES": "nan"})


def test_estimate_counts_every_call_with_throttle():
    calls, secs = ja.estimate(48, Config())
    assert calls == 48 * 3 * 2
    assert secs == pytest.approx(calls * (ja.THROTTLE_S + ja.EST_LATENCY_S))


def test_only_unlimited_run_is_canonical():
    assert ja.output_dir(0) == (ja.OUT, True)
    path, canonical = ja.output_dir(5)
    assert not canonical and path.is_relative_to(ja.RUNS)


# --- re-judging through the production judge() ------------------------------------------


class _FakeJudgeClient:
    """Records every request; replies from a script (a str, or an Exception to raise)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if self.replies else self.default
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])

    default = '{"answered": true, "faithfulness": 1.0, "context_relevance": 0.8}'


def _run(rows, client, cfg=Config(k=2, temperatures=(0.0, 0.7)), **kw):
    questions = {r["query_id"]: f"claim {r['query_id']}" for r in rows}
    contexts = {r["query_id"]: rebuild_contexts(r["retrieved_doc_ids"], BY_ID) for r in rows}
    return run_repeats(
        client, rows, questions, contexts, cfg, sleep=lambda _: None, log=lambda _: None, **kw
    )


def test_run_uses_production_prompt_and_overrides_only_temperature():
    rows = [_row("1")]
    client = _FakeJudgeClient([])
    records, requested = _run(rows, client)
    assert requested == ja.PRODUCTION_TEMPERATURE  # what judge() itself asked for
    assert [c["temperature"] for c in client.calls] == [0.0, 0.0, 0.7, 0.7]
    first = client.calls[0]
    assert first["model"] == settings.judge_model
    assert first["messages"][0]["content"] == rag_eval.JUDGE_SYSTEM
    user = first["messages"][1]["content"]
    for i, ctx in enumerate(rebuild_contexts(["d1", "d2"], BY_ID), start=1):
        assert f"[{i}] {ctx}" in user
    assert user.startswith("QUESTION: claim 1") and user.endswith("ANSWER: An answer [1].")
    assert set(records["1"]["repeats"]) == {"0.0", "0.7"}


def test_empty_answer_rows_are_never_judged():
    keep, skipped = select_rows([_row("1"), _row("2", answer="")], limit=0, seed=13)
    client = _FakeJudgeClient([])
    records, _ = _run(keep, client)
    assert skipped == ["2"] and set(records) == {"1"}
    assert len(client.calls) == 2 * 2
    assert all("ANSWER: An answer" in c["messages"][1]["content"] for c in client.calls)


def test_parse_failure_is_recorded_not_fatal():
    client = _FakeJudgeClient(["no json here"])
    records, _ = _run([_row("1")], client)
    reps = records["1"]["repeats"]["0.0"]
    assert reps[0] == {"error": "JudgeParseError", "error_kind": "parse"}
    assert "error" not in reps[1]


def test_consecutive_api_errors_abort_and_keep_completed_rows():
    saved = {}
    client = _FakeJudgeClient([_FakeJudgeClient.default] * 4 + [RuntimeError("429")] * 3)
    with pytest.raises(ja.AbortRun):
        _run([_row("1"), _row("2")], client, save=saved.update)
    assert set(saved) == {"1"}  # row 2 died mid-way and will be redone on resume


def test_checkpointed_rows_are_not_rejudged():
    client = _FakeJudgeClient([])
    first, _ = _run([_row("1")], client)
    client2 = _FakeJudgeClient([])
    again, _ = _run([_row("1"), _row("2")], client2, done=first)
    assert again["1"] == first["1"]
    assert len(client2.calls) == 4  # only row 2


# --- summary metrics --------------------------------------------------------------------


def _rep(answered, faith, ctx=0.5):
    return {"answered": answered, "faithfulness": faith, "context_relevance": ctx}


def _rec(qid, reps, *, verdict="SUPPORTED", answered=True, source="verdict", orig=None):
    return {
        "query_id": qid,
        "verdict": verdict,
        "answered": answered,
        "answered_source": source,
        "original": orig or _rep(True, 1.0, 1.0),
        "repeats": {"0.0": reps},
    }


def test_summarize_hand_example():
    recs = [
        # Verdict-decided: judge disagreement can't flip `answered`.
        _rec("1", [_rep(True, 1.0), _rep(False, 0.5), _rep(True, 1.0)]),
        # Judge-decided (no verdict), published answered=True: a False repeat flips it.
        _rec("2", [_rep(True, 1.0), _rep(True, 1.0), _rep(False, 1.0)],
             verdict=None, source="judge"),
        _rec("3", [_rep(True, 1.0), {"error": "JudgeParseError", "error_kind": "parse"},
                   _rep(True, 1.0)]),
    ]
    s = summarize(recs, 0.0, 3)
    a = s["answered"]
    assert s["parse_failures"] == 1 and s["api_errors"] == 0
    assert a["rows_all_repeats_valid"] == 2
    assert a["unanimous_rate"] == 0.0  # rows 1 and 2 both split
    assert a["agreement_with_original"] == pytest.approx(round(6 / 8, 4))
    assert a["agreement_with_original_per_repeat"] == [1.0, 0.5, 0.6667]
    assert a["disagrees_with_published_answered"] == pytest.approx(round(2 / 8, 4))
    assert a["would_flip_published_answered"] == pytest.approx(round(1 / 8, 4))
    assert a["judge_decided_rows"] == 1 and a["rows_with_any_flip"] == 1
    f = s["faithfulness"]
    assert f["mean_abs_diff_vs_original"] == pytest.approx(round(0.5 / 8, 4))
    assert f["max_row_std"] == pytest.approx(round(float(np.std([1, 0.5, 1], ddof=1)), 4))
    # context_relevance is constant at 0.5 in every repeat: alpha over repeats alone is
    # undefined (None, not NaN); against the original 1.0 it is defined.
    assert s["context_relevance"]["alpha_repeats"] is None
    assert s["context_relevance"]["mean_row_std"] == 0.0
    assert s["context_relevance"]["alpha_with_original"] is not None


def test_summarize_perfect_reproducibility():
    recs = [_rec(str(i), [_rep(True, 0.9, 0.7)] * 3, orig=_rep(True, 0.9, 0.7)) for i in range(4)]
    recs.append(_rec("x", [_rep(False, 0.2, 0.1)] * 3, orig=_rep(False, 0.2, 0.1),
                     verdict=None, answered=False, source="judge"))
    s = summarize(recs, 0.0, 3)
    assert s["identical_to_original_rate"] == 1.0
    assert s["answered"]["unanimous_rate"] == 1.0
    assert s["answered"]["fleiss_kappa"] == pytest.approx(1.0)
    assert s["answered"]["would_flip_published_answered"] == 0.0
    assert s["faithfulness"]["alpha_repeats"] == pytest.approx(1.0)
    assert s["faithfulness"]["mean_abs_diff_vs_original"] == 0.0


def test_markdown_renders_with_undefined_stats():
    recs = [_rec("1", [_rep(True, 1.0, 1.0)] * 2, orig=_rep(True, 1.0, 1.0))]
    result = {
        "n_rows": 1,
        "skipped_empty_answer": ["9"],
        "summary": {"0.0": summarize(recs, 0.0, 2)},
        "run": {
            "source": {"git_sha": "abc1234def"},
            "judge_model": settings.judge_model,
            "repeats": 2,
            "judge_prompt_hash": "0" * 64,
            "canonical": True,
            "row_limit": 0,
        },
    }
    md = ja.to_markdown(result)
    assert "| `answered`: Fleiss' kappa | n/a |" in md
    assert "Skipped (empty answer): 9." in md
