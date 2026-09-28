"""web_eval tests on fake data — no network, no index."""
import json

import httpx
import pytest

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.eval import web_eval
from app.retrieve.semantic_scholar import SemanticScholarRetriever


class _Limiter:
    def acquire(self, timeout=None):
        return True


def test_titles_match_tolerates_punctuation_and_case_only():
    assert web_eval.titles_match("Microstructural development of X.", "microstructural Development of X")
    assert not web_eval.titles_match("A totally different paper", "Microstructural development of X")
    assert not web_eval.titles_match("", "x")


class FakeBatch:
    def __init__(self, papers):
        self.papers = papers
        self.calls = []

    def batch(self, ids, fields):
        self.calls.append(ids)
        return [self.papers.get(i.removeprefix("CorpusId:")) for i in ids]


def test_check_mapping_confirms_only_above_threshold():
    titles = {str(i): f"Paper number {i}" for i in range(1, 11)}
    good = {str(i): {"corpusId": i, "title": f"Paper number {i}."} for i in range(1, 11)}
    r = web_eval.check_mapping(FakeBatch(good), titles, list(titles))
    assert r["confirmed"] and r["match_rate"] == 1.0 and r["found_in_s2"] == 10

    # 2 of 10 unknown to S2, 1 with a different paper behind the id -> 0.7 < 0.9.
    bad = dict(good)
    del bad["1"], bad["2"]
    bad["3"] = {"corpusId": 3, "title": "Something else entirely"}
    r = web_eval.check_mapping(FakeBatch(bad), titles, list(titles))
    assert not r["confirmed"] and r["found_in_s2"] == 8 and r["title_matches"] == 7
    assert r["mismatch_examples"][0]["doc_id"] == "3"


def test_check_mapping_batches_at_500_ids():
    titles = {str(i): f"t{i}" for i in range(1, 1202)}
    fake = FakeBatch({})
    web_eval.check_mapping(fake, titles, list(titles))
    assert [len(c) for c in fake.calls] == [500, 500, 201]


def test_score_on_fake_runs():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 1}}
    runs = {
        "hybrid": {"q1": {"a": 3.0, "x": 2.0}, "q2": {"b": 2.0}, "q3": {"y": 1.0, "c": 0.5}},
        "web": {"q1": {"x": 2.0, "a": 1.0}, "q2": {}, "q3": {"c": 1.0}},  # q2: S2 found nothing
        "web_any": {"q1": {"a": 1.0}, "q2": {"b": 1.0}, "q3": {"c": 1.0}},
    }
    r = web_eval.score(qrels, runs)
    assert r["n_queries"] == 3
    assert r["scores"]["web"]["recall@10"] == pytest.approx(2 / 3)
    assert r["scores"]["web_any"]["recall@10"] == 1.0
    assert r["scores"]["hybrid"]["recall@100"] == 1.0
    c = r["comparisons"]["web_vs_hybrid"]["ndcg@10"]
    assert c["delta"] == pytest.approx(r["scores"]["web"]["ndcg@10"] - r["scores"]["hybrid"]["ndcg@10"])
    assert 0.0 <= c["p"] <= 1.0 and sum(c["wtl"].values()) == 3


def test_to_run_is_rank_derived():
    hits = [SearchHit("a", 0.1), SearchHit("b", 0.9)]  # raw scores ignored: order is the contract
    assert web_eval.to_run(hits) == {"a": 2.0, "b": 1.0}


def test_output_dir_guards_the_committed_artifact(monkeypatch):
    monkeypatch.setattr(settings, "eval_dataset", web_eval.CANONICAL_DATASET)
    assert web_eval.output_dir(0, 300) == (web_eval.OUT, True)
    out, canonical = web_eval.output_dir(2, 2)
    assert not canonical and out.parent == web_eval.RUNS and out.name.startswith("web_")


def _fake_s2(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/paper/batch"):
            ids = json.loads(request.content)["ids"]
            return httpx.Response(
                200, json=[{"corpusId": int(i.split(":")[1]), "title": f"Title {i.split(':')[1]}"} for i in ids]
            )
        q = request.url.params["query"]
        data = {
            "claim one": [{"corpusId": 11, "title": "Title 11", "abstract": "a"},
                          {"corpusId": 99, "title": "noise", "abstract": "b"}],
            "claim two": [{"corpusId": 22, "title": "Title 22", "abstract": None}],  # no abstract
        }[q]
        return httpx.Response(200, json={"total": len(data), "offset": 0, "data": data})

    return handler


def _patch_main(monkeypatch, tmp_path, requests):
    queries = {"1": "claim one", "2": "claim two"}
    qrels = {"1": {"11": 1}, "2": {"22": 1}}
    monkeypatch.setattr("app.ingest.corpus.load_queries_qrels", lambda: (queries, qrels))
    monkeypatch.setattr(
        "app.ingest.corpus.load_documents",
        lambda: [{"doc_id": "11", "title": "Title 11"}, {"doc_id": "22", "title": "Title 22"}],
    )
    monkeypatch.setattr(web_eval, "load_hybrid_run", lambda n: {"1": {"11": 2.0}, "2": {"5": 1.0}})
    monkeypatch.setattr(settings, "s2_cache_dir", str(tmp_path / "s2_cache"))
    monkeypatch.setattr(settings, "eval_dataset", web_eval.CANONICAL_DATASET)
    monkeypatch.setattr(web_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(web_eval, "RUNS", tmp_path / "runs")
    monkeypatch.delenv("SSR_EVAL_LIMIT", raising=False)
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    monkeypatch.delenv("SSR_WEB_EVAL_ALLOW_UNMAPPED", raising=False)

    def factory(**kw):
        return SemanticScholarRetriever(
            api_key="",
            client=httpx.Client(transport=httpx.MockTransport(_fake_s2(requests))),
            limiter=_Limiter(),
            sleep=lambda s: None,
            **kw,
        )

    monkeypatch.setattr(web_eval, "SemanticScholarRetriever", factory)


def test_main_end_to_end_on_fakes_and_resumes_from_cache(monkeypatch, tmp_path):
    requests: list = []
    _patch_main(monkeypatch, tmp_path, requests)
    assert web_eval.main() == 0
    assert len(requests) == 3  # 1 mapping batch + 2 searches, nothing else
    blob = json.loads((tmp_path / "results" / "web_retrieval.json").read_text())
    assert blob["mapping"]["confirmed"] and blob["n_queries"] == 2
    assert blob["scores"]["web"]["recall@10"] == 0.5  # claim two's paper has no abstract
    assert blob["scores"]["web_any"]["recall@10"] == 1.0  # ...but S2 did find it
    assert blob["scores"]["hybrid"]["recall@10"] == 0.5
    assert blob["dropped_no_abstract"] == 1 and blob["fetched_from"]
    assert "~200M papers" in (tmp_path / "results" / "web_retrieval.md").read_text()

    # A re-run is served entirely from the on-disk cache: zero new requests.
    assert web_eval.main() == 0
    assert len(requests) == 3


def test_main_stops_before_searching_when_the_id_mapping_fails(monkeypatch, tmp_path):
    requests: list = []
    _patch_main(monkeypatch, tmp_path, requests)
    monkeypatch.setattr(
        "app.ingest.corpus.load_documents",
        lambda: [{"doc_id": "11", "title": "Unrelated"}, {"doc_id": "22", "title": "Other"}],
    )
    assert web_eval.main() == 1
    assert len(requests) == 1  # only the batch lookup; no search request spent
    assert not (tmp_path / "results").exists()
