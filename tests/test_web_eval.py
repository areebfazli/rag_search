"""web_eval tests on fake data — no network, no index."""
import json

import httpx
import numpy as np
import pytest

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.eval import web_eval
from app.retrieve.query_rewrite import MAX_TERMS
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
    monkeypatch.delenv("SSR_WEB_EVAL_POOL", raising=False)
    monkeypatch.delenv("SSR_WEB_EVAL_ALLOW_UNMAPPED", raising=False)
    assert web_eval.output_dir(0, 300) == (web_eval.OUT, True)
    out, canonical = web_eval.output_dir(2, 2)
    assert not canonical and out.parent == web_eval.RUNS and out.name.startswith("web_")


@pytest.mark.parametrize("env,suffix", [
    ({"SSR_WEB_EVAL_POOL": "0"}, "_nopool"),
    ({"SSR_WEB_EVAL_ALLOW_UNMAPPED": "1"}, "_unmapped"),
    ({"SSR_WEB_EVAL_POOL": "0", "SSR_WEB_EVAL_ALLOW_UNMAPPED": "1"}, "_nopool_unmapped"),
])
def test_output_dir_refuses_the_committed_artifact_without_pool_or_with_unmapped_ids(monkeypatch, env, suffix):
    monkeypatch.setattr(settings, "eval_dataset", web_eval.CANONICAL_DATASET)
    monkeypatch.delenv("SSR_WEB_EVAL_POOL", raising=False)
    monkeypatch.delenv("SSR_WEB_EVAL_ALLOW_UNMAPPED", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out, canonical = web_eval.output_dir(0, 300)
    assert not canonical and out == web_eval.RUNS / f"web_beir-scifact-test_300{suffix}"


CLAIM_ONE = "Zeta kinase activates macrophages via lysosomal cathepsin."
CLAIM_TWO = "Omega receptor binds ligand."


class FakeEmbedder:
    def __init__(self):
        self.documents: list[str] = []

    def encode_query(self, text):
        return np.ones(3, dtype=np.float32) / np.sqrt(3)

    def encode_documents(self, texts, batch_size=32, show_progress=True):
        self.documents += list(texts)
        return np.stack([np.array([len(t), 1.0, 2.0], dtype=np.float32) / np.linalg.norm([len(t), 1.0, 2.0]) for t in texts])


def _fake_s2(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/paper/batch"):
            ids = json.loads(request.content)["ids"]
            return httpx.Response(
                200, json=[{"corpusId": int(i.split(":")[1]), "title": f"Title {i.split(':')[1]}"} for i in ids]
            )
        q = request.url.params["query"]
        if q.startswith("Zeta"):  # claim one: raw and rewrite queries
            data = [{"corpusId": 11, "title": "Title 11", "abstract": "a"},
                    {"corpusId": 99, "title": "noise", "abstract": "b"}]
        elif q.startswith("Omega"):
            data = [{"corpusId": 22, "title": "Title 22", "abstract": None}]  # no abstract
        else:
            data = []
        return httpx.Response(200, json={"total": len(data), "offset": 0, "data": data})

    return handler


def _patch_main(monkeypatch, tmp_path, requests):
    monkeypatch.setenv("SSR_WEB_EVAL_POOL", "0")  # pooled rows: tests/test_web_search.py
    queries = {"1": CLAIM_ONE, "2": CLAIM_TWO}
    qrels = {"1": {"11": 1}, "2": {"22": 1}}
    monkeypatch.setattr("app.ingest.corpus.load_queries_qrels", lambda: (queries, qrels))
    monkeypatch.setattr(
        "app.ingest.corpus.load_documents",
        lambda: [{"doc_id": "11", "title": "Title 11"}, {"doc_id": "22", "title": "Title 22"}],
    )
    monkeypatch.setattr(web_eval, "load_hybrid_run", lambda n: {"1": {"11": 2.0}, "2": {"5": 1.0}})
    embedders: list = []
    monkeypatch.setattr(web_eval, "make_embedder", lambda: embedders.append(FakeEmbedder()) or embedders[-1])
    monkeypatch.setattr(settings, "s2_cache_dir", str(tmp_path / "s2_cache"))
    monkeypatch.setattr(settings, "eval_dataset", web_eval.CANONICAL_DATASET)
    monkeypatch.setattr(web_eval, "OUT", tmp_path / "results")
    monkeypatch.setattr(web_eval, "RUNS", tmp_path / "runs")
    for var in ("SSR_EVAL_LIMIT", "SSR_EVAL_REFRESH", "SSR_WEB_EVAL_ALLOW_UNMAPPED", "SSR_WEB_EVAL_N"):
        monkeypatch.delenv(var, raising=False)

    def factory(**kw):
        return SemanticScholarRetriever(
            api_key="",
            client=httpx.Client(transport=httpx.MockTransport(_fake_s2(requests))),
            limiter=_Limiter(),
            sleep=lambda s: None,
            **kw,
        )

    monkeypatch.setattr(web_eval, "SemanticScholarRetriever", factory)
    return embedders


def _nopool_dir(tmp_path):
    return tmp_path / "runs" / "web_beir-scifact-test_2_nopool"


def test_main_end_to_end_on_fakes_and_resumes_from_cache(monkeypatch, tmp_path):
    requests: list = []
    embedders = _patch_main(monkeypatch, tmp_path, requests)
    assert web_eval.main() == 0
    # SSR_WEB_EVAL_POOL=0 lacks the pooled rows: never the committed eval/results/.
    assert not (tmp_path / "results").exists()
    sent = [r.url.params.get("query") for r in requests if r.url.path.endswith("/search")]
    # 1 mapping batch + 2 raw claims + 2 rewrite primaries + 2 pooled fallbacks (neither
    # primary filled the page). The rerank rows reuse these responses: no extra requests.
    assert len(requests) == 7 and len(sent) == 6
    assert CLAIM_ONE in sent and CLAIM_TWO in sent
    assert "Zeta kinase activates macrophages lysosomal cathepsin" in sent
    blob = json.loads((_nopool_dir(tmp_path) / "web_retrieval.json").read_text())
    assert blob["mapping"]["confirmed"] and blob["n_queries"] == 2
    assert blob["scores"]["web"]["recall@10"] == 0.5  # claim two's paper has no abstract
    assert blob["scores"]["web_any"]["recall@10"] == 1.0  # ...but S2 did find it
    assert blob["scores"]["hybrid"]["recall@10"] == 0.5
    for key in ("web_rewrite", "web_rerank", "web_rewrite_rerank"):
        assert blob["scores"][key]["recall@100"] == 0.5
        assert blob["scores"][f"{key}_any"]["recall@100"] == 1.0
    assert blob["dropped_no_abstract"] == 1 and blob["fetched_from"]
    assert blob["rewrite"]["fallback_used"] == 2 and blob["rewrite"]["max_terms"] == MAX_TERMS
    assert blob["embeddings"]["computed"] > 0
    assert "~200M papers" in (_nopool_dir(tmp_path) / "web_retrieval.md").read_text()
    assert len(embedders) == 1  # one embedder shared by every rerank row

    # A re-run is served entirely from the on-disk caches: zero requests, zero embeddings.
    assert web_eval.main() == 0
    assert len(requests) == 7
    assert embedders[-1].documents == []
    blob = json.loads((_nopool_dir(tmp_path) / "web_retrieval.json").read_text())
    assert blob["embeddings"]["computed"] == 0 and blob["embeddings"]["cached"] > 0


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


def test_sampled_run_is_never_canonical(monkeypatch, tmp_path):
    requests: list = []
    _patch_main(monkeypatch, tmp_path, requests)
    monkeypatch.setenv("SSR_WEB_EVAL_N", "2")
    assert web_eval.main() == 0
    assert not (tmp_path / "results").exists()
    (run_dir,) = (tmp_path / "runs").iterdir()
    blob = json.loads((run_dir / "web_retrieval.json").read_text())
    assert blob["n_queries"] == 2 and blob["sample_seed"] == web_eval.SEED
    assert run_dir.name == "web_beir-scifact-test_2_nopool"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # identical toy runs: zero-variance t-tests
def test_rows_and_pairs_cover_every_variant_in_both_views():
    keys = [k for k, _ in web_eval.ROWS]
    assert keys[0] == "hybrid"
    for v in web_eval.VARIANTS:
        assert v in keys and f"{v}_any" in keys
        assert (v, "hybrid") in web_eval.PAIRS and (f"{v}_any", "hybrid") in web_eval.PAIRS
        if v != "web":
            assert (v, "web") in web_eval.PAIRS and (f"{v}_any", "web_any") in web_eval.PAIRS
    assert "recall@5" in web_eval.METRICS

    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    runs = {k: {"q1": {"a": 2.0, "x": 1.0}, "q2": {"b": 1.0}} for k in keys}
    runs["web"] = {"q1": {"x": 1.0}, "q2": {}}
    r = web_eval.score(qrels, runs)
    assert set(r["scores"]) == set(keys)
    assert set(r["comparisons"]) == {f"{a}_vs_{b}" for a, b in web_eval.PAIRS}
    assert r["comparisons"]["web_rewrite_vs_web"]["recall@5"]["delta"] == 1.0
    meta = {
        "title": "t", "fetched_from": "a", "fetched_to": "b", "n_queries": 2, "requests_sent": 0,
        "authenticated": False, "dropped_no_abstract": 0,
        "mapping": {"found_in_s2": 1, "n_gold_docs": 1, "title_matches": 1, "match_rate": 1.0, "threshold": 0.9},
        "rewrite": {"max_terms": 6, "fallback_terms": 3, "fallback_used": 0, "no_terms": 0},
        "embeddings": {"cached": 0, "computed": 0},
    }
    md = web_eval.to_markdown(r, meta)
    for _, label in web_eval.ROWS:
        assert f"| {label} |" in md
    assert "Recall@5" in md and "web_rewrite_rerank_any vs web_any" in md


def test_pooled_rows_are_compared_with_the_previous_best_and_built_on_request():
    assert ("web_pool_live", "web_rewrite_rerank") in web_eval.PAIRS
    assert ("web_pool_offline", "web_pool_live") in web_eval.PAIRS
    keys = [k for k, _ in web_eval.ROWS]
    assert {"web_pool_live", "web_pool_offline"} <= set(keys) and "web_pool_live_any" not in keys

    class R:  # resolver/pubmed stand-ins: construction only, never called
        pass

    out = web_eval.pipelines(object(), lambda: None, None, None, {"pubmed": R(), "resolver": R()})
    live = out["web_pool_live"]
    assert live.pooled and live.strict and live.dense_cap == 50 and live.snippet_queries == ("claim",)
    assert not live.keep_no_abstract and out["web_pool_offline"].multi_query == 3
    assert "web_pool_live" not in web_eval.pipelines(object(), lambda: None, None, None)
