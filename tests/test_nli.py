"""Tests for the NLI verifier (app/verify/nli.py). Fake tokenizer + fake model: no
downloads, no network, no torch model weights."""
import json
from types import SimpleNamespace

import pytest
import torch

from app.verify import nli
from app.verify.nli import (
    LabelMappingError,
    NLIVerifier,
    PairCache,
    PairScore,
    aggregate_claim,
    label_order,
    pair_key,
    score_pairs_cached,
    split_sentences,
    window_spans,
)

MNLI_ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

# --- label mapping --------------------------------------------------------------------------


def test_label_order_follows_id2label_not_an_assumed_order():
    assert label_order(MNLI_ID2LABEL) == ["SUPPORT", "NEI", "CONTRADICT"]
    # Another checkpoint's order, string keys and upper case (as some configs ship them).
    assert label_order({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}) == [
        "CONTRADICT", "NEI", "SUPPORT"
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {0: "entailment", 1: "not_entailment"},  # 2-way head
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},  # unnamed head
        {0: "entailment", 2: "neutral", 3: "contradiction"},  # gap in indices
        {0: "entailment", 1: "entailment", 2: "contradiction"},  # duplicate
    ],
)
def test_label_order_refuses_anything_but_the_three_nli_classes(bad):
    with pytest.raises(LabelMappingError):
        label_order(bad)


# --- fake HF tokenizer / model ---------------------------------------------------------------


class FakeTokenizer:
    """Whitespace tokens. A batch call records the (premise, claim) pairs it was given,
    so the fake model can 'read' them."""

    def __init__(self):
        self.batches = []

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def __call__(self, text, text_pair=None, add_special_tokens=True, **kw):
        if isinstance(text, str) and text_pair is None:
            return {"input_ids": text.split()}
        self.batches.append(list(zip(text, text_pair, strict=True)))
        return {"row": torch.arange(len(text))}


class FakeModel:
    """Logits keyed on words in the premise: SUPPORTS -> entailment, REFUTES ->
    contradiction, otherwise neutral; indices follow the given id2label."""

    def __init__(self, tokenizer, id2label=MNLI_ID2LABEL):
        self.tok = tokenizer
        self.config = SimpleNamespace(id2label=id2label)
        self.index = {v: int(k) for k, v in id2label.items()}

    def eval(self):
        return self

    def __call__(self, row):
        logits = []
        for premise, _claim in self.tok.batches[-1]:
            name = (
                "entailment" if "SUPPORTS" in premise
                else "contradiction" if "REFUTES" in premise
                else "neutral"
            )
            z = [0.0] * 3
            z[self.index[name]] = 4.0
            logits.append(z)
        return SimpleNamespace(logits=torch.tensor(logits))


def make_verifier(windowing="truncate", id2label=MNLI_ID2LABEL, max_length=512, batch_size=2):
    tok = FakeTokenizer()
    return NLIVerifier(
        model_name="fake/nli", windowing=windowing, int8=False, threads=0,
        batch_size=batch_size, max_length=max_length, tokenizer=tok, model=FakeModel(tok, id2label),
    )


def test_verifier_maps_probabilities_through_id2label():
    for id2label in (MNLI_ID2LABEL, {0: "contradiction", 1: "entailment", 2: "neutral"}):
        v = make_verifier(id2label=id2label)
        s_sup, s_ref, s_neu = v.score(
            [("T", "This SUPPORTS it.", "c"), ("T", "This REFUTES it.", "c"), ("T", "Other.", "c")]
        )
        assert s_sup.probs["SUPPORT"] > 0.9
        assert s_ref.probs["CONTRADICT"] > 0.9
        assert s_neu.probs["NEI"] > 0.9
        assert sum(s_sup.probs.values()) == pytest.approx(1.0)


def test_premise_is_first_and_is_title_plus_abstract():
    v = make_verifier()
    v.score([("A title", "The abstract.", "the claim")])
    ((premise, claim),) = v.tokenizer.batches[-1]
    assert premise == "A title. The abstract."
    assert claim == "the claim"


def test_results_keep_input_order_despite_length_sorting():
    v = make_verifier(batch_size=2)
    items = [
        ("T", "long " * 20 + "SUPPORTS", "c"),
        ("T", "REFUTES", "c"),
        ("T", "mid " * 5, "c"),
    ]
    got = [max(s.probs, key=s.probs.get) for s in v.score(items)]
    assert got == ["SUPPORT", "CONTRADICT", "NEI"]
    assert v.pairs_scored == 3 and v.windows_scored == 3


def test_model_loaded_int8_flag_names_dtype():
    assert make_verifier().dtype == "fp32"


# --- windowing ------------------------------------------------------------------------------


def test_split_sentences():
    text = "BACKGROUND We did X. Results were 3.5 fold (p<0.01). 12 patients died! (A) note? end"
    assert split_sentences(text) == [
        "BACKGROUND We did X.", "Results were 3.5 fold (p<0.01).", "12 patients died!", "(A) note? end"
    ]


@pytest.mark.parametrize(
    ("lengths", "budget", "overlap"),
    [([3, 3, 3, 3, 3], 7, 1), ([5] * 10, 12, 1), ([1, 9, 1, 9, 1], 10, 0), ([20, 1, 1], 5, 1), ([2], 5, 1)],
)
def test_window_spans_cover_everything_within_budget_and_terminate(lengths, budget, overlap):
    spans = window_spans(lengths, budget, overlap)
    covered = {i for a, b in spans for i in range(a, b)}
    assert covered == set(range(len(lengths)))
    for a, b in spans:
        assert b > a
        # Over budget only when a single sentence alone exceeds it.
        assert sum(lengths[a:b]) <= budget or b - a == 1
    starts = [a for a, _ in spans]
    assert starts == sorted(set(starts))  # strictly advancing


def test_window_spans_overlap_neighbours():
    assert window_spans([3, 3, 3, 3], 6, overlap=1) == [(0, 2), (1, 3), (2, 4)]
    assert window_spans([3, 3, 3, 3], 6, overlap=0) == [(0, 2), (2, 4)]


def test_long_abstract_is_windowed_and_the_strongest_window_wins():
    v = make_verifier(windowing="window", max_length=30)
    sentences = [f"Filler sentence number {i} here." for i in range(12)]
    sentences[9] = "This finding SUPPORTS the claim."
    abstract = " ".join(sentences)
    claim = "short claim"
    assert v.window_tag("Title", abstract, claim).startswith("window30")
    premises = v.premises("Title", abstract, claim)
    assert len(premises) > 1
    budget = 30 - 3 - len(claim.split())
    for p in premises:
        assert p.startswith("Title. ")  # the title rides along in every window
        assert len(p.split()) <= budget
    (s,) = v.score([("Title", abstract, claim)])
    assert s.n_windows == len(premises)
    assert s.probs["SUPPORT"] > 0.9
    assert "SUPPORTS" in premises[s.window]


def test_truncate_mode_scores_one_window_and_misses_late_evidence():
    v = make_verifier(windowing="truncate", max_length=30)
    abstract = " ".join(["Filler words go here."] * 12 + ["This SUPPORTS it."])
    assert v.window_tag("T", abstract, "c").startswith("truncate30")
    assert len(v.premises("T", abstract, "c")) == 1


def test_short_pair_is_full_in_both_modes():
    for w in ("truncate", "window"):
        v = make_verifier(windowing=w, max_length=64)
        assert v.window_tag("T", "Short abstract.", "claim") == "full"
        assert v.premises("T", "Short abstract.", "claim") == ["T. Short abstract."]


# --- aggregation ------------------------------------------------------------------------------


def P(s, c):
    return {"SUPPORT": s, "CONTRADICT": c, "NEI": round(1 - s - c, 6)}


def test_max_rule_takes_the_most_confident_non_nei_label():
    d = aggregate_claim([P(0.1, 0.1), P(0.7, 0.2), P(0.1, 0.6)], tau=0.5)
    assert (d.label, d.index, d.candidate) == ("SUPPORT", 1, "SUPPORT")
    assert d.confidence == pytest.approx(0.7)


def test_below_tau_is_nei_but_keeps_the_candidate():
    d = aggregate_claim([P(0.3, 0.1), P(0.2, 0.45)], tau=0.5)
    assert d.label == "NEI" and d.candidate == "CONTRADICT" and d.index == 1
    assert aggregate_claim([P(0.3, 0.1), P(0.2, 0.45)], tau=0.45).label == "CONTRADICT"


def test_conflict_resolves_to_higher_confidence():
    assert aggregate_claim([P(0.8, 0.1), P(0.05, 0.9)], tau=0.5).label == "CONTRADICT"
    assert aggregate_claim([P(0.95, 0.0), P(0.05, 0.9)], tau=0.5).label == "SUPPORT"


def test_exact_ties_go_to_the_earlier_passage_then_support():
    d = aggregate_claim([P(0.1, 0.6), P(0.6, 0.1)], tau=0.5)
    assert (d.label, d.index) == ("CONTRADICT", 0)
    d = aggregate_claim([P(0.4, 0.4)], tau=0.3)
    assert d.label == "SUPPORT"


def test_k_limits_the_passages_considered():
    ps = [P(0.1, 0.1), P(0.1, 0.1), P(0.9, 0.0)]
    assert aggregate_claim(ps, tau=0.5, k=2).label == "NEI"
    assert aggregate_claim(ps, tau=0.5, k=3).label == "SUPPORT"


def test_mean_rule():
    ps = [P(0.9, 0.0), P(0.1, 0.0), P(0.2, 0.0)]
    d = aggregate_claim(ps, tau=0.5, rule="mean", k=3)
    assert d.label == "NEI" and d.confidence == pytest.approx(0.4) and d.index == 0
    assert aggregate_claim(ps, tau=0.39, rule="mean", k=3).label == "SUPPORT"


def test_no_passages_is_nei_and_bad_arguments_raise():
    assert aggregate_claim([], tau=0.0).label == "NEI"
    with pytest.raises(ValueError):
        aggregate_claim([P(0.5, 0.1)], tau=0.5, rule="vote")
    with pytest.raises(ValueError):
        aggregate_claim([P(0.5, 0.1)], tau=0.5, k=0)


# --- cache ------------------------------------------------------------------------------------


class CountingVerifier:
    model_name = "fake/nli"
    dtype = "fp32"
    max_length = 512

    def __init__(self, windowing="truncate"):
        self.windowing = windowing
        self.calls = []

    def window_tag(self, title, abstract, claim):
        return "full" if len(abstract) < 50 else self.windowing

    def score(self, items):
        self.calls.append(len(items))
        return [PairScore(P(0.6, 0.1)) for _ in items]


PAIRS = [("d1", "T1", "abstract one", "claim a"), ("d2", "T2", "x" * 80, "claim a"),
         ("d1", "T1", "abstract one", "claim b")]


def test_cache_hit_skips_the_model_and_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    path = tmp_path / "c.jsonl"
    v = CountingVerifier()
    first = score_pairs_cached(v, PairCache(path), PAIRS, chunk=2)
    assert v.calls == [2, 1]  # chunked, appended after each chunk
    cache = PairCache(path)  # a fresh process reads the file back
    assert len(cache.entries) == 3
    again = score_pairs_cached(v, cache, PAIRS)
    assert v.calls == [2, 1]  # no new model call
    assert cache.hits == 3 and cache.misses == 0
    assert [s.probs for s in again] == [s.probs for s in first]


def test_cache_key_depends_on_dtype_window_tag_claim_and_premise():
    base = ("m", "fp32", "full", 512, "d1", "T. text", "claim")
    k = pair_key(*base)
    for i, alt in ((1, "int8-dynamic"), (2, "window512o1m4"), (3, 256), (4, "d2"), (5, "T. edited"), (6, "other claim")):
        changed = list(base)
        changed[i] = alt
        assert pair_key(*changed) != k


def test_short_pairs_share_cache_entries_across_windowing_modes(tmp_path, monkeypatch):
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    cache = PairCache(tmp_path / "c.jsonl")
    score_pairs_cached(CountingVerifier("truncate"), cache, PAIRS)
    v = CountingVerifier("window")
    score_pairs_cached(v, cache, PAIRS)
    assert v.calls == [1]  # only the long pair (d2) is re-scored


def test_torn_or_corrupt_cache_lines_are_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    path = tmp_path / "c.jsonl"
    good = {"key": "k1", "probs": P(0.5, 0.2), "n_windows": 2, "window": 1}
    path.write_text(json.dumps(good) + "\nnot json\n" + '{"key": "k2", "probs": {}}\n{"key": "k3", "pro')
    cache = PairCache(path)
    assert set(cache.entries) == {"k1"}
    assert cache.get("k1").n_windows == 2


def test_duplicate_pairs_in_one_call_are_scored_once(tmp_path, monkeypatch):
    monkeypatch.delenv("SSR_EVAL_REFRESH", raising=False)
    v = CountingVerifier()
    out = score_pairs_cached(v, PairCache(tmp_path / "c.jsonl"), [PAIRS[0], PAIRS[0]])
    assert v.calls == [1] and len(out) == 2 and out[0] == out[1]


def test_frozen_settings_are_well_formed():
    f = nli.FROZEN
    assert f["windowing"] in nli.WINDOWINGS and f["rule"] in nli.RULES
    assert 1 <= f["k"] <= 5 and 0.0 <= f["tau"] <= 1.0
    # Tuned on the train split only, never the test/dev claims.
    assert f["tuned_on"] is None or f["tuned_on"].startswith("beir/scifact/train")
