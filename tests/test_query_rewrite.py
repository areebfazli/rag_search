"""Claim -> keyword query rewrite (app/retrieve/query_rewrite.py). Pure functions, offline."""
from app.retrieve.query_rewrite import (
    FALLBACK_TERMS,
    MAX_TERMS,
    TermRarity,
    claim_terms,
    keyword_query,
    rewrite_queries,
    top_terms,
)
from app.retrieve.semantic_scholar import normalize_query


def test_drops_function_words_hedges_and_generic_relations():
    terms = claim_terms("A deficiency of vitamin B12 significantly decreases blood levels of homocysteine.")
    assert terms == ["deficiency", "vitamin", "B12", "blood", "homocysteine"]
    terms = claim_terms("Side effects associated with antidepressants may lead to an increased risk of stroke.")
    assert terms == ["Side", "antidepressants", "stroke"]


def test_keeps_code_names_and_entity_numbers():
    terms = claim_terms("CK-2017357 increases muscle force in Ly6C+ monocytes.")
    assert terms == ["CK-2017357", "muscle", "force", "Ly6C+", "monocytes"]
    assert "APOE4" in claim_terms("APOE4 expression in neurons results in decreased tau phosphorylation.")
    assert claim_terms("HIV-1 and PD-1 in 4-PBA treated cells") == ["HIV-1", "PD-1", "4-PBA", "treated", "cells"]


def test_hyphen_compounds_split_on_generic_parts_and_reach_s2_without_hyphens():
    assert claim_terms("iPSC-derived neurons") == ["iPSC", "neurons"]
    assert claim_terms("T-cell loss in 7-day-old mice") == ["T-cell", "loss", "mice"]
    # S2 zero-matches hyphenated terms; the retriever's normalize_query spaces them out.
    assert normalize_query(keyword_query("CK-2017357 increases muscle force")) == "CK 2017357 muscle force"


def test_drops_standalone_numbers_doses_abbreviation_parentheticals_and_lone_letters():
    terms = claim_terms(
        "61% of sudden infant death syndrome (SIDS) cases given 40mg/day in 2001 involve M. tuberculosis."
    )
    assert terms == ["sudden", "infant", "death", "syndrome", "tuberculosis"]


def test_dedupes_by_stem_case_insensitively():
    assert claim_terms("Cancer cells and cancers: tumor, tumors, Tumor.") == ["Cancer", "cells", "tumor"]


def test_cap_keeps_claim_order_and_prefers_rare_terms_with_rarity():
    terms = ["cells", "expression", "ALDH1", "prognosis", "breast"]
    rarity = TermRarity({"cell": 900, "express": 800, "prognosi": 50, "breast": 100}, 1000)
    assert top_terms(terms, 3, rarity) == ["ALDH1", "prognosis", "breast"]  # unseen = rarest
    # Without rarity: entities first, then longer words; still returned in claim order.
    assert top_terms(terms, 2) == ["expression", "ALDH1"]
    assert top_terms(terms, 10) == terms


def test_rarity_from_texts_counts_documents_not_occurrences():
    r = TermRarity.from_texts(["cells cells cells", "cells tumour", "rare"])
    assert r.n_docs == 3 and r.df["cell"] == 2 and r.df["tumour"] == 1
    assert r.idf("tumour") > r.idf("cells")
    assert r.idf("never-seen") == max(r.idf(w) for w in ("cells", "tumour", "never"))


def test_rewrite_queries_primary_and_shorter_fallback():
    claim = (
        "Citrullinated proteins externalized in neutrophil extracellular traps act "
        "indirectly to disrupt the inflammatory cycle in mice."
    )
    primary, fallback = rewrite_queries(claim)
    assert len(primary.split()) == MAX_TERMS and len(fallback.split()) == FALLBACK_TERMS
    assert set(fallback.split()) <= set(primary.split())
    # A short claim has one query (the fallback would repeat it); nothing -> nothing.
    assert rewrite_queries("Charcoal treats paraquat.") == ["Charcoal treats paraquat"]
    assert rewrite_queries("It may be the case that it is.") == []
    assert rewrite_queries("") == []


def test_split_terms_at_the_first_relational_word():
    from app.retrieve.query_rewrite import split_terms

    assert split_terms("A deficiency of vitamin B12 increases blood levels of homocysteine.") == (
        ["deficiency", "vitamin", "B12"], ["blood", "homocysteine"]
    )
    assert split_terms("HAND2 methylation in endometrial carcinogenesis.") == (
        ["HAND2", "methylation", "endometrial", "carcinogenesis"], []
    )
    assert split_terms("") == ([], [])


def test_multi_queries_are_distinct_from_the_rewrite_and_each_other():
    from app.retrieve.query_rewrite import _norm_q, multi_queries

    claim = "Increased diastolic blood pressure (DBP) is associated with abdominal aortic aneurysm."
    extra = multi_queries(claim, n=5)
    base = {_norm_q(q) for q in rewrite_queries(claim)}
    assert extra and len(extra) <= 5
    assert len({_norm_q(q) for q in extra}) == len(extra)
    assert not base & {_norm_q(q) for q in extra}
    assert all(len(q.split()) >= 2 for q in extra)
    assert multi_queries(claim, n=1) == extra[:1]  # deterministic, best first
    assert multi_queries("Charcoal.") == [] and multi_queries("") == []
