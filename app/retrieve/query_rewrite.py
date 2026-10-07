"""Claim -> keyword query for Semantic Scholar's relevance search. Deterministic, no LLM.

Why: S2 paper/search behaves close to an AND over the query terms. Fed a full SciFact claim
("A deficiency of vitamin B12 decreases blood levels of homocysteine.") it matches nothing
for most claims: 168 of the 300 test claims returned zero results in the committed raw-query
run. So the claim is reduced to its content terms:

  * function words, hedges ("may", "significantly") and generic relational verbs/nouns
    ("is associated with", "increases", "leads to", "role", "effect") are dropped. Direction
    words matter here beyond noise: SciFact's CONTRADICT claims are often written by
    flipping them ("increases" -> "decreases"), so they are exactly the terms least likely
    to be in the paper;
  * standalone numbers, percentages and doses ("61%", "40mg/day", "2001") are dropped —
    SciFact also mutates numbers to build false claims — but a number that belongs to an
    entity is kept with it ("HIV-1", "4-PBA", "CK-2017357", "APOE4", "B12"), and so is a
    short number after a name part or a type/class/stage/grade/phase head
    ("interleukin-2", "caspase-11", "Type 1 diabetes" -> "Type-1"; "2-fold" still drops);
  * a one-token parenthetical is an abbreviation restating the words before it
    ("sudden infant death syndrome (SIDS)") and is dropped: under AND semantics a paper
    that never uses the abbreviation would be excluded;
  * hyphenated compounds stay whole when every part is content (and S2 gets them with
    spaces — "Hyphenated query terms yield no matches", see
    semantic_scholar.normalize_query) and are split when a part is generic
    ("iPSC-derived" -> "iPSC");
  * terms are deduplicated by stem (a word already inside a kept compound, "TDP" next to
    "TDP-43", is dropped) and capped; when there are too many, entity-like tokens (codes,
    gene/drug names) outrank plain words, then longer words, then earlier ones, and the
    kept terms go out in claim order. With corpus rarity, a plain word the corpus never
    saw (often a typo or run-together, "panicprone") ranks last.

`rewrite_queries` returns the primary query plus a shorter fallback (the highest-priority
terms only). The web pipeline sends both unless the primary alone fills the requested page
with abstract-bearing papers, and pools their results (deduped): the two queries surface
different papers, and the local rerank sorts the union.

Chosen on a seeded 100-claim sample of beir/scifact/train (the 300 test claims are
SciFact's public dev set; see web_eval) and then frozen. Gold papers in S2's top 100
(papers with an abstract / any paper), train, no rerank:
  raw claim 0.08 / 0.09 · 8 terms 0.10 / 0.17 · 6 terms 0.15 / 0.24 · 4 terms 0.17 / 0.24
  · 3 terms 0.18 / 0.27 — S2's ranking does better with FEWER terms than the "~8-10"
  first planned. Pooling two queries beat any single one: after the BM25 rerank of the
  pool, 6+3 terms reached 0.22 / 0.34 (8+4: 0.19 / 0.25; 5+3: 0.22 / 0.32; 6+4: 0.21 /
  0.30), and sending the fallback only when the primary found < 10 papers gave 0.16 /
  0.25. Term priority is local-corpus IDF (TermRarity).
"""
from __future__ import annotations

import math
import re

import Stemmer

# Frozen on beir/scifact/train (web_eval dev runs); see the module docstring.
MAX_TERMS = 6
FALLBACK_TERMS = 3

_STEMMER = Stemmer.Stemmer("english")

# English function words (articles, pronouns, auxiliaries, prepositions, conjunctions,
# quantifiers). Lower-case; matched against the lower-cased token.
STOPWORDS = frozenset(
    """
    a an the this that these those there here it its itself they them their theirs we us
    our ours you your he him his she her hers i me my one ones who whom whose which what
    whatever whichever when where why how whether if then than so such thus hence
    and or nor but yet also either neither both each every all any some no not none only
    own same other another more most less least much many few several various
    is am are was were be been being do does did doing done have has had having
    will would shall should can cannot could may might must ought need needs
    of in on at by for from to into onto upon with within without about above below
    over under between among amongst through throughout during before after since until
    against across along around behind beyond near toward towards via per versus vs
    despite unlike like as because due while whereas although though even just very too
    rather quite again further still already ever never always often usually sometimes
    etc ie eg two three four five six seven eight nine ten first second third
    myself yourself himself herself ourselves yourselves themselves up out
    """.split()
)

# Hedges, intensifiers and evaluative adverbs/adjectives that carry no topic.
HEDGES = frozenset(
    """
    likely unlikely possibly potentially probably presumably perhaps apparently generally
    typically commonly frequently rarely significantly substantially strongly weakly
    markedly highly largely partly partially fully completely directly indirectly
    primarily mainly mostly solely exclusively specifically particularly especially
    approximately roughly nearly almost around about relatively comparatively
    significant substantial strong weak marked major minor important key critical crucial
    essential necessary sufficient novel new known well better worse best worst good bad
    high higher highest low lower lowest great greater greatest large larger largest
    small smaller smallest long longer short shorter early earlier late later
    specific dependent positive positively negative negatively effective ineffective
    """.split()
)

# Generic relational verbs and nouns, as roots: every inflection generated by _inflect
# (increase, increases, increased, increasing, ...) is dropped. These state HOW two
# things relate, not WHAT they are, and SciFact flips them to write false claims.
_GENERIC_ROOTS = """
    associate association correlate correlation relate relation relationship link
    increase decrease reduce reduction raise lower elevate elevation enhance improve impair
    promote inhibit suppress prevent induce cause lead result affect influence regulate
    modulate mediate contribute depend require involve play role effect impact
    show exhibit display demonstrate reveal find observe report suggest indicate predict
    have make take give use occur appear become remain exist contain lack
    help allow enable fail tend seem produce provide
    change alter level amount rate number percentage proportion ratio
    risk factor outcome response activity function process mechanism
    patient people individual person subject case
    year month week day hour time period old
    study trial analysis evidence data result finding
    type form kind way part derive base target
    """.split()


def _inflect(root: str) -> set[str]:
    forms = {root, root + "s", root + "es", root + "ed", root + "d", root + "ing"}
    if root.endswith("e"):
        forms |= {root[:-1] + "ing", root[:-1] + "ed"}
    if root.endswith("y"):
        forms |= {root[:-1] + "ies", root[:-1] + "ied"}
    if root.endswith("sis"):
        forms.add(root[:-3] + "ses")
    if root.endswith("on"):
        forms.add(root + "s")
    return forms


GENERIC = frozenset(f for root in _GENERIC_ROOTS for f in _inflect(root)) | {
    "led", "made", "took", "taken", "gave", "given", "shown", "found", "ran", "people",
    "children", "men", "women", "persons", "analyses", "data",
}

DROP = STOPWORDS | HEDGES | GENERIC

# A bare number, optionally signed, comparator-prefixed ("<0.05", "~50") or a currency
# amount ("$750", "€1,200"), or an ordinal ("21st").
_NUMBER = re.compile(r"^[~<>≤≥±]?[$€£¥]?[+-]?\d+(?:[.,]\d+)*%?$|^\d+(?:st|nd|rd|th)$")
# The number in a numbered name: "interleukin-2", "caspase-11", "glucose-6-phosphate",
# "Type 1 diabetes", "class II". Short, and only AFTER a name part (a leading number,
# "2-fold" / "12-week", is a measure and is still dropped).
_NAME_NUMBER = re.compile(r"\d{1,3}")
_NUMBERED_HEADS = frozenset({"type", "class", "stage", "grade", "phase"})
_HEAD_NUMBER = re.compile(r"\d{1,2}|I{1,3}|IV|VI{0,3}")
# A dose/measure: a number glued to a unit, optionally "/unit" ("40mg/day", "100g", "5mm").
_MEASURE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:mg|g|kg|µg|ug|mcg|ng|ml|l|dl|mm|cm|m|km|nm|um|µm|h|hr|hrs|min|s|"
    r"d|wk|y|yr|yrs|mmol|umol|µmol|nmol|mol|iu|u|kcal|cal|bp|kb|mb|x|fold)(?:/\w+)?$",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[^\s()\[\]{}]+|[()\[\]{}]")
_STRIP = "\"'`.,;:!?*“”‘’…"


def _is_entity(tok: str) -> bool:
    """Code-like or name-like token: mixes letters and digits (APOE4, B12, 6MP), has an
    upper-case letter after the first character (iPSC, mTOR, Ly6C), or is an acronym of 2+
    capitals (ART, GAVI)."""
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if has_alpha and has_digit:
        return True
    return any(c.isupper() for c in tok[1:]) or (len(tok) >= 2 and tok.isupper())


def _is_number(tok: str) -> bool:
    return bool(_NUMBER.match(tok) or _MEASURE.match(tok))


def _clean(tok: str) -> str:
    tok = tok.strip(_STRIP)
    for suffix in ("'s", "’s"):
        if tok.lower().endswith(suffix):
            tok = tok[: -len(suffix)]
    return tok.strip(_STRIP + "-/")


def _content_parts(tok: str) -> list[str]:
    """The content term(s) in one whitespace token (see the module docstring)."""
    tok = _clean(tok)
    if not tok or not any(c.isalnum() for c in tok):
        return []
    parts = [p for p in re.split(r"[-/]", tok) if p]
    if len(parts) == 1:
        p = parts[0]
        if _is_number(p) or (p.lower() in DROP and not _is_entity(p)):
            return []
        if len(p) == 1:  # a lone letter ("M. stadtmanae", "B cells") matches everything
            return []
        return [p]

    def generic(p: str) -> bool:
        return (p.lower() in DROP and not _is_entity(p)) or bool(_MEASURE.match(p))

    def name_number(i: int) -> bool:  # "interleukin-2": a short number right after a name part
        return (
            i > 0
            and bool(_NAME_NUMBER.fullmatch(parts[i]))
            and any(c.isalpha() for c in parts[i - 1])
            and not generic(parts[i - 1])
        )

    has_entity = any(_is_entity(p) for p in parts)
    # "HIV-1", "4-PBA": with an entity part the number names the entity; without one,
    # only a number that follows a name part does ("caspase-11", not "2-fold").
    numbers_ok = has_entity or all(name_number(i) for i, p in enumerate(parts) if _is_number(p))
    if not any(generic(p) for p in parts) and numbers_ok:
        return ["-".join(parts)]
    kept: list[str] = []
    prev_kept = False
    for i, p in enumerate(parts):
        if _is_number(p) and prev_kept and name_number(i):
            kept[-1] += "-" + p  # "TDP-43-induced" -> "TDP-43"
            continue
        prev_kept = False
        if generic(p) or _is_number(p) or (len(p) == 1 and not p.isupper()):
            continue
        kept.append(p)
        prev_kept = True
    return kept


def _token_parts(tokens: list[str]) -> list[tuple[str, list[str], bool]]:
    """(token, its content terms, joined) for each token; a numbered-name head followed by its
    number ("Type 1", "class II", "stage 3") becomes ONE term ("Type-1") — alone, the
    head is generic and the number is dropped, which would turn "Type 1 diabetes" into
    "diabetes"."""
    out: list[tuple[str, list[str], bool]] = []
    i = 0
    while i < len(tokens):
        word = _clean(tokens[i])
        if word.lower() in _NUMBERED_HEADS and i + 1 < len(tokens):
            nxt = _clean(tokens[i + 1])
            if _HEAD_NUMBER.fullmatch(nxt):
                out.append((tokens[i], [f"{word}-{nxt}"], True))
                i += 2
                continue
        out.append((tokens[i], _content_parts(tokens[i]), False))
        i += 1
    return out


def _strip_abbreviations(tokens: list[str]) -> list[str]:
    """Drop one-token parentheticals ("(SIDS)", "(Th2)"); keep longer ones' words."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in "([{" and i + 2 < len(tokens) and tokens[i + 2] in ")]}":
            i += 3
            continue
        if tokens[i] not in "()[]{}":
            out.append(tokens[i])
        i += 1
    return out


def _stem_key(term: str) -> str:
    return " ".join(_STEMMER.stemWord(w) for w in term.lower().split("-"))


def claim_terms(claim: str) -> list[str]:
    """All content terms of `claim`, in claim order, deduplicated by stem."""
    tokens = _strip_abbreviations(_TOKEN.findall(claim or ""))
    terms: list[str] = []
    seen: set[str] = set()
    for _tok, parts, _joined in _token_parts(tokens):
        for part in parts:
            key = _stem_key(part)
            if key in seen:
                continue
            seen.add(key)
            terms.append(part)
    return _drop_compound_parts(terms)


def _drop_compound_parts(terms: list[str]) -> list[str]:
    """Drop a one-word term that is also a part of a kept compound ("TDP" next to
    "TDP-43", "cells" next to "T-cell"): S2 already gets that word inside the compound,
    and a duplicate would only take a slot under the term cap."""
    parts = {w for t in terms if "-" in t for w in _stem_key(t).split()}
    return [t for t in terms if "-" in t or _stem_key(t) not in parts]


_WORD = re.compile(r"\w+")


class TermRarity:
    """Document frequencies of stemmed words over a corpus (the local SciFact abstracts),
    used only to decide WHICH terms survive the cap: a word common in biomedical
    abstracts ("cells", "expression", "population") is the first to go. Unsupervised
    corpus statistics — the same kind BM25 uses — no labels involved."""

    def __init__(self, df: dict[str, int], n_docs: int):
        self.df = df
        self.n_docs = n_docs

    @classmethod
    def from_texts(cls, texts) -> "TermRarity":
        df: dict[str, int] = {}
        n = 0
        for text in texts:
            n += 1
            for stem in {_STEMMER.stemWord(w) for w in _WORD.findall((text or "").lower())}:
                df[stem] = df.get(stem, 0) + 1
        return cls(df, n)

    def idf(self, term: str) -> float:
        """Smoothed IDF of the term's RAREST word (a compound is as specific as its most
        specific part); a word the corpus never saw gets the maximum."""
        words = _WORD.findall(term.lower()) or [term.lower()]
        best_df = min(self.df.get(_STEMMER.stemWord(w), 0) for w in words)
        return math.log((self.n_docs + 1) / (best_df + 1))

    def seen_idf(self, term: str) -> float | None:
        """idf() over only the words the corpus has seen; None if it saw none of them."""
        words = _WORD.findall(term.lower()) or [term.lower()]
        dfs = [d for d in (self.df.get(_STEMMER.stemWord(w), 0) for w in words) if d > 0]
        return math.log((self.n_docs + 1) / (min(dfs) + 1)) if dfs else None


def _priority(term: str, position: int, rarity: TermRarity | None) -> tuple:
    if rarity is not None:
        if _is_entity(term):  # a code/gene/drug name the corpus never saw is still specific
            return (0, -rarity.idf(term), position)
        # A plain word the corpus never saw is more often a typo or a run-together
        # ("panicprone", "methyltrasnferase") than a specific term, and S2's near-AND
        # matching loses the paper on it: such words rank LAST, and a compound is ranked
        # on its seen words only ("glucose-6-phospate" is not boosted by its typo).
        idf = rarity.seen_idf(term)
        return (1, 0.0, position) if idf is None else (0, -idf, position)
    return (not _is_entity(term), -len(term), position)


def top_terms(terms: list[str], n: int, rarity: TermRarity | None = None) -> list[str]:
    """The `n` highest-priority terms, returned in their original order. Priority is
    corpus rarity when `rarity` is given, else entity-likeness then word length."""
    if len(terms) <= n:
        return list(terms)
    ranked = sorted(range(len(terms)), key=lambda i: _priority(terms[i], i, rarity))[:n]
    return [terms[i] for i in sorted(ranked)]


def keyword_query(
    claim: str, max_terms: int = MAX_TERMS, rarity: TermRarity | None = None
) -> str:
    return " ".join(top_terms(claim_terms(claim), max_terms, rarity))


def rewrite_queries(
    claim: str,
    rarity: TermRarity | None = None,
    max_terms: int = MAX_TERMS,
    fallback_terms: int = FALLBACK_TERMS,
) -> list[str]:
    """[primary, fallback] keyword queries for `claim` (fallback omitted when it would
    equal the primary). Empty list if the claim has no content terms at all."""
    terms = claim_terms(claim)
    if not terms:
        return []
    primary = " ".join(top_terms(terms, max_terms, rarity))
    fallback = " ".join(top_terms(terms, fallback_terms, rarity))
    return [primary] if fallback == primary else [primary, fallback]


# --- Extra deterministic queries for a wider S2 candidate pool (multi-query) ---------
#
# rewrite_queries' two queries share their rarest terms, so under S2's near-AND matching
# they miss a gold paper that does not use one of those words. These variants each drop a
# different part of the claim. The split is at the first generic relational word
# ("increases", "is associated with", "regulates", ...) that follows a content term: in a
# claim that word separates what is acted on/exposed (subject) from the outcome (object).

_RELATIONAL = frozenset(f for root in _GENERIC_ROOTS for f in _inflect(root)) - STOPWORDS


def split_terms(claim: str) -> tuple[list[str], list[str]]:
    """(subject terms, object terms): content terms before / after the first generic
    relational word that follows at least one content term. ([], terms) is never
    returned — no split means (terms, [])."""
    tokens = _strip_abbreviations(_TOKEN.findall(claim or ""))
    head: list[str] = []
    tail: list[str] = []
    seen: set[str] = set()
    split = False
    for tok, parts, joined in _token_parts(tokens):
        word = _clean(tok).lower()
        if not split and head and not joined and word in _RELATIONAL:
            split = True
            continue
        for part in parts:
            key = _stem_key(part)
            if key in seen:
                continue
            seen.add(key)
            (tail if split else head).append(part)
    return head, tail


def multi_queries(claim: str, rarity: TermRarity | None = None, n: int = 3) -> list[str]:
    """Up to `n` extra keyword queries, distinct from rewrite_queries(claim) and from
    each other, best-first:

      1. subject + object: the top-2 terms of each side of the relational split;
      2. entity-focused: entity-like terms (codes, genes, drugs, acronyms) plus the
         rarest other terms, 3 in all;
      3. the rarest pair of terms;
      4. object side alone (top 3);
      5. subject side alone (top 3).

    Empty when the claim has fewer than 2 content terms."""
    terms = claim_terms(claim)
    if len(terms) < 2:
        return []
    head, tail = split_terms(claim)
    entities = [t for t in terms if _is_entity(t)]
    others = [t for t in terms if not _is_entity(t)]
    candidates: list[list[str]] = []
    if head and tail:
        candidates.append(top_terms(head, 2, rarity) + top_terms(tail, 2, rarity))
    if entities:
        ent = top_terms(entities, 3, rarity)
        rest = top_terms(others, max(0, 3 - len(ent)), rarity)
        candidates.append([t for t in terms if t in ent or t in rest])
    candidates.append(top_terms(terms, 2, rarity))
    if len(tail) >= 2:
        candidates.append(top_terms(tail, 3, rarity))
    if len(head) >= 2:
        candidates.append(top_terms(head, 3, rarity))
    taken = {_norm_q(q) for q in rewrite_queries(claim, rarity)}
    out: list[str] = []
    for c in candidates:
        q = " ".join(c)
        if len(c) < 2 or _norm_q(q) in taken:
            continue
        taken.add(_norm_q(q))
        out.append(q)
        if len(out) >= n:
            break
    return out


def _norm_q(q: str) -> str:
    return " ".join(sorted(_stem_key(w) for w in q.split()))
