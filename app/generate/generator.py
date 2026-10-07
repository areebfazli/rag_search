"""LLM answer generation over the OpenAI-compatible interface.

Points at the endpoint app.core.llm_endpoints resolves for the generator role
(free models on OpenRouter by default, under its free-only guard; Groq, Ollama or another
OpenAI-compatible backend via SSR_LLM_PROVIDER=openai_compat + SSR_LLM_BASE_URL, with no
code change). Maps the model's [n] citations back to
document ids so answers are traceable to sources, and parses the optional final
verdict line (claims only) into Answer.verdict.

On a reasoning model (gpt-oss; the default free Ling generator also reasons by default)
the hidden reasoning is billed against the same completion budget as the answer, so a
too-small budget truncates the reply — possibly to nothing. generate() reads finish_reason, retries a truncated reply once with a
larger budget, and flags one that is still cut off instead of passing it off as whole.

If the input is a claim and the reply still carries no parseable verdict (cut off to
nothing, or prose without a verdict line), generate() makes exactly ONE more call: the
verdict-only re-ask (prompts.reask_messages, REASK_MAX_TOKENS; settings.llm_reask, on by
default). Its verdict fills Answer.verdict (verdict_source "reask"); the displayed text
stays the first reply's prose, minus any verdict-like line that would contradict it
(reask_display_text). Only claims are re-asked (looks_like_claim): never a question, a
keyword query or an instruction.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from openai import APIStatusError, OpenAI, OpenAIError

from app.core.config import settings
from app.core.llm_endpoints import (
    EmptyCompletionError,
    GuardedClient,
    LLMEndpoint,
    build_client,
    completion_choice,
    endpoint_for_url,
    resolve_endpoint,
    response_cost,
    response_provider,
)
from app.core.interfaces import Answer, SearchHit, hit_passage
from app.generate.prompts import SYSTEM, VERDICTS, build_user_prompt, reask_messages

_CITE = re.compile(r"\[(\d+)\]")
# Grouped markers the model also writes: "[1, 2]", "[1,3]", "[3-4]", "[2–4]", "[1, 3-5]".
# Each part is a number or an ascending range; a range is expanded (and, like a single
# marker, range-checked against the hits by map_citations). _CITE's [n] is the one-part case.
_CITE_PART = r"\d+(?:\s*[-–—]\s*\d+)?"
_CITE_GROUP_SRC = rf"\[\s*{_CITE_PART}(?:\s*[,;]\s*{_CITE_PART})*\s*\]"
_CITE_GROUP = re.compile(rf"\[\s*({_CITE_PART}(?:\s*[,;]\s*{_CITE_PART})*)\s*\]")
_CITE_RANGE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")
# gpt-oss sometimes cites in its native CJK-bracket style — 【2】, occasionally with a
# browsing-style tail like 【2†L3-L5】 — or with fullwidth square brackets ［2］. \d also
# matches fullwidth digits, which int() reads, so ［２］ becomes [2] too.
_WIDE_CITE = re.compile(r"[【［]\s*(\d+)\s*(?:†[^】］\n]*)?[】］]")
# A well-formed verdict line: the whole line, tolerating the markdown emphasis, quotes,
# trailing period and trailing [n] citation markers models like to add, but nothing else
# (no prose) — anything looser starts matching sentences that merely mention a verdict.
# Citations are allowed because gpt-oss writes "Verdict: REFUTED[1]" routinely: 5 of 8
# unparsed verdicts in the 2026-09-24 run were exactly that, all matching the gold label.
_MARKUP = r"[\s*_`'\"]*"
_VERDICT_ALT = "|".join(v.replace(" ", r"\s+") for v in VERDICTS)
_VERDICT_LINE = re.compile(
    rf"^{_MARKUP}verdict{_MARKUP}:{_MARKUP}({_VERDICT_ALT}){_MARKUP}(?:\s*{_CITE_GROUP_SRC})*\.?{_MARKUP}$",
    re.IGNORECASE,
)
# Any line that *starts* like a verdict — used to detect a second, competing one.
_VERDICT_LIKE = re.compile(rf"^{_MARKUP}verdict{_MARKUP}:", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    """Rewrite 【n】 / ［n］ citation markers as plain ASCII [n]. Idempotent."""
    return _WIDE_CITE.sub(lambda m: f"[{int(m.group(1))}]", text)


def _cited_numbers(group: str, n_hits: int) -> set[int]:
    """The passage numbers one marker's inside names ("1, 3-4" -> {1, 3, 4}), clipped to
    1..n_hits; a descending range ("4-2") names nothing."""
    out: set[int] = set()
    for part in re.split(r"[,;]", group):
        if m := _CITE_RANGE.search(part):
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                out.update(range(max(lo, 1), min(hi, n_hits) + 1))
        elif part.strip():
            out.add(int(part))
    return {n for n in out if 1 <= n <= n_hits}


def map_citations(text: str, hits: list[SearchHit]) -> list[str]:
    """Map [n] markers in the answer to hit doc_ids — 1-based, deduped, ordered,
    ignoring out-of-range indices the model may hallucinate. Fullwidth markers are
    normalised first, so they map (and are range-checked) exactly like [n]; grouped
    markers ("[1, 2]", "[3-4]") map every passage they name."""
    cited: set[int] = set()
    for m in _CITE_GROUP.finditer(normalize_citations(text)):
        cited |= _cited_numbers(m.group(1), len(hits))
    return [hits[n - 1].doc_id for n in sorted(cited)]


def split_verdict(text: str) -> tuple[str, str | None]:
    """Split the optional trailing ``Verdict: ...`` line off an answer.

    Returns (display_text, verdict). The verdict counts only if the LAST non-empty line
    is a well-formed verdict line and no earlier line also looks like one. That is what
    keeps a verdict string quoted from a passage (or planted in one) from being read as
    the model's own: mid-line mentions never match, and an echoed verdict line followed
    by the real one is ambiguous, so it yields None rather than a guess.

    A valid line is stripped from the display text — it is a machine field, surfaced
    separately, and the prose already says the same thing in words. A malformed one is
    left in place, so nothing the model wrote is silently dropped. If the verdict was
    the ENTIRE reply, the text is kept as-is rather than serving an empty answer.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return text, None
    m = _VERDICT_LINE.match(lines[-1])
    if m is None or any(_VERDICT_LIKE.match(line) for line in lines[:-1]):
        return text, None
    verdict = " ".join(m.group(1).upper().split())
    body = "\n".join(lines[:-1]).rstrip()
    return (body or text.strip()), verdict


# --- verdict recovery when the final Verdict line is missing -----------------------------
#
# Designed on the unparsed replies of a 100-claim beir/scifact/train run (never on test).
# Both fallbacks are deliberately narrow, and both keep split_verdict's guarantee that
# text quoted from a passage never counts as the model's own verdict:
#
# * "inline": ``Verdict: X`` closing the LAST line after the prose, e.g. "... [2].
#   Verdict: REFUTED". It must follow a finished sentence (., !, ?, ) or ]), carry no
#   quote marks on either side, and be the reply's only "verdict:" mention anywhere.
# * "stance": the reply's own first sentence opens with the context as its subject and
#   a stance verb as its main verb ("The passages support ...", "The context does not
#   provide enough evidence ..."). The subject must come first and the matched subject +
#   verb must be quote-free, so "Passage [2] says 'the context supports ...'" can't
#   match; a first sentence with a hedge outside quotes (partially, but, however, ...),
#   or a reply with any "verdict:" mention, is not read at all.
#   It is on only for claims (looks_like_claim): an answer to a question, a keyword query
#   or an instruction may well say "the passages support X" without the input being a
#   claim to judge. A match is also rejected when the rest of the sentence turns the
#   stance around or away from the claim (_STANCE_SCOPE: "support neither ... nor",
#   "supports the opposite", "a different mechanism", "the claim is false", a ';' clause
#   with a competing stance verb), and when a leading "No," / "Yes," contradicts it.
VERDICT_SOURCES = ("line", "inline", "stance")
_QUOTE_CHARS = "\"“”«»„"
_INLINE_MARKUP = r"[\s*_`]*"  # no quotes: a quoted "Verdict: X" is a passage's, not ours
_INLINE_VERDICT = re.compile(
    rf"(?<=[.!?)\]])\s+{_INLINE_MARKUP}verdict{_INLINE_MARKUP}:{_INLINE_MARKUP}"
    rf"({_VERDICT_ALT}){_INLINE_MARKUP}(?:\s*{_CITE_GROUP_SRC})*\.?{_INLINE_MARKUP}$",
    re.IGNORECASE,
)
_VERDICT_MENTION = re.compile(r"verdict[\s*_`'\"]*:", re.IGNORECASE)
_SUBJECT = (
    r"(?:the\s+|these\s+|this\s+|both\s+|all\s+)?"
    r"(?:(?:provided|given|retrieved|available|cited|supplied)\s+)?"
    r"(?:context(?:\s+passages?)?|passages?|evidence|abstracts?)"
    r"(?:\s*\[\d+\])*"
)
_ADVERB = r"(?:\s+(?:directly|clearly|strongly|explicitly|consistently))?"
_YES_NO = r"(?:(?:yes|no)\s*[,.;:]\s*)?"
_CLAIM_IS = rf"{_YES_NO}(?:the\s+|this\s+)?(?:claim|statement)\s+is{_ADVERB}\s+"


def _stance(active: str, passive: str) -> re.Pattern[str]:
    """Anchored at the sentence start: the context as subject ("The passages support"),
    or the claim as subject with the context as agent ("The claim is supported by
    passage [1]")."""
    return re.compile(
        rf"^(?:{_YES_NO}{_SUBJECT}{_ADVERB}\s+(?:{active})"
        rf"|{_CLAIM_IS}(?:{passive})(?:\s+by\s+{_SUBJECT}|\s*[.,;:]))",
        re.IGNORECASE,
    )


_STANCE = (
    ("NOT ENOUGH EVIDENCE", re.compile(
        rf"^{_YES_NO}{_SUBJECT}\s+(?:do|does)\s+not{_ADVERB}\s+"
        r"(?:provide|contain|include|offer|give|present|report|mention|address|discuss|"
        r"describe|state|show|establish)\b",  # not "support": it can mean refute (train)
        re.IGNORECASE,
    )),
    ("REFUTED", _stance(r"refutes?|contradicts?|disproves?", r"refuted|contradicted|disproved")),
    ("SUPPORTED", _stance(r"supports?|confirms?", r"supported|confirmed")),
)
_HEDGE = re.compile(
    r"\b(?:partial(?:ly)?|partly|but|however|although|though|whereas|while|only|mixed|"
    r"indirect(?:ly)?|suggests?|may|might|"
    # degree hedges ("supports the claim weakly at best", "somewhat supports")
    r"weak(?:ly)?|at\s+best|somewhat|arguably|tentative(?:ly)?|in\s+part|"
    r"to\s+(?:some|a\s+limited)\s+(?:extent|degree))\b",
    re.IGNORECASE,
)
# The prompt's own "Answer (cite with [n]):" marker, which a model sometimes echoes.
_ANSWER_MARKER = re.compile(r"^answer\s*\(cite with \[n\]\)\s*:\s*", re.IGNORECASE)
_QUOTED_SPAN = re.compile(r'["“][^"“”]*["”]')
_FIRST_SENTENCE = re.compile(r"^(.*?[.!?])(?=\s|$)", re.DOTALL)
_QUESTION_START = re.compile(
    r"^(?:what|how|why|when|where|which|who|whom|whose|is|are|was|were|do|does|did|can|"
    r"could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)


def looks_like_question(query: str) -> bool:
    """True for an input phrased as a question (ends in '?' or opens with a wh-word or
    auxiliary): only a claim is judged from the reply's first sentence."""
    q = query.strip()
    return q.endswith("?") or bool(_QUESTION_START.match(q))


# --- is the input a claim? -----------------------------------------------------------------
#
# Gates both claim-only behaviours: the verdict-only re-ask (needs_reask) and the
# first-sentence stance fallback. A deliberately simple heuristic, checked to accept all
# 300 beir/scifact/test and all 809 beir/scifact/train claims:
#   1. not a question (looks_like_question) and not an instruction (_IMPERATIVE_START:
#      "Explain ...", "Tell me ...", "List ...", "Please ...");
#   2. a finite verb, found by _verb_signal, at one of two strengths:
#      strong — an auxiliary/modal/"not" anywhere ("is", "does not", "can"), or a common
#        claim verb (_CLAIM_VERBS) in its base or -s form ("Statins lower LDL", "Smoking
#        causes lung cancer"), not as the first word, not right after a determiner or
#        preposition ("an increase", "to reduce", "with lower") and not right before
#        "of" / "in" / "for" / ... unless that pair is the verb's own ("results in",
#        "consists of": _VERB_PREPOSITIONS);
#      weak — a later word ending in -s / -ed. An -s word right before a preposition or
#        conjunction, or at the very end, reads as a plural noun ("effects of statins on
#        LDL", "... in elderly patients"), and an -ed word right after a determiner or
#        preposition as an adjective ("in treated patients"); neither counts.
#      -ing words never count: without an auxiliary they are gerunds or modifiers
#      ("effects of smoking on lung function", "TNF-alpha signaling pathways").
#   3. enough words for that signal: with a final '.', two or more words and any signal
#      ("Statins decrease blood cholesterol."); without, three or more with a strong one,
#      or at least CLAIM_MIN_WORDS with a weak one.
# Keyword queries ("BRCA1 breast cancer risk", "statins") and noun phrases, with or
# without a period ("Effects of exercise on depression.", "Stem cells."), are never
# claims. Known misses, by design on the safe side: a claim whose only verb is outside
# _CLAIM_VERBS and has no -s/-ed ending ("Statins help."); it gets no re-ask and no stance
# verdict, just as a question. Known false positive: a noun phrase whose noun is also a
# claim verb in the -s form before its own preposition ("trial results in elderly
# patients").
_IMPERATIVE_START = re.compile(
    r"^(?:please|kindly|explain|tell|list|describe|summari[sz]e|compare|contrast|show|find|"
    r"give|define|outline|discuss|identify|name|provide|search|look|review|elaborate|"
    r"clarify|help|suggest|recommend|enumerate|detail|write|get|fetch|lookup)\b",
    re.IGNORECASE,
)
_AUXILIARIES = frozenset(
    "is are was were be been being am has have had do does did can could may might must "
    "shall should will would cannot not".split()
)
_CLAIM_VERBS = frozenset(
    "accelerate act activate affect allow alter ameliorate arise associate attenuate bind "
    "block cause compensate confer consist contain contribute control correlate cure decline decrease "
    "delay deplete depend derive determine develop differ display drink drive elevate encode "
    "enforce enhance exacerbate exceed exhibit express extend facilitate form impair improve "
    "increase induce influence inhibit interact kill lack lead limit localize lower maintain "
    "mediate modulate need occur originate outperform participate persist play precede "
    "predict prevent produce promote protect raise range receive reduce regulate rely repress require "
    "reside restrict result shorten stimulate suppress sustain switch target trigger worsen "
    # intransitive ones a claim can end on ("The association holds.", "Statins work.")
    "die exist fail hold matter survive vary work"
    .split()
)
# A claim verb followed by its own preposition is still a verb ("results in", "consists of",
# "develops from"), where the same word before another preposition reads as a noun ("an
# increase in", "causes of").
_VERB_PREPOSITIONS = frozenset({
    ("result", "in"), ("consist", "of"), ("occur", "in"), ("develop", "from"),
    ("develop", "in"), ("allow", "for"), ("switch", "from"), ("depend", "on"), ("rely", "on"),
    ("account", "for"), ("participate", "in"), ("compensate", "for"), ("differ", "from"),
    ("differ", "in"), ("arise", "from"), ("arise", "in"), ("derive", "from"),
    ("originate", "from"), ("originate", "in"), ("reside", "in"), ("persist", "in"),
    ("act", "on"), ("localize", "in"), ("range", "from"), ("play", "in"),
})
# Pairs that read as a verb only in the -s form ("Autophagy declines in aged organisms",
# but "cognitive decline in elderly patients" is a noun phrase).
_VERB_S_PREPOSITIONS = frozenset({("decline", "in")})
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_VERB_SUFFIX = re.compile(r"[a-z]{2,}(?:s|ed)")
_NOT_VERB_ENDING = re.compile(r"(?:ss|us|is|ics|ous)$")  # class, virus, analysis, ...
_NOUN_CONTEXT = frozenset("of in on for among at from and or versus vs".split())
# Words a verb never follows directly: a determiner or preposition makes the next word a
# noun or adjective ("the increase", "a decrease", "to reduce", "with lower", "in treated").
_NOT_BEFORE_VERB = frozenset(
    "the a an this that these those its their his her our my your no any each every some "
    "of in on for with by to from at among than as into per via".split()
)
CLAIM_MIN_WORDS = 5  # without a final period, for a weak (-s / -ed) verb signal


def _claim_verb_stem(word: str) -> str | None:
    """The _CLAIM_VERBS stem `word` is a base or -s/-es form of ("causes" -> "cause",
    "suppresss" -> "suppress", "relies" -> "rely"), or None."""
    candidates = [word, word[:-1], word[:-2]] if word.endswith("s") else [word]
    if word.endswith("ies"):
        candidates.append(word[:-3] + "y")
    return next((c for c in candidates if c in _CLAIM_VERBS), None)


def _verb_signal(words: Sequence[str]) -> str | None:
    """"strong", "weak" or None: the best finite-verb evidence in `words` (see above)."""
    low = [w.lower() for w in words]
    weak = False
    for i, w in enumerate(low):
        if w in _AUXILIARIES:
            return "strong"
        if i == 0:  # the first word is the subject ("Statins", "Increased", "Increase of")
            continue
        prev = low[i - 1]
        nxt = low[i + 1] if i + 1 < len(low) else None
        last = w.rsplit("-", 1)[-1]  # "up-regulates" -> "regulates"
        stem = _claim_verb_stem(last)
        if stem is not None and prev not in _NOT_BEFORE_VERB and (
            nxt not in _NOUN_CONTEXT or (stem, nxt) in _VERB_PREPOSITIONS
            or (last != stem and (stem, nxt) in _VERB_S_PREPOSITIONS)
        ):
            return "strong"
        if weak or not _VERB_SUFFIX.fullmatch(last) or _NOT_VERB_ENDING.search(last):
            continue
        if last.endswith("s") and (nxt is None or nxt in _NOUN_CONTEXT):
            continue  # a plural noun, not a verb
        if last.endswith("ed") and prev in _NOT_BEFORE_VERB:
            continue  # an adjective ("in treated patients")
        weak = True
    return "weak" if weak else None


def looks_like_claim(query: str) -> bool:
    """True for an input that reads as a claim to judge (see the heuristic above): the
    only inputs that get a verdict-only re-ask or a first-sentence stance verdict."""
    q = query.strip()
    if not q or looks_like_question(q) or _IMPERATIVE_START.match(q):
        return False
    n_words = len(q.split())
    signal = _verb_signal(_WORD.findall(q))
    if signal is None or n_words < 2:
        return False
    if q.endswith("."):
        return True
    return n_words >= (3 if signal == "strong" else CLAIM_MIN_WORDS)


# After the matched subject + stance verb, words that turn the stance around or point it
# away from the claim ("support neither the claim nor its negation", "supports the
# opposite conclusion", "a different mechanism", "the claim is false", "confirm nothing").
# Checked only on the stance clause: the rest of the sentence up to the first ':', since
# what follows a colon is the evidence being described ("The context refutes the claim:
# X was suppressed, rather than facilitated [1]" is a refutation, not a reversal of it).
# Quotes included (being stricter is the safe side).
# * strong words, anywhere in that clause. The bare negations are matched lowercase only,
#   so the abbreviation "NO" (nitric oxide) in a biomedical answer is never read as "no".
_SCOPE_STRONG = (
    r"(?i:opposite|contrary|negat(?:ion|e|es|ed|ing)|wrong|false|incorrect|untrue|"
    r"not\s+(?:the|this|that)\s+(?:claim|statement))"
)
_SCOPE_NEGATIONS = r"neither|nor|none|nothing|no"
_STANCE_SCOPE = {
    "SUPPORTED": re.compile(rf"\b(?:{_SCOPE_NEGATIONS}|{_SCOPE_STRONG})\b"),
    "REFUTED": re.compile(rf"\b(?:{_SCOPE_NEGATIONS}|{_SCOPE_STRONG})\b"),
    # "do not mention X, so they neither support nor refute the claim" / "nor do they
    # discuss Y" restate NOT ENOUGH EVIDENCE: the negations alone don't shift it.
    "NOT ENOUGH EVIDENCE": re.compile(rf"\b(?:{_SCOPE_STRONG})\b"),
}
# * softer words only as the verb's own object ("support a different mechanism",
#   "supports the reverse", "an inverse association"): within the first few words after
#   the verb, and not inside a "that ..." content clause, where they describe the finding
#   ("supports that damage-induced fork reversal requires ...").
_SCOPE_SOFT = re.compile(
    r"\b(?:different(?:ly)?|revers(?:e|ed|al)|inverse|converse|instead|rather)\b", re.IGNORECASE
)
_SCOPE_SOFT_WORDS = 4
_THAT_CLAUSE = re.compile(r"^\s*that\b", re.IGNORECASE)


# * in the verb's own object — the clause up to its first "that" / "which" / "who", after
#   which the words describe the finding ("refutes the claim that X is not Y", "supports
#   that ... the ALDH2 null variant ...") — for SUPPORTED / REFUTED:
#   - a negation, which points the stance away from (part of) the claim: "support a link,
#     not the causal claim", "support X, not Y", "support the claim (they do not)", "never"
#     (lowercase only, as above);
#   - an object other than the claim: "support the null hypothesis", "a null result",
#     "refute the alternative", "contradict each other".
_SCOPE_OBJECT = re.compile(
    r"\b(?:not|never|null|alternatives?|each\s+other|one\s+another|themselves)\b|n't\b"
)
_CONTENT_CLAUSE = re.compile(r"\b(?:that|which|who|whom|whose|where|whereby)\b", re.IGNORECASE)


def _scope_shifted(verdict: str, rest: str) -> bool:
    """True when the words after the matched stance verb turn it around (see above)."""
    clause = rest.split(":", 1)[0]
    if _STANCE_SCOPE[verdict].search(clause):
        return True
    if _THAT_CLAUSE.match(clause):
        return False
    head_clause = _CONTENT_CLAUSE.split(clause, 1)[0]
    if verdict != "NOT ENOUGH EVIDENCE" and _SCOPE_OBJECT.search(head_clause):
        return True
    head = " ".join(clause.split()[:_SCOPE_SOFT_WORDS])
    return bool(_SCOPE_SOFT.search(head))


# The opposite stance verb later in the same sentence competes with the first ("The
# evidence contradicts the null hypothesis, supporting the claim").
_COMPETING_STANCE = {
    "SUPPORTED": re.compile(r"\b(?:refut|contradict|disprov)", re.IGNORECASE),
    "REFUTED": re.compile(r"\b(?:support|confirm)", re.IGNORECASE),
}
# A ';' clause carrying its own stance verb competes with the first one ("...does not
# provide evidence that contradicts the claim; it supports it"). A ';' clause without one
# just continues the thought ("...do not mention E. coli; they study Serratia").
_SEMICOLON_STANCE = re.compile(
    r";.*\b(?:support|refut|contradict|confirm|disprov)", re.IGNORECASE | re.DOTALL
)
_LEADING_YES_NO = re.compile(r"^(yes|no)\b", re.IGNORECASE)
# The sentence right after the stance one opening with a contrast takes it back ("The
# claim is supported. However, the passages only show it in mice."): not read.
_CONTRAST_NEXT = re.compile(
    r"^[\s*_`]*(?:however|but|although|though|yet|nevertheless|nonetheless|that said|"
    # a correction ("Not really, though.", "In fact they refute it.", "Actually, ...")
    r"not|in\s+fact|actually|on\s+the\s+contrary|or\s+rather|wait)\b",
    re.IGNORECASE,
)
# A leading "No," never goes with SUPPORTED, and a leading "Yes," never with REFUTED or
# NOT ENOUGH EVIDENCE: the two halves disagree, so the sentence is not read.
_YES_NO_CONFLICT = {"no": {"SUPPORTED"}, "yes": {"REFUTED", "NOT ENOUGH EVIDENCE"}}


def _stance_verdict(text: str) -> str | None:
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    first_line = _ANSWER_MARKER.sub("", first_line.strip().lstrip("*_` "))
    m = _FIRST_SENTENCE.match(first_line)
    sentence = m.group(1) if m else first_line.strip()
    after = first_line[m.end():] if m else ""
    if not after.strip():  # the next sentence may start on the next non-empty line
        later = [ln for ln in text.splitlines() if ln.strip()][1:2]
        after = later[0] if later else ""
    if _CONTRAST_NEXT.match(after.strip()):
        return None
    if sentence.rstrip("*_` ").endswith("?"):  # "The passages support the claim?" asks
        return None
    # Hedges count only in the model's own words, not inside a span it quotes.
    if _HEDGE.search(_QUOTED_SPAN.sub(" ", sentence)):
        return None
    if _SEMICOLON_STANCE.search(sentence):
        return None
    for verdict, pattern in _STANCE:
        # The anchored match must itself be quote-free: the stance is the model's own
        # subject and verb, never words it is quoting (a later quoted span is fine).
        if (hit := pattern.match(sentence)) and not any(c in hit.group(0) for c in _QUOTE_CHARS):
            rest = sentence[hit.end():]
            if _scope_shifted(verdict, rest):
                return None
            if verdict in _COMPETING_STANCE and _COMPETING_STANCE[verdict].search(rest):
                return None
            yn = _LEADING_YES_NO.match(sentence)
            if yn and verdict in _YES_NO_CONFLICT[yn.group(1).lower()]:
                return None
            return verdict
    return None


def parse_verdict(text: str, allow_stance: bool = True) -> tuple[str, str | None, str | None]:
    """(display_text, verdict, source): split_verdict first; failing that, the two
    narrow fallbacks above, tried only when the reply has no verdict-like line at all
    (a malformed or competing one stays unparsed — ambiguity is never guessed through).
    source is "line", "inline", "stance", or None with no verdict."""
    body, verdict = split_verdict(text)
    if verdict is not None:
        return body, verdict, "line"
    lines = text.rstrip().splitlines()
    if not lines or any(_VERDICT_LIKE.match(ln) for ln in lines):
        return text, None, None
    mentions = len(_VERDICT_MENTION.findall(text))
    if mentions == 1:
        last = lines[-1]
        m = _INLINE_VERDICT.search(last)
        if m and last[: m.start()].count('"') % 2 == 0 and not any(
            c in last[: m.start()][-3:] for c in _QUOTE_CHARS
        ):
            head = last[: m.start()].rstrip()
            display = "\n".join([*lines[:-1], head]).rstrip()
            return display, " ".join(m.group(1).upper().split()), "inline"
    if mentions == 0 and allow_stance and (verdict := _stance_verdict(text)) is not None:
        return text, verdict, "stance"
    return text, None, None


_QUOTE_SPAN = re.compile(r'["“]([^"“”\n]{12,})["”]')
_ELLIPSIS = re.compile(r"\.\.\.|…")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
# How far after a closing quote (or before an opening one) its [n] may sit.
_QUOTE_CITE_WINDOW = 12


def _norm_for_match(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form for substring matching:
    NFKC (fullwidth, ligatures), [n] markers dropped, every non-alphanumeric run -> one
    space. Punctuation drift (curly vs straight quotes, dashes, spacing around '%') is
    what a faithful copy usually differs by; words and numbers must still match."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _CITE.sub(" ", t)
    return _NON_ALNUM.sub(" ", t).strip()


@dataclass(frozen=True)
class EvidenceQuote:
    """The first quoted span in a reply and whether it is really in the context.

    passage is the 1-based [n] next to the quote (None if it carries none); found is
    True when the quote occurs verbatim (after _norm_for_match) in THAT passage — or in
    any passage when it names none — and False otherwise. An ellipsis splits the quote
    into fragments that must each occur, in order."""

    quote: str
    passage: int | None
    found: bool


def check_evidence_quote(text: str, hits: list[SearchHit]) -> EvidenceQuote | None:
    """Find the reply's first quoted span and check it against the cited passage.
    None when the reply quotes nothing. A measurement only: it never changes the
    verdict."""
    m = _QUOTE_SPAN.search(text)
    if m is None:
        return None
    quote = m.group(1).strip()
    after = text[m.end(): m.end() + _QUOTE_CITE_WINDOW]
    before = text[max(0, m.start() - _QUOTE_CITE_WINDOW): m.start()]
    cite = _CITE.search(after) or (list(_CITE.finditer(before)) or [None])[-1]
    passage = int(cite.group(1)) if cite else None
    if passage is not None and 1 <= passage <= len(hits):
        pool = [hits[passage - 1]]
    else:
        pool = hits if passage is None else []
    frags = [f for f in (_norm_for_match(p) for p in _ELLIPSIS.split(quote)) if f]

    def occurs(hay: str) -> bool:
        pos = 0
        for f in frags:
            i = hay.find(f" {f} ", pos)  # whole words: "reduced" never matches "unreduced"
            if i < 0:
                return False
            pos = i + len(f) + 1
        return bool(frags)

    found = any(occurs(" " + _norm_for_match(hit_passage(h)) + " ") for h in pool)
    return EvidenceQuote(quote=quote, passage=passage, found=found)


# Appended to the display text when the reply is still cut off after the retry: a
# half-sentence must not read as a complete answer (an empty one least of all).
TRUNCATION_NOTE = "[Answer truncated: the model ran out of its token budget.]"
TRUNCATION_RETRY_FACTOR = 2  # one retry, at twice the configured budget

# --- the verdict-only re-ask --------------------------------------------------------------
#
# One call, no truncation retry (the budget is the retry): Ling reasons by default and the
# replies that needed a re-ask had spent all 4,096 retry tokens reasoning; Ling's endpoint
# allows 32,768 completion tokens. Frozen with the prompt (prompts.REASK_SYSTEM).
REASK_MAX_TOKENS = 8192
# Timeout / retry policy. Every generation request goes through the generator's client,
# built in __init__ with CLIENT_TIMEOUT_S and an explicit SDK max_retries (never the SDK's
# own default, 2 in openai 2.x). 30 s per attempt keeps a hung request from tying up an
# API worker for minutes. The retry count depends on the caller:
# * interactive (the API, LLMGenerator()'s default): CLIENT_MAX_RETRIES = 1. Each SDK retry
#   re-sends a request that may already have waited its full timeout, so retries multiply
#   the time an /answer worker is held: with 5 retries one /answer could take
#   2 attempts x 6 x 30 s + 2 x 120 s re-ask ~ 10 min before any backoff sleeps. With 1
#   (and API_REASK_MAX_RETRIES = 0) the worst case is 2 x 2 x 30 s + 120 s = 4 min plus
#   the SDK's backoff sleeps; one retry still absorbs a single transient 429 / 5xx.
# * batch (rag_eval passes max_retries=BATCH_MAX_RETRIES): 5, as before. A batch run has
#   no user waiting, and quick SDK retries absorb transient free-tier 429s, 5xx and
#   timeouts before rag_eval's own layer (with_rate_limit_retries: a 60 s wait on a 429
#   or empty completion, then the same claim again) has to.
CLIENT_TIMEOUT_S = 30.0
CLIENT_MAX_RETRIES = 1
BATCH_MAX_RETRIES = 5
# The re-ask has its own explicit policy, applied per call in reask_verdict (a copy of the
# same client via with_options: same connection pool, endpoint and free-only guard). It
# differs from the first call's on purpose: the re-ask may spend up to REASK_MAX_TOKENS
# (4x the default first-call budget) reasoning, and a 30 s timeout would cut off exactly
# those long replies, then silently resend them, each a request against the free daily
# cap. A 4,096-token first-call retry fits inside 30 s on Ling, so 8,192 tokens needs
# ~60 s: 120 s is 2x that. Its SDK retries also depend on the caller: rag_eval (batch)
# uses REASK_MAX_RETRIES = 1, which absorbs a transient 429/5xx while bounding a timed-out
# resend to one extra request (~4 min worst case); the API (interactive) uses
# API_REASK_MAX_RETRIES = 0, since a failed re-ask never fails /answer — the first answer
# is served with a warning — so a retry would only hold the worker longer. (The archived
# post-hoc re-ask experiment, which fetched the committed re-ask replies, sent with no SDK
# retry and a 300 s timeout.) None of these values is a request field: no cache key
# or prompt hash depends on them.
# Above this call, the two callers differ deliberately too: the API is interactive, so a
# re-ask that still fails is dropped and the first answer is served (reask_error
# recorded, a warning returned); rag_eval is a batch that must give every claim the same
# treatment, so it additionally waits out 429s and empty completions
# (rag_eval.with_rate_limit_retries) and stops resumably on the daily cap.
REASK_TIMEOUT_S = 120.0
REASK_MAX_RETRIES = 1
API_REASK_MAX_RETRIES = 0
# Shown instead of the bare truncation note when the first reply had no prose at all and
# the re-ask supplied the verdict: the reader gets a verdict with no explanation, and is
# told why rather than shown "answer truncated" next to a confident verdict badge. The
# wording follows the actual cause: REASK_NOTE when the first reply was cut off (it ends in
# TRUNCATION_NOTE), REASK_NOTE_NO_VERDICT when it finished but carried nothing besides
# verdict-like lines that did not parse (or nothing at all).
REASK_NOTE = (
    "[No explanation available: the model's answer ran out of its token budget, so this "
    "verdict comes from a second, verdict-only check of the same passages.]"
)
REASK_NOTE_NO_VERDICT = (
    "[No explanation available: the model's answer had no usable verdict line, so this "
    "verdict comes from a second, verdict-only check of the same passages.]"
)
REASK_NOTES = (REASK_NOTE, REASK_NOTE_NO_VERDICT)


def needs_reask(query: str, verdict: str | None) -> bool:
    """A claim (looks_like_claim) whose reply yielded no verdict. A question, keyword
    query or instruction never is (it gets no verdict by design), and a reply that did
    yield one — even a truncated one — is left alone."""
    return verdict is None and looks_like_claim(query)


def parse_reask(raw: str | None) -> tuple[str | None, str | None]:
    """(verdict, parse source) of a re-ask reply. The prompt asks for an explicit Verdict
    line, so the first-sentence stance fallback is off."""
    _, verdict, source = parse_verdict(normalize_citations((raw or "").strip()), allow_stance=False)
    return verdict, source


# Verdict-like lines in a first reply that yielded no parseable verdict ("Verdict -
# SUPPORTED (maybe)", "**Verdict**: probably supports", a doubled "Verdict:" line, a
# trailing "Final answer: SUPPORTED"). Once the re-ask supplies the verdict they would sit
# next to the verdict badge and could contradict it, so reask_display_text drops them.
# All shapes are conservative — a whole line, or a whole trailing sentence, that does
# nothing but state a verdict:
#  * a line labelled "verdict" (any value, any separator, markdown / bullet / blockquote /
#    heading markers allowed: "- Verdict: X", "> Verdict: X", "## Verdict"), or labelled
#    "final answer" / "conclusion" / "label" with a verdict-ish value;
#  * a line that is a bare verdict value ("**SUPPORTED**", "REFUTED."), or any short label
#    (one to three words) with exactly a verdict value ("Overall: SUPPORTED", "Assessment:
#    REFUTED (high confidence)"), or a short verdict statement ("The answer is SUPPORTED.",
#    "Conclusion: the claim is supported.");
#  * the label + verdict-ish value, or the verdict statement, closing a line after a
#    finished sentence ("... [1][2]. Final answer: SUPPORTED", "... [2]. Thus, Verdict:
#    SUPPORTED (high confidence)", "... [1]. In conclusion, the claim is supported."), the
#    label form at most _TRAILING_VERDICT_MAX chars long.
# A sentence that merely mentions a verdict or uses the word in prose ("the data supported
# the hypothesis", "The verdict of the trial was clear: ...") stays.
_VERDICTISH = (
    r"(?:supported|supports?|refuted|refutes?|contradict\w*|not\s+enough|nei\b|"
    r"insufficient|true|false|probably|likely|yes\b|no\b)"
)
# Markdown emphasis, quotes, and heading / blockquote / bullet markers opening a line.
_LINE_LEAD = r"[\s*_`'\"#>•+\-–—]*"
_LABEL_SEP = rf"{_MARKUP}\s*(?:[:=\-–—]|\s(?={_MARKUP}{_VERDICTISH}))"
_LABEL_PREFIX = r"(?:(?:the|my|final|overall)\s+)*"
_VERDICT_LABEL_LINE = re.compile(
    rf"^{_LINE_LEAD}\(?{_MARKUP}{_LABEL_PREFIX}verdict{_LABEL_SEP}",
    re.IGNORECASE,
)
_VERDICT_HEADING = re.compile(rf"^{_LINE_LEAD}{_LABEL_PREFIX}verdict{_MARKUP}:?{_MARKUP}$", re.IGNORECASE)
_OTHER_LABEL = rf"{_LABEL_PREFIX}(?:final\s+answer|conclusion|label)"
_OTHER_LABEL_LINE = re.compile(
    rf"^{_LINE_LEAD}\(?{_MARKUP}{_OTHER_LABEL}{_LABEL_SEP}{_MARKUP}{_VERDICTISH}",
    re.IGNORECASE,
)
# Exactly a verdict value, then only markup, [n] markers, one short parenthetical, a period.
_VERDICT_VALUE = (
    r"(?:supported|refuted|not\s+enough\s+evidence|not\s+supported|insufficient\s+evidence|"
    r"nei|supports|refutes)"
)
_VALUE_TAIL = (
    rf"{_MARKUP}(?:\s*\[\d+\])*(?:\s*\([^()\n]{{0,60}}\))?{_MARKUP}(?:\s*\[\d+\])*\.?{_MARKUP}"
)
_SHORT_LABEL = rf"(?:[a-z]+\s+){{0,2}}[a-z]+{_MARKUP}\s*[:=\-–—]{_MARKUP}\s*"
_CONNECTOR = r"(?:(?:so|thus|hence|therefore|overall|in\s+(?:conclusion|summary|short|sum))\s*,?\s+)?"
_VERDICT_STATEMENT = (
    rf"{_CONNECTOR}{_LABEL_PREFIX}(?:answer|verdict|label|conclusion|result|assessment|claim|"
    rf"statement)\s+is\s+(?:(?:therefore|thus|hence|then)\s+)?{_VERDICT_VALUE}{_VALUE_TAIL}"
)
_VERDICT_ONLY_LINES = (
    re.compile(rf"^{_LINE_LEAD}{_VERDICT_VALUE}{_VALUE_TAIL}$", re.IGNORECASE),
    re.compile(rf"^{_LINE_LEAD}{_SHORT_LABEL}{_VERDICT_VALUE}{_VALUE_TAIL}$", re.IGNORECASE),
    re.compile(rf"^{_LINE_LEAD}(?:{_SHORT_LABEL})?{_VERDICT_STATEMENT}$", re.IGNORECASE),
)
_TRAILING_VERDICT_MAX = 60
_SENTENCE_END = r"(?<=[.!?)\]\"”])"
_TRAILING_VERDICT = re.compile(
    rf"{_SENTENCE_END}\s+\(?{_MARKUP}{_CONNECTOR}(?:{_LABEL_PREFIX}verdict|{_OTHER_LABEL})"
    rf"{_LABEL_SEP}{_MARKUP}{_VERDICTISH}[^\n]{{0,{_TRAILING_VERDICT_MAX}}}$",
    re.IGNORECASE,
)
_TRAILING_STATEMENT = re.compile(rf"{_SENTENCE_END}\s+{_VERDICT_STATEMENT}$", re.IGNORECASE)


def _verdict_only_line(line: str) -> bool:
    return bool(
        _VERDICT_LABEL_LINE.match(line) or _OTHER_LABEL_LINE.match(line)
        or _VERDICT_HEADING.match(line) or any(p.match(line) for p in _VERDICT_ONLY_LINES)
    )


def strip_verdict_lines(text: str) -> str:
    """text without its verdict-like lines, and without a verdict statement closing any
    remaining line; see the shapes above."""
    kept = [
        _TRAILING_STATEMENT.sub("", _TRAILING_VERDICT.sub("", ln)).rstrip() if ln.strip() else ln
        for ln in text.splitlines()
        if not (ln.strip() and _verdict_only_line(ln))
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept).strip()


def reask_display_text(text: str) -> str:
    """The answer text to show once the re-ask supplied the verdict: the first reply's
    prose as served (truncation note kept), minus any verdict-like line that could
    contradict the re-ask's verdict (strip_verdict_lines). If no prose is left, a note
    saying why, by cause: REASK_NOTE for a reply cut off by its token budget (it ends in
    TRUNCATION_NOTE), REASK_NOTE_NO_VERDICT for one that finished without a usable
    verdict line."""
    body = text.rstrip()
    truncated = body.endswith(TRUNCATION_NOTE)
    if truncated:
        body = body[: -len(TRUNCATION_NOTE)]
    stripped = strip_verdict_lines(body)
    if not stripped:
        return REASK_NOTE if truncated else REASK_NOTE_NO_VERDICT
    if stripped == body.strip():
        return text  # nothing verdict-like: served exactly as before
    return f"{stripped}\n\n{TRUNCATION_NOTE}" if truncated else stripped


def reask_prompt_hash() -> str:
    """sha256 of the re-ask request rendered on fixed placeholders (the value
    the post-hoc re-ask experiment froze and rag.json records as run.reask.prompt_hash): a
    wording or layout edit moves it."""
    ph = [SearchHit("{doc_id}", 0.0, "{text}", {"title": "{title}"})] * 2
    blob = json.dumps(reask_messages("{claim}", ph), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class ReaskReply:
    """How one re-ask call ended (raw reply as returned, before parsing)."""

    raw: str
    finish_reason: str | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    provider: str | None


@dataclass
class GeneratedAnswer(Answer):
    """An Answer plus how the generation call ended — a side channel for the eval.

    Every field is defaulted, so this is a drop-in Answer (the API response is built
    from the base fields only). Token counts are those of the FINAL attempt and are
    None when the provider does not report them (reasoning_tokens is only reported by
    reasoning models, under usage.completion_tokens_details)."""

    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    attempts: int = 1
    truncated: bool = False
    # OpenRouter's reported cost (USD), summed over ALL attempts (a truncation retry is
    # billed too) — None if any attempt went unreported — and the upstream provider that
    # served the final attempt (None when the backend doesn't say).
    cost_usd: float | None = None
    provider: str | None = None
    # Which parse produced `verdict` (VERDICT_SOURCES: "line" | "inline" | "stance"), None
    # when there is none; and the evidence-quote check (check_evidence_quote) — None when
    # the reply quotes nothing. Measurements for the eval; the API does not expose them.
    verdict_source: str | None = None
    evidence_quote: EvidenceQuote | None = None
    # The verdict-only re-ask: whether it was sent, its reply (None if not sent or if it
    # failed), and the error type when it failed (the first answer is then served as is).
    # finish_reason / token counts above stay the FIRST reply's; cost_usd includes the
    # re-ask.
    reask_attempted: bool = False
    reask: ReaskReply | None = None
    reask_error: str | None = None
    # Short notes for the API response (main.py merges getattr(ans, "warnings")).
    warnings: list[str] = field(default_factory=list)


# Sent with every generation and re-ask request (the value every committed number was
# generated with).
GENERATION_TEMPERATURE = 0.1


def resolve_reasoning_effort(setting: str) -> str | None:
    """The reasoning effort to send, or None to omit the parameter.

    See Settings.llm_reasoning_effort: "auto", "" and "off" never send it (the model's own
    default applies; a non-reasoning model may reject the parameter outright); anything
    else is an explicit choice, sent as-is.
    """
    value = setting.strip().lower()
    return None if value in {"", "off", "auto"} else value


def _usage_counts(resp) -> tuple[int | None, int | None]:
    """(completion_tokens, reasoning_tokens) from resp.usage, tolerating providers that
    omit usage or its completion_tokens_details breakdown."""
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(usage, "completion_tokens", None), getattr(details, "reasoning_tokens", None)


def reasoning_params(provider: str, effort: str | None) -> dict:
    """The request field that carries `effort`, in the provider's own form: OpenRouter's
    unified ``reasoning: {"effort": ...}`` object (docs: openrouter.ai/docs/use-cases/
    reasoning-tokens), or the top-level ``reasoning_effort`` Groq, Ollama and OpenAI accept.
    Empty when there is nothing to send."""
    if not effort:
        return {}
    if provider == "openrouter":
        return {"reasoning": {"effort": effort}}
    return {"reasoning_effort": effort}


def _choice_or_raise(resp, costs: list[float | None]):
    """completion_choice(resp), with any EmptyCompletionError carrying what this
    generation's calls so far reported costing (None if any is unknown)."""
    try:
        return completion_choice(resp)
    except EmptyCompletionError as e:
        e.cost_usd = None if any(c is None for c in costs) else sum(costs)
        raise


class LLMGenerator:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
        endpoint: LLMEndpoint | None = None,
        reask: bool | None = None,
        max_retries: int = CLIENT_MAX_RETRIES,
        reask_max_retries: int = API_REASK_MAX_RETRIES,
    ):
        # Where requests go. An explicit endpoint wins; an explicit base_url builds one
        # whose provider (and default key) follows that URL; otherwise the settings
        # decide (llm_endpoints.resolve_endpoint). `model` overrides the model id only.
        if endpoint is None:
            if base_url is not None:
                endpoint = endpoint_for_url(
                    "generator", base_url, model or settings.llm_model, api_key
                )
            else:
                endpoint = resolve_endpoint("generator", require_key=False)
        if model is not None or api_key is not None:
            endpoint = LLMEndpoint(
                "generator",
                endpoint.provider,
                endpoint.base_url,
                model or endpoint.model,
                endpoint.api_key if api_key is None else api_key,
            )
        endpoint.check()  # free-only guard: refuse a disallowed model before any request
        self.endpoint = endpoint
        self.model = endpoint.model
        self.max_completion_tokens = max_completion_tokens or settings.llm_max_completion_tokens
        self.reasoning_effort = resolve_reasoning_effort(
            settings.llm_reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        # generate() re-asks a claim with no verdict (settings.llm_reask). rag_eval turns
        # it off here and applies the same re-ask as a cached post-step instead.
        self.reask_enabled = settings.llm_reask if reask is None else reask
        # SDK retry counts (see CLIENT_MAX_RETRIES): the defaults are the API's; rag_eval
        # passes the batch policy (BATCH_MAX_RETRIES, REASK_MAX_RETRIES).
        self.reask_max_retries = reask_max_retries
        self.client = build_client(
            endpoint,
            factory=OpenAI,
            max_retries=max_retries,
            timeout=CLIENT_TIMEOUT_S,  # don't let a hung request tie up a worker for minutes
        )

    def _reask_client(self):
        """self.client with the re-ask policy (REASK_TIMEOUT_S / self.reask_max_retries): a
        GuardedClient over the SDK client's with_options copy. A client without that
        shape (a test fake, or one a caller swapped in) is used as it is."""
        raw = getattr(self.client, "_client", None)
        with_options = getattr(raw, "with_options", None)
        if not isinstance(self.client, GuardedClient) or not callable(with_options):
            return self.client
        return GuardedClient(
            with_options(timeout=REASK_TIMEOUT_S, max_retries=self.reask_max_retries), self.endpoint
        )

    def _complete(self, messages: list[dict], max_tokens: int, client=None):
        # Reasoning rides in extra_body and only when resolved, in the provider's own
        # form: it is a passthrough, and a backend that doesn't know it must never see
        # it. OpenRouter requests also carry the endpoint's free provider routing, and
        # the free-only guard is checked here as well as in the client wrapper, so it
        # holds even if self.client is swapped out.
        body = {**reasoning_params(self.endpoint.provider, self.reasoning_effort),
                **self.endpoint.extra_body()}
        self.endpoint.check(model=self.model, body=body)
        extra: dict = {"extra_body": body} if body else {}
        extra["temperature"] = GENERATION_TEMPERATURE
        return (client or self.client).chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            **extra,
        )

    def _complete_counted(self, messages: list[dict], max_tokens: int, costs: list[float | None]):
        """_complete, with an OpenAIError carrying ``cost_usd``: what this generation's
        earlier attempts reported costing plus the failed one — 0 for an HTTP error status
        (an error response is not a billed completion), unknown (None) for a timeout or
        connection error (the provider may have generated and billed). None overall if
        any part is unknown."""
        try:
            return self._complete(messages, max_tokens)
        except OpenAIError as e:
            failed = 0.0 if isinstance(e, APIStatusError) else None
            parts = [*costs, failed]
            e.cost_usd = None if any(c is None for c in parts) else sum(parts)
            raise

    def reask_verdict(self, query: str, hits: Sequence[SearchHit]) -> ReaskReply:
        """Exactly one verdict-only call (prompts.reask_messages, REASK_MAX_TOKENS) through
        the same _complete as generation: same endpoint, free-only check, provider
        routing and params (temperature / reasoning), with the re-ask's own
        timeout / retry policy (REASK_TIMEOUT_S, REASK_MAX_RETRIES) whichever caller sends
        it. Raises like any call (EmptyCompletionError for a 200 with no completion); what
        happens then is the caller's (see REASK_TIMEOUT_S)."""
        resp = self._complete(reask_messages(query, hits), REASK_MAX_TOKENS, self._reask_client())
        cost = response_cost(resp)
        choice = _choice_or_raise(resp, [cost])
        completion_tokens, reasoning_tokens = _usage_counts(resp)
        return ReaskReply(
            raw=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            provider=response_provider(resp),
        )

    def generate(self, query: str, hits: list[SearchHit]) -> Answer:
        """One grounded answer: generation (+ one truncation retry) and, for a claim with
        no verdict, the verdict-only re-ask when it is on."""
        if not hits:
            return GeneratedAnswer(text="No relevant documents were found.", citations=[], hits=[])
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(query, hits)},
        ]
        # finish_reason "length" means the budget ran out mid-reply (on a reasoning model
        # it can run out before the first answer token). One retry at a larger budget
        # recovers the common case at the cost of one extra call, only when needed;
        # retrying further would just burn quota on a reply that is pathologically long.
        budget = self.max_completion_tokens
        costs: list[float | None] = []
        resp = self._complete_counted(messages, budget, costs)
        attempts = 1
        costs.append(response_cost(resp))
        choice = _choice_or_raise(resp, costs)
        if choice.finish_reason == "length":
            # A failure here must not lose what the first attempt was billed: the
            # exception carries it (cost_usd), as an empty completion's already does.
            resp = self._complete_counted(messages, budget * TRUNCATION_RETRY_FACTOR, costs)
            attempts = 2
            costs.append(response_cost(resp))
            choice = _choice_or_raise(resp, costs)
        truncated = choice.finish_reason == "length"
        raw = normalize_citations((choice.message.content or "").strip())
        # A cut-off reply can't have been cut inside a valid verdict line (the whole line
        # must match a full verdict value), so the line / inline parse is safe on a
        # truncated reply. The first-sentence stance fallback is not: a cut-off reply's
        # opening may be about to be hedged or turned around, so a truncated reply gets no
        # stance verdict (a claim then goes to the verdict-only re-ask instead).
        text, verdict, verdict_source = parse_verdict(
            raw, allow_stance=looks_like_claim(query) and not truncated
        )
        if truncated:
            text = f"{text}\n\n{TRUNCATION_NOTE}" if text else TRUNCATION_NOTE
        completion_tokens, reasoning_tokens = _usage_counts(resp)
        provider = response_provider(resp)
        # A claim with no verdict gets exactly one verdict-only re-ask. A failed re-ask
        # (provider error, empty completion) never fails the request: the first answer is
        # served as it is, and the error type is recorded.
        reask_attempted, reask, reask_error = False, None, None
        if self.reask_enabled and needs_reask(query, verdict):
            reask_attempted = True
            try:
                reask = self.reask_verdict(query, hits)
            except EmptyCompletionError as e:
                costs.append(e.cost_usd)  # the re-ask's own reported cost (None if unknown)
                reask_error = type(e).__name__
            except OpenAIError as e:
                costs.append(None)  # may have been billed; unknown
                reask_error = type(e).__name__
            else:
                costs.append(reask.cost_usd)
                reask_verdict, _ = parse_reask(reask.raw)
                if reask_verdict is not None:
                    verdict, verdict_source = reask_verdict, "reask"
                    text = reask_display_text(text)
        # Citations come from the displayed text, so every listed source is one the
        # reader can see referenced — plus any [n] on the verdict line the parser took off
        # it ("Verdict: REFUTED [3]": the model's own citation for its verdict). For a
        # "line" / "inline" verdict the reply is exactly the display text + that line.
        cite_text = raw if verdict_source in ("line", "inline") else text
        return GeneratedAnswer(
            text=text,
            citations=map_citations(cite_text, hits),
            hits=hits,
            verdict=verdict,
            finish_reason=choice.finish_reason,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            attempts=attempts,
            truncated=truncated,
            # All-or-nothing: a partly reported cost would undercount.
            cost_usd=None if any(c is None for c in costs) else sum(costs),
            provider=provider,
            verdict_source=verdict_source,
            evidence_quote=check_evidence_quote(text, hits),
            reask_attempted=reask_attempted,
            reask=reask,
            reask_error=reask_error,
        )
