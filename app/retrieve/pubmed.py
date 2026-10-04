"""PubMed (NCBI E-utilities) as an extra open-web candidate source for claim -> paper.

Two requests per search, both to the fixed host eutils.ncbi.nlm.nih.gov (never a setting):

1. ESearch for PMIDs in PubMed's Best Match order. "The E-utilities In-Depth: Parameters,
   Syntax and More" (https://www.ncbi.nlm.nih.gov/books/NBK25499/, last update March 4,
   2026): "Values of sort for PubMed are as follows: ... relevance – default sort order,
   (“Best Match”) on web PubMed", and "the default value used by ESearch for a given
   database may differ from that used on NCBI web search pages" — so sort=relevance is
   sent explicitly. retmode: "'json' is also supported to return output in JSON format".
   retmax: "up to a maximum of 10,000 records ... For PubMed and PMC, ESearch can only
   retrieve the first 10,000 records matching the query"; retstart defaults to 0. term:
   "For very long queries (more than several hundred characters long), consider using an
   HTTP POST call" — we cap the term (MAX_TERM_CHARS) instead and stay on GET.
2. ONE EFetch for those PMIDs (db=pubmed, id=comma list, rettype=abstract, retmode=xml).
   NBK25499: "There is no set maximum for the number of UIDs that can be passed to
   EFetch, but if more than about 200 UIDs are to be provided, the request should be
   made using the HTTP POST method" — so a search fetches at most MAX_IDS (200) and a
   GET suffices (200 PMIDs is a ~2.6 KB URL).

Usage policy ("A General Introduction to the E-utilities",
https://www.ncbi.nlm.nih.gov/books/NBK25497/, last update November 17, 2022): "Without an
API key, any site (IP address) posting more than 3 requests per second to the
E-utilities will receive an error message" ({"error":"API rate limit exceeded",...});
"NCBI recommends that users post no more than three URL requests per second and limit
large jobs to either weekends or between 9:00 PM and 5:00 AM Eastern time during
weekdays". No API key is used here, so a process-wide limiter (web_sources.host_limiter)
spaces every request, retries included, to DEFAULT_RATE_PER_S = 2/s, and a rate above 3
is refused.

tool/email: NBK25499 says "The following two parameters should be included in all
E-utility requests. tool ... email" — recommended, not required (the NLM data guide,
https://dataguide.nlm.nih.gov/eutilities/utilities.html: "the tool and email parameters
are allowed (and encouraged) on any E-utilities URL"). NBK25497: if NCBI blocks an IP,
"service will not be restored unless the developers ... register values of the tool and
email parameters", and email "should be a complete and valid e-mail address of the
software developer and not that of a third-party end user". We send tool=rag_search and
NO email: none has been provided, and inventing one would be wrong. A deployment that
expects sustained traffic should register a real developer email with NCBI and add it.

Query syntax (PubMed User Guide, https://pubmed.ncbi.nlm.nih.gov/help/, last update
September 30, 2026): untagged terms go through Automatic Term Mapping (ATM) and "PubMed
applies an AND operator between concepts"; "Enter Boolean operators in uppercase
characters"; [tag] field tags, "double quotes" phrases and "*" wildcards all change or
"turn off Automatic Term Mapping"; and "If you use a hyphen and the phrase is not found
in the phrase index, the search will not return any results for that phrase".
pubmed_term() therefore reduces untrusted input to plain words so ATM keeps working and
nobody's claim text can steer the search syntax. Best Match itself: "'weight' is
calculated for citations depending on how many search terms are found and in which
fields ... re-ranked for better relevance by a new machine-learning algorithm".

Everything PubMed returns is UNTRUSTED (anyone can publish an abstract). The XML is
refused if it declares entities or has an internal DTD subset (only the plain external
DOCTYPE line PubMed itself sends is stripped), parsed with xml.etree.ElementTree, and all
text goes through semantic_scholar.clean_untrusted with the same caps as S2 text.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable

import httpx

from app.retrieve.semantic_scholar import (
    MAX_ABSTRACT_CHARS,
    MAX_TITLE_CHARS,
    RateLimiter,
    ResponseCache,
    clean_untrusted,
)
from app.retrieve.web_sources import (
    MAX_RESPONSE_BYTES,
    ExternalPaper,
    HttpFetcher,
    SourceError,
    host_limiter,
    normalize_doi,
    normalize_pmid,
)

# Fixed, not a setting: requests only ever go to NCBI's E-utilities host.
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_PATH = "/esearch.fcgi"
EFETCH_PATH = "/efetch.fcgi"
SOURCE = "pubmed"
TOOL = "rag_search"  # NBK25499 tool: "a string with no internal spaces"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

DEFAULT_RATE_PER_S = 2.0  # NBK25497: > 3 req/s without an API key "will receive an error"
MAX_RATE_PER_S = 3.0
DEFAULT_MAX_RETRIES = 3
MAX_IDS = 200  # NBK25499: "more than about 200 UIDs ... should be made using the HTTP POST method"
MAX_TERM_CHARS = 300  # NBK25499: POST advised for queries "more than several hundred characters"

# Neutralising PubMed query syntax (see module docstring).
_FIELD_TAG = re.compile(r"\[[^\]]*\]")  # asthma[tiab], "x"[Title:~3]
_POSSESSIVE = re.compile(r"['’]s\b", re.IGNORECASE)
_APOSTROPHE = re.compile(r"['’`]")
_NOT_WORD = re.compile(r"[^\w.]+")  # anything but letters/digits/underscore/period -> space
_EDGE_DOTS = re.compile(r"^[._]+|[._]+$")
_BOOLEAN = {"and", "or", "not"}

# XML safety (see _parse_xml).
_ENTITY_DECL = re.compile(r"<!ENTITY", re.IGNORECASE)
_DOCTYPE_ANY = re.compile(r"<!DOCTYPE", re.IGNORECASE)
_QUOTED = r"(?:\"[^\"<>\[\]]*\"|'[^'<>\[\]]*')"
_DOCTYPE_EXTERNAL = re.compile(
    rf"<!DOCTYPE\s+[A-Za-z_][\w.:-]*\s+(?:PUBLIC\s+{_QUOTED}\s+{_QUOTED}|SYSTEM\s+{_QUOTED})\s*>"
)
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d|21\d\d)\b")


def pubmed_term(query: object) -> str:
    """Untrusted claim/keyword text -> a plain-word PubMed `term`, or "" if nothing is left.

    Drops [field tags] whole, and turns quotes, parentheses, wildcards (*), hyphens,
    slashes, colons, '#', '&' and every other non-word character into spaces, so the
    result is bare words that PubMed's Automatic Term Mapping handles normally (ATM still
    maps "vitamin b12" or "heart attack" to MeSH). Boolean operators are only operators in
    upper case; AND/OR/NOT in any case are dropped (lower-case "not" is not a PubMed
    stopword, so keeping it would AND the literal word into the search). Possessive "'s"
    is removed and other apostrophes join ("Alzheimer's" -> "Alzheimer"). A term made only
    of numbers is dropped (PubMed treats it as a PMID lookup, not a search). Capped at
    MAX_TERM_CHARS on a word boundary.
    """
    s = clean_untrusted(query, 10 * MAX_TERM_CHARS)
    s = _FIELD_TAG.sub(" ", s)
    s = _POSSESSIVE.sub("", s)
    s = _APOSTROPHE.sub("", s)
    s = _NOT_WORD.sub(" ", s)
    words = []
    for tok in s.split():
        tok = _EDGE_DOTS.sub("", tok)
        if tok and tok.lower() not in _BOOLEAN:
            words.append(tok)
    if all(w.isdigit() for w in words):
        return ""
    out = ""
    for w in words:
        nxt = f"{out} {w}" if out else w
        if len(nxt) > MAX_TERM_CHARS:
            break
        out = nxt
    return out or words[0][:MAX_TERM_CHARS]


def esearch_params(term: str, limit: int) -> dict:
    """ESearch query parameters (the fetcher adds db/tool) — also the cache key."""
    return {
        "term": term,
        "retmax": max(1, min(int(limit), MAX_IDS)),
        "retstart": 0,
        "sort": "relevance",
        "retmode": "json",
    }


def parse_esearch(payload: object, limit: int) -> list[str]:
    """ESearch JSON -> PMIDs in Best Match order, validated, deduped, capped at `limit`."""
    result = payload.get("esearchresult") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raise SourceError("unexpected esearch response shape")
    if result.get("ERROR"):
        raise SourceError("esearch reported an error")
    ids = result.get("idlist", [])
    if not isinstance(ids, list):
        raise SourceError("unexpected esearch response shape")
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        pmid = normalize_pmid(raw)
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
            if len(out) >= limit:
                break
    return out


def _parse_xml(text: object) -> ET.Element:
    """Third-party XML -> Element, refusing what ElementTree should never be fed.

    PubMed's efetch XML starts with a plain external DOCTYPE
    (<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, ..." "https://dtd...">);
    that one line is stripped (expat never fetches it anyway). Anything else DTD-shaped —
    an internal subset ("[...]"), any <!ENTITY declaration (billion-laughs / external
    entity tricks), a second DOCTYPE — refuses the whole payload, as does an oversized or
    malformed one. Undeclared entities then fail to parse rather than resolve.
    """
    if not isinstance(text, str):
        raise SourceError("unexpected efetch response type")
    if len(text) > MAX_RESPONSE_BYTES:
        raise SourceError("response too large")
    if _ENTITY_DECL.search(text):
        raise SourceError("refused XML: entity declaration")
    text = _DOCTYPE_EXTERNAL.sub("", text, count=1)
    if _DOCTYPE_ANY.search(text):
        raise SourceError("refused XML: DOCTYPE with an internal subset")
    try:
        return ET.fromstring(text)
    except ET.ParseError as e:
        raise SourceError("malformed XML response") from e


def _text(el: ET.Element | None) -> str:
    """All text under an element, inline markup (<i>, <sup>, MathML) flattened away."""
    return "".join(el.itertext()) if el is not None else ""


def _abstract(container: ET.Element | None, include_labels: bool) -> str:
    """Every <AbstractText> section of <Abstract>, in order, optionally "LABEL: text".
    OtherAbstract (translations) and CopyrightInformation are not included."""
    if container is None:
        return ""
    parts = []
    for sec in container.findall("Abstract/AbstractText"):
        body = clean_untrusted(_text(sec), MAX_ABSTRACT_CHARS)
        if not body:
            continue
        label = clean_untrusted(sec.get("Label"), 80)
        parts.append(f"{label}: {body}" if include_labels and label else body)
    return clean_untrusted(" ".join(parts), MAX_ABSTRACT_CHARS)


def _year(*dates: ET.Element | None) -> int | None:
    """First plausible year from <Year> (or a free-text <MedlineDate>, e.g. "1998 Dec-1999 Jan")."""
    for d in dates:
        if d is None:
            continue
        for tag in ("Year", "MedlineDate"):
            m = _YEAR.search(_text(d.find(tag)))
            if m:
                return int(m.group(1))
    return None


def _doi(id_list: ET.Element | None, elocations: Iterable[ET.Element] = ()) -> str | None:
    """DOI from the record's own ArticleIdList (IdType="doi"), else a valid ELocationID."""
    if id_list is not None:
        for aid in id_list.findall("ArticleId"):
            if aid.get("IdType") == "doi" and (doi := normalize_doi(_text(aid))):
                return doi
    for loc in elocations:
        if loc.get("EIdType") == "doi" and loc.get("ValidYN", "Y") != "N":
            if doi := normalize_doi(_text(loc)):
                return doi
    return None


def _journal_article(rec: ET.Element, include_labels: bool) -> ExternalPaper | None:
    cit = rec.find("MedlineCitation")
    pmid = normalize_pmid(_text(cit.find("PMID")).strip()) if cit is not None else None
    if pmid is None:
        return None
    art = cit.find("Article")
    if art is None:
        return ExternalPaper(source=SOURCE, rank=-1, pmid=pmid, url=PUBMED_URL.format(pmid=pmid))
    return ExternalPaper(
        source=SOURCE,
        rank=-1,
        title=clean_untrusted(_text(art.find("ArticleTitle")), MAX_TITLE_CHARS),
        abstract=_abstract(art, include_labels),
        year=_year(art.find("Journal/JournalIssue/PubDate"), art.find("ArticleDate")),
        url=PUBMED_URL.format(pmid=pmid),
        pmid=pmid,
        # Only PubmedData's own list: ReferenceList entries carry their own ArticleIdLists.
        doi=_doi(rec.find("PubmedData/ArticleIdList"), art.findall("ELocationID")),
    )


def _book_article(rec: ET.Element, include_labels: bool) -> ExternalPaper | None:
    doc = rec.find("BookDocument")
    pmid = normalize_pmid(_text(doc.find("PMID")).strip()) if doc is not None else None
    if pmid is None:
        return None
    title = _text(doc.find("ArticleTitle")) or _text(doc.find("Book/BookTitle"))
    return ExternalPaper(
        source=SOURCE,
        rank=-1,
        title=clean_untrusted(title, MAX_TITLE_CHARS),
        abstract=_abstract(doc, include_labels),
        year=_year(doc.find("Book/PubDate"), doc.find("ContributionDate")),
        url=PUBMED_URL.format(pmid=pmid),
        pmid=pmid,
        doi=_doi(doc.find("ArticleIdList"), doc.findall("ELocationID")),
    )


def parse_efetch(text: object, include_labels: bool = True) -> dict[str, ExternalPaper]:
    """EFetch PubMed XML -> {pmid: ExternalPaper} (rank -1; the caller assigns ranks).

    Handles <PubmedArticle> and <PubmedBookArticle>; records without a usable PMID are
    skipped, a record without an abstract keeps abstract "". An <ERROR> reply, a root
    other than PubmedArticleSet, or unsafe/malformed XML raises SourceError.
    """
    root = _parse_xml(text)
    if root.tag != "PubmedArticleSet":
        raise SourceError("efetch reported an error" if root.find(".//ERROR") is not None
                          else "unexpected efetch response shape")
    out: dict[str, ExternalPaper] = {}
    for rec in root:
        if rec.tag == "PubmedArticle":
            paper = _journal_article(rec, include_labels)
        elif rec.tag == "PubmedBookArticle":
            paper = _book_article(rec, include_labels)
        else:
            continue  # e.g. DeleteCitation
        if paper is not None and paper.pmid not in out:
            out[paper.pmid] = paper
    return out


class PubMedSource:
    """PubMed Best Match search -> ExternalPaper candidates (see module docstring).

    `fetcher` replaces the whole HTTP layer (it must point at EUTILS_BASE_URL); otherwise
    one HttpFetcher is built from cache/client/limiter/max_retries/max_wait_s/
    max_backoff_s. Without a `limiter`, the process-wide "pubmed" host limiter is used at
    `rate` req/s (first caller's rate wins for the process). `cache=None` disables
    caching; with a ResponseCache, esearch is cached by term/retmax/sort and efetch by its
    exact id list, as raw payloads.
    """

    def __init__(
        self,
        *,
        fetcher: HttpFetcher | None = None,
        cache: ResponseCache | None = None,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        rate: float = DEFAULT_RATE_PER_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_wait_s: float | None = None,
        max_backoff_s: float = 30.0,
        timeout_s: float = 10.0,
        include_labels: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if fetcher is None:
            if not 0 < rate <= MAX_RATE_PER_S:
                raise ValueError(f"rate must be in (0, {MAX_RATE_PER_S}] req/s without an API key")
            fetcher = HttpFetcher(
                EUTILS_BASE_URL,
                limiter=limiter or host_limiter(SOURCE, rate),
                cache=cache,
                client=client,
                timeout_s=timeout_s,
                max_retries=max_retries,
                max_wait_s=max_wait_s,
                max_backoff_s=max_backoff_s,
                default_params={"db": "pubmed", "tool": TOOL},
                sleep=sleep,
            )
        elif fetcher.base_url != EUTILS_BASE_URL:
            raise ValueError("PubMedSource fetcher must target the NCBI E-utilities host")
        self._fetchers = (fetcher,)
        self.include_labels = include_labels

    @property
    def _fetcher(self) -> HttpFetcher:
        return self._fetchers[0]

    @property
    def requests_sent(self) -> int:
        """Network attempts so far, retries included (cache hits are free)."""
        return sum(f.requests_sent for f in self._fetchers)

    def esearch(self, query: str, limit: int = 100) -> list[str]:
        """PMIDs for `query` (sanitised by pubmed_term) in Best Match order. An empty
        term sends nothing and returns []."""
        term = pubmed_term(query)
        if not term:
            return []
        params = esearch_params(term, limit)
        entry = self._fetcher.get(
            ESEARCH_PATH, params, parse="json",
            validate=lambda p: isinstance(p, dict) and isinstance(p.get("esearchresult"), dict),
        )
        return parse_esearch(entry["response"], params["retmax"])

    def efetch(self, pmids: Iterable[str]) -> dict[str, ExternalPaper]:
        """ONE EFetch GET for up to MAX_IDS PMIDs -> {pmid: ExternalPaper} for those PubMed
        returned (only requested PMIDs are kept). Invalid ids are dropped, duplicates
        collapsed; nothing to fetch sends nothing."""
        ids: list[str] = []
        for raw in pmids:
            p = normalize_pmid(raw)
            if p and p not in ids:
                ids.append(p)
        if not ids:
            return {}
        if len(ids) > MAX_IDS:
            raise ValueError(f"efetch takes at most {MAX_IDS} PMIDs per GET")
        params = {"id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        entry = self._fetcher.get(EFETCH_PATH, params, parse="text")
        records = parse_efetch(entry["response"], include_labels=self.include_labels)
        wanted = set(ids)
        return {p: rec for p, rec in records.items() if p in wanted}

    def search(self, query: str, limit: int = 100) -> list[ExternalPaper]:
        """ESearch (Best Match) then one EFetch: candidates in PubMed's rank order.

        `rank` is the 0-based Best Match position. A PMID that EFetch did not return is
        still listed (pmid + url only, empty text) — it exists in PubMed and the PMID
        alone can be resolved to an S2 corpus id downstream.
        """
        pmids = self.esearch(query, limit)
        if not pmids:
            return []
        records = self.efetch(pmids)
        out = []
        for rank, pmid in enumerate(pmids):
            rec = records.get(pmid) or ExternalPaper(
                source=SOURCE, rank=rank, pmid=pmid, url=PUBMED_URL.format(pmid=pmid)
            )
            rec.rank = rank
            out.append(rec)
        return out
