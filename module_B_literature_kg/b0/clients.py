"""module_B_literature_kg/b0/clients.py: talking to Semantic Scholar and the arXiv API, safely.

Every call goes through three protections:
    1. the cache       : if we already asked this exact question, reuse the stored answer
    2. polite pacing   : a short pause between calls so we are not blocked
    3. retry           : 3 tries with growing waits (1 s, 2 s, ...), then give up with an error
Fresh (non-cached) replies are also saved as raw files for the run (runs/<id>/raw_replies/).

Every paper that leaves this file has the same plain shape (a dict, JSON-friendly):
    key, s2_id, arxiv_id, doi, title, abstract, year, venue, citation_count, authors, open_access_pdf_url
"key" is a temporary name used inside B0 (the final "P1", "P2" ids are assigned after de-duplication).

NOTE: the Semantic Scholar and arXiv response shapes below follow their public documentation as I
know it. They were tested here only against fake replies (no internet in the build sandbox), so run
one real call per method once and compare (see the "first live check" list in the answer).
"""
from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import requests

try:                                           # works whether cache.py sits in module_a/ or next to config.py
    from module_A_tooling.cache import Cache, get_default_cache, make_key
except ImportError:                            # pragma: no cover
    from module_A_tooling.cache import Cache, get_default_cache, make_key

from module_A_tooling.config import (ARXIV_API_URL, ARXIV_MIN_INTERVAL_SECONDS, REQUEST_TIMEOUT_SECONDS,
                    RETRY_BACKOFF_FACTOR, RETRY_BASE_SECONDS, RETRY_MAX_ATTEMPTS,
                    S2_API_KEY, S2_BASE_URL, S2_MIN_INTERVAL_SECONDS,
                    S2_NOT_FOUND_TTL_SECONDS, S2_REFERENCES_PAGE_SIZE, USER_AGENT)
from .manifest import atomic_write_text

S2_PAPER_FIELDS = "paperId,title,abstract,year,venue,citationCount,externalIds,openAccessPdf,authors"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 30.0
_MISSING = object()          # "not in the cache" (a cached None means "known not to exist")


# --------------------------------------------------------------------------- errors

class NotFound(Exception):
    """The server says this paper does not exist (HTTP 404)."""


class HttpError(Exception):
    """The server kept answering with an error after all retries."""

    def __init__(self, status: int, what: str):
        super().__init__(f"HTTP {status} for {what}")
        self.status = status


# --------------------------------------------------------------------------- id helpers

_ARXIV_NEW = re.compile(r"\d{4}\.\d{4,5}")
_ARXIV_OLD = re.compile(r"[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7}")
_DOI = re.compile(r"10\.\d{4,9}/\S+")
_S2_ID = re.compile(r"[0-9a-f]{40}")


def normalize_arxiv_id(text: Optional[str]) -> Optional[str]:
    """'arXiv:2305.12345v2', 'https://arxiv.org/abs/2305.12345' -> '2305.12345'. None if not an arXiv id."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = re.sub(r"v\d+$", "", s)
    if _ARXIV_NEW.fullmatch(s) or _ARXIV_OLD.fullmatch(s):
        return s
    return None


def normalize_doi(text: Optional[str]) -> Optional[str]:
    """'https://doi.org/10.1145/ABC' or 'doi:10.1145/abc' -> '10.1145/abc' (lower case). None if not a DOI."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)
    s = s.lower()
    return s if _DOI.fullmatch(s) else None


def looks_like_s2_id(text: Optional[str]) -> bool:
    return bool(text) and bool(_S2_ID.fullmatch(text.strip().lower()))


def normalize_title(title: str) -> str:
    """Lower case, letters and digits only, single spaces. Used for matching and for keys."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (title or "").lower())).strip()


def paper_key(paper: dict) -> str:
    """Temporary name of a paper inside B0: best identifier we have."""
    if paper.get("s2_id"):
        return f"s2:{paper['s2_id']}"
    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"
    if paper.get("doi"):
        return f"doi:{paper['doi']}"
    return "title:" + hashlib.sha1(normalize_title(paper.get("title", "")).encode()).hexdigest()[:12]


def new_paper(**fields: Any) -> dict:
    """A paper dict with every field present (missing ones are None / '' / [])."""
    paper = {"s2_id": None, "arxiv_id": None, "doi": None, "title": "", "abstract": "",
             "year": None, "venue": None, "citation_count": None, "authors": [],
             "open_access_pdf_url": None}
    paper.update(fields)
    paper["key"] = paper_key(paper)
    return paper


def normalize_s2(raw: dict) -> dict:
    """Semantic Scholar paper JSON -> our plain paper dict."""
    ext = raw.get("externalIds") or {}
    return new_paper(
        s2_id=raw.get("paperId"),
        arxiv_id=normalize_arxiv_id(ext.get("ArXiv")),
        doi=normalize_doi(ext.get("DOI")),
        title=(raw.get("title") or "").strip(),
        abstract=(raw.get("abstract") or "").strip(),
        year=raw.get("year"),
        venue=raw.get("venue") or None,
        citation_count=raw.get("citationCount"),
        authors=[a["name"] for a in (raw.get("authors") or []) if a.get("name")],
        open_access_pdf_url=(raw.get("openAccessPdf") or {}).get("url"),
    )


# --------------------------------------------------------------------------- retry, pacing, raw replies

def call_with_retry(send: Callable[[], Any], *, what: str = "request",
                    max_attempts: int = RETRY_MAX_ATTEMPTS, base: float = RETRY_BASE_SECONDS,
                    factor: float = RETRY_BACKOFF_FACTOR,
                    sleep: Callable[[float], None] = time.sleep) -> Any:
    """Call send() (which returns an HTTP response). Retries on network errors and on HTTP 429/5xx,
    waiting base, base*factor, ... seconds (or the server's Retry-After). 404 raises NotFound at once;
    any other 4xx raises HttpError at once (retrying cannot help)."""
    last: Exception = RuntimeError(f"{what}: no attempt was made")
    for attempt in range(1, max_attempts + 1):
        wait = base * factor ** (attempt - 1)
        try:
            resp = send()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        else:
            status = resp.status_code
            if status == 404:
                raise NotFound(what)
            if status < 400:
                return resp
            if status not in RETRYABLE_STATUS:
                raise HttpError(status, what)
            last = HttpError(status, what)
            retry_after = str(resp.headers.get("Retry-After", "")).strip()
            if retry_after.isdigit():
                wait = min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
        if attempt < max_attempts:
            sleep(wait)
    raise last


class _Throttle:
    """Make sure at least `interval` seconds pass between two calls."""

    def __init__(self, interval: float, sleep: Callable[[float], None], clock: Callable[[], float]):
        self.interval, self._sleep, self._clock = interval, sleep, clock
        self._last: Optional[float] = None

    def wait(self) -> None:
        if self._last is not None:
            gap = self.interval - (self._clock() - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()


class RawReplyStore:
    """Saves every fresh API reply under runs/<id>/raw_replies/ (for debugging and the record).
    directory=None switches it off."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory is not None else None

    def save(self, kind: str, key: str, text: str, ext: str = "json") -> None:
        if self.directory is not None:
            atomic_write_text(self.directory / f"{kind}_{key[:16]}.{ext}", text)


# --------------------------------------------------------------------------- Semantic Scholar

class S2Client:
    def __init__(self, cache: Optional[Cache] = None, session: Any = None,
                 raw: Optional[RawReplyStore] = None, api_key: Optional[str] = S2_API_KEY,
                 min_interval: float = S2_MIN_INTERVAL_SECONDS,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.cache = cache if cache is not None else get_default_cache()
        self.session = session if session is not None else requests.Session()
        self.raw = raw if raw is not None else RawReplyStore(None)
        self.api_key = api_key
        self._sleep = sleep
        self._throttle = _Throttle(min_interval, sleep, clock)

    # -- low level
    def _get_json(self, path: str, params: dict, kind: str, key: str) -> dict:
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        def send():
            self._throttle.wait()
            return self.session.get(S2_BASE_URL + path, params=params, headers=headers,
                                    timeout=REQUEST_TIMEOUT_SECONDS)

        resp = call_with_retry(send, what=f"Semantic Scholar {path}", sleep=self._sleep)
        self.raw.save(kind, key, resp.text if isinstance(getattr(resp, "text", None), str) else "")
        return resp.json()

    # -- one paper
    def get_paper(self, identifier: str) -> Optional[dict]:
        """identifier: 'ARXIV:2305.12345', 'DOI:10.x/y' or a Semantic Scholar paperId.
        Returns a plain paper dict, or None if Semantic Scholar does not know it."""
        key = make_key("s2.paper", {"id": identifier, "fields": S2_PAPER_FIELDS})
        cached = self.cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        try:
            raw = self._get_json(f"/paper/{quote(identifier, safe=':/')}",
                                 {"fields": S2_PAPER_FIELDS}, "s2paper", key)
        except NotFound:
            self.cache.set(key, None, kind="s2", ttl_seconds=S2_NOT_FOUND_TTL_SECONDS)
            return None
        paper = normalize_s2(raw) if raw.get("paperId") else None
        _store(self.cache, key, paper, "s2")
        return paper

    # -- title search
    def search_title(self, title: str, limit: int) -> list[dict]:
        key = make_key("s2.search", {"q": title, "limit": limit, "fields": S2_PAPER_FIELDS})
        cached = self.cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        try:
            payload = self._get_json("/paper/search",
                                     {"query": title, "limit": limit, "fields": S2_PAPER_FIELDS},
                                     "s2search", key)
        except NotFound:
            payload = {"data": []}
        papers = [normalize_s2(p) for p in (payload.get("data") or []) if p.get("paperId")]
        self.cache.set(key, papers, kind="s2")
        return papers

    # -- references (backward citations)
    def get_references(self, s2_id: str, cap: int) -> dict:
        """What this paper cites, at most `cap` papers.
        Returns {"references": [paper dicts], "partial": bool, "skipped_no_id": int}
        partial=True means the paper has more references than the cap (the list was cut)."""
        key = make_key("s2.references", {"id": s2_id, "fields": S2_PAPER_FIELDS, "cap": cap})
        cached = self.cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached

        refs: list[dict] = []
        skipped = 0
        offset = 0
        partial = False
        while True:
            payload = self._get_json(
                f"/paper/{quote(s2_id, safe=':')}/references",
                {"fields": S2_PAPER_FIELDS, "offset": offset, "limit": S2_REFERENCES_PAGE_SIZE},
                "s2refs", f"{key}-{offset}")
            for item in payload.get("data") or []:
                cited = item.get("citedPaper") or {}
                if not cited.get("paperId"):          # reference Semantic Scholar could not resolve
                    skipped += 1
                    continue
                refs.append(normalize_s2(cited))
            nxt = payload.get("next")
            if nxt is None:
                break
            offset = nxt
            if len(refs) >= cap:
                partial = True
                break
        if len(refs) > cap:
            refs = refs[:cap]
            partial = True
        result = {"references": refs, "partial": partial, "skipped_no_id": skipped}
        self.cache.set(key, result, kind="s2")
        return result


def _store(cache: Cache, key: str, paper: Optional[dict], kind: str) -> None:
    """Save a lookup result. A found paper uses the normal expiry of its kind; 'not found' (None) is
    remembered only briefly, because the paper may appear in the index later."""
    if paper is None:
        cache.set(key, None, kind=kind, ttl_seconds=S2_NOT_FOUND_TTL_SECONDS)
    else:
        cache.set(key, paper, kind=kind)


# --------------------------------------------------------------------------- arXiv API

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def _squash(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class ArxivClient:
    """Backup source, used only when Semantic Scholar has no abstract (or does not know the paper)."""

    def __init__(self, cache: Optional[Cache] = None, session: Any = None,
                 raw: Optional[RawReplyStore] = None,
                 min_interval: float = ARXIV_MIN_INTERVAL_SECONDS,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.cache = cache if cache is not None else get_default_cache()
        self.session = session if session is not None else requests.Session()
        self.raw = raw if raw is not None else RawReplyStore(None)
        self._sleep = sleep
        self._throttle = _Throttle(min_interval, sleep, clock)

    def get_metadata(self, arxiv_id: str) -> Optional[dict]:
        """Plain paper dict from the arXiv API (title, abstract, year, authors), or None."""
        aid = normalize_arxiv_id(arxiv_id)
        if aid is None:
            return None
        key = make_key("arxiv.meta", {"id": aid})
        cached = self.cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached

        def send():
            self._throttle.wait()
            return self.session.get(ARXIV_API_URL, params={"id_list": aid, "max_results": 1},
                                    headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)

        try:
            resp = call_with_retry(send, what=f"arXiv API {aid}", sleep=self._sleep)
        except NotFound:
            self.cache.set(key, None, kind="arxiv_api", ttl_seconds=S2_NOT_FOUND_TTL_SECONDS)
            return None
        self.raw.save("arxiv", key, resp.text, ext="xml")
        paper = parse_arxiv_feed(resp.text)
        _store(self.cache, key, paper, "arxiv_api")
        return paper


def parse_arxiv_feed(xml_text: str) -> Optional[dict]:
    """Atom XML from the arXiv API -> plain paper dict (None if the feed holds no real paper)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return None
    entry_id = _squash(entry.findtext(f"{_ATOM}id"))
    if "api/errors" in entry_id:                  # arXiv answers a bad id with an "Error" entry
        return None
    arxiv_id = normalize_arxiv_id(entry_id)
    if arxiv_id is None:
        return None
    published = _squash(entry.findtext(f"{_ATOM}published"))
    return new_paper(
        arxiv_id=arxiv_id,
        doi=normalize_doi(entry.findtext(f"{_ARXIV}doi")),
        title=_squash(entry.findtext(f"{_ATOM}title")),
        abstract=_squash(entry.findtext(f"{_ATOM}summary")),
        year=int(published[:4]) if published[:4].isdigit() else None,
        authors=[_squash(a.findtext(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author")
                 if _squash(a.findtext(f"{_ATOM}name"))],
        open_access_pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )
