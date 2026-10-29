"""module_B_literature_kg/b0/fulltext.py: step 3 of B0. Get the full text of each seed paper, and PROVE it is usable.

Why this file is careful
    Downloading a page and getting usable text are two different things. ar5iv (arXiv papers converted
    to HTML) and PDFs can both give text with lost sections, dropped tables, garbled characters, or
    even the wrong paper. So every source is checked and gets a report card:
        status   "ok" | "degraded" | "failed"
        warnings [{code, severity: critical|minor|info, detail}]
        source   "ar5iv" | "arxiv_pdf" | "s2_pdf"
        provenance (url, fetched_at, raw hash, extractor version)
    A short or empty text is NEVER success.

Source order for a seed that has an arXiv ID
    1. ar5iv HTML                -> if "ok", done.
    2. arXiv PDF                 -> then 3. the Semantic Scholar openAccessPdf link
    Every try is kept as a candidate (a degraded ar5iv text is NOT thrown away), and `pick_best` chooses:
        status (ok > degraded > failed), then fewer/milder warnings, then ar5iv before arXiv PDF before S2 PDF.
    If every candidate failed: has_full_text = False (abstract-only) and the attempt log says why.

Classification rule (same for HTML and PDF)
    any hard check fails                                   -> failed
    any critical warning, or >= MINOR_WARNINGS_FOR_DEGRADED minor warnings -> degraded
    otherwise                                              -> ok
    "info" notes are recorded but do not count (they apply to almost every PDF).

Outputs: runs/<id>/fulltext/ with, per seed, <name>.txt (the text) and <name>.json (report card),
         plus index.json (one line per seed).

NOTE: the ar5iv HTML class names (ltx_*) are written from my memory of LaTeXML's output. They were
tested only on synthetic pages. Check them on 2 or 3 real ar5iv pages before trusting the numbers.
PyMuPDF (`import fitz`) is the PDF reader; pypdf is used if PyMuPDF is not installed (flagged as a
minor warning, because it cannot keep two-column pages in order).
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from bs4 import BeautifulSoup, FeatureNotFound, NavigableString

try:
    from module_A_tooling.config import Cache, get_default_cache, make_key
except ImportError:                            # pragma: no cover
    from module_A_tooling.cache import Cache, get_default_cache, make_key

from module_A_tooling.config import (ARXIV_MIN_INTERVAL_SECONDS, ARXIV_PDF_BASE_URL, AR5IV_BASE_URL, EXTRACTOR_VERSION,
                    FAILED_FULLTEXT_RETRY_SECONDS, MAX_GARBAGE_RATIO, MAX_MINOR_ERROR_SPANS,
                    MAX_PDF_BYTES, MIN_CHARS_PER_PAGE, MIN_COVERAGE_RATIO, MIN_FULLTEXT_CHARS,
                    MIN_READABLE_WORD_RATIO, MIN_SECTIONS, MINOR_WARNINGS_FOR_DEGRADED,
                    PDF_CACHE_DIR, PDF_EXTRACTOR_VERSION, REQUEST_TIMEOUT_SECONDS, RUN_FILES,
                    TITLE_MATCH_MIN, USER_AGENT)
from .clients import HttpError, NotFound, _Throttle, call_with_retry, normalize_title
from .manifest import (Manifest, MissingInputError, StepResult, atomic_write_bytes, atomic_write_json,
                       atomic_write_text, read_json, run_step)
from .seeds import SEEDS_FILE

FULLTEXT_DIR = RUN_FILES["fulltext"]
INDEX_FILE = "index.json"
_MISSING = object()
SOURCE_RANK = {"ar5iv": 0, "arxiv_pdf": 1, "s2_pdf": 2}
STATUS_RANK = {"ok": 0, "degraded": 1, "failed": 2}


class PdfLibraryMissing(RuntimeError):
    """Neither PyMuPDF nor pypdf is installed."""


# --------------------------------------------------------------------------- thresholds

@dataclass(frozen=True)
class Thresholds:
    """All the numbers the checks use. Defaults come from config.py; tests can pass their own."""
    min_chars: int = MIN_FULLTEXT_CHARS
    title_min: float = TITLE_MATCH_MIN
    min_coverage: float = MIN_COVERAGE_RATIO
    min_sections: int = MIN_SECTIONS
    max_error_spans: int = MAX_MINOR_ERROR_SPANS
    minor_for_degraded: int = MINOR_WARNINGS_FOR_DEGRADED
    min_chars_per_page: int = MIN_CHARS_PER_PAGE
    max_garbage: float = MAX_GARBAGE_RATIO
    min_readable: float = MIN_READABLE_WORD_RATIO


def warn(code: str, severity: str, detail: str = "") -> dict:
    return {"code": code, "severity": severity, "detail": detail}


# --------------------------------------------------------------------------- shared text helpers

def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _nonspace_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def title_overlap(expected: str, candidate: str) -> float:
    """Share of the expected title's words found in `candidate` (0..1)."""
    want = set(normalize_title(expected).split())
    have = set(normalize_title(candidate).split())
    if not want:
        return 0.0
    return len(want & have) / len(want)


def normalize_for_matching(text: str) -> str:
    """Make text comparable: used by B2b's quote check on BOTH the paper text and the quote.
    Replaces ligatures, joins words split by a hyphen at a line end, unifies quotes and spaces."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", t)
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", t).strip().lower()


def classify(hard_failure: Optional[str], warnings: list[dict], th: Thresholds) -> str:
    if hard_failure:
        return "failed"
    critical = [w for w in warnings if w["severity"] == "critical"]
    minor = [w for w in warnings if w["severity"] == "minor"]
    if critical or len(minor) >= th.minor_for_degraded:
        return "degraded"
    return "ok"


def _penalty(candidate: dict) -> int:
    return sum(10 if w["severity"] == "critical" else 1 if w["severity"] == "minor" else 0
               for w in candidate["warnings"])


def pick_best(candidates: list[dict]) -> Optional[dict]:
    """status first, then fewer/milder warnings, then ar5iv before arXiv PDF before S2 PDF."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: (STATUS_RANK[c["status"]], _penalty(c), SOURCE_RANK[c["source"]]))


# --------------------------------------------------------------------------- ar5iv HTML extraction

_CHROME_SELECTORS = [".ltx_page_navbar", ".ltx_page_header", ".ltx_page_footer", ".ar5iv-footer",
                     ".ltx_pagination", ".ltx_authors", ".ltx_dates", ".ltx_role_affiliation"]
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_ERROR_PAGE_PHRASES = ("page not found", "not found", "conversion failed", "fatal error", "no such paper",
                       "404")
_INTRO_RE = re.compile(r"^(?:[A-Za-z]?\d*(?:\.\d+)*\.?\s*)?introduction\b", re.I)
_ABSTRACT_RE = re.compile(r"^abstract\b", re.I)
_REFERENCES_RE = re.compile(r"^(?:\d+\.?\s*)?(references|bibliography)\b", re.I)


def _make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:                          # pragma: no cover
        return BeautifulSoup(html, "html.parser")


def _text_of(tag) -> str:
    return _squash(tag.get_text(""))


def extract_html(html: str) -> dict:
    """ar5iv HTML -> {title, text, sections, stats}. Does not judge quality (see classify_html)."""
    soup = _make_soup(html)
    page_title = _squash(soup.title.get_text()) if soup.title else ""
    doc = (soup.select_one("article.ltx_document") or soup.select_one("div.ltx_document")
           or soup.select_one(".ltx_page_content"))
    stats = {"has_document": doc is not None, "page_title": page_title, "baseline_chars": 0,
             "kept_chars": 0, "n_math": 0, "n_math_kept": 0, "n_tables": 0, "n_tables_kept": 0,
             "n_error_spans": 0, "has_abstract": False, "has_intro": False, "has_references": False,
             "looks_like_error_page": False}
    if doc is None:
        stats["looks_like_error_page"] = True            # no converted paper on the page at all
        return {"title": "", "text": "", "sections": [], "stats": stats}

    for tag in doc.find_all(["script", "style", "noscript", "nav", "button", "svg"]):
        tag.decompose()
    for selector in _CHROME_SELECTORS:
        for tag in doc.select(selector):
            tag.decompose()
    stats["n_error_spans"] = len(doc.select(".ltx_ERROR"))

    title = ""
    title_tag = doc.select_one("h1.ltx_title_document") or doc.select_one(".ltx_title_document")
    if title_tag is not None:
        title = _text_of(title_tag)
        title_tag.decompose()

    # formulas: keep the LaTeX that LaTeXML stores in the alttext attribute
    for m in doc.find_all("math"):
        stats["n_math"] += 1
        alt = (m.get("alttext") or "").strip()
        if not alt:
            ann = m.find("annotation", attrs={"encoding": re.compile("tex", re.I)})
            alt = _squash(ann.get_text()) if ann is not None else ""
        if alt:
            stats["n_math_kept"] += 1
            m.replace_with(NavigableString(f" ${alt}$ "))
        else:
            m.decompose()

    stats["baseline_chars"] = _nonspace_len(doc.get_text(""))
    blocks: list[tuple[str, str, int]] = []            # (kind, text, heading level)
    _walk(doc, blocks, stats)

    lines = [f"# {title}"] if title else []
    sections = []
    for kind, text, level in blocks:
        if kind == "heading":
            lines.append("#" * max(level, 2) + " " + text)
            sections.append({"title": text, "level": level})
        else:
            lines.append(text)
        stats["kept_chars"] += _nonspace_len(text)
    text = "\n\n".join(lines)

    heads = [s["title"] for s in sections]
    stats["has_abstract"] = bool(doc.select_one(".ltx_abstract")) or any(_ABSTRACT_RE.match(h) for h in heads)
    stats["has_intro"] = any(_INTRO_RE.match(h) for h in heads)
    stats["has_references"] = bool(doc.select_one(".ltx_bibliography")) or any(_REFERENCES_RE.match(h) for h in heads)
    if stats["baseline_chars"] < 2000:
        low = (page_title + " " + _squash(doc.get_text(" "))).lower()
        stats["looks_like_error_page"] = any(p in low for p in _ERROR_PAGE_PHRASES)
    return {"title": title, "text": text, "sections": sections, "stats": stats}


def _walk(node, blocks: list, stats: dict) -> None:
    """Emit text only from known content blocks. Text sitting in a wrapper we do not understand is
    NOT emitted, which lowers the coverage ratio and so shows up as a warning instead of passing silently."""
    for child in node.children:
        if isinstance(child, NavigableString):
            continue
        name = child.name
        classes = set(child.get("class") or [])
        if name in _HEADINGS:
            text = _text_of(child)
            if text:
                level = 2 if "ltx_title_abstract" in classes else int(name[1])
                blocks.append(("heading", text, level))
        elif name == "p":
            text = _text_of(child)
            if text:
                blocks.append(("para", text, 0))
        elif name == "table":
            _emit_table(child, classes, blocks, stats)
        elif name in ("figcaption", "caption"):
            text = _text_of(child)
            if text:
                blocks.append(("caption", text, 0))
        elif name == "li" and "ltx_bibitem" in classes:
            text = _text_of(child)
            if text:
                blocks.append(("bibitem", text, 0))
        else:
            _walk(child, blocks, stats)


def _emit_table(table, classes: set, blocks: list, stats: dict) -> None:
    is_data = "ltx_tabular" in classes
    if is_data:
        stats["n_tables"] += 1
    rows = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:        # row of a nested table
            continue
        cells = [_text_of(td) for td in tr.find_all(["td", "th"]) if td.find_parent("tr") is tr]
        line = " | ".join(c for c in cells if c)
        if line:
            rows.append(line)
    text = "\n".join(rows) if is_data else " ".join(rows)   # equation layouts are one paragraph
    if text:
        blocks.append(("table" if is_data else "para", text, 0))
        if is_data:
            stats["n_tables_kept"] += 1


def classify_html(ex: dict, expected_title: str, th: Thresholds) -> tuple[str, Optional[str], list[dict]]:
    """-> (status, failure_reason, warnings)"""
    s = ex["stats"]
    if not s["has_document"] or s["looks_like_error_page"]:
        return "failed", "ar5iv_error_page", []
    if len(ex["text"]) < th.min_chars:
        return "failed", "too_short", []
    overlap = title_overlap(expected_title, ex["title"] or s["page_title"] or ex["text"][:300])
    if overlap < th.title_min:
        return "failed", "title_mismatch", [warn("title_mismatch", "critical", f"overlap {overlap:.2f}")]

    warnings: list[dict] = []
    coverage = s["kept_chars"] / s["baseline_chars"] if s["baseline_chars"] else 0.0
    if coverage < th.min_coverage:
        warnings.append(warn("low_coverage", "critical", f"extracted {coverage:.2f} of the page text"))
    if len(ex["sections"]) < th.min_sections:
        warnings.append(warn("few_sections", "critical", f"{len(ex['sections'])} headings found"))
    if not s["has_abstract"] and not s["has_intro"]:
        warnings.append(warn("no_abstract_or_intro", "critical"))
    elif not s["has_abstract"]:
        warnings.append(warn("no_abstract_heading", "minor"))
    elif not s["has_intro"]:
        warnings.append(warn("no_intro_heading", "minor"))
    if not s["has_references"]:
        warnings.append(warn("no_references_section", "minor"))
    if s["n_math"] > 0 and s["n_math_kept"] / s["n_math"] < 0.5:
        warnings.append(warn("equations_lost", "minor", f"{s['n_math_kept']} of {s['n_math']} kept"))
    if s["n_tables"] > 0 and s["n_tables_kept"] / s["n_tables"] < 0.5:
        warnings.append(warn("tables_lost", "minor", f"{s['n_tables_kept']} of {s['n_tables']} kept"))
    if s["n_error_spans"] > 0:
        severity = "critical" if s["n_error_spans"] > th.max_error_spans else "minor"
        warnings.append(warn("conversion_errors", severity, f"{s['n_error_spans']} error markers"))
    return classify(None, warnings, th), None, warnings


# --------------------------------------------------------------------------- PDF text extraction

def order_blocks(blocks: list, page_width: float) -> list:
    """Put PDF text blocks into reading order. blocks: (x0, y0, x1, y1, text, ...) tuples.
    One-column pages: top to bottom. Two-column pages: full-width blocks (title, wide figures) act as
    separators; between them the left column is read first, then the right column."""
    if not blocks:
        return []
    mid, wide = page_width / 2.0, 0.6 * page_width
    by_y = sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))
    narrow = [b for b in by_y if (b[2] - b[0]) <= wide]
    left = [b for b in narrow if b[2] <= mid + 0.05 * page_width]
    right = [b for b in narrow if b[0] >= mid - 0.05 * page_width]
    overlap = any(l[1] < r[3] and r[1] < l[3] for l in left for r in right)
    if not (len(left) >= 2 and len(right) >= 2 and overlap):
        return by_y                                    # looks like a one-column page
    ordered, col_l, col_r = [], [], []

    def flush():
        ordered.extend(sorted(col_l, key=lambda b: b[1]))
        ordered.extend(sorted(col_r, key=lambda b: b[1]))
        col_l.clear()
        col_r.clear()

    for b in by_y:
        if b in left:
            col_l.append(b)
        elif b in right:
            col_r.append(b)
        else:                                          # full width, or straddles the middle
            flush()
            ordered.append(b)
    flush()
    return ordered


def _read_with_pymupdf(pdf_bytes: bytes) -> list[str]:
    import fitz                                        # PyMuPDF
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            blocks = [b for b in page.get_text("blocks") if b[6] == 0]     # text blocks only
            ordered = order_blocks(blocks, page.rect.width)
            pages.append("\n\n".join(b[4].strip() for b in ordered if b[4].strip()))
    return pages


def _read_with_pypdf(pdf_bytes: bytes) -> list[str]:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(io.BytesIO(pdf_bytes)).pages]


def read_pdf_pages(pdf_bytes: bytes) -> tuple[list[str], str]:
    """Default PDF reader: (page texts, reader name). PyMuPDF if installed, else pypdf."""
    try:
        import fitz                                    # noqa: F401
    except ImportError:
        try:
            return _read_with_pypdf(pdf_bytes), "pypdf"
        except ImportError as exc:
            raise PdfLibraryMissing("install PyMuPDF (pip install pymupdf) to read PDFs") from exc
    return _read_with_pymupdf(pdf_bytes), "pymupdf"


_HEADING_WORDS = (r"abstract|introduction|related work|background|preliminaries|methods?|methodology|approach|"
                  r"model|experiments?|results?|evaluation|analysis|discussion|limitations|conclusions?"
                  r"(?: and future work)?|future work|acknowledg(?:e)?ments?|references|bibliography|"
                  r"appendix(?:\s+[A-Z])?")
_HEADING_RE = re.compile(rf"^(?:(?:[A-Z]|\d+)(?:\.\d+)*\.?\s+)?(?:{_HEADING_WORDS})\b[\s:.]*$", re.I)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z\-:, &]{2,70}$")
_PAGE_NUMBER_RE = re.compile(r"^\s*[-\u2013]?\s*\d{1,4}\s*[-\u2013]?\s*$")


def _is_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 80 and bool(_HEADING_RE.match(line) or _NUMBERED_HEADING_RE.match(line))


def _remove_headers_footers(pages: list[str]) -> tuple[list[str], int]:
    """Drop lines that repeat at the top/bottom of many pages (running titles, page numbers)."""
    n = len(pages)
    split = [p.split("\n") for p in pages]

    def edge_positions(lines):
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return set(idx[:2] + idx[-2:])

    def norm(line):
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter = Counter()
    for lines in split:
        counts.update({norm(lines[i]) for i in edge_positions(lines)})
    threshold = max(3, math.ceil(0.5 * n))
    repeated = {k for k, c in counts.items() if c >= threshold and k}
    removed, out = 0, []
    for lines in split:
        edges = edge_positions(lines)
        keep = []
        for i, line in enumerate(lines):
            if i in edges and line.strip() and (norm(line) in repeated or _PAGE_NUMBER_RE.match(line)):
                removed += 1
                continue
            keep.append(line)
        out.append("\n".join(keep))
    return out, removed


def _paragraphs(page: str, sections: list[dict]) -> list[str]:
    """Join the lines of each text block into paragraphs; fix hyphenated line breaks."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", page):
        current = ""
        for raw in block.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if _is_heading(line):
                if current:
                    out.append(current)
                    current = ""
                out.append(line)
                sections.append({"title": line, "level": 1})
                continue
            if current.endswith("-") and len(current) > 1 and current[-2].isalpha() and line[:1].islower():
                current = current[:-1] + line                  # "trans-" + "former" -> "transformer"
            else:
                current = f"{current} {line}".strip()
        if current:
            out.append(current)
    return out


_CID_RE = re.compile(r"\(cid:\d+\)")


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(len(m) for m in _CID_RE.findall(text))
    for ch in _CID_RE.sub("", text):
        o = ord(ch)
        if ch == "\ufffd" or 0xE000 <= o <= 0xF8FF or (o < 32 and ch not in "\n\t\r"):
            bad += 1
    return bad / len(text)


def readable_word_ratio(text: str) -> float:
    """Share of normal-looking words. Tokens made only of symbols are ignored (equations are full of them)."""
    word = re.compile(r"^(?:[^\W_]+(?:[-'\u2019/][^\W_]+)*|\d+(?:[.,]\d+)+)$")     # words, x1, 28.4, 1,000
    good = total = 0
    for tok in text.split():
        core = tok.strip(".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019*\u2020\u2021\u00a7%")
        if not any(ch.isalnum() for ch in core):
            continue
        total += 1
        if word.match(core):
            good += 1
    return good / total if total else 0.0


def extract_pdf_text(pages: list[str], reader: str) -> dict:
    """Page texts -> {text, sections, stats}. Cleans ligatures, running headers/footers, hyphenation."""
    pages = [unicodedata.normalize("NFKC", p) for p in pages]
    pages, removed = _remove_headers_footers(pages)
    sections: list[dict] = []
    first_page = ""
    paragraphs: list[str] = []
    for i, page in enumerate(pages):
        paras = _paragraphs(page, sections)
        if i == 0:
            first_page = " ".join(paras)
        paragraphs.extend(paras)
    text = "\n\n".join(paragraphs)
    heads = [s["title"] for s in sections]
    return {"text": text, "sections": sections,
            "stats": {"n_pages": len(pages), "chars": len(text), "reader": reader,
                      "removed_header_footer_lines": removed, "first_page": first_page[:3000],
                      "garbage_ratio": round(garbage_ratio(text), 5),
                      "readable_ratio": round(readable_word_ratio(text), 5),
                      "has_abstract": any(_ABSTRACT_RE.match(re.sub(r"^[\dA-Z.]+\s+", "", h)) for h in heads),
                      "has_intro": any(_INTRO_RE.match(h) for h in heads),
                      "has_references": any(_REFERENCES_RE.match(h) for h in heads)}}


def classify_pdf(ex: dict, expected_title: str, th: Thresholds) -> tuple[str, Optional[str], list[dict]]:
    s = ex["stats"]
    if len(ex["text"]) < th.min_chars:
        return "failed", "too_short", []
    if s["n_pages"] and len(ex["text"]) / s["n_pages"] < th.min_chars_per_page:
        return "failed", "low_text_per_page", []
    overlap = title_overlap(expected_title, s["first_page"])
    if overlap < th.title_min:
        return "failed", "title_mismatch", [warn("title_mismatch", "critical", f"overlap {overlap:.2f}")]

    warnings: list[dict] = []
    if s["garbage_ratio"] > th.max_garbage:
        warnings.append(warn("high_garbage", "critical", f"{s['garbage_ratio']:.3f} garbage characters"))
    if s["readable_ratio"] < th.min_readable:
        warnings.append(warn("low_readable_words", "critical", f"{s['readable_ratio']:.2f} readable words"))
    if len(ex["sections"]) < th.min_sections:
        warnings.append(warn("few_sections", "critical", f"{len(ex['sections'])} headings found"))
    if not s["has_references"]:
        warnings.append(warn("no_references_section", "minor"))
    if s["reader"] != "pymupdf":
        warnings.append(warn("pdf_reader_fallback", "minor", f"read with {s['reader']}: column order not kept"))
    if s["removed_header_footer_lines"]:
        warnings.append(warn("header_footer_removed", "info", f"{s['removed_header_footer_lines']} lines"))
    warnings.append(warn("equations_unreliable", "info", "PDF text cannot keep formulas intact"))
    return classify(None, warnings, th), None, warnings


# --------------------------------------------------------------------------- downloading

class FulltextFetcher:
    """Downloads ar5iv pages and PDFs with retry, pacing and caching.
    Raw HTML goes to the SQLite cache. PDFs are files in pdf_dir (too big for SQLite).
    A failed download is remembered for FAILED_FULLTEXT_RETRY_SECONDS, then retried."""

    def __init__(self, cache: Optional[Cache] = None, session: Any = None,
                 pdf_dir: Optional[Path] = None, min_interval: float = ARXIV_MIN_INTERVAL_SECONDS,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], float] = time.time,
                 pdf_reader: Callable[[bytes], tuple[list[str], str]] = read_pdf_pages,
                 max_pdf_bytes: int = MAX_PDF_BYTES):
        self.cache = cache if cache is not None else get_default_cache()
        self.session = session if session is not None else requests.Session()
        self.pdf_dir = Path(pdf_dir) if pdf_dir is not None else Path(PDF_CACHE_DIR)
        self._sleep, self._now = sleep, now
        self._throttle = _Throttle(min_interval, sleep, clock)
        self.pdf_reader = pdf_reader
        self.max_pdf_bytes = max_pdf_bytes

    def _send(self, url: str, stream: bool = False):
        def send():
            self._throttle.wait()
            return self.session.get(url, headers={"User-Agent": USER_AGENT},
                                    timeout=REQUEST_TIMEOUT_SECONDS, stream=stream)
        return call_with_retry(send, what=url, sleep=self._sleep)

    def _remember_failure(self, key: str, status: str) -> dict:
        result = {"status": status}
        self.cache.set(key, result, kind="fulltext", ttl_seconds=FAILED_FULLTEXT_RETRY_SECONDS)
        return result

    def fetch_html(self, url: str) -> dict:
        key = make_key("fulltext.html", {"url": url})
        cached = self.cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        try:
            resp = self._send(url)
        except NotFound:
            return self._remember_failure(key, "not_found")
        except (HttpError, requests.RequestException) as exc:       # not cached: try again next run
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result = {"status": "ok", "url": url, "html": resp.text, "fetched_at": self._now()}
        self.cache.set(key, result, kind="fulltext")                 # never expires
        return result

    def fetch_pdf(self, url: str) -> dict:
        key = make_key("fulltext.pdf.fail", {"url": url})
        cached = self.cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        stem = hashlib.sha256(url.encode()).hexdigest()
        pdf_path, meta_path = self.pdf_dir / f"{stem}.pdf", self.pdf_dir / f"{stem}.json"
        meta = read_json(meta_path)
        if meta and pdf_path.exists():
            return {"status": "ok", "url": url, "path": str(pdf_path), "sha256": meta["sha256"],
                    "fetched_at": meta["fetched_at"]}
        try:
            resp = self._send(url, stream=True)
        except NotFound:
            return self._remember_failure(key, "not_found")
        except (HttpError, requests.RequestException) as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        try:
            length = str(resp.headers.get("Content-Length", "")).strip()
            if length.isdigit() and int(length) > self.max_pdf_bytes:
                return self._remember_failure(key, "too_large")
            data = bytearray()
            for chunk in resp.iter_content(chunk_size=1 << 16):
                data += chunk
                if len(data) > self.max_pdf_bytes:
                    return self._remember_failure(key, "too_large")
        finally:
            close = getattr(resp, "close", None)
            if close:
                close()
        if not bytes(data[:5]) == b"%PDF-":
            return self._remember_failure(key, "not_a_pdf")      # e.g. a paywall page
        digest, fetched = hashlib.sha256(bytes(data)).hexdigest(), self._now()
        atomic_write_bytes(pdf_path, bytes(data))
        atomic_write_json(meta_path, {"url": url, "sha256": digest, "fetched_at": fetched})
        return {"status": "ok", "url": url, "path": str(pdf_path), "sha256": digest, "fetched_at": fetched}


# --------------------------------------------------------------------------- one source -> one candidate

def _candidate(source: str, url: str, status: str, failure_reason: Optional[str], ex: Optional[dict],
               warnings: list[dict], raw_hash: Optional[str], fetched_at: Optional[float],
               extractor_version: str) -> dict:
    return {"source": source, "url": url, "status": status, "failure_reason": failure_reason,
            "text": ex["text"] if ex and status != "failed" else "",
            "sections": ex["sections"] if ex and status != "failed" else [],
            "warnings": warnings,
            "stats": {k: v for k, v in (ex or {}).get("stats", {}).items() if k != "first_page"},
            "provenance": {"url": url, "fetched_at": fetched_at, "raw_hash": raw_hash,
                           "extractor_version": extractor_version}}


def _failed(source: str, url: str, reason: str, detail: str = "") -> dict:
    return _candidate(source, url, "failed", reason, None,
                      [warn(reason, "critical", detail)] if detail else [], None, None, "")


def try_ar5iv(arxiv_id: str, expected_title: str, fetcher: FulltextFetcher, th: Thresholds) -> dict:
    url = AR5IV_BASE_URL + arxiv_id
    got = fetcher.fetch_html(url)
    if got["status"] != "ok":
        return _failed("ar5iv", url, f"download_{got['status']}", got.get("error", ""))
    html = got["html"]
    raw_hash = hashlib.sha256(html.encode("utf-8", "replace")).hexdigest()
    ex = fetcher.cache.get_or_compute(                       # keyed by page hash + extractor version
        make_key("fulltext.extract.html", {"raw": raw_hash}, revision=EXTRACTOR_VERSION),
        lambda: extract_html(html), kind="fulltext")
    status, reason, warnings = classify_html(ex, expected_title, th)
    return _candidate("ar5iv", url, status, reason, ex, warnings, raw_hash, got["fetched_at"], EXTRACTOR_VERSION)


def try_pdf(source: str, url: str, expected_title: str, fetcher: FulltextFetcher, th: Thresholds) -> dict:
    got = fetcher.fetch_pdf(url)
    if got["status"] != "ok":
        return _failed(source, url, f"download_{got['status']}", got.get("error", ""))
    key = make_key("fulltext.extract.pdf", {"raw": got["sha256"]}, revision=PDF_EXTRACTOR_VERSION)
    cached = fetcher.cache.get(key, _MISSING)
    if cached is _MISSING:
        try:
            pages, reader = fetcher.pdf_reader(Path(got["path"]).read_bytes())
        except PdfLibraryMissing:
            raise
        except Exception as exc:                             # corrupt, encrypted, unreadable
            return _failed(source, url, "pdf_unreadable", f"{type(exc).__name__}: {exc}")
        cached = extract_pdf_text(pages, reader)
        fetcher.cache.set(key, cached, kind="fulltext")
    status, reason, warnings = classify_pdf(cached, expected_title, th)
    return _candidate(source, url, status, reason, cached, warnings, got["sha256"], got["fetched_at"],
                      PDF_EXTRACTOR_VERSION)


def get_seed_fulltext(paper: dict, fetcher: FulltextFetcher, th: Optional[Thresholds] = None) -> dict:
    """Full text for one seed. Returns
        {"has_full_text": bool, "best": candidate or None, "attempts": [...attempt log...]}"""
    th = th or Thresholds()
    arxiv_id, title = paper.get("arxiv_id"), paper.get("title", "")
    if not arxiv_id:
        return {"has_full_text": False, "best": None,
                "attempts": [{"source": None, "status": "skipped", "reason": "no_arxiv_id"}]}

    candidates: list[dict] = []
    first = try_ar5iv(arxiv_id, title, fetcher, th)
    candidates.append(first)
    if first["status"] != "ok":
        urls = [("arxiv_pdf", f"{ARXIV_PDF_BASE_URL}{arxiv_id}")]
        s2_url = paper.get("open_access_pdf_url")
        if s2_url and s2_url != urls[0][1]:
            urls.append(("s2_pdf", s2_url))
        for source, url in urls:
            cand = try_pdf(source, url, title, fetcher, th)
            candidates.append(cand)
            if cand["status"] == "ok":
                break

    attempts = [{"source": c["source"], "url": c["url"], "status": c["status"],
                 "reason": c["failure_reason"], "warnings": [w["code"] for w in c["warnings"]]}
                for c in candidates]
    best = pick_best(candidates)
    if best is None or best["status"] == "failed":
        return {"has_full_text": False, "best": None, "attempts": attempts}
    return {"has_full_text": True, "best": best, "attempts": attempts}


# --------------------------------------------------------------------------- the step

def _file_stem(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)


def validate_fulltext_output(staging: Path, expected_keys: set[str]) -> list[str]:
    folder = staging / FULLTEXT_DIR
    index = read_json(folder / INDEX_FILE)
    if not isinstance(index, dict):
        return [f"{FULLTEXT_DIR}/{INDEX_FILE} is missing or not valid JSON"]
    problems = []
    for key in expected_keys:
        entry = index.get(key)
        if entry is None:
            problems.append(f"no entry for seed {key!r}")
            continue
        if entry["has_full_text"]:
            txt = folder / f"{_file_stem(key)}.txt"
            if not txt.exists() or len(txt.read_text(encoding="utf-8")) < 1 or entry["status"] == "failed":
                problems.append(f"seed {key!r} claims full text but has no valid text file")
        elif (folder / f"{_file_stem(key)}.txt").exists():
            problems.append(f"seed {key!r} has a text file but has_full_text is False")
    return problems


def run_fulltext_step(manifest: Manifest, fetcher: FulltextFetcher,
                      th: Optional[Thresholds] = None) -> StepResult:
    """Step 3 under the manifest. Reads seeds.json; writes fulltext/ (one .txt + .json per seed)."""
    seeds_doc = read_json(manifest.root / SEEDS_FILE)
    if seeds_doc is None:
        raise MissingInputError(f"{SEEDS_FILE} not found: run step 1 (b0_seeds) first")
    th = th or Thresholds()
    papers = [rec["paper"] for rec in seeds_doc.get("seeds", [])]

    def work(staging: Path) -> dict:
        folder = staging / FULLTEXT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        index: dict[str, dict] = {}
        counts = Counter()
        for paper in papers:
            res = get_seed_fulltext(paper, fetcher, th)
            best, stem = res["best"], _file_stem(paper["key"])
            report = {"key": paper["key"], "title": paper["title"], "has_full_text": res["has_full_text"],
                      "attempts": res["attempts"]}
            if best is not None:
                atomic_write_text(folder / f"{stem}.txt", best["text"])
                report.update({k: best[k] for k in ("status", "source", "warnings", "sections", "stats",
                                                    "provenance")})
                counts[best["status"]] += 1
            else:
                report.update({"status": "failed" if paper.get("arxiv_id") else "skipped", "source": None,
                               "warnings": [], "sections": [], "stats": {}, "provenance": None})
                counts["unavailable"] += 1
            atomic_write_json(folder / f"{stem}.json", report)
            index[paper["key"]] = {"has_full_text": res["has_full_text"], "status": report["status"],
                                   "source": report["source"], "text_file": f"{stem}.txt" if best else None,
                                   "n_warnings": len([w for w in report["warnings"] if w["severity"] != "info"])}
        atomic_write_json(folder / INDEX_FILE, index)
        return {"ok": counts["ok"], "degraded": counts["degraded"], "unavailable": counts["unavailable"]}

    return run_step(manifest, "b0_fulltext", work, outputs=[FULLTEXT_DIR], inputs=[SEEDS_FILE],
                    extra_inputs={"th": th.__dict__},
                    validate=lambda staging: validate_fulltext_output(staging, {p["key"] for p in papers}))
