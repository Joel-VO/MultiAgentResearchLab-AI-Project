"""module_B_literature_kg/b2/b2a_review.py: B2a core. Review ONE paper from its title and abstract.

What goes in:   paper id, title, abstract, an LLMClient, a model nickname.
What comes out: a ReviewOutcome holding the interface.PaperReview plus numbers about how it went.

How one paper is reviewed
    1. clean the abstract (tags, "Abstract:" prefix, spaces, max length)
    2. ONE LLM call fills a light form: idea, method, pros (the results the authors claim), concepts, methods,
       datasets, metrics. Every claim must carry a quote copied from the text.
       cons and future_work are NOT asked for: abstracts almost never state them and models tend to guess them.
       They stay [] for abstract-only papers; only seeds get them (B2b, full text).
    3. plain code checks every quote against the title and abstract. A claim whose quote is not found is
       DROPPED (idea/method become "not stated", pros items disappear).
    4. datasets, metrics and methods must appear in the text, otherwise they are dropped
       (an invented dataset would create false "conflicts" later). Concepts are topic labels: no check, capped.
    5. if MORE than B2A_RETRY_FAIL_FRACTION of the claims failed the check, ask ONCE more, telling the model
       which quotes were not found. The better of the two answers is kept.
    6. an LLM that keeps failing gives an EMPTY review (status "failed"), never a crash.

evidence_quotes keys: "idea", "method", and "pros.0", "pros.1", ... (the index is the position of the claim in
the FINAL list).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..interface import PaperReview
from module_A_tooling.config import (B2A_PROMPT_VERSION, B2A_RETRY_FAIL_FRACTION, MAX_ABSTRACT_CHARS,
                                     MAX_CONCEPTS, REVIEW_MODEL_B2A)
from module_A_tooling.llm_client import LLMClient, LLMError
from .quote_check import name_supported, quote_supported

NOT_STATED = "not stated"


# --------------------------------------------------------------------------- the answer form

class Claim(BaseModel):
    text: str = ""
    quote: str = ""


class ReviewDraftB2a(BaseModel):
    """What the LLM returns. Missing parts default to "nothing stated"."""
    idea: Claim = Field(default_factory=Claim)
    method: Claim = Field(default_factory=Claim)
    pros: list[Claim] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = ("You extract structured information from the title and abstract of a scientific paper. "
                 "You use ONLY what the text says and never guess. Reply with one JSON object and nothing else.")

_PROMPT = """TITLE: {title}
ABSTRACT: {abstract}

Fill the JSON form below. Rules:
1. Use only information stated in the title or abstract above. If something is not stated, write "not stated" (for idea and method) or use an empty list.
2. Every claim needs a "quote": an exact copy of 5 to 40 consecutive words from the text above that supports the claim. Copy it character by character. No "...", no rewording, no joining of separate sentences.
3. idea: the main contribution, one sentence. method: how it is done, one or two sentences.
4. pros: results or strengths the authors claim (for example an improvement over a baseline). Only what the text states; if it states no result, use [].
5. concepts: up to {max_concepts} short topic labels. methods, datasets, metrics: names written exactly as they appear in the text, only if they appear in it.

JSON form:
{{"idea": {{"text": "...", "quote": "..."}}, "method": {{"text": "...", "quote": "..."}},
 "pros": [{{"text": "...", "quote": "..."}}],
 "concepts": ["..."], "methods": ["..."], "datasets": ["..."], "metrics": ["..."]}}"""


def build_prompt(title: str, abstract: str) -> str:
    return _PROMPT.format(title=title, abstract=abstract, max_concepts=MAX_CONCEPTS)


def clean_abstract(text: Optional[str], max_chars: int = MAX_ABSTRACT_CHARS) -> str:
    """Tags (<jats:p>), HTML entities, a leading "Abstract:", extra spaces; cut at max_chars on a word end."""
    t = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", " ", text or ""))
    t = re.sub(r"^\s*abstract\s*[:.\-\u2013\u2014]?\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0]
    return t


# --------------------------------------------------------------------------- checking one answer

def _dedupe(names: list[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        n = (n or "").strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def build_review(paper_id: str, title: str, abstract: str, draft: dict) -> tuple[PaperReview, dict, list[str]]:
    """Apply the quote check and the name check to an LLM answer.
    Returns (review, counts, failed_quotes). counts: claims_made, claims_kept, entities_made, entities_kept."""
    texts = [title, abstract]
    counts = {"claims_made": 0, "claims_kept": 0, "entities_made": 0, "entities_kept": 0}
    failed: list[str] = []
    quotes: dict[str, str] = {}

    def check(claim: dict) -> Optional[tuple[str, str]]:
        text = (claim.get("text") or "").strip()
        if not text or text.lower() == NOT_STATED:
            return None
        counts["claims_made"] += 1
        quote = (claim.get("quote") or "").strip()
        if quote_supported(quote, texts):
            counts["claims_kept"] += 1
            return text, quote
        failed.append(quote[:120] or "(no quote given)")
        return None

    single = {}
    for name in ("idea", "method"):
        got = check(draft.get(name) or {})
        single[name] = got[0] if got else NOT_STATED
        if got:
            quotes[name] = got[1]
    pros: list[str] = []
    for claim in draft.get("pros") or []:
        got = check(claim)
        if got:
            quotes[f"pros.{len(pros)}"] = got[1]
            pros.append(got[0])

    entities: dict[str, list[str]] = {}
    for name in ("methods", "datasets", "metrics"):
        kept_names = []
        for n in _dedupe(draft.get(name) or []):
            counts["entities_made"] += 1
            if name_supported(n, texts):
                counts["entities_kept"] += 1
                kept_names.append(n)
        entities[name] = kept_names
    concepts = _dedupe(draft.get("concepts") or [])[:MAX_CONCEPTS]

    review = PaperReview(paper_id=paper_id, title=title, idea=single["idea"], method=single["method"],
                         pros=pros, concepts=concepts, methods=entities["methods"], datasets=entities["datasets"],
                         metrics=entities["metrics"], evidence_quotes=quotes)
    return review, counts, failed


def _is_empty(r: PaperReview) -> bool:
    return (r.idea == NOT_STATED and r.method == NOT_STATED
            and not (r.pros or r.concepts or r.methods or r.datasets or r.metrics))


# --------------------------------------------------------------------------- one paper

@dataclass
class ReviewOutcome:
    review: PaperReview
    status: str                              # "ok" | "empty" (nothing supported) | "failed" (LLM unusable)
    claims_made: int = 0
    claims_kept: int = 0
    entities_made: int = 0
    entities_kept: int = 0
    retried: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    error: str = ""
    failed_quotes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """JSON-friendly record (for the sidecar file)."""
        d = {k: getattr(self, k) for k in ("status", "claims_made", "claims_kept", "entities_made",
                                           "entities_kept", "retried", "input_tokens", "output_tokens",
                                           "seconds", "cached", "error", "failed_quotes")}
        d["review"] = self.review.model_dump()
        return d


def review_paper(paper_id: str, title: str, abstract: str, client: LLMClient, model: str = REVIEW_MODEL_B2A,
                 *, version: str = B2A_PROMPT_VERSION) -> ReviewOutcome:
    """Review one paper. LLMConfigError (bad setup) is NOT caught: it must stop the run, not hide in a report."""
    title = " ".join((title or "").split())
    abstract = clean_abstract(abstract)
    prompt = build_prompt(title, abstract)

    def ask(p: str):
        return client.call("b2a_review", SYSTEM_PROMPT, p, ReviewDraftB2a, model=model, version=version)

    try:
        first = ask(prompt)
    except LLMError as exc:
        return ReviewOutcome(PaperReview(paper_id=paper_id, title=title), "failed", error=str(exc)[:300])
    review, counts, failed = build_review(paper_id, title, abstract, first.data)
    tokens_in, tokens_out, seconds, cached = first.input_tokens, first.output_tokens, first.seconds, first.cached
    retried = False

    made, kept = counts["claims_made"], counts["claims_kept"]
    if made and (made - kept) / made > B2A_RETRY_FAIL_FRACTION:
        feedback = ("\n\nIn a previous attempt these quotes were NOT found word for word in the text: "
                    + "; ".join(repr(q) for q in failed[:5])
                    + ". Copy each quote exactly from the text above, or leave the claim out.")
        try:
            second = ask(prompt + feedback)
        except LLMError:
            second = None
        if second is not None:
            retried = True
            tokens_in += second.input_tokens
            tokens_out += second.output_tokens
            seconds += second.seconds
            cached = cached and second.cached
            review2, counts2, failed2 = build_review(paper_id, title, abstract, second.data)
            if counts2["claims_kept"] > counts["claims_kept"]:
                review, counts, failed = review2, counts2, failed2

    return ReviewOutcome(review, "empty" if _is_empty(review) else "ok", retried=retried,
                         input_tokens=tokens_in, output_tokens=tokens_out, seconds=seconds, cached=cached,
                         failed_quotes=failed, **counts)
