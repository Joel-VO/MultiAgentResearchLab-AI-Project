"""module_B_literature_kg/b2/quote_check.py: the "show me where the paper says that" test, shared by B2a and B2b.

Plain-language idea
    An LLM can write a claim that sounds right but is not in the paper. So every claim must come with a quote,
    and plain code (no AI) checks that the quote really is in the paper's text. A claim without a valid
    quote is thrown away. Names of datasets, metrics and methods get a simpler check: the name must appear
    in the text.

Both sides are cleaned with the SAME function (fulltext.normalize_for_matching: ligatures, curly quotes,
dashes, line-end hyphens, spacing, lower case), so harmless typography differences never cause a miss.
"""
from __future__ import annotations

import re
from typing import Sequence

from module_A_tooling.config import MAX_QUOTE_CHARS, MIN_QUOTE_WORDS
from module_B_literature_kg.b0.fulltext import normalize_for_matching


def quote_supported(quote: str, texts: Sequence[str], *, min_words: int = MIN_QUOTE_WORDS,
                    max_chars: int = MAX_QUOTE_CHARS) -> bool:
    """True if `quote` is a real stretch of one of `texts` (e.g. [title, abstract]).
    Refused: empty, too short to be evidence, too long, or containing "..." (a stitched-together quote)."""
    quote = quote or ""
    if "..." in quote or "\u2026" in quote or len(quote) > max_chars:
        return False
    q = normalize_for_matching(quote)
    if not q or len(q.split()) < min_words:
        return False
    return any(q in normalize_for_matching(t) for t in texts)


def _fold(text: str) -> str:
    """Normalised, and hyphens/slashes/underscores read as spaces ("Question-Answering" == "question answering")."""
    return re.sub(r"\s+", " ", re.sub(r"[-_/]", " ", normalize_for_matching(text))).strip()


def name_supported(name: str, texts: Sequence[str]) -> bool:
    """True if `name` appears in one of `texts` as a whole word or phrase (case, hyphens and spacing ignored)."""
    n = _fold(name)
    if len(n) < 2:
        return False
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])")
    return any(pattern.search(_fold(t)) for t in texts)
