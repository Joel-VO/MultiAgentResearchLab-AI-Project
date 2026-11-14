"""module_B_literature_kg/b2/b2b_review.py: B2b. Review each SEED paper from its FULL TEXT.

Same trick as B2a: the LLM must give a quote for every claim, and plain code checks the quote is really in the text.
The paper is cut into sections and three small calls each read only the sections they need:
    A: idea, method, concepts              (introduction, method sections)
    B: pros, methods, datasets, metrics    (experiment, result sections)
    C: cons, future_work                   (limitation, discussion, conclusion sections)
A claim whose quote is not found gets ONE correction round. If it still fails, it is dropped.

Outputs: reviews_b2b.json          {paper_id: PaperReview}
         reviews_b2b_records.json  {paper_id: status, flags, provenance (call + section + quote), dropped claims}
Run:     python -m module_B_literature_kg.b2.b2b_review
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..interface import PaperReview
from module_A_tooling.config import (B2B_MAX_INPUT_CHARS, B2B_MIN_ROUTED_CHARS, B2B_PROMPT_VERSION,
                                     MAX_CONCEPTS, REVIEW_MODEL_B2B, RUN_FILES)
from module_A_tooling.llm_client import LLMClient, LLMError
from module_B_literature_kg.b0.assemble import PAPERS_FILE, RECORDS_FILE
from module_B_literature_kg.b0.fulltext import FULLTEXT_DIR, _is_heading
from module_B_literature_kg.b0.manifest import Manifest, atomic_write_json, read_json, run_step
from .b2a_review import NOT_STATED, SYSTEM_PROMPT, Claim, _dedupe, clean_abstract, review_paper
from .quote_check import name_supported, quote_supported

REVIEWS_FILE = RUN_FILES["reviews_b2b"]
RECORDS_B2B_FILE = RUN_FILES["reviews_b2b_records"]
WEAK_OVERLAP = 0.3          # a claim sharing fewer words than this with its quote is only LISTED, never dropped


# --------------------------------------------------------------------------- the answer forms

class Draft(BaseModel):
    """One form for all three calls. Each call only reads the fields it owns."""
    idea: Claim = Field(default_factory=Claim)
    method: Claim = Field(default_factory=Claim)
    pros: list[Claim] = Field(default_factory=list)
    cons: list[Claim] = Field(default_factory=list)
    future_work: list[Claim] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class Fix(BaseModel):
    index: int
    quote: str = ""


class FixDraft(BaseModel):
    fixes: list[Fix] = Field(default_factory=list)


# fields each call owns: (claim fields with quotes, name fields)
FIELDS = {"A": (("idea", "method"), ("concepts",)),
          "B": (("pros",), ("methods", "datasets", "metrics")),
          "C": (("cons", "future_work"), ())}

ROUTES = {"A": ("introduction", "background", "overview", "approach", "method", "model", "architecture",
                "framework", "problem", "preliminar"),
          "B": ("experiment", "evaluation", "result", "setup", "implementation", "baseline", "dataset",
                "ablation", "analysis", "comparison", "conclusion"),
          "C": ("limitation", "discussion", "conclusion", "future work", "broader impact", "error analysis")}

RULES = """Fill the JSON form below. Rules:
1. Use only what the text above says. If something is not stated, write "not stated" (idea, method) or use an empty list.
2. Every claim needs a "quote": an exact copy of 5 to 40 consecutive words from the text above. Copy it character by character. No "...", no rewording.
"""

TASKS = {"A": (f"3. idea: the main contribution, one sentence. method: how it is done, one or two sentences. "
               f"concepts: up to {MAX_CONCEPTS} short topic labels.",
               '{"idea": {"text": "...", "quote": "..."}, "method": {"text": "...", "quote": "..."}, "concepts": ["..."]}'),
         "B": ("3. pros: results or strengths the authors report (for example an improvement over a baseline), at most 6. "
               "methods, datasets, metrics: names written exactly as in the text, only if they appear in it "
               "(the paper's own method and the baselines it compares with are both fine).",
               '{"pros": [{"text": "...", "quote": "..."}], "methods": ["..."], "datasets": ["..."], "metrics": ["..."]}'),
         "C": ("3. cons: limitations or weaknesses that the AUTHORS THEMSELVES state, at most 5. "
               "future_work: future directions that the AUTHORS THEMSELVES state, at most 5. "
               "Never add your own opinion. If the authors state none, return an empty list.",
               '{"cons": [{"text": "...", "quote": "..."}], "future_work": [{"text": "...", "quote": "..."}]}')}


# --------------------------------------------------------------------------- cutting and routing the text

def split_sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)]. Works for ar5iv text ('## 1 Introduction') and PDF text (plain heading lines)."""
    sections, title, lines = [], "front", []
    for line in text.split("\n"):
        if line.startswith("#") or _is_heading(line):
            sections.append((title, "\n".join(lines)))
            title, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    sections.append((title, "\n".join(lines)))
    return sections


def drop_tail(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Cut everything from References / Acknowledgements / Appendix on, but keep a section about limitations."""
    out, cutting = [], False
    for title, body in sections:
        name = re.sub(r"^[\d.]+\s*", "", title.lower())          # "6 References" -> "references"
        cutting = cutting or name.startswith(("references", "bibliography", "appendix", "acknowledg"))
        if not cutting or "limitation" in name:
            out.append((title, body))
    return out


def route(sections: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Give every section to the calls whose keywords its heading contains.
    A heading with no keyword (e.g. '3.1 Retriever') takes the route of the heading above it."""
    out, current = {k: [] for k in ROUTES}, []
    for title, body in sections:
        hit = [k for k, words in ROUTES.items() if any(w in title.lower() for w in words)]
        current = hit or current
        for k in current:
            out[k].append((title, body))
    return out


# --------------------------------------------------------------------------- small helpers

def get_claims(draft: dict, claim_fields: tuple) -> list[dict]:
    claims = []
    for field in claim_fields:
        items = draft.get(field) or []
        for c in (items if isinstance(items, list) else [items]):
            if (c.get("text") or "").strip().lower() not in ("", NOT_STATED):
                claims.append({"field": field, "text": c["text"].strip(), "quote": (c.get("quote") or "").strip()})
    return claims


def overlap(claim: str, quote: str) -> float:
    """Share of the claim's longer words that also appear in its quote (only a hint, see WEAK_OVERLAP)."""
    words = lambda s: set(re.findall(r"[a-z]{4,}", s.lower()))
    a = words(claim)
    return round(len(a & words(quote)) / len(a), 2) if a else 1.0


def correct(client: LLMClient, model: str, bad: list[dict], context: str) -> None:
    """ONE small call for the failed claims only: the model gives a better quote or says 'remove'."""
    lines = "\n".join(f'{i}. [{c["field"]}] {c["text"]} (your quote: "{c["quote"]}")' for i, c in enumerate(bad))
    prompt = (f"{context}\n\nThe quotes of these claims were NOT found word for word in the text above:\n{lines}\n\n"
              'For each claim copy an exact quote of 5 to 40 consecutive words from the text that supports it, '
              'or use "remove" as the quote. Reply with JSON only: {"fixes": [{"index": 0, "quote": "..."}]}')
    try:
        fixes = client.call("b2b_fix", SYSTEM_PROMPT, prompt, FixDraft, model=model, version=B2B_PROMPT_VERSION).data["fixes"]
    except LLMError:
        return
    for f in fixes:
        if 0 <= f["index"] < len(bad):
            bad[f["index"]]["quote"] = f["quote"]


def fallback(pid: str, title: str, abstract: str, client: LLMClient, why: str):
    """No full text, or every call failed: use the B2a abstract review."""
    return review_paper(pid, title, abstract, client).review, {"status": "fallback_abstract", "flags": [why]}


# --------------------------------------------------------------------------- one seed paper

def review_seed(pid: str, title: str, abstract: str, text: str, client: LLMClient,
                model: str = REVIEW_MODEL_B2B, cap: int = B2B_MAX_INPUT_CHARS):
    """-> (PaperReview, record dict)"""
    if not text:
        return fallback(pid, title, abstract, client, "no_full_text")
    abstract = clean_abstract(abstract)
    sections = drop_tail(split_sections(text))
    routed = route(sections)

    f = {"idea": NOT_STATED, "method": NOT_STATED, "pros": [], "cons": [], "future_work": [],
         "concepts": [], "methods": [], "datasets": [], "metrics": []}
    quotes, prov, weak, dropped, flags = {}, {}, [], [], []
    made = kept = failed_calls = 0

    for letter in "ABC":
        chosen = routed[letter]
        if sum(len(b) for _, b in chosen) < B2B_MIN_ROUTED_CHARS:     # headings not recognised: send everything
            chosen = sections
            flags.append(f"{letter}_routed_by_fallback")
        per = cap // len(chosen)                                       # every section gets an equal share, from its start
        shown = [(t, b[:per]) for t, b in chosen]
        if any(len(b) > per for _, b in chosen):
            flags.append(f"{letter}_truncated")
        body = "\n\n".join(f"## {t}\n{b}" for t, b in shown)
        context = f"ABSTRACT: {abstract}\n\nPAPER TEXT:\n{body}"
        texts = [title, abstract] + [b for _, b in shown]              # quotes are checked against what the model saw

        task, form = TASKS[letter]
        prompt = f"TITLE: {title}\n{context}\n\n{RULES}{task}\n\nJSON form:\n{form}"
        try:
            draft = client.call("b2b_review", SYSTEM_PROMPT, prompt, Draft, model=model, version=B2B_PROMPT_VERSION).data
        except LLMError:
            failed_calls += 1
            continue

        claim_fields, name_fields = FIELDS[letter]
        claims = get_claims(draft, claim_fields)
        bad = [c for c in claims if not quote_supported(c["quote"], texts)]
        good = [c for c in claims if c not in bad]
        if bad:                                                        # one correction round, failed claims only
            correct(client, model, bad, context)
            fixed = [c for c in bad if quote_supported(c["quote"], texts)]
            good += fixed
            bad = [c for c in bad if c not in fixed]
        dropped += [{"field": c["field"], "reason": "quote_not_found", "quote": c["quote"]} for c in bad]
        made += len(claims)
        kept += len(good)

        for c in good:
            if c["field"] in ("idea", "method"):
                key = c["field"]
                f[key] = c["text"]
            else:
                key = f"{c['field']}.{len(f[c['field']])}"
                f[c["field"]].append(c["text"])
            quotes[key] = c["quote"]
            section = next((t for t, b in shown if quote_supported(c["quote"], [b])), "abstract")
            ov = overlap(c["text"], c["quote"])
            prov[key] = {"call": letter, "section": section, "quote": c["quote"], "overlap": ov}
            if ov < WEAK_OVERLAP:
                weak.append(key)

        for field in name_fields:
            names = _dedupe(draft.get(field) or [])
            if field == "concepts":
                f[field] = names[:MAX_CONCEPTS]
            else:
                kept_names = [n for n in names if name_supported(n, texts)]
                dropped += [{"field": field, "reason": "name_not_found", "quote": n} for n in names if n not in kept_names]
                f[field] = kept_names

    if failed_calls == 3:
        return fallback(pid, title, abstract, client, "all_calls_failed")
    review = PaperReview(paper_id=pid, title=title, evidence_quotes=quotes, **f)
    record = {"status": "ok" if failed_calls == 0 else "partial", "flags": flags, "claims_made": made,
              "claims_kept": kept, "quote_pass_rate": round(kept / made, 2) if made else None,
              "provenance": prov, "weak_overlap": weak, "dropped": dropped}
    return review, record


# --------------------------------------------------------------------------- the step

def run_b2b_step(manifest: Manifest, client: LLMClient, model: str = REVIEW_MODEL_B2B):
    """Step b2b_review under the manifest. Runs after b0_assemble. Reviews the seed papers only."""
    root = manifest.root
    papers, records = read_json(root / PAPERS_FILE), read_json(root / RECORDS_FILE)

    def work(staging):
        reviews, info = {}, {}
        for pid, p in papers.items():
            if not p["is_seed"]:
                continue
            ft = records[pid].get("fulltext") or {}
            text = ""
            if ft.get("has_full_text"):
                text = (root / FULLTEXT_DIR / ft["text_file"]).read_text(encoding="utf-8")
            review, info[pid] = review_seed(pid, p["title"], p["abstract"], text, client, model)
            reviews[pid] = review.model_dump()
        atomic_write_json(staging / REVIEWS_FILE, reviews)
        atomic_write_json(staging / RECORDS_B2B_FILE, info)
        return {"n_seeds": len(reviews)}

    return run_step(manifest, "b2b_review", work, outputs=[REVIEWS_FILE, RECORDS_B2B_FILE],
                    inputs=[PAPERS_FILE, RECORDS_FILE, FULLTEXT_DIR],
                    extra_inputs={"model": model, "version": B2B_PROMPT_VERSION, "cap": B2B_MAX_INPUT_CHARS})


if __name__ == "__main__":                       # pragma: no cover
    run_b2b_step(Manifest(), LLMClient())
