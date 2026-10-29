"""module_B_literature_kg/b0/seeds.py: step 1 of B0. Find each seed paper and get its metadata and abstract.

What goes in:   the seed list from the request (arXiv IDs mainly; titles, DOIs, S2 ids also accepted).
What comes out: runs/<id>/seeds.json
    {"seeds": [...resolved, unique seeds...], "duplicates": [...], "unresolved": [...],
     "needs_confirmation": [...inputs to double-check...], "warnings": [...]}

Rules (all agreed in the B0 final spec)
    * arXiv ID      -> Semantic Scholar first; if it does not know the paper, ask the arXiv API.
    * Title         -> search Semantic Scholar, take the best match, ALWAYS log the match score.
                       score < SEED_TITLE_MIN                     -> unresolved
                       SEED_TITLE_MIN <= score < SEED_TITLE_CONFIRM -> accepted but flagged "confirm_title_match"
    * DOI / S2 id   -> accepted as an extension (same lookup). Such a seed has full text only if its
                       record contains an arXiv ID, otherwise it is flagged "no_arxiv_id".
    * Abstract      -> Semantic Scholar, else arXiv API, else flagged "no_abstract" (B0 then embeds the title only).
    * Two seeds that are the same paper (same S2 id / arXiv id / DOI) are merged into the first one.
    * Some seeds unresolved -> continue with a warning. ALL unresolved -> the step fails (run stops).
    * A network failure for one seed marks that seed unresolved ("lookup_error"); other seeds go on.
"""
from __future__ import annotations
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import requests

from module_A_tooling.config import RUN_FILES, S2_TITLE_SEARCH_LIMIT, SEED_TITLE_CONFIRM, SEED_TITLE_MIN
from .clients import (ArxivClient, HttpError, S2Client, looks_like_s2_id, normalize_arxiv_id,
                      normalize_doi, normalize_title, paper_key)
from .manifest import Manifest, StepResult, atomic_write_json, read_json, run_step

SEEDS_FILE = RUN_FILES["seeds"]
SEEDS_VERSION = 1


class NoUsableSeedsError(RuntimeError):
    """Not a single seed could be resolved, so the run cannot continue."""


# --------------------------------------------------------------------------- small helpers

def classify_seed(raw: str) -> tuple[str, str]:
    """Decide what kind of thing the user typed. Returns (kind, cleaned value).
    kind is 'arxiv', 'doi', 's2' (Semantic Scholar paper id) or 'title'."""
    text = (raw or "").strip()
    arxiv_id = normalize_arxiv_id(text)
    if arxiv_id:
        return "arxiv", arxiv_id
    doi = normalize_doi(text)
    if doi:
        return "doi", doi
    if looks_like_s2_id(text):
        return "s2", text.lower()
    return "title", text


def title_similarity(a: str, b: str) -> float:
    """0..1. Same words in the same order -> 1.0. Punctuation, case and spacing are ignored."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _identities(paper: dict) -> set[str]:
    ids = set()
    for field in ("s2_id", "arxiv_id", "doi"):
        if paper.get(field):
            ids.add(f"{field}:{paper[field]}")
    return ids


def _new_record(raw: str, kind: str) -> dict:
    return {"input": raw, "input_kind": kind, "status": "unresolved", "paper": None,
            "match": {"method": None, "confidence": None, "candidates": []},
            "abstract_source": None, "flags": [], "notes": []}


# --------------------------------------------------------------------------- resolving one seed

def resolve_seed(raw: str, s2: S2Client, arxiv: ArxivClient, *,
                 title_min: float = SEED_TITLE_MIN, title_confirm: float = SEED_TITLE_CONFIRM,
                 search_limit: int = S2_TITLE_SEARCH_LIMIT) -> dict:
    kind, value = classify_seed(raw)
    rec = _new_record(raw, kind)
    try:
        paper = _lookup(rec, kind, value, s2, arxiv, title_min, title_confirm, search_limit)
        if paper is None:
            return rec
        _finish_paper(rec, paper, arxiv, forced_arxiv_id=value if kind == "arxiv" else None)
    except (HttpError, requests.RequestException) as exc:
        rec["status"] = "unresolved"
        rec["paper"] = None
        rec["flags"].append("lookup_error")
        rec["notes"].append(f"network or server error after retries: {exc}")
    return rec


def _lookup(rec: dict, kind: str, value: str, s2: S2Client, arxiv: ArxivClient,
            title_min: float, title_confirm: float, search_limit: int) -> Optional[dict]:
    """Find the paper. Fills rec['match'] and rec['notes']. Returns the paper dict or None."""
    if kind == "title":
        candidates = s2.search_title(value, search_limit)
        scored = sorted(((title_similarity(value, c["title"]), c) for c in candidates),
                        key=lambda t: -t[0])
        rec["match"]["method"] = "s2_title_search"
        rec["match"]["candidates"] = [{"title": c["title"], "s2_id": c["s2_id"], "score": round(sc, 4)}
                                      for sc, c in scored[:3]]
        if not scored:
            rec["notes"].append("Semantic Scholar returned no results for this title")
            return None
        best_score, best = scored[0]
        rec["match"]["confidence"] = round(best_score, 4)
        if best_score < title_min:
            rec["notes"].append(f"best title match {best['title']!r} scored {best_score:.2f}, "
                                f"below the minimum {title_min:.2f}")
            return None
        if best_score < title_confirm:
            rec["flags"].append("confirm_title_match")
            rec["notes"].append(f"title match {best['title']!r} scored {best_score:.2f}: please confirm")
        return best

    if kind == "arxiv":
        paper = s2.get_paper(f"ARXIV:{value}")
        rec["match"]["method"] = "s2_arxiv_id"
        if paper is None:                              # Semantic Scholar does not know it: arXiv API backup
            paper = arxiv.get_metadata(value)
            rec["match"]["method"] = "arxiv_api"
            if paper is not None:
                rec["flags"].append("not_in_semantic_scholar")
                rec["notes"].append("Semantic Scholar has no record; metadata came from the arXiv API")
    elif kind == "doi":
        paper = s2.get_paper(f"DOI:{value}")
        rec["match"]["method"] = "s2_doi"
    else:
        paper = s2.get_paper(value)
        rec["match"]["method"] = "s2_paper_id"

    if paper is None:
        rec["notes"].append("paper not found")
    else:
        rec["match"]["confidence"] = 1.0               # an exact identifier is a certain match
    return paper


def _finish_paper(rec: dict, paper: dict, arxiv: ArxivClient, forced_arxiv_id: Optional[str]) -> None:
    """Fill the abstract, add flags, and mark the seed resolved."""
    if forced_arxiv_id and not paper.get("arxiv_id"):
        paper["arxiv_id"] = forced_arxiv_id            # we know it: the user gave it
        paper["key"] = paper_key(paper)

    if paper.get("abstract"):
        rec["abstract_source"] = "arxiv_api" if rec["match"]["method"] == "arxiv_api" else "s2"
    elif paper.get("arxiv_id"):
        backup = arxiv.get_metadata(paper["arxiv_id"])
        if backup and backup.get("abstract"):
            paper["abstract"] = backup["abstract"]
            rec["abstract_source"] = "arxiv_api"
    if not paper.get("abstract"):
        rec["flags"].append("no_abstract")
        rec["notes"].append("no abstract found; embedding will use the title only")
    if not paper.get("arxiv_id"):
        rec["flags"].append("no_arxiv_id")
        rec["notes"].append("no arXiv ID: no ar5iv or arXiv-PDF full text, abstract only")
    if not paper.get("s2_id"):
        rec["flags"].append("no_s2_id")
        rec["notes"].append("not in Semantic Scholar: references cannot be fetched")

    paper["is_seed"] = True
    rec["paper"] = paper
    rec["status"] = "resolved"


# --------------------------------------------------------------------------- resolving all seeds

def resolve_seeds(raw_seeds: list[str], s2: S2Client, arxiv: ArxivClient, **kwargs) -> dict:
    """Resolve every seed, merge duplicates, and summarise. Order of the input list is kept."""
    seeds, duplicates, unresolved = [], [], []
    seen: dict[str, str] = {}                          # identity -> key of the seed that owns it
    for raw in raw_seeds:
        rec = resolve_seed(raw, s2, arxiv, **kwargs)
        if rec["status"] != "resolved":
            unresolved.append(rec)
            continue
        paper = rec["paper"]
        owner = next((seen[i] for i in sorted(_identities(paper)) if i in seen), None)
        if owner is not None:
            duplicates.append({"input": raw, "duplicate_of": owner, "title": paper["title"]})
            continue
        for ident in _identities(paper):
            seen[ident] = paper["key"]
        seeds.append(rec)

    warnings = []
    if unresolved:
        warnings.append(f"{len(unresolved)} of {len(raw_seeds)} seeds could not be resolved: "
                        + "; ".join(repr(r["input"]) for r in unresolved))
    if duplicates:
        warnings.append(f"{len(duplicates)} seed(s) were duplicates of another seed and were merged")
    needs_confirmation = [r["input"] for r in seeds if "confirm_title_match" in r["flags"]]
    if needs_confirmation:
        warnings.append("please confirm these title matches: " + "; ".join(map(repr, needs_confirmation)))
    for r in seeds:
        for flag in ("no_abstract", "no_arxiv_id", "no_s2_id"):
            if flag in r["flags"]:
                warnings.append(f"seed {r['paper']['title'][:60]!r}: {flag}")
    return {"version": SEEDS_VERSION, "seeds": seeds, "duplicates": duplicates,
            "unresolved": unresolved, "needs_confirmation": needs_confirmation,
            "warnings": warnings}


# --------------------------------------------------------------------------- the step

def validate_seeds_output(staging: Path) -> list[str]:
    doc = read_json(staging / SEEDS_FILE)
    if not isinstance(doc, dict):
        return [f"{SEEDS_FILE} is missing or not valid JSON"]
    problems = []
    keys = set()
    for rec in doc.get("seeds", []):
        paper = rec.get("paper") or {}
        if not paper.get("title"):
            problems.append(f"seed {rec.get('input')!r} has no title")
        if paper.get("key") in keys:
            problems.append(f"seed key {paper.get('key')!r} appears twice")
        keys.add(paper.get("key"))
    if not doc.get("seeds"):
        problems.append("no resolved seeds")
    return problems


def run_seeds_step(manifest: Manifest, raw_seeds: list[str], s2: S2Client, arxiv: ArxivClient,
                   **kwargs) -> StepResult:
    """Run step 1 under the manifest. Raises NoUsableSeedsError (and marks the step failed) if no
    seed could be resolved."""
    def work(staging: Path) -> dict:
        result = resolve_seeds(raw_seeds, s2, arxiv, **kwargs)
        if not result["seeds"]:
            raise NoUsableSeedsError("none of the " f"{len(raw_seeds)} seeds could be resolved: "
                                     + "; ".join(result["warnings"]))
        atomic_write_json(staging / SEEDS_FILE, result)
        return {"n_seeds": len(result["seeds"]), "n_unresolved": len(result["unresolved"]),
                "n_duplicates": len(result["duplicates"]), "warnings": result["warnings"]}

    return run_step(manifest, "b0_seeds", work, outputs=[SEEDS_FILE],
                    extra_inputs={"seeds": list(raw_seeds)}, validate=validate_seeds_output)
