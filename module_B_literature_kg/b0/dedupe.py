"""module_B_literature_kg/b0/dedupe.py: step 4 of B0. Merge duplicate papers SAFELY, give each paper its final id
(P1, P2, ...), and rewrite the citation edges with those ids.

The two mistakes to avoid
    too eager : two DIFFERENT papers with similar titles get merged for good
    too shy   : the arXiv preprint and the published version have different IDs and stay as two papers

How a pair of papers is judged (strongest evidence first)
    Level 1  exact identifier (same DOI, or same arXiv id, or same Semantic Scholar id)
             -> merge, confidence 1.0
    Level 2  identical normalised title AND same first-author surname AND years within
             DEDUPE_YEAR_TOLERANCE (both authors and both years must be known)
             -> merge, confidence 0.9
    Level 3  title similarity >= DEDUPE_TITLE_SIM but not Level 2
             -> only a "candidate": flagged in the log, NEVER merged
    Titles shorter than DEDUPE_MIN_TITLE_CHARS (after normalising) are too generic and never match
    on title.

Nothing is destroyed
    * Every decision is written to dedupe_log.json with evidence and confidence.
    * alias_map.json maps every original key to its final id, so the original records can always be found.
    * Undo a merge: put the pair under "undo" in runs/<id>/dedupe_overrides.json and re-run. The same
      file can promote a candidate with "merge". (This file is optional and written by hand.)

Outputs: canonical_papers.json, alias_map.json, dedupe_log.json, citation_edges.json
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

from module_A_tooling.config import (DEDUPE_MIN_TITLE_CHARS, DEDUPE_TITLE_SIM, DEDUPE_YEAR_TOLERANCE, RUN_FILES)
from .clients import normalize_title
from .manifest import (Manifest, MissingInputError, StepResult, atomic_write_json, read_json, run_step)
from .references import REFERENCES_FILE
from .seeds import SEEDS_FILE

CANONICAL_FILE = RUN_FILES["canonical"]
ALIAS_FILE = RUN_FILES["alias_map"]
LOG_FILE = RUN_FILES["dedupe_log"]
EDGES_FILE = RUN_FILES["citation_edges"]
OVERRIDES_FILE = RUN_FILES["dedupe_overrides"]
DEDUPE_VERSION = 1
ID_FIELDS = ("doi", "arxiv_id", "s2_id")
SCALAR_FIELDS = ("title", "abstract", "year", "venue", "doi", "arxiv_id", "s2_id", "open_access_pdf_url")


# --------------------------------------------------------------------------- helpers

def first_author_surname(authors: list[str]) -> str:
    """'Ada Lovelace' -> 'lovelace', 'Lovelace, Ada' -> 'lovelace' (accents removed). '' if unknown."""
    if not authors or not authors[0]:
        return ""
    name = unicodedata.normalize("NFKD", authors[0])
    name = "".join(ch for ch in name if not unicodedata.combining(ch)).lower()
    if "," in name:                                            # "Lovelace, Ada": the surname comes first
        name = name.split(",")[0]
    words = re.findall(r"[a-z']+", name)
    return words[-1] if words else ""


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def collect_records(seeds_doc: dict, refs_doc: dict) -> list[dict]:
    """Seeds first (in seed order), then reference stubs (in order of first appearance)."""
    records: list[dict] = []
    for rec in seeds_doc.get("seeds", []):
        paper = dict(rec["paper"])
        paper["is_seed"] = True
        paper["seed_flags"] = list(rec.get("flags", []))
        records.append(paper)
    for stub in refs_doc.get("stubs", {}).values():
        paper = dict(stub)
        paper["is_seed"] = False
        records.append(paper)
    keys = [r["key"] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("two records share the same key; seeds.json and references.json are inconsistent")
    return records


def _author_year_agree(a: dict, b: dict, year_tolerance: int) -> tuple[bool, str]:
    sa, sb = first_author_surname(a.get("authors", [])), first_author_surname(b.get("authors", []))
    if not sa or not sb:
        return False, "first author unknown on at least one record"
    if sa != sb:
        return False, f"first authors differ ({sa} vs {sb})"
    ya, yb = a.get("year"), b.get("year")
    if ya is None or yb is None:
        return False, "year unknown on at least one record"
    if abs(ya - yb) > year_tolerance:
        return False, f"years differ by more than {year_tolerance} ({ya} vs {yb})"
    return True, f"first author {sa}, years {ya}/{yb}"


# --------------------------------------------------------------------------- finding matches

def find_decisions(records: list[dict], *, undo: Iterable[tuple[str, str]] = (),
                   force_merge: Iterable[tuple[str, str]] = (), title_sim: float = DEDUPE_TITLE_SIM,
                   year_tolerance: int = DEDUPE_YEAR_TOLERANCE,
                   min_title_chars: int = DEDUPE_MIN_TITLE_CHARS) -> list[dict]:
    """Judge pairs of records. Returns decisions: {a, b, rule, evidence, confidence, status}
    status: "merged" (will be merged), "candidate" (flagged only), "undone" (a merge a person cancelled)."""
    undo_pairs = {_pair(a, b) for a, b in undo}
    by_pair: dict[tuple[str, str], dict] = {}
    title_of = {r["key"]: r.get("title", "") for r in records}

    def add(a: str, b: str, rule: str, evidence: str, confidence: float, status: str) -> None:
        pair = _pair(a, b)
        old = by_pair.get(pair)
        if old is not None:                                   # same pair found again: add the evidence
            if evidence not in old["evidence"]:
                old["evidence"] += "; " + evidence
            return
        by_pair[pair] = {"a": pair[0], "b": pair[1], "a_title": title_of[pair[0]][:90],
                         "b_title": title_of[pair[1]][:90], "rule": rule, "evidence": evidence,
                         "confidence": round(confidence, 4), "status": status}

    def merge_or_undo(a: str, b: str, rule: str, evidence: str, confidence: float) -> None:
        add(a, b, rule, evidence, confidence, "undone" if _pair(a, b) in undo_pairs else "merged")

    # Level 1: exact identifiers
    for field in ID_FIELDS:
        groups: dict[str, list[dict]] = {}
        for r in records:
            if r.get(field):
                groups.setdefault(r[field].lower(), []).append(r)
        for value, members in groups.items():
            for other in members[1:]:
                merge_or_undo(members[0]["key"], other["key"], "exact_id", f"same {field}: {value}", 1.0)

    norm = {r["key"]: normalize_title(r.get("title", "")) for r in records}
    usable = [r for r in records if len(norm[r["key"]]) >= min_title_chars]

    # Level 2: identical title + same first author + years close. Identical title with other
    # authors/years is only a candidate.
    by_title: dict[str, list[dict]] = {}
    for r in usable:
        by_title.setdefault(norm[r["key"]], []).append(r)
    for members in by_title.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if _pair(a["key"], b["key"]) in by_pair:
                    continue
                ok, why = _author_year_agree(a, b, year_tolerance)
                if ok:
                    merge_or_undo(a["key"], b["key"], "title_author_year", "identical title; " + why, 0.9)
                else:
                    add(a["key"], b["key"], "similar_title", f"identical title but {why}", 1.0, "candidate")

    # Level 3: nearly identical titles -> candidates only. Compare only papers whose titles start with
    # the same 3 words or end with the same 3 words (cheap blocking; a title changed at both ends is missed).
    blocks: dict[str, list[dict]] = {}
    for r in usable:
        words = norm[r["key"]].split()
        blocks.setdefault("f:" + " ".join(words[:3]), []).append(r)
        blocks.setdefault("l:" + " ".join(words[-3:]), []).append(r)
    for members in blocks.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if _pair(a["key"], b["key"]) in by_pair:
                    continue
                matcher = SequenceMatcher(None, norm[a["key"]], norm[b["key"]])
                if matcher.real_quick_ratio() < title_sim or matcher.quick_ratio() < title_sim:
                    continue
                ratio = matcher.ratio()
                if ratio >= title_sim:
                    add(a["key"], b["key"], "similar_title", f"title similarity {ratio:.3f}", ratio, "candidate")

    # manual promotion of a pair to "merged"
    known = {r["key"] for r in records}
    for a, b in force_merge:
        if a in known and b in known and a != b:
            pair = by_pair.get(_pair(a, b))
            if pair is None:
                add(a, b, "manual", "merged by hand in dedupe_overrides.json", 1.0, "merged")
            elif pair["status"] != "merged":
                pair.update(rule="manual", status="merged", confidence=1.0,
                            evidence=pair["evidence"] + "; merged by hand in dedupe_overrides.json")
    return list(by_pair.values())


# --------------------------------------------------------------------------- applying the decisions

def _merge_group(pid: str, members: list[tuple[int, dict]]) -> dict:
    """One final paper from the records that are the same paper. Fields come from the best record;
    blanks are filled from the others."""
    ordered = sorted(members, key=lambda t: (not t[1].get("is_seed"), not t[1].get("abstract"),
                                              not t[1].get("s2_id"), t[0]))
    primary = ordered[0][1]
    merged = {"paper_id": pid, "key": primary["key"],
              "aliases": [r["key"] for _, r in sorted(members, key=lambda t: t[0])]}
    for field in SCALAR_FIELDS:
        merged[field] = next((r[field] for _, r in ordered if r.get(field)),
                             "" if field in ("title", "abstract") else None)
    counts = [r["citation_count"] for _, r in ordered if r.get("citation_count") is not None]
    merged["citation_count"] = max(counts) if counts else None
    merged["authors"] = next((r["authors"] for _, r in ordered if r.get("authors")), [])
    merged["is_seed"] = any(r.get("is_seed") for _, r in members)
    merged["seed_flags"] = sorted({f for _, r in members for f in r.get("seed_flags", [])})
    merged["cited_by_seeds"] = []                              # filled in by dedupe() from the final edges
    conflicts = []
    for field in ID_FIELDS:
        values = sorted({r[field].lower() for _, r in members if r.get(field)})
        if len(values) > 1:
            conflicts.append(f"{field}: " + " vs ".join(values))
    merged["id_conflicts"] = conflicts
    return merged


def build_canonical(records: list[dict], decisions: list[dict]) -> tuple[dict, dict]:
    """Group records joined by "merged" decisions and number the groups P1, P2, ...
    Order is fixed: groups are numbered by their first record (seeds first, then references).
    Returns (canonical {pid: record}, alias_map {original key: pid})."""
    index = {r["key"]: i for i, r in enumerate(records)}
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d in decisions:
        if d["status"] == "merged":
            ra, rb = find(index[d["a"]]), find(index[d["b"]])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)              # the earliest record is the root

    groups: dict[int, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        groups.setdefault(find(i), []).append((i, r))
    canonical, alias = {}, {}
    for n, root in enumerate(sorted(groups), start=1):
        pid = f"P{n}"
        canonical[pid] = _merge_group(pid, groups[root])
        for _, r in groups[root]:
            alias[r["key"]] = pid
    return canonical, alias


def remap_edges(edges: list[dict], alias: dict) -> list[dict]:
    """Rewrite edges with final ids; drop self-loops and repeated edges."""
    seen: set[tuple[str, str]] = set()
    for e in edges:
        src, dst = alias[e["source"]], alias[e["target"]]
        if src != dst:
            seen.add((src, dst))
    order = lambda pid: int(pid[1:])
    return [{"source": s, "target": t, "kind": "cites"}
            for s, t in sorted(seen, key=lambda p: (order(p[0]), order(p[1])))]


def dedupe(seeds_doc: dict, refs_doc: dict, overrides: Optional[dict] = None, **kwargs) -> dict:
    overrides = overrides or {}
    records = collect_records(seeds_doc, refs_doc)
    decisions = find_decisions(records, undo=[tuple(p) for p in overrides.get("undo", [])],
                               force_merge=[tuple(p) for p in overrides.get("merge", [])], **kwargs)
    canonical, alias = build_canonical(records, decisions)
    edges = remap_edges(refs_doc.get("edges", []), alias)
    cited_by: dict[str, set[str]] = {pid: set() for pid in canonical}      # "which seeds cite this paper",
    for e in edges:                                                        # worked out from the final edges
        if canonical[e["source"]]["is_seed"]:
            cited_by[e["target"]].add(e["source"])
    for pid, paper in canonical.items():
        paper["cited_by_seeds"] = sorted(cited_by[pid], key=lambda x: int(x[1:]))
    counts = {s: sum(1 for d in decisions if d["status"] == s) for s in ("merged", "candidate", "undone")}
    warnings = []
    if counts["candidate"]:
        warnings.append(f"{counts['candidate']} possible duplicate(s) were flagged but NOT merged (see dedupe_log.json)")
    for p in canonical.values():
        if p["id_conflicts"]:
            warnings.append(f"{p['paper_id']} merged records disagree on identifiers: " + "; ".join(p["id_conflicts"]))
    return {"canonical": {"version": DEDUPE_VERSION, "papers": canonical},
            "alias_map": alias,
            "log": {"version": DEDUPE_VERSION, "n_records": len(records), "n_papers": len(canonical),
                    "counts": counts, "decisions": decisions},
            "edges": edges, "warnings": warnings}


# --------------------------------------------------------------------------- the step

def validate_dedupe_output(staging: Path, expected_keys: set[str], seed_keys: set[str]) -> list[str]:
    canonical = (read_json(staging / CANONICAL_FILE) or {}).get("papers", {})
    alias = read_json(staging / ALIAS_FILE) or {}
    edges = read_json(staging / EDGES_FILE)
    if not canonical or not alias or edges is None:
        return ["one of the dedupe output files is missing or empty"]
    problems = []
    if set(alias) != expected_keys:
        problems.append("alias map does not cover exactly the original records")
    expected_ids = {f"P{i}" for i in range(1, len(canonical) + 1)}
    if set(canonical) != expected_ids:
        problems.append("paper ids are not P1..Pn without gaps")
    if set(alias.values()) != set(canonical):
        problems.append("alias map points at ids that do not exist (or leaves some unused)")
    for key in seed_keys:
        if not canonical.get(alias.get(key), {}).get("is_seed"):
            problems.append(f"seed {key!r} lost its seed flag")
    seen = set()
    for e in edges:
        pair = (e["source"], e["target"])
        if e["source"] not in canonical or e["target"] not in canonical:
            problems.append(f"edge {pair} points at an unknown paper")
        if e["source"] == e["target"]:
            problems.append(f"self edge {pair}")
        if pair in seen:
            problems.append(f"repeated edge {pair}")
        seen.add(pair)
    return problems


def run_dedupe_step(manifest: Manifest) -> StepResult:
    """Step 4 under the manifest. Reads seeds.json and references.json (and dedupe_overrides.json if present)."""
    root = manifest.root
    seeds_doc, refs_doc = read_json(root / SEEDS_FILE), read_json(root / REFERENCES_FILE)
    if seeds_doc is None or refs_doc is None:
        raise MissingInputError(f"{SEEDS_FILE} and {REFERENCES_FILE} are needed: run steps 1 and 2 first")
    inputs = [SEEDS_FILE, REFERENCES_FILE] + ([OVERRIDES_FILE] if (root / OVERRIDES_FILE).exists() else [])
    overrides = read_json(root / OVERRIDES_FILE, {}) if len(inputs) == 3 else {}
    records = collect_records(seeds_doc, refs_doc)
    expected_keys = {r["key"] for r in records}
    seed_keys = {r["key"] for r in records if r["is_seed"]}

    def work(staging: Path) -> dict:
        out = dedupe(seeds_doc, refs_doc, overrides)
        atomic_write_json(staging / CANONICAL_FILE, out["canonical"])
        atomic_write_json(staging / ALIAS_FILE, out["alias_map"])
        atomic_write_json(staging / LOG_FILE, out["log"])
        atomic_write_json(staging / EDGES_FILE, out["edges"])
        return {"n_records": out["log"]["n_records"], "n_papers": out["log"]["n_papers"],
                **out["log"]["counts"], "warnings": out["warnings"]}

    return run_step(manifest, "b0_dedupe", work,
                    outputs=[CANONICAL_FILE, ALIAS_FILE, LOG_FILE, EDGES_FILE], inputs=inputs,
                    validate=lambda staging: validate_dedupe_output(staging, expected_keys, seed_keys))
