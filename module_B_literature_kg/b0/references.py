"""module_B_literature_kg/b0/references.py: step 2 of B0. For each seed, fetch the papers it cites.

What goes in:   runs/<id>/seeds.json            (from step 1)
What comes out: runs/<id>/references.json
    {"cap": 200,
     "by_seed": {seed_key: {"status": "ok" | "unavailable" | "error", "n_references": n,
                            "partial": bool, "skipped_no_id": n, "reason": "..."}},
     "stubs":   {paper_key: paper dict},        # cited papers that are not seeds (B1 will score them)
     "edges":   [{"source": seed_key, "target": cited_key, "kind": "cites"}],   # citing -> cited
     "warnings": [...]}

Rules
    * Backward references only (what the paper cites). Forward citations are out of scope.
    * At most MAX_REFERENCES_PER_PAPER per seed. If the paper has more, partial=True is recorded.
    * Edges are a SET of (source, target): a repeated or self edge is impossible, so re-running the
      step can never duplicate anything.
    * A seed that is not in Semantic Scholar has no reference list ("unavailable"). A network error for
      one seed is recorded ("error") and the others go on. If NO seed gives references, the step fails,
      because B1 would have nothing to expand.
"""
from __future__ import annotations

from pathlib import Path

import requests

from module_A_tooling.config import MAX_REFERENCES_PER_PAPER, RUN_FILES
from .clients import HttpError, S2Client
from .manifest import (Manifest, MissingInputError, StepResult, atomic_write_json, read_json,
                       run_step)
from .seeds import SEEDS_FILE

REFERENCES_FILE = RUN_FILES["references"]
REFERENCES_VERSION = 1


class ReferencesUnavailableError(RuntimeError):
    """No seed produced a reference list."""


def fetch_references(seeds_doc: dict, s2: S2Client, cap: int = MAX_REFERENCES_PER_PAPER) -> dict:
    seeds = [rec["paper"] for rec in seeds_doc.get("seeds", [])]
    seed_keys = {p["key"] for p in seeds}
    by_seed: dict[str, dict] = {}
    stubs: dict[str, dict] = {}
    cited_by: dict[str, list[str]] = {}
    edges: set[tuple[str, str]] = set()
    warnings: list[str] = []

    for seed in seeds:
        key = seed["key"]
        if not seed.get("s2_id"):
            by_seed[key] = {"status": "unavailable", "reason": "seed has no Semantic Scholar id",
                            "n_references": 0, "partial": False, "skipped_no_id": 0}
            warnings.append(f"seed {seed['title'][:60]!r}: references unavailable (no Semantic Scholar id)")
            continue
        try:
            got = s2.get_references(seed["s2_id"], cap)
        except (HttpError, requests.RequestException) as exc:
            by_seed[key] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                            "n_references": 0, "partial": False, "skipped_no_id": 0}
            warnings.append(f"seed {seed['title'][:60]!r}: reference fetch failed ({exc})")
            continue

        kept = 0
        for ref in got["references"]:
            if ref["key"] == key:                      # a paper citing itself: ignore
                continue
            edges.add((key, ref["key"]))
            kept += 1
            if ref["key"] in seed_keys:                # a seed cited by another seed: the seed record wins
                continue
            old = stubs.get(ref["key"])
            if old is None or (not old["abstract"] and ref["abstract"]):
                stubs[ref["key"]] = {**ref, "is_seed": False}
            cited_by.setdefault(ref["key"], [])
            if key not in cited_by[ref["key"]]:
                cited_by[ref["key"]].append(key)
        by_seed[key] = {"status": "ok", "n_references": kept, "partial": got["partial"],
                        "skipped_no_id": got["skipped_no_id"]}
        if got["partial"]:
            warnings.append(f"seed {seed['title'][:60]!r}: more than {cap} references, list was cut (partial)")

    for k, stub in stubs.items():
        stub["cited_by_seeds"] = cited_by.get(k, [])

    return {"version": REFERENCES_VERSION, "cap": cap, "by_seed": by_seed, "stubs": stubs,
            "edges": [{"source": s, "target": t, "kind": "cites"} for s, t in sorted(edges)],
            "warnings": warnings}


def validate_references_output(staging: Path, seed_keys: set[str]) -> list[str]:
    doc = read_json(staging / REFERENCES_FILE)
    if not isinstance(doc, dict):
        return [f"{REFERENCES_FILE} is missing or not valid JSON"]
    problems = []
    known = seed_keys | set(doc.get("stubs", {}))
    seen = set()
    for e in doc.get("edges", []):
        pair = (e["source"], e["target"])
        if pair in seen:
            problems.append(f"duplicate edge {pair}")
        seen.add(pair)
        if e["source"] not in seed_keys:
            problems.append(f"edge source {e['source']!r} is not a seed")
        if e["target"] not in known:
            problems.append(f"edge target {e['target']!r} is neither a seed nor a stub")
        if e["source"] == e["target"]:
            problems.append(f"self edge on {e['source']!r}")
    if not any(v["status"] == "ok" for v in doc.get("by_seed", {}).values()):
        problems.append("no seed produced a reference list")
    return problems


def run_references_step(manifest: Manifest, s2: S2Client, cap: int = MAX_REFERENCES_PER_PAPER) -> StepResult:
    """Run step 2 under the manifest. Reads seeds.json (step 1's output)."""
    seeds_path = manifest.root / SEEDS_FILE
    seeds_doc = read_json(seeds_path)
    if seeds_doc is None:
        raise MissingInputError(f"{SEEDS_FILE} not found: run step 1 (b0_seeds) first")
    seed_keys = {rec["paper"]["key"] for rec in seeds_doc.get("seeds", [])}

    def work(staging: Path) -> dict:
        doc = fetch_references(seeds_doc, s2, cap)
        if not any(v["status"] == "ok" for v in doc["by_seed"].values()):
            raise ReferencesUnavailableError("no seed produced a reference list: " + "; ".join(doc["warnings"]))
        atomic_write_json(staging / REFERENCES_FILE, doc)
        return {"n_stubs": len(doc["stubs"]), "n_edges": len(doc["edges"]), "warnings": doc["warnings"]}

    return run_step(manifest, "b0_references", work, outputs=[REFERENCES_FILE], inputs=[SEEDS_FILE],
                    extra_inputs={"cap": cap},
                    validate=lambda staging: validate_references_output(staging, seed_keys))
