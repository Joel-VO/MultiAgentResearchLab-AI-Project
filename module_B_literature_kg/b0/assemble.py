"""module_B_literature_kg/b0/assemble.py: step 6 (the last step) of B0. Put the pieces together, CHECK everything, and
only then declare B0 finished.

What it builds
    papers.json          {paper_id: PaperMeta}       exactly the shape interface.py expects
    paper_records.json   {paper_id: {...}}           the sidecar: where each fact came from (provenance)
    b0_report.json       counts, warnings, and the result of every quality gate
Files written by the earlier steps (ids.json, embeddings.npy, topic.npy, citation_edges.json) are
checked here but not rewritten.

Quality gates (ALL must pass, otherwise the step fails and "done" is NOT written)
    steps_intact     steps 1-5 are "done" and their output files still match the recorded hashes
    seed_resolved    at least one seed paper was resolved
    paper_ids        ids are P1..Pn without gaps; every paper has a title
    alias_map        every original record maps to a final id
    edges_valid      every citation edge joins two known papers; no self-loops, no repeats
    vectors_valid    rows match ids, no NaN, every vector has length 1, topic vector has the same size,
                     every seed is embedded
    embedding_status every paper is accounted for (embedded or skipped), and it agrees with ids.json
    fulltext_files   every paper marked has_full_text has a non-empty text file with status ok/degraded
    paper_meta       every papers.json entry has the PaperMeta fields with the right types
If a gate fails, the diagnostics stay in runs/<id>/.staging/b0_assemble/ (including b0_report.json).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from module_A_tooling.config import RUN_FILES, VECTOR_NORM_TOLERANCE
from .dedupe import ALIAS_FILE, CANONICAL_FILE, EDGES_FILE, LOG_FILE
from .embed import EMBEDDINGS_FILE, IDS_FILE, STATUS_FILE, TOPIC_FILE
from .fulltext import FULLTEXT_DIR, INDEX_FILE, _file_stem
from .manifest import Manifest, MissingInputError, StepResult, atomic_write_json, read_json, run_step
from .references import REFERENCES_FILE
from .seeds import SEEDS_FILE

PAPERS_FILE = RUN_FILES["papers"]
RECORDS_FILE = RUN_FILES["paper_records"]
REPORT_FILE = RUN_FILES["b0_report"]
UPSTREAM_STEPS = ["b0_seeds", "b0_references", "b0_fulltext", "b0_dedupe", "b0_embed"]
REPORT_VERSION = 1

# PaperMeta (interface.py) field -> accepted Python types. Kept here so the check works even where
# pydantic is not installed; if interface.py can be imported, PaperMeta itself is used as well.
PAPER_META_TYPES: dict[str, tuple] = {
    "paper_id": (str,), "title": (str,), "abstract": (str,), "year": (int, type(None)),
    "venue": (str, type(None)), "citation_count": (int, type(None)), "s2_id": (str, type(None)),
    "arxiv_id": (str, type(None)), "doi": (str, type(None)), "is_seed": (bool,), "has_full_text": (bool,)}


class QualityGateError(RuntimeError):
    """One or more quality gates failed."""


def _num(pid: str) -> int:
    return int(pid[1:])


# --------------------------------------------------------------------------- loading what steps 1-5 wrote

def load_run(root: Path) -> dict:
    """Read every file the gates and the builders need. Raises MissingInputError if one is absent."""
    need = [SEEDS_FILE, REFERENCES_FILE, CANONICAL_FILE, ALIAS_FILE, LOG_FILE, EDGES_FILE, IDS_FILE,
            EMBEDDINGS_FILE, TOPIC_FILE, STATUS_FILE, f"{FULLTEXT_DIR}/{INDEX_FILE}"]
    missing = [n for n in need if not (root / n).exists()]
    if missing:
        raise MissingInputError("B0 is incomplete, these files are missing: " + ", ".join(missing))
    return {"seeds": read_json(root / SEEDS_FILE), "references": read_json(root / REFERENCES_FILE),
            "canonical": read_json(root / CANONICAL_FILE)["papers"], "alias": read_json(root / ALIAS_FILE),
            "log": read_json(root / LOG_FILE), "edges": read_json(root / EDGES_FILE),
            "ids": read_json(root / IDS_FILE), "matrix": np.load(root / EMBEDDINGS_FILE),
            "topic": np.load(root / TOPIC_FILE), "emb_status": read_json(root / STATUS_FILE),
            "ft_index": read_json(root / FULLTEXT_DIR / INDEX_FILE)}


# --------------------------------------------------------------------------- building the outputs

def _fulltext_by_pid(run: dict) -> dict[str, dict]:
    """Full-text index entries re-keyed from original keys to final paper ids."""
    out: dict[str, dict] = {}
    for key, entry in run["ft_index"].items():
        pid = run["alias"].get(key)
        if pid is not None and (pid not in out or entry["has_full_text"]):
            out[pid] = {**entry, "raw_key": key}
    return out


def build_papers(run: dict) -> dict[str, dict]:
    """papers.json: one PaperMeta-shaped dict per final paper id."""
    ft = _fulltext_by_pid(run)
    papers = {}
    for pid in sorted(run["canonical"], key=_num):
        c = run["canonical"][pid]
        papers[pid] = {"paper_id": pid, "title": c["title"], "abstract": c["abstract"] or "",
                       "year": c["year"], "venue": c["venue"], "citation_count": c["citation_count"],
                       "s2_id": c["s2_id"], "arxiv_id": c["arxiv_id"], "doi": c["doi"],
                       "is_seed": bool(c["is_seed"]),
                       "has_full_text": bool(ft.get(pid, {}).get("has_full_text", False))}
    return papers


def build_records(root: Path, run: dict) -> dict[str, dict]:
    """paper_records.json: provenance for every paper (where each fact came from and how sure we are)."""
    alias, ft = run["alias"], _fulltext_by_pid(run)
    seed_rec = {rec["paper"]["key"]: rec for rec in run["seeds"]["seeds"]}
    decisions = run["log"]["decisions"]
    records = {}
    for pid in sorted(run["canonical"], key=_num):
        c = run["canonical"][pid]
        aliases = set(c["aliases"])
        rec: dict[str, Any] = {"aliases": c["aliases"], "is_seed": c["is_seed"], "seed_flags": c["seed_flags"],
                               "id_conflicts": c["id_conflicts"], "cited_by_seeds": c["cited_by_seeds"],
                               "merge_evidence": [], "possible_duplicates": []}
        for d in decisions:
            if d["a"] in aliases or d["b"] in aliases:
                other = d["b"] if d["a"] in aliases else d["a"]
                item = {"other_key": other, "other_id": alias.get(other), "rule": d["rule"],
                        "evidence": d["evidence"], "confidence": d["confidence"]}
                if d["status"] == "merged" and alias.get(other) == pid:
                    rec["merge_evidence"].append(item)
                elif d["status"] == "candidate":
                    rec["possible_duplicates"].append(item)
        for key in c["aliases"]:
            if key in seed_rec:                                 # seed-only facts
                s = seed_rec[key]
                rec["seed_resolution"] = {"input": s["input"], "input_kind": s["input_kind"],
                                          "match": s["match"], "abstract_source": s["abstract_source"],
                                          "flags": s["flags"], "notes": s["notes"]}
                rec["references"] = run["references"]["by_seed"].get(key)
        entry = ft.get(pid)
        if entry is not None:
            report = read_json(root / FULLTEXT_DIR / f"{_file_stem(entry['raw_key'])}.json", {})
            rec["fulltext"] = {"has_full_text": entry["has_full_text"], "status": entry["status"],
                               "source": entry["source"], "text_file": entry["text_file"],
                               "warnings": report.get("warnings", []), "attempts": report.get("attempts", []),
                               "provenance": report.get("provenance")}
        rec["embedding"] = run["emb_status"].get(pid)
        records[pid] = rec
    return records


# --------------------------------------------------------------------------- the quality gates

def _gate_steps_intact(manifest: Manifest) -> list[str]:
    return [f"{step}: {why}" for step in UPSTREAM_STEPS if (why := manifest.output_problem(step))]


def _gate_seed_resolved(run: dict) -> list[str]:
    return [] if run["seeds"]["seeds"] else ["no seed was resolved"]


def _gate_paper_ids(run: dict) -> list[str]:
    papers, problems = run["canonical"], []
    if set(papers) != {f"P{i}" for i in range(1, len(papers) + 1)}:
        problems.append("paper ids are not P1..Pn without gaps")
    problems += [f"{pid} has no title" for pid, p in papers.items() if not (p.get("title") or "").strip()]
    return problems


def _gate_alias_map(run: dict) -> list[str]:
    problems = []
    all_keys = {k for p in run["canonical"].values() for k in p["aliases"]}
    if set(run["alias"]) != all_keys:
        problems.append("alias map and the papers' alias lists disagree")
    if not set(run["alias"].values()) <= set(run["canonical"]):
        problems.append("alias map points at unknown paper ids")
    return problems


def _gate_edges(run: dict) -> list[str]:
    problems, seen = [], set()
    for e in run["edges"]:
        pair = (e["source"], e["target"])
        if pair[0] not in run["canonical"] or pair[1] not in run["canonical"]:
            problems.append(f"edge {pair} joins an unknown paper")
        if pair[0] == pair[1]:
            problems.append(f"self edge {pair}")
        if pair in seen:
            problems.append(f"repeated edge {pair}")
        seen.add(pair)
    return problems


def _gate_vectors(run: dict) -> list[str]:
    ids, m, topic = run["ids"], run["matrix"], run["topic"]
    if m.ndim != 2 or m.shape[0] != len(ids) or not len(ids):
        return ["embeddings.npy and ids.json do not line up (or are empty)"]
    problems = []
    if len(set(ids)) != len(ids) or not set(ids) <= set(run["canonical"]):
        problems.append("ids.json has duplicates or unknown paper ids")
    if not np.isfinite(m).all():
        problems.append("embeddings contain NaN or infinite values")
    elif np.abs(np.linalg.norm(m, axis=1) - 1.0).max() > VECTOR_NORM_TOLERANCE:
        problems.append("some vectors are not unit length")
    if topic.shape != (m.shape[1],) or not np.isfinite(topic).all() \
            or abs(np.linalg.norm(topic) - 1.0) > VECTOR_NORM_TOLERANCE:
        problems.append("topic vector has the wrong size or is not unit length")
    seeds_missing = [pid for pid, p in run["canonical"].items() if p["is_seed"] and pid not in set(ids)]
    if seeds_missing:
        problems.append("seed papers without a vector: " + ", ".join(seeds_missing))
    return problems


def _gate_embedding_status(run: dict) -> list[str]:
    status, ids = run["emb_status"], set(run["ids"])
    problems = []
    if set(status) != set(run["canonical"]):
        problems.append("embedding_status.json does not cover every paper")
    if {p for p, s in status.items() if s["status"] == "embedded"} != ids:
        problems.append("embedding_status.json and ids.json disagree")
    return problems


def _gate_fulltext_files(root: Path, run: dict) -> list[str]:
    problems = []
    for pid, entry in _fulltext_by_pid(run).items():
        if not entry["has_full_text"]:
            continue
        txt = root / FULLTEXT_DIR / (entry["text_file"] or "")
        if not entry["text_file"] or not txt.is_file() or not txt.read_text(encoding="utf-8").strip():
            problems.append(f"{pid} is marked has_full_text but its text file is missing or empty")
        if entry["status"] not in ("ok", "degraded"):
            problems.append(f"{pid} is marked has_full_text but its status is {entry['status']!r}")
        if not run["canonical"][pid]["is_seed"]:
            problems.append(f"{pid} has full text but is not a seed")
    return problems


def _gate_paper_meta(papers: dict[str, dict]) -> list[str]:
    problems = []
    for pid, p in papers.items():
        if set(p) != set(PAPER_META_TYPES):
            problems.append(f"{pid}: fields differ from PaperMeta")
            continue
        for field, types in PAPER_META_TYPES.items():
            ok = isinstance(p[field], types) and not (isinstance(p[field], bool) and bool not in types)
            if not ok:
                problems.append(f"{pid}.{field} has the wrong type ({type(p[field]).__name__})")
    try:                                                    # also use the real contract when it can be imported
        from interface import PaperMeta
    except Exception:
        return problems
    for pid, p in papers.items():
        try:
            PaperMeta(**p)
        except Exception as exc:
            problems.append(f"{pid}: rejected by interface.PaperMeta ({exc})")
    return problems


def run_gates(root: Path, manifest: Manifest, run: dict, papers: dict[str, dict]) -> list[dict]:
    checks = [("steps_intact", _gate_steps_intact(manifest)), ("seed_resolved", _gate_seed_resolved(run)),
              ("paper_ids", _gate_paper_ids(run)), ("alias_map", _gate_alias_map(run)),
              ("edges_valid", _gate_edges(run)), ("vectors_valid", _gate_vectors(run)),
              ("embedding_status", _gate_embedding_status(run)),
              ("fulltext_files", _gate_fulltext_files(root, run)), ("paper_meta", _gate_paper_meta(papers))]
    return [{"gate": name, "passed": not problems, "problems": problems} for name, problems in checks]


# --------------------------------------------------------------------------- the report

def build_report(manifest: Manifest, run: dict, papers: dict, gates: list[dict]) -> dict:
    steps = {s: {"status": e.get("status"), "notes": e.get("notes", {})}
             for s, e in manifest.steps().items() if s in UPSTREAM_STEPS}
    warnings: list[str] = []
    for s in UPSTREAM_STEPS:
        warnings += [f"[{s}] {w}" for w in steps.get(s, {}).get("notes", {}).get("warnings", [])]
    ft = _fulltext_by_pid(run)
    for pid, p in papers.items():
        if p["is_seed"] and not p["has_full_text"]:
            reason = ft.get(pid, {}).get("status", "unknown")
            warnings.append(f"seed {pid} ({p['title'][:50]!r}) has no full text ({reason}): abstract-only")
    skipped = [pid for pid, s in run["emb_status"].items() if s["status"] == "skipped"]
    if skipped:
        warnings.append(f"{len(skipped)} paper(s) have no title or abstract and were not embedded: {skipped[:5]}")
    degraded = [pid for pid, e in ft.items() if e.get("status") == "degraded"]
    if degraded:
        warnings.append(f"full text is degraded for: {degraded}")
    partial = [k for k, v in run["references"]["by_seed"].items() if v.get("partial")]
    counts = {"n_seeds": sum(p["is_seed"] for p in papers.values()), "n_papers": len(papers),
              "n_records_before_dedupe": run["log"]["n_records"], "n_edges": len(run["edges"]),
              "n_embedded": len(run["ids"]), "n_skipped_embedding": len(skipped),
              "n_with_full_text": sum(p["has_full_text"] for p in papers.values()),
              "n_merged_pairs": run["log"]["counts"]["merged"],
              "n_possible_duplicates": run["log"]["counts"]["candidate"],
              "n_seeds_partial_references": len(partial), "embedding_dim": int(run["matrix"].shape[1])}
    return {"version": REPORT_VERSION, "counts": counts, "steps": steps,
            "needs_confirmation": run["seeds"].get("needs_confirmation", []),
            "warnings": warnings, "gates": gates, "all_gates_passed": all(g["passed"] for g in gates)}


# --------------------------------------------------------------------------- the step

def run_assemble_step(manifest: Manifest) -> StepResult:
    """Step 6 under the manifest. Fails (no 'done') if any quality gate fails."""
    root = manifest.root
    inputs = [SEEDS_FILE, REFERENCES_FILE, FULLTEXT_DIR, CANONICAL_FILE, ALIAS_FILE, LOG_FILE, EDGES_FILE,
              IDS_FILE, EMBEDDINGS_FILE, TOPIC_FILE, STATUS_FILE]

    def work(staging: Path) -> dict:
        run = load_run(root)
        papers = build_papers(run)
        gates = run_gates(root, manifest, run, papers)
        report = build_report(manifest, run, papers, gates)
        atomic_write_json(staging / REPORT_FILE, report)          # written first: it is the diagnostics
        atomic_write_json(staging / PAPERS_FILE, papers)
        atomic_write_json(staging / RECORDS_FILE, build_records(root, run))
        return {**report["counts"], "warnings": report["warnings"]}

    def validate(staging: Path) -> list[str]:
        report = read_json(staging / REPORT_FILE, {})
        return [f"gate {g['gate']} failed: " + "; ".join(g["problems"][:5])
                for g in report.get("gates", []) if not g["passed"]]

    return run_step(manifest, "b0_assemble", work, outputs=[PAPERS_FILE, RECORDS_FILE, REPORT_FILE],
                    inputs=inputs, validate=validate)
