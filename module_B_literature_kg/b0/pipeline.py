"""module_B_literature_kg/b0/pipeline.py: run all of B0 in order. This file only CALLS the steps, nothing else.

    from module_B_literature_kg.b0.pipeline import run_b0
    result = run_b0(request)                 # request has .topic and .seed_papers (ResearchRequest or a dict)
    print(format_summary(result))

Order (each step is skipped when nothing it depends on changed, so re-running is cheap and resumes a
crashed run):
    1 b0_seeds       find the seed papers
    2 b0_references  fetch what the seeds cite
    3 b0_fulltext    full text of the seeds (ar5iv, then PDF), checked
    4 b0_dedupe      merge duplicates, give ids P1..Pn, remap citation edges
    5 b0_embed       vectors for the papers and the topic
    6 b0_assemble    quality gates, then papers.json, paper_records.json, b0_report.json

If a step fails, run_b0 raises B0Error(step, cause) and later steps do not run. Fix the cause and call
run_b0 again: finished steps are skipped.

From the command line (real internet, real BGE-M3):
    python -m module_B_literature_kg.b0.pipeline --topic "LLM self-verification" --seed 2305.12345 --seed "Self-Refine"
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from module_A_tooling.config import DEFAULT_RUN_ID, RUN_FILES, run_dir
from .assemble import PAPERS_FILE, RECORDS_FILE, REPORT_FILE, run_assemble_step
from .clients import ArxivClient, RawReplyStore, S2Client
from .dedupe import run_dedupe_step
from .embed import Embedder, run_embed_step
from .fulltext import FulltextFetcher, Thresholds, run_fulltext_step
from .manifest import Manifest, StepResult, atomic_write_json, read_json
from .references import run_references_step
from .seeds import run_seeds_step

log = logging.getLogger("b0")

STEP_ORDER = ["b0_seeds", "b0_references", "b0_fulltext", "b0_dedupe", "b0_embed", "b0_assemble"]


class B0Error(RuntimeError):
    """A step failed. .step says which, .cause is the original error."""

    def __init__(self, step: str, cause: BaseException):
        super().__init__(f"B0 stopped at {step}: {type(cause).__name__}: {cause}")
        self.step, self.cause = step, cause


@dataclass
class B0Result:
    run_id: str
    root: Path
    steps: list[StepResult] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    completed: bool = False                      # False only when stop_after ended the run early

    @property
    def paths(self) -> dict[str, Optional[str]]:
        """Files for the next module. The names match interface.KBHandle; graph and reviews come from later steps."""
        def p(name: str) -> str:
            return str(self.root / RUN_FILES[name])
        return {"emb_path": p("embeddings"), "ids_path": p("ids"), "meta_path": p("papers"),
                "topic_emb_path": p("topic_embedding"), "citation_edges_path": p("citation_edges"),
                "paper_records_path": p("paper_records"), "report_path": p("b0_report"),
                "graph_path": None, "reviews_path": None}


# --------------------------------------------------------------------------- request handling

def _field(request: Any, name: str) -> Any:
    if isinstance(request, dict):
        return request[name]
    return getattr(request, name)


def read_request(request: Any) -> tuple[str, list[str]]:
    """(topic, seed list) from a ResearchRequest, any object with those attributes, or a dict."""
    topic = str(_field(request, "topic")).strip()
    seeds = [str(s).strip() for s in _field(request, "seed_papers") if str(s).strip()]
    if not topic:
        raise ValueError("the request has no topic")
    if not seeds:
        raise ValueError("the request has no seed papers")
    return topic, list(dict.fromkeys(seeds))                       # same text twice -> once


# --------------------------------------------------------------------------- the run

def run_b0(request: Any, run_id: str = DEFAULT_RUN_ID, *, root: Optional[Path] = None,
           s2: Optional[S2Client] = None, arxiv: Optional[ArxivClient] = None,
           fetcher: Optional[FulltextFetcher] = None, embedder: Optional[Embedder] = None,
           thresholds: Optional[Thresholds] = None, force: bool = False,
           stop_after: Optional[str] = None) -> B0Result:
    """Run B0 for one request.

    root         folder of the run (default runs/<run_id>/). Tests pass a temporary folder.
    s2, arxiv, fetcher, embedder   the services. Left out, the real ones are created (internet, BGE-M3).
    force        ignore "nothing changed" and redo every step.
    stop_after   stop after this step (e.g. "b0_references") to look at partial results.
    """
    if stop_after is not None and stop_after not in STEP_ORDER:
        raise ValueError(f"stop_after must be one of {STEP_ORDER}")
    topic, seeds = read_request(request)
    manifest = Manifest(run_id, root=root)
    stale = manifest.validate_prior_state()                         # outputs changed behind our back -> redo
    for step, why in stale.items():
        log.warning("%s will re-run: %s", step, why)
    if force:
        for step in STEP_ORDER:
            manifest.set(step, {"status": "stale", "stale_reason": "forced"})
    atomic_write_json(manifest.root / RUN_FILES["request"], {"topic": topic, "seed_papers": seeds})

    raw = RawReplyStore(manifest.root / RUN_FILES["raw_replies"])
    s2 = s2 or S2Client(raw=raw)
    arxiv = arxiv or ArxivClient(raw=raw)
    fetcher = fetcher or FulltextFetcher()
    embedder = embedder or Embedder()

    runners = {"b0_seeds": lambda: run_seeds_step(manifest, seeds, s2, arxiv),
               "b0_references": lambda: run_references_step(manifest, s2),
               "b0_fulltext": lambda: run_fulltext_step(manifest, fetcher, thresholds),
               "b0_dedupe": lambda: run_dedupe_step(manifest),
               "b0_embed": lambda: run_embed_step(manifest, topic, embedder),
               "b0_assemble": lambda: run_assemble_step(manifest)}

    result = B0Result(run_id=run_id, root=manifest.root)
    for step in STEP_ORDER:
        try:
            res = runners[step]()
        except Exception as exc:
            log.error("%s FAILED: %s", step, exc)
            raise B0Error(step, exc) from exc
        result.steps.append(res)
        log.info("%s: %s", step, res.status)
        if step == stop_after:
            return result
    result.report = read_json(manifest.root / REPORT_FILE, {})
    result.completed = True
    return result


# --------------------------------------------------------------------------- reading the outcome

def format_summary(result: B0Result) -> str:
    """A short plain-text summary: what ran, what was found, what needs a human look."""
    lines = [f"B0 run {result.run_id}: " + ("COMPLETE" if result.completed else "stopped early")]
    for s in result.steps:
        lines.append(f"  {s.step:<14} {s.status}")
    c = result.report.get("counts")
    if c:
        lines += ["", f"  papers: {c['n_papers']} ({c['n_seeds']} seeds), citation edges: {c['n_edges']}",
                  f"  embedded: {c['n_embedded']} (vector size {c['embedding_dim']}), "
                  f"seeds with full text: {c['n_with_full_text']}",
                  f"  duplicates merged: {c['n_merged_pairs']}, possible duplicates flagged: {c['n_possible_duplicates']}"]
    if result.report.get("needs_confirmation"):
        lines += ["", "  PLEASE CONFIRM these title matches: " + "; ".join(result.report["needs_confirmation"])]
    if result.report.get("warnings"):
        lines += ["", "  warnings:"] + [f"   - {w}" for w in result.report["warnings"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m module_B_literature_kg.b0.pipeline", description="Run Module B step B0.")
    ap.add_argument("--topic", required=True, help="the research topic (text)")
    ap.add_argument("--seed", action="append", default=[], help="a seed paper: arXiv id, title, DOI (repeatable)")
    ap.add_argument("--seeds-file", help="text file with one seed per line")
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument("--force", action="store_true", help="redo every step even if nothing changed")
    ap.add_argument("--stop-after", choices=STEP_ORDER, help="stop after this step")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(args.seed)
    if args.seeds_file:
        seeds += [ln.strip() for ln in Path(args.seeds_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = run_b0({"topic": args.topic, "seed_papers": seeds}, args.run_id,
                        force=args.force, stop_after=args.stop_after)
    except (B0Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_summary(result))
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
