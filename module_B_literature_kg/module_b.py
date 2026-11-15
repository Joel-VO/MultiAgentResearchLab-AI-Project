"""module_B_literature_kg/module_b.py: run ALL of Module B and return what Module C expects (ModuleBResult).

    B0 (seeds, references, discovery, full text, dedupe, embeddings)   b0/pipeline.py
    B2 (paper reviews -> reviews.json)                                 b2/pipeline.py
    B3 (knowledge graph -> graph.json)                                 b3/b3_step.py
    B4 (conflicts -> conflicts.json)                                   b4/b4_step.py

Use it in the LangGraph node (see interface.module_b_node):
    from module_B_literature_kg.module_b import run_module_b
    state_update = module_b_node(state, run_module_b)

Every step is skipped when nothing changed, so calling it again resumes a crashed run.

From the command line (the local LLM server must be running):
    python -m module_B_literature_kg.module_b --run-id run_003 --topic "LLM self-verification" --seed 2305.12345
    python -m module_B_literature_kg.module_b --run-id run_002      # reuses runs/run_002/request.json
"""
import argparse
import logging
import sys
import json
from typing import Literal, Optional

from pydantic import BaseModel

from module_A_tooling.config import B4_MAX_PAIRS, B4_MODEL, B4_PROMPT_VERSION

from module_A_tooling.config import DEFAULT_RUN_ID, RUN_FILES, run_dir
from module_A_tooling.llm_client import LLMClient
from module_A_tooling.schemas import LedgerEntry
from .interface import Conflict, ModuleBResult, PaperReview,KnowledgeBase
from module_B_literature_kg.b3.b3_normalize import normalize_name
from .b0.manifest import Manifest, read_json
from .b0.pipeline import run_b0
from .b2.b2_merge import REVIEWS_FILE
from .b2.pipeline import run_b2
from .b3.b3_step import make_kb_handle, run_b3_step
from .b4.b4_step import CONFLICTS_FILE, run_b4_step


class ConflictCheck(BaseModel):
    """The answer form the LLM must fill for one pair of papers."""
    relation: Literal["contradict", "agree", "unrelated"]
    summary: str = ""
    evidence_text: str = ""
    claim_a: Optional[str] = None
    claim_b: Optional[str] = None

def conflict_ledger_entries(conflicts):
    """One ledger line per conflict. A finding conflict is the LLM's judgement (hypothesis);
    a reviewer conflict is computed from real scores (fact)."""
    return [LedgerEntry(id=f"L-{c.id}", claim=c.summary, tag="hypothesis" if c.type == "finding" else "fact",
                        source_ids=c.paper_ids, produced_by="B.conflicts")
            for c in conflicts]


def run_module_b(request, run_id=DEFAULT_RUN_ID, client=None, root=None) -> ModuleBResult:
    client = client or LLMClient()

    run_b0(request, run_id, root=root)
    run_b2(run_id, client, root=root)

    manifest = Manifest(run_id, root=root)                # open again: it now holds what B0 and B2 wrote
    run_b3_step(manifest)
    run_b4_step(manifest, client)

    reviews = [PaperReview(**r) for r in read_json(manifest.root / REVIEWS_FILE).values()]
    conflicts = [Conflict(**c) for c in read_json(manifest.root / CONFLICTS_FILE)]
    return ModuleBResult(kb=make_kb_handle(manifest.root), reviews=reviews, conflicts=conflicts,
                         ledger_entries=conflict_ledger_entries(conflicts))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all of Module B (B0, B2, B3, B4).")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--topic", help="the research topic")
    parser.add_argument("--seed", action="append", default=[], help="a seed paper: arXiv id, title or DOI (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.topic and args.seed:
        request = {"topic": args.topic, "seed_papers": args.seed}
    elif args.topic or args.seed:
        print("ERROR: give both --topic and --seed, or neither")
        return 1
    else:                                                 # neither: use the request that B0 saved in the run folder
        saved = run_dir(args.run_id) / RUN_FILES["request"]
        if not saved.exists():
            print(f"ERROR: {saved} not found: give --topic and --seed")
            return 1
        request = read_json(saved)

    result = run_module_b(request, args.run_id)
    print(f"Module B finished: {len(result.reviews)} reviews, {len(result.conflicts)} conflicts, "
          f"files are in runs/{args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
