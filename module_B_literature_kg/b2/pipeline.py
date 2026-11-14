"""module_B_literature_kg/b2/pipeline.py: run all of B2 in order. This file only CALLS the steps.

    1 b2a_review   review every non-seed paper from its abstract
    2 b2b_review   review every seed paper from its full text
    3 b2_merge     combine both into reviews.json

It needs B0 to be finished first (b0_assemble done). Like B0, each step is skipped when nothing it
depends on changed, so running it again resumes where it stopped.

From the command line (needs the local LLM server running):
    python -m module_B_literature_kg.b2.pipeline --run-id run_001
"""
import argparse
import sys

from module_A_tooling.config import DEFAULT_RUN_ID
from module_A_tooling.llm_client import LLMClient
from module_B_literature_kg.b0.manifest import Manifest
from .b2a_step import run_b2a_step
from .b2b_review import run_b2b_step
from .b2_merge import run_merge_step


def run_b2(run_id=DEFAULT_RUN_ID, client=None, root=None):
    """Run B2 for one run. Returns the list of StepResult. root = folder of the run (tests pass a temp folder)."""
    manifest = Manifest(run_id, root=root)
    manifest.validate_prior_state()                       # outputs changed behind our back -> mark stale
    if manifest.status("b0_assemble") != "done":
        raise RuntimeError("B0 is not finished: run the B0 pipeline first (b0_assemble must be done)")
    client = client or LLMClient()

    steps = [("b2a_review", lambda: run_b2a_step(manifest, client)),
             ("b2b_review", lambda: run_b2b_step(manifest, client)),
             ("b2_merge", lambda: run_merge_step(manifest))]
    results = []
    for name, run in steps:
        try:
            result = run()
        except Exception as exc:
            print(f"B2 stopped at {name}: {type(exc).__name__}: {exc}")
            raise
        print(f"{name}: {result.status}")
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Module B step B2 (paper reviews).")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    run_b2(args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
