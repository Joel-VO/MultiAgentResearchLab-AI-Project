"""module_B_literature_kg/b4/b4_step.py: find conflicts and save them as conflicts.json.

Two kinds of conflict:
    finding   two similar papers report results that contradict each other (LLM judges, b4_conflicts.py)
    reviewer  the reviewers of one paper gave very different scores (plain maths, b4_openreview.py)

Reviewer conflicts are OPTIONAL. They need a file  openreview_scores.json  in the run folder, a list like
    [{"paper_id": "P3", "reviewer_scores": [3, 8, 4, 9], "reviewer_comments": ["...", "..."]}]
Nothing in this project downloads it yet (write it by hand or add a small downloader later).
If the file is missing, only finding conflicts are produced.

Reads:  the graph, reviews and vectors (through interface.open_kb)
Writes: conflicts.json   a list of interface.Conflict (as dicts)
"""
from module_A_tooling.config import B4_MAX_PAIRS, B4_MODEL, RUN_FILES
from module_B_literature_kg.b0.assemble import PAPERS_FILE
from module_B_literature_kg.b0.embed import EMBEDDINGS_FILE, IDS_FILE, TOPIC_FILE
from module_B_literature_kg.b0.manifest import atomic_write_json, read_json, run_step
from ..interface import open_kb
from ..b2.b2_merge import REVIEWS_FILE
from ..b3.b3_step import GRAPH_FILE, make_kb_handle
from .b4_conflicts import detect_finding_conflicts, make_compare_fn
from .b4_openreview import detect_reviewer_conflicts

CONFLICTS_FILE = RUN_FILES["conflicts"]          # "conflicts.json"
OPENREVIEW_FILE = "openreview_scores.json"       # optional input, see the top of this file


def run_b4_step(manifest, client, model=B4_MODEL):
    root = manifest.root
    inputs = [GRAPH_FILE, REVIEWS_FILE, PAPERS_FILE, IDS_FILE, EMBEDDINGS_FILE, TOPIC_FILE]
    if (root / OPENREVIEW_FILE).exists():
        inputs.append(OPENREVIEW_FILE)

    def work(staging):
        kb = open_kb(make_kb_handle(root))
        conflicts = detect_finding_conflicts(kb, make_compare_fn(client, model))
        n_finding = len(conflicts)
        if OPENREVIEW_FILE in inputs:
            data = read_json(root / OPENREVIEW_FILE, [])
            items = [i for i in data if isinstance(i, dict) and i.get("paper_id") in kb.ids]   # only papers we have
            conflicts += detect_reviewer_conflicts(items)
        atomic_write_json(staging / CONFLICTS_FILE, [c.model_dump() for c in conflicts])
        return {"n_finding": n_finding, "n_reviewer": len(conflicts) - n_finding}

    return run_step(manifest, "b4_conflicts", work, outputs=[CONFLICTS_FILE], inputs=inputs,
                    extra_inputs={"model": model, "max_pairs": B4_MAX_PAIRS})
