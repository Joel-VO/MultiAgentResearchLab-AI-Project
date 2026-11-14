"""module_B_literature_kg/b2/b2_merge.py: put the B2a and B2b reviews into ONE file, reviews.json.

B3 (graph) and the KBHandle expect a single {paper_id: PaperReview} for every paper.
Rule for each paper:
    seed paper        -> the B2b review (full text; B2b falls back to the abstract review by itself)
    non-seed paper    -> the B2a review (abstract)
    neither exists    -> an empty review, so that every paper has an entry

Reads:  papers.json, reviews_b2a.json, reviews_b2b.json
Writes: reviews.json
"""
from module_A_tooling.config import RUN_FILES
from module_B_literature_kg.b0.assemble import PAPERS_FILE
from module_B_literature_kg.b0.manifest import atomic_write_json, read_json, run_step
from ..interface import PaperReview
from .b2a_step import REVIEWS_B2A_FILE
from .b2b_review import REVIEWS_FILE as REVIEWS_B2B_FILE

REVIEWS_FILE = RUN_FILES["reviews"]          # "reviews.json"


def run_merge_step(manifest):
    root = manifest.root

    def work(staging):
        papers = read_json(root / PAPERS_FILE)
        b2a = read_json(root / REVIEWS_B2A_FILE)
        b2b = read_json(root / REVIEWS_B2B_FILE)

        merged = {}
        counts = {"from_b2b": 0, "from_b2a": 0, "empty": 0}
        for pid, paper in papers.items():
            if pid in b2b:
                review = b2b[pid]
                counts["from_b2b"] += 1
            elif pid in b2a:
                review = b2a[pid]
                counts["from_b2a"] += 1
            else:
                review = PaperReview(paper_id=pid, title=paper["title"]).model_dump()
                counts["empty"] += 1
            merged[pid] = PaperReview(**review).model_dump()    # fails loudly if the shape is wrong
        atomic_write_json(staging / REVIEWS_FILE, merged)
        return counts

    return run_step(manifest, "b2_merge", work, outputs=[REVIEWS_FILE],
                    inputs=[PAPERS_FILE, REVIEWS_B2A_FILE, REVIEWS_B2B_FILE])
