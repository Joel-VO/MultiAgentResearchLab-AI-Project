"""module_B_literature_kg/b2/b2a_step.py: run the B2a abstract review over ALL non-seed papers.

b2a_review.py reviews ONE paper. This file just loops over the papers and saves the results.

Reads:  papers.json                  (written by B0)
Writes: reviews_b2a.json             {paper_id: PaperReview}
        reviews_b2a_records.json     {paper_id: how the review went (status, tokens, retried ...)}
"""
from module_A_tooling.config import REVIEW_MODEL_B2A
from module_B_literature_kg.b0.assemble import PAPERS_FILE
from module_B_literature_kg.b0.manifest import MissingInputError, atomic_write_json, read_json, run_step
from ..interface import PaperReview
from .b2a_review import review_paper

REVIEWS_B2A_FILE = "reviews_b2a.json"
RECORDS_B2A_FILE = "reviews_b2a_records.json"


def run_b2a_step(manifest, client, model=REVIEW_MODEL_B2A):
    papers = read_json(manifest.root / PAPERS_FILE)
    if papers is None:
        raise MissingInputError(f"{PAPERS_FILE} not found: run B0 first")

    def work(staging):
        reviews, records = {}, {}
        todo = [p for p in papers.values() if not p["is_seed"]]   # seeds are reviewed by B2b
        for i, paper in enumerate(todo, start=1):
            pid = paper["paper_id"]
            if not paper["abstract"].strip():
                # nothing to read: empty review, no LLM call
                reviews[pid] = PaperReview(paper_id=pid, title=paper["title"]).model_dump()
                records[pid] = {"status": "no_abstract"}
                continue
            outcome = review_paper(pid, paper["title"], paper["abstract"], client, model)
            record = outcome.to_record()
            reviews[pid] = record.pop("review")
            records[pid] = record
            print(f"B2a {i}/{len(todo)} {pid}: {outcome.status}")
        atomic_write_json(staging / REVIEWS_B2A_FILE, reviews)
        atomic_write_json(staging / RECORDS_B2A_FILE, records)
        return {"n_reviewed": len(reviews)}

    return run_step(manifest, "b2a_review", work,
                    outputs=[REVIEWS_B2A_FILE, RECORDS_B2A_FILE], inputs=[PAPERS_FILE],
                    extra_inputs={"model": model})
