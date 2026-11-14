"""Offline test of the whole B2 pipeline. No LLM server and no internet: the LLM is a fake function.

Run from the project root:
    python -m module_B_literature_kg.b2.test_b2_integration
(also works with pytest)

The fake papers:
    P1 seed, has an abstract, no full text   -> B2b falls back to the abstract review
    P2 normal paper with an abstract         -> B2a review
    P3 normal paper without an abstract      -> empty review, no LLM call
"""
import json
import tempfile
from pathlib import Path

from module_A_tooling.cache import Cache
from module_A_tooling.config import REVIEW_MODEL_B2A, REVIEW_MODEL_B2B
from module_A_tooling.llm_client import LLMClient
from module_B_literature_kg.b0.manifest import Manifest
from module_B_literature_kg.b2.pipeline import run_b2

ABSTRACT = ("We propose RetrieverX, a retrieval method that improves answer accuracy on SQuAD "
            "by five points over a strong baseline. Experiments show clear gains.")
QUOTE = "We propose RetrieverX, a retrieval method that improves"          # 8 words, copied from ABSTRACT


def fake_llm(spec, system, prompt, schema):
    """Pretends to be the local model: always answers with a real quote from the abstract."""
    answer = {"idea": {"text": "A retrieval method", "quote": QUOTE},
              "method": {"text": "", "quote": ""}, "pros": [],
              "concepts": ["retrieval"], "methods": [], "datasets": ["SQuAD"], "metrics": []}
    return json.dumps(answer), 10, 10


def make_paper(pid, is_seed, abstract):
    return {"paper_id": pid, "title": f"Paper {pid}", "abstract": abstract, "year": 2020, "venue": None,
            "citation_count": 1, "s2_id": None, "arxiv_id": None, "doi": None,
            "is_seed": is_seed, "has_full_text": False}


def test_b2_pipeline():
    root = Path(tempfile.mkdtemp())
    papers = {"P1": make_paper("P1", True, ABSTRACT),
              "P2": make_paper("P2", False, ABSTRACT),
              "P3": make_paper("P3", False, "")}
    (root / "papers.json").write_text(json.dumps(papers))
    (root / "paper_records.json").write_text(json.dumps({"P1": {}, "P2": {}, "P3": {}}))
    (root / "fulltext").mkdir()                                    # B2b reads this folder (empty: no full text)
    Manifest(root=root).set("b0_assemble", {"status": "done"})     # pretend B0 finished

    models = {name: {"model": "fake", "base_url": "http://nowhere"}
              for name in (REVIEW_MODEL_B2A, REVIEW_MODEL_B2B)}
    client = LLMClient(cache=Cache(":memory:"), models=models, provider=fake_llm)

    results = run_b2(client=client, root=root)
    assert [r.status for r in results] == ["done", "done", "done"]

    reviews = json.loads((root / "reviews.json").read_text())
    assert set(reviews) == {"P1", "P2", "P3"}                      # every paper has a review
    assert reviews["P2"]["datasets"] == ["SQuAD"]                  # B2a worked
    assert reviews["P2"]["evidence_quotes"]["idea"] == QUOTE       # the quote check kept the claim
    assert reviews["P1"]["datasets"] == ["SQuAD"]                  # seed fallback worked
    assert reviews["P3"]["idea"] == "not stated"                   # no abstract -> empty review
    assert reviews["P3"]["datasets"] == []

    again = run_b2(client=client, root=root)                       # second run: nothing changed
    assert [r.status for r in again] == ["skipped", "skipped", "skipped"]


if __name__ == "__main__":
    test_b2_pipeline()
    print("B2 integration test passed")
