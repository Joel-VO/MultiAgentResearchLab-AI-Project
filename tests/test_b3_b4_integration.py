"""Offline test of B3 (graph) and B4 (conflicts). No LLM server, no internet: the LLM is a fake function.

Run from the project root:
    python -m module_B_literature_kg.test_b3_b4_integration
(also works with pytest)

Fake data: 4 papers. P1 and P2 share the dataset SQuAD, P3 and P4 share nothing.
So B4 must ask the LLM about exactly one pair (P1, P2), and the fake LLM says "contradict".
P1 also has very different reviewer scores, so there must be one reviewer conflict too.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from module_A_tooling.cache import Cache
from module_A_tooling.config import B4_MODEL
from module_A_tooling.llm_client import LLMClient
from module_B_literature_kg.b0.manifest import Manifest
from module_B_literature_kg.b3.b3_step import run_b3_step
from module_B_literature_kg.b4.b4_step import run_b4_step


def fake_llm(spec, system, prompt, schema):
    answer = {"relation": "contradict", "summary": "They disagree about the gain on SQuAD.",
              "evidence_text": "A reports a gain, B reports none.",
              "claim_a": "improves accuracy by 5 points", "claim_b": "no gain over the baseline"}
    return json.dumps(answer), 10, 10


def make_paper(pid):
    return {"paper_id": pid, "title": f"Paper {pid}", "abstract": "some abstract", "year": 2020, "venue": None,
            "citation_count": 1, "s2_id": None, "arxiv_id": None, "doi": None,
            "is_seed": pid == "P1", "has_full_text": False}


def make_review(pid, datasets, metrics, pros):
    return {"paper_id": pid, "title": f"Paper {pid}", "idea": "an idea", "method": "a method",
            "pros": pros, "datasets": datasets, "metrics": metrics}


def test_b3_b4():
    root = Path(tempfile.mkdtemp())
    ids = ["P1", "P2", "P3", "P4"]
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    (root / "papers.json").write_text(json.dumps({i: make_paper(i) for i in ids}))
    (root / "ids.json").write_text(json.dumps(ids))
    np.save(root / "embeddings.npy", vectors.astype(np.float32))
    np.save(root / "topic.npy", vectors[0].astype(np.float32))
    (root / "citation_edges.json").write_text(json.dumps([{"source": "P1", "target": "P2", "kind": "cites"}]))
    (root / "reviews.json").write_text(json.dumps({
        "P1": make_review("P1", ["SQuAD"], ["accuracy"], ["improves accuracy by 5 points"]),
        "P2": make_review("P2", ["squad"], [], ["no gain over the baseline"]),
        "P3": make_review("P3", ["CIFAR-10"], ["error rate"], ["lower error"]),
        "P4": make_review("P4", ["ImageNet"], ["top-1"], ["higher top-1"])}))
    (root / "openreview_scores.json").write_text(json.dumps(
        [{"paper_id": "P1", "reviewer_scores": [1, 9, 2, 8], "reviewer_comments": ["weak", "great"]}]))

    manifest = Manifest(root=root)

    # ---- B3
    assert run_b3_step(manifest).status == "done"
    graph = json.loads((root / "graph.json").read_text())
    node_ids = [n["id"] for n in graph["nodes"]]
    assert "P1" in node_ids and "dataset:squad" in node_ids        # papers + a concept node

    # ---- B4
    client = LLMClient(cache=Cache(":memory:"), models={B4_MODEL: {"model": "fake", "base_url": "http://nowhere"}},
                       provider=fake_llm)
    assert run_b4_step(manifest, client).status == "done"
    conflicts = json.loads((root / "conflicts.json").read_text())
    kinds = sorted(c["type"] for c in conflicts)
    assert kinds == ["finding", "reviewer"]
    finding = next(c for c in conflicts if c["type"] == "finding")
    assert finding["paper_ids"] == ["P1", "P2"]
    assert finding["shared_dataset"] == "squad"

    # ---- nothing changed: both steps are skipped
    assert run_b3_step(manifest).status == "skipped"
    assert run_b4_step(manifest, client).status == "skipped"


if __name__ == "__main__":
    test_b3_b4()
    print("B3 + B4 integration test passed")
