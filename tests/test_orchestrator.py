# Offline tests for the LangGraph orchestration (module_A_tooling/orchestrator.py). No Ollama,
# internet or embedding model: B, C, D and the LLM calls of F are stubs that return the
# contract types, built from the DESIGN.md Section 7 samples. E, F.Belief maths and F's
# rule-based review are the real code.
# Run from the root directory: "pytest tests/test_orchestrator.py"

import functools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from module_A_tooling.orchestrator import (ContractError, Modules, ProposalRejection, _entry,
                                           run_lab, write_placeholder_report)
from module_A_tooling.schemas import Evidence
from module_B_literature_kg.interface import KBHandle, ModuleBResult, save_graph
from schemas import Gap, ResearchRequest, Resources
from module_E_experimenter.runner import SimParams, run_experiment
from module_E_experimenter.samples import HYPOTHESIS, PAPER_REVIEWS, PAPERS, PROPOSAL
from module_F_reviewer_belief.reviewer import fallback_review

REQUEST = ResearchRequest(
    topic="Improve the factual reliability of question answering with a verification agent",
    seed_papers=["2309.11495", "Self-Refine"],
    resources=Resources(compute_budget_gpu_hours=10, time_budget_hours=24, licenses_allowed=["mit"]))

H2 = HYPOTHESIS.model_copy(update={"id": "H2", "statement": "Longer prompts reduce hallucination."})


def write_kb(root: Path) -> KBHandle:
    """A KB on disk in the Module B format, holding the four sample papers."""
    ids = [p.paper_id for p in PAPERS]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(ids, type="paper")
    save_graph(graph, str(root / "graph.json"))
    np.save(root / "embeddings.npy", np.eye(len(ids)))
    np.save(root / "topic.npy", np.full(len(ids), 0.5))
    (root / "ids.json").write_text(json.dumps(ids))
    (root / "papers.json").write_text(json.dumps({p.paper_id: p.model_dump() for p in PAPERS}))
    (root / "reviews.json").write_text(json.dumps({k: r.model_dump() for k, r in PAPER_REVIEWS.items()}))
    return KBHandle(graph_path=str(root / "graph.json"), emb_path=str(root / "embeddings.npy"),
                    ids_path=str(root / "ids.json"), meta_path=str(root / "papers.json"),
                    reviews_path=str(root / "reviews.json"), topic_emb_path=str(root / "topic.npy"))


def stub_modules(tmp_path: Path, proposals=None, **overrides) -> Modules:
    calls = {}

    def run_module_b(request):
        calls["b"] = request
        return ModuleBResult(kb=write_kb(tmp_path), reviews=list(PAPER_REVIEWS.values()), conflicts=[])

    def mine_gaps(kb, reviews, conflicts, request):
        calls["c"] = (kb, reviews, conflicts, request)
        return [Gap(id="G1", source="future_work", description="external verifier", paper_ids=["P1"])]

    def literature_evidence(h, paper, review, proposal):
        return Evidence(id=f"{h.id}:{paper.paper_id}", hypothesis_id=h.id, source_id=paper.paper_id,
                        direction="support", type="direct", reliability=0.8, similarity=0.6)

    def review(hypothesis, proposal, results, belief, open_tasks, review_id):
        return fallback_review(hypothesis, results, belief, open_tasks, review_id)

    m = Modules(
        run_module_b=run_module_b,
        mine_gaps=mine_gaps,
        hypothesise=lambda kb, reviews, conflicts, request, gaps: [HYPOTHESIS.model_copy(update={"status": "candidate"}), H2],
        build_proposals=lambda hyps, kb, resources: (
            [PROPOSAL] if proposals is None else proposals,
            [ProposalRejection(hypothesis_id="H2", reasons=["license: removed all 1 candidate datasets"])]),
        embed_text=lambda text: np.zeros(len(PAPERS)),
        literature_evidence=literature_evidence,
        sim_params=lambda rng_seed: SimParams(p_fail=0.0, rng_seed=rng_seed),
        run_experiment=functools.partial(run_experiment, use_llm=False),
        review=review,
        generate_report=lambda state: write_placeholder_report(state, tmp_path),
    )
    for k, v in overrides.items():
        setattr(m, k, v)
    m.calls = calls
    return m


def test_full_run_passes_data_between_modules(tmp_path):
    m = stub_modules(tmp_path)
    state = run_lab(REQUEST, m)

    # B -> C: C receives B's KB handle, reviews and the request
    kb, reviews, conflicts, request = m.calls["c"]
    assert kb == state["kb"] and len(reviews) == 4 and conflicts == [] and request.topic == REQUEST.topic
    # D: accepted proposal stored, rejection kept for C
    assert list(state["proposals"]) == ["PR1"]
    assert state["rejections"][0].hypothesis_id == "H2"
    # F.Belief picks the hypothesis that has a proposal
    assert state["selected_hypothesis_id"] == "H1" and state["selected_proposal_id"] == "PR1"
    assert state["hypotheses"]["H1"].status == "selected"
    # E -> F.Reviewer -> controller -> E: R1 asks for a baseline, E2 runs it, R2 closes it
    assert list(state["experiments"]) == ["E1", "E2"]
    assert state["experiments"]["E2"].task_id == "T1" and state["experiments"]["E2"].kind == "baseline"
    assert [r.id for r in state["reviews"]] == ["R1", "R2"]
    assert state["closed_task_ids"] == ["T1"]
    assert state["round"] == 2 and state["stop_reason"] == "no open task an experiment can address"
    # belief is recomputed from literature (P1, P2) and both experiments
    assert {"H1:P1", "H1:P2", "H1:E1", "H1:E2"} <= set(state["evidence"])
    assert state["beliefs"]["H1"].alpha >= 1 + 2 * 0.8 * 0.6
    report = Path(state["report_path"]).read_text()
    assert "H1 [selected]" in report and "rejected H2" in report and "E2 [baseline] ok for T1" in report


def test_max_rounds_stops_the_loop(tmp_path):
    state = run_lab(REQUEST, stub_modules(tmp_path), max_rounds=1)
    assert list(state["experiments"]) == ["E1"] and state["stop_reason"] == "reached 1 rounds"


def test_no_proposal_goes_straight_to_report(tmp_path):
    state = run_lab(REQUEST, stub_modules(tmp_path, proposals=[]))
    assert not state["experiments"] and not state["reviews"]
    assert state["stop_reason"] == "no hypothesis passed the CSP"
    assert Path(state["report_path"]).exists()


def test_wrong_output_shape_names_the_module(tmp_path):
    m = stub_modules(tmp_path, run_module_b=lambda request: {"reviews": [], "conflicts": []})
    with pytest.raises(ContractError, match="Module B run_module_b .*ModuleBResult"):
        run_lab(REQUEST, m)


def test_rejected_proposal_in_wrong_format(tmp_path):
    m = stub_modules(tmp_path)
    m.build_proposals = lambda hyps, kb, resources: ([PROPOSAL], [{"hypothesis_id": "H2", "rejection_reasons": ["x"]}])
    with pytest.raises(ContractError, match="Module D build_proposals \\(rejected\\)"):
        run_lab(REQUEST, m)


def test_missing_entry_point_is_reported(tmp_path):
    m = stub_modules(tmp_path, mine_gaps=_entry("module_C_hypothesis.no_such_file:mine_gaps"))
    with pytest.raises(ContractError, match="cannot load module_C_hypothesis.no_such_file:mine_gaps"):
        run_lab(REQUEST, m)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
