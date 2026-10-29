"""Tests for D.2 (module_D_workflow_csp_rl/csp.py). Run from the repo root: pytest tests/test_csp.py

Uses the shared Pydantic types from schemas.py. The KB is a small fake that follows the
Module B contract (concept nodes '<type>:<name>' with type/name attrs, paper --has_*--> concept
edges, closest_to_text returning objects with .paper_id and .score).
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets `python tests/test_csp.py` work
from module_D_workflow_csp_rl.csp import (DatasetInfo, build_proposal, build_proposals,  # noqa: E402
                                          kb_domains, relevance)
from schemas import Hypothesis, Resources  # noqa: E402

H1_TEXT = ("A separate retrieval-grounded verifier agent improves factual accuracy "
           "over self-verification")


# ---------------------------------------------------------------- builders
def make_graph(extra_concepts=(), extra_edges=()):
    g = nx.MultiDiGraph()
    for p in ("P1", "P2", "P3", "P4"):
        g.add_node(p, type="paper")
    for t, n in [("dataset", "TruthfulQA"), ("dataset", "HotpotQA"), ("metric", "accuracy"),
                 ("method", "self-verification"), ("method", "verifier agent"),
                 ("method", "retrieval"), *extra_concepts]:
        g.add_node(f"{t}:{n.lower()}", type=t, name=n)
    edges = [("P1", "has_method", "method:self-verification"),
             ("P1", "has_dataset", "dataset:truthfulqa"), ("P1", "has_metric", "metric:accuracy"),
             ("P2", "has_method", "method:retrieval"), ("P2", "has_dataset", "dataset:truthfulqa"),
             ("P3", "has_method", "method:verifier agent"),
             ("P3", "has_dataset", "dataset:hotpotqa"), ("P3", "has_metric", "metric:accuracy"),
             ("P4", "has_method", "method:self-verification"),
             ("P4", "has_dataset", "dataset:truthfulqa"), *extra_edges]
    for u, k, v in edges:
        g.add_edge(u, v, kind=k)
    return g


def make_kb(sim=0.5, graph=None):
    g = graph if graph is not None else make_graph()
    return NS(graph=g, closest_to_text=lambda text, k: [NS(paper_id="P3", score=sim)])


def hyp(id="H1", statement=H1_TEXT, pred="Verifier beats self-verification"):
    return Hypothesis(id=id, statement=statement, testable_prediction=pred, rationale="",
                      supporting_paper_ids=[], assumptions=[])


def resources(gpu_hours=5.0, licenses=("apache-2.0", "cc-by-4.0")):
    return Resources(compute_budget_gpu_hours=gpu_hours, time_budget_hours=10,
                     licenses_allowed=list(licenses))


def catalog(**over):  # GPU hours per dataset; a run also costs 2 x 0.5 for method + baseline
    c = {"truthfulqa": DatasetInfo("apache-2.0", 1.5), "hotpotqa": DatasetInfo("cc-by-4.0", 2.0)}
    c.update(over)
    return c


def rejected(p):
    return p.status == "rejected"


# ---------------------------------------------------------------- helpers
def test_relevance_is_fraction_of_name_tokens():
    assert relevance("verifier agent", "a verifier agent helps") == 1.0
    assert relevance("verifier agent", "an agent helps") == 0.5
    assert relevance("", "anything") == 0.0


def test_kb_domains_reads_concept_layer_only():
    d = kb_domains(make_graph())
    assert d["dataset"]["dataset:truthfulqa"] == "TruthfulQA"
    assert set(d["metric"]) == {"metric:accuracy"}
    assert "P1" not in str(d)


# ---------------------------------------------------------------- happy path
def test_accepts_feasible_hypothesis_with_correct_roles():
    p = build_proposal(hyp(), make_kb(), resources(), catalog())
    assert p.status == "active" and p.rejection_reasons == []
    assert (p.method, p.baselines, p.metric) == ("verifier agent", ["self-verification"], "accuracy")
    assert p.dataset == "TruthfulQA"            # cheapest of the equally relevant datasets
    assert p.constraint_report.violated == []
    assert p.hypothesis_id == "H1" and p.id.startswith("PR-")
    assert p.est_compute_gpu_hours == 2.5 and p.higher_is_better is True


def test_lower_is_better_metric_is_flagged():
    g = make_graph(extra_concepts=[("metric", "perplexity")],
                   extra_edges=[("P1", "has_metric", "metric:perplexity"),
                                ("P4", "has_metric", "metric:perplexity")])
    p = build_proposal(hyp(statement="A separate verifier agent reduces perplexity over self-verification"), make_kb(graph=g), resources(), catalog())
    assert p.metric == "perplexity" and p.higher_is_better is False


def test_baseline_never_equals_method():
    p = build_proposal(hyp(statement="verifier agent verifier agent accuracy"), make_kb(),
                       resources(), catalog())
    assert p.status == "active" and p.method != p.baselines[0]


# ---------------------------------------------------------------- scenario: closely related prior work
def test_rejects_duplicate_of_existing_paper():
    p = build_proposal(hyp(), make_kb(sim=0.93), resources(), catalog())
    assert rejected(p) and p.rejection_reasons[0].startswith("novelty")
    assert p.dataset is None and p.method is None


def test_novelty_boundary_is_inclusive_at_threshold():
    assert rejected(build_proposal(hyp(), make_kb(sim=0.85), resources(), catalog()))
    assert not rejected(build_proposal(hyp(), make_kb(sim=0.84), resources(), catalog()))


# ---------------------------------------------------------------- scenario: limited resources
def test_tight_budget_rejects_with_compute_reason():
    p = build_proposal(hyp(), make_kb(), resources(gpu_hours=0.5), catalog())
    assert rejected(p) and any("compute budget" in x for x in p.rejection_reasons)


def test_budget_forces_cheaper_dataset():
    # self-verification also runs on HotpotQA here, so the cheaper dataset is feasible
    g = make_graph(extra_edges=[("P3", "has_method", "method:self-verification")])
    cat = catalog(truthfulqa=DatasetInfo("apache-2.0", 4.0), hotpotqa=DatasetInfo("cc-by-4.0", 2.0))
    p = build_proposal(hyp(), make_kb(graph=g), resources(gpu_hours=3.5), cat)
    assert p.status == "active" and p.dataset == "HotpotQA"


def test_rejects_when_only_affordable_dataset_cannot_run_the_baseline():
    cat = catalog(truthfulqa=DatasetInfo("apache-2.0", 4.0), hotpotqa=DatasetInfo("cc-by-4.0", 2.0))
    p = build_proposal(hyp(), make_kb(), resources(gpu_hours=3.5), cat)
    assert rejected(p) and any("baseline runs on dataset" in x for x in p.rejection_reasons)


def test_budget_checks_full_run_cost_not_just_dataset():
    # dataset alone (3.0) fits a 3.5 budget, dataset + method + baseline (4.0) does not
    cat = {"truthfulqa": DatasetInfo("apache-2.0", 3.0), "hotpotqa": DatasetInfo("cc-by-4.0", 3.0)}
    assert rejected(build_proposal(hyp(), make_kb(), resources(gpu_hours=3.5), cat))


# ---------------------------------------------------------------- licensing and ethics
def test_disallowed_license_rejects():
    p = build_proposal(hyp(), make_kb(), resources(licenses=("mit",)), catalog())
    assert rejected(p) and any("license" in x for x in p.rejection_reasons)


def test_unknown_dataset_license_fails_closed():
    p = build_proposal(hyp(), make_kb(), resources(), catalog=None)
    assert rejected(p) and any("license" in x for x in p.rejection_reasons)


def test_license_match_is_case_insensitive():
    p = build_proposal(hyp(), make_kb(), resources(licenses=("Apache-2.0", "CC-BY-4.0")), catalog())
    assert p.status == "active"


def test_sensitive_dataset_needs_ethics_clearance():
    g = make_graph(extra_concepts=[("dataset", "MedRecords")],
                   extra_edges=[("P1", "has_dataset", "dataset:medrecords")])
    cat = catalog(medrecords=DatasetInfo("apache-2.0", 0.1, sensitive=True))
    p = build_proposal(hyp(), make_kb(graph=g), resources(), cat)
    assert p.status == "active" and p.dataset != "MedRecords"
    cat["medrecords"] = DatasetInfo("apache-2.0", 0.1, sensitive=True, ethics_approved=True)
    assert build_proposal(hyp(), make_kb(graph=g), resources(), cat).status == "active"


# ---------------------------------------------------------------- compatibility / testability
def test_untestable_hypothesis_rejected():
    p = build_proposal(hyp(statement="Quantum annealing improves QA", pred=""), make_kb(),
                       resources(), catalog())
    assert rejected(p) and any("testability" in x for x in p.rejection_reasons)


def test_metric_must_co_occur_with_dataset():
    g = make_graph()
    g.remove_edges_from([(u, v, k) for u, v, k, d in list(g.edges(keys=True, data=True))
                         if d["kind"] == "has_metric"])
    p = build_proposal(hyp(), make_kb(graph=g), resources(), catalog())
    assert rejected(p) and any("metric fits dataset" in x for x in p.rejection_reasons)


def test_baseline_must_run_on_chosen_dataset_and_is_pinned_by_the_hypothesis():
    # HotpotQA is cheaper but only the verifier agent runs there; the hypothesis pins
    # self-verification as the baseline, so the solver must fall back to TruthfulQA.
    cat = catalog(truthfulqa=DatasetInfo("apache-2.0", 2.5), hotpotqa=DatasetInfo("cc-by-4.0", 0.5))
    p = build_proposal(hyp(), make_kb(), resources(gpu_hours=5.0), cat)
    assert p.status == "active" and p.dataset == "TruthfulQA"
    assert p.baselines == ["self-verification"]


def test_empty_kb_domains_reject_with_reason():
    p = build_proposal(hyp(), make_kb(graph=nx.MultiDiGraph()), resources(), catalog())
    assert rejected(p) and p.rejection_reasons


# ---------------------------------------------------------------- batch API
def test_build_proposals_splits_accepted_and_rejected():
    hs = [hyp("H1"), hyp("H9", statement="Quantum annealing improves QA", pred="")]
    ok, bad = build_proposals(hs, make_kb(), resources(), catalog())
    assert [p.hypothesis_id for p in ok] == ["H1"]
    assert [p.hypothesis_id for p in bad] == ["H9"] and bad[0].rejection_reasons


def test_is_deterministic_apart_from_the_generated_id():
    a = build_proposal(hyp(), make_kb(), resources(), catalog()).model_dump()
    b = build_proposal(hyp(), make_kb(), resources(), catalog()).model_dump()
    a.pop("id"), b.pop("id")
    assert a == b


if __name__ == "__main__":  # tiny runner so it also works without pytest
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", type(e).__name__, e)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
