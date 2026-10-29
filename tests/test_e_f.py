# Offline tests for Modules E and F (no Ollama needed; a fake client returns canned JSON).
# Run from the root directory: "python3 -m tests.test_e_f" (or "pytest tests/test_e_f.py").

import json

from module_A_tooling.schemas import BeliefState, Evidence, ExperimentResult
from module_E_experimenter.runner import SimParams, run_experiment
from module_E_experimenter.samples import (HYPOTHESIS, PAPER_REVIEWS, PAPERS, PROPOSAL,
                                           REVISION_TASKS)
from module_F_reviewer_belief.belief import (compute_belief, experiment_evidence,
                                             literature_evidence, rank, reliability, similarity)
from module_F_reviewer_belief.reviewer import review


class FakeClient:
    """Stands in for OllamaClient: returns the given replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, user_prompt, system_prompt=None, format=None, **options):
        self.calls += 1
        return self.replies.pop(0) if self.replies else ""


def result(diffs, kind="main", status="ok", rid="E1"):
    base = [0.60] * len(diffs)
    return ExperimentResult(
        id=rid, proposal_id="PR1", hypothesis_id="H1", kind=kind, design="d", mode="simulated",
        metrics={"accuracy": [b + d for b, d in zip(base, diffs)]},
        baseline_metrics={"accuracy": base}, seeds=list(range(len(diffs))), status=status)


def close(a, b, tol=0.002):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- Module E


def test_experiment_is_deterministic_and_paired():
    sim = SimParams(p_fail=0.0, rng_seed=7)
    r1 = run_experiment(PROPOSAL, HYPOTHESIS, sim=sim, use_llm=False)
    r2 = run_experiment(PROPOSAL, HYPOTHESIS, sim=sim, use_llm=False)
    assert r1 == r2
    assert r1.status == "ok" and r1.kind == "main" and r1.mode == "simulated"
    assert len(r1.metrics["accuracy"]) == len(r1.baseline_metrics["accuracy"]) == 3
    assert r1.proposal_id == "PR1" and r1.hypothesis_id == "H1"


def test_task_type_sets_kind():
    sim = SimParams(p_fail=0.0)
    kinds = [run_experiment(PROPOSAL, HYPOTHESIS, task=t, sim=sim, use_llm=False).kind
             for t in REVISION_TASKS]
    assert kinds == ["ablation", "baseline", "analysis"]
    r = run_experiment(PROPOSAL, HYPOTHESIS, task=REVISION_TASKS[0], sim=sim, use_llm=False)
    assert r.task_id == "T1"


def test_failed_run():
    r = run_experiment(PROPOSAL, HYPOTHESIS, sim=SimParams(p_fail=1.0), use_llm=False)
    assert r.status == "failed" and r.error and not r.metrics


def test_llm_design_and_fallback():
    r = run_experiment(PROPOSAL, HYPOTHESIS, sim=SimParams(p_fail=0.0),
                       client=FakeClient("Run the verifier on TruthfulQA."))
    assert r.design == "Run the verifier on TruthfulQA."
    r = run_experiment(PROPOSAL, HYPOTHESIS, sim=SimParams(p_fail=0.0), client=FakeClient(""))
    assert "TruthfulQA" in r.design      # empty LLM reply -> template


# --------------------------------------------------------------------------- Module F.1


def test_experiment_evidence_directions():
    # DESIGN 7.6: diffs +0.022, +0.013, +0.018 -> CI [0.010, 0.025] -> support
    e = experiment_evidence(result([0.022, 0.013, 0.018]))
    assert e.direction == "support" and e.type == "direct" and e.id == "H1:E1"
    assert experiment_evidence(result([-0.02, -0.03, -0.025])).direction == "contradict"
    assert experiment_evidence(result([0.02, -0.02, 0.001])).direction == "neutral"
    assert experiment_evidence(result([0.02, 0.02, 0.02], status="failed")) is None
    assert experiment_evidence(result([0.02, 0.03, 0.02], kind="analysis")) is None


def test_compute_belief_matches_design_example():
    # DESIGN 7.5: P1-P3 support, P4 contradicts -> alpha 2.185, beta 1.68, CI [0.179, 0.910]
    ev = [
        Evidence(id="H1:P1", hypothesis_id="H1", source_id="P1", direction="support", type="direct", reliability=0.90, similarity=0.75),
        Evidence(id="H1:P2", hypothesis_id="H1", source_id="P2", direction="support", type="direct", reliability=0.70, similarity=0.50),
        Evidence(id="H1:P3", hypothesis_id="H1", source_id="P3", direction="support", type="indirect", reliability=0.80, similarity=2 / 3),
        Evidence(id="H1:P4", hypothesis_id="H1", source_id="P4", direction="contradict", type="direct", reliability=0.85, similarity=0.80),
        Evidence(id="H2:P1", hypothesis_id="H2", source_id="P1", direction="support", type="direct", reliability=1, similarity=1),
    ]
    b = compute_belief("H1", ev)
    assert close(b.alpha, 2.185) and close(b.beta, 1.68)
    assert close(b.mean, 0.565) and close(b.ci_low, 0.179) and close(b.ci_high, 0.910)
    assert b.preliminary


def test_rank_by_ci_low():
    a = BeliefState(hypothesis_id="H1", alpha=1, beta=1, mean=0.56, ci_low=0.18, ci_high=0.9, total_weight=1, preliminary=True)
    b = BeliefState(hypothesis_id="H2", alpha=1, beta=1, mean=0.59, ci_low=0.13, ci_high=0.96, total_weight=1, preliminary=True)
    assert [x.hypothesis_id for x in rank([b, a])] == ["H1", "H2"]


def test_reliability_and_similarity():
    assert reliability(PAPERS[0]) > reliability(PAPERS[2])      # venue + more citations
    assert close(similarity(PAPER_REVIEWS["P1"], PROPOSAL), 0.75)
    assert similarity(None, PROPOSAL) == 0.5


def test_literature_label_quote_check():
    p1 = PAPERS[0]
    real = json.dumps({"reason": "same claim", "direction": "support", "type": "direct",
                       "quote": "Self-verification improves accuracy on TruthfulQA compared to the baseline model."})
    e = literature_evidence(HYPOTHESIS, p1, PAPER_REVIEWS["P1"], PROPOSAL, client=FakeClient(real))
    assert e is not None and e.direction == "support" and e.id == "H1:P1"
    made_up = json.dumps({"reason": "r", "direction": "support", "type": "direct", "quote": "Verifiers always win."})
    assert literature_evidence(HYPOTHESIS, p1, client=FakeClient(made_up)) is None


def test_literature_label_retries_then_gives_up():
    client = FakeClient("not json", "still not json", "{}")
    assert literature_evidence(HYPOTHESIS, PAPERS[0], client=client) is None
    assert client.calls == 3                                     # 1 try + 2 retries


# --------------------------------------------------------------------------- Module F.2


def test_review_from_llm():
    draft = {
        "strengths": ["Tests the P1/P4 conflict directly"],
        "weaknesses": ["Baseline not compute-matched (assumption)"],
        "questions": [],
        "closed_task_ids": ["T3", "T99"],
        "new_tasks": [{"text": "Add a compute-matched baseline", "type": "baseline", "severity": "major"}],
        "scores": {"novelty": 6, "soundness": 5, "evidence": 5, "clarity": 7, "overall": 12},
    }
    res = [result([0.011, 0.004, 0.012])]
    res[0].task_id = "T1"
    belief = compute_belief("H1", [experiment_evidence(res[0])])
    rv = review(HYPOTHESIS, PROPOSAL, res, belief, open_tasks=REVISION_TASKS,
                client=FakeClient(json.dumps(draft)))
    assert rv.closed_task_ids == ["T1", "T3"]                    # T1 from the run, T99 unknown
    assert rv.revision_tasks[0].id == "T4"                       # next free ID after open T1-T3
    assert rv.scores["overall"] == 10.0                          # clamped to 0-10


def test_review_fallback_on_bad_llm():
    res = [result([0.02, 0.02, 0.02], status="failed")]
    res[0].error = "OOM"
    belief = compute_belief("H1", [])
    rv = review(HYPOTHESIS, PROPOSAL, res, belief, client=FakeClient("garbage", "garbage", "garbage"))
    assert any("failed" in w for w in rv.weaknesses)
    assert rv.revision_tasks and rv.revision_tasks[0].type == "baseline"
    assert set(rv.scores) == {"novelty", "soundness", "evidence", "clarity", "overall"}


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\nAll {len(tests)} tests passed.")
