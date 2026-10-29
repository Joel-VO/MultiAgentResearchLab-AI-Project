# Live demo of Modules E and F on the DESIGN.md Section 7 example. Needs Ollama running
# with the models in config.py (reasoning is disabled, so each call takes a few seconds).
# Run from the root directory: "python3 -m tests.test_ef_standalone"

import time

from config import GENERATOR_MODEL, REVIEWER_MODEL
from module_A_tooling.local_llm_inference import OllamaClient
from module_A_tooling.schemas import BeliefState, Evidence, Review, RevisionTask
from module_E_experimenter.runner import SimParams, run_experiment
from module_E_experimenter.samples import (HYPOTHESIS, PAPER_REVIEWS, PAPERS, PROPOSAL,
                                           REVISION_TASKS)
from module_F_reviewer_belief.belief import compute_belief, experiment_evidence, literature_evidence
from module_F_reviewer_belief.reviewer import review

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}
SIM = SimParams(true_effect=0.015, noise=0.005, baseline_mean=0.60, p_fail=0.1)


def step(title):
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))
    return time.time()


def done(t0):
    print(f"  ({time.time() - t0:.1f}s)")


def show_belief(b: BeliefState):
    print(f"  Belief {b.hypothesis_id}: alpha={b.alpha:.3f} beta={b.beta:.3f} mean={b.mean:.3f} "
          f"90% CI=[{b.ci_low:.3f}, {b.ci_high:.3f}] weight={b.total_weight:.2f}"
          f"{'  PRELIMINARY' if b.preliminary else ''}")


def show_review(rv: Review):
    print(f"  Review {rv.id}: overall {rv.scores.get('overall')}/10, scores {rv.scores}")
    for s in rv.strengths:
        print(f"    + {s}")
    for w in rv.weaknesses:
        print(f"    - {w}")
    for t in rv.revision_tasks:
        print(f"    task {t.id} [{t.type}, {t.severity}] {t.text}")
    if rv.closed_task_ids:
        print(f"    closed: {rv.closed_task_ids}")


def demo():
    t_start = time.time()
    generator = OllamaClient(model=GENERATOR_MODEL, think=False)
    reviewer = OllamaClient(model=REVIEWER_MODEL, think=False)
    print(f"Hypothesis {HYPOTHESIS.id}: {HYPOTHESIS.statement}")
    print(f"Proposal {PROPOSAL.id}: {PROPOSAL.method} vs {PROPOSAL.baselines} on {PROPOSAL.dataset}, "
          f"metric {PROPOSAL.metric}")

    # 1. Literature evidence -> initial belief
    t0 = step("F.1 literature evidence (generator labels, quote-checked)")
    evidence: list[Evidence] = []
    for paper in PAPERS:
        e = literature_evidence(HYPOTHESIS, paper, PAPER_REVIEWS[paper.paper_id], PROPOSAL, client=generator)
        if e:
            evidence.append(e)
            print(f"  {paper.paper_id}: {e.direction:<10} {e.type:<8} rel={e.reliability:.2f} "
                  f"sim={e.similarity:.2f}  \"{e.quote}\"")
    belief = compute_belief(HYPOTHESIS.id, evidence)
    show_belief(belief)
    done(t0)

    # 2. First experiment -> belief update
    t0 = step("E main experiment (simulated)")
    results = [run_experiment(PROPOSAL, HYPOTHESIS, sim=SIM.model_copy(update={"rng_seed": 1}),
                              client=generator, result_id="E1")]
    belief = show_run(results[-1], evidence)
    done(t0)

    # 3. First review
    t0 = step("F.2 review 1")
    rv1 = review(HYPOTHESIS, PROPOSAL, results, belief, open_tasks=[], client=reviewer, review_id="R1")
    show_review(rv1)
    done(t0)

    # 4. Address the most severe open task with a second experiment
    open_tasks: list[RevisionTask] = rv1.revision_tasks
    runnable = [t for t in open_tasks if t.type in ("baseline", "ablation", "analysis")]
    if not runnable:                     # reviewer asked for no experiment: use the DESIGN 7.7 tasks
        open_tasks = runnable = REVISION_TASKS
    target = min(runnable, key=lambda t: SEVERITY_ORDER[t.severity])
    t0 = step(f"E experiment for {target.id} [{target.type}]")
    results.append(run_experiment(PROPOSAL, HYPOTHESIS, task=target,
                                  sim=SIM.model_copy(update={"rng_seed": 2}), client=generator,
                                  result_id="E2"))
    belief = show_run(results[-1], evidence)
    done(t0)

    # 5. Second review
    t0 = step("F.2 review 2")
    rv2 = review(HYPOTHESIS, PROPOSAL, results, belief, open_tasks=open_tasks, client=reviewer,
                 review_id="R2")
    show_review(rv2)
    done(t0)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


def show_run(result, evidence: list[Evidence]) -> BeliefState:
    """Prints a run, adds its evidence (if any) and returns the updated belief."""
    print(f"  {result.id} [{result.kind}] status={result.status}"
          + (f" error={result.error}" if result.error else ""))
    print(f"  design: {result.design}")
    if result.status == "ok":
        print(f"  method   {result.metrics}")
        print(f"  baseline {result.baseline_metrics}")
    e = experiment_evidence(result)
    if e:
        evidence.append(e)
        print(f"  evidence: {e.direction} (weight {e.reliability:.2f})")
    else:
        print("  evidence: none (failed or analysis run)")
    belief = compute_belief(HYPOTHESIS.id, evidence)
    show_belief(belief)
    return belief


if __name__ == "__main__":
    demo()
