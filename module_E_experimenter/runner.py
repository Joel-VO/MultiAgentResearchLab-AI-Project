"""
Module E: Experimenter
----------------------
Runs an experiment for a Proposal (optionally addressing a RevisionTask) and returns an
ExperimentResult in the DESIGN.md format.

Only simulated runs exist for now: per seed, the baseline metric is drawn from
Normal(baseline_mean, noise) and the method's metric from the same baseline plus
Normal(true_effect, noise), so results are paired by seed. Real runs need Module A's
sandbox and dataset gate.
"""
import random
from typing import Literal, Optional

from pydantic import BaseModel

from config import GENERATOR_MODEL, MIN_SEEDS
from module_A_tooling.local_llm_inference import OllamaClient
from module_A_tooling.schemas import ExperimentResult, Hypothesis, Proposal, RevisionTask

TASK_TO_KIND = {"baseline": "baseline", "ablation": "ablation", "analysis": "analysis"}
N_SAMPLES = 500


class SimParams(BaseModel):
    """Simulator settings for one scenario."""
    true_effect: float = 0.02      # mean gain of the method over the baseline
    noise: float = 0.01            # per-seed standard deviation
    baseline_mean: float = 0.60    # baseline metric value
    p_fail: float = 0.1            # probability that a run crashes
    rng_seed: int = 0


def experiment_kind(task: Optional[RevisionTask]) -> Literal["main", "baseline", "ablation", "analysis"]:
    return TASK_TO_KIND.get(task.type, "main") if task else "main"


def template_design(proposal: Proposal, kind: str, n_seeds: int, task: Optional[RevisionTask]) -> str:
    """Fixed design text, used when the LLM is off or unreachable."""
    goal = f" Addresses {task.id}: {task.text}" if task else ""
    return (f"[{kind}] Evaluate '{proposal.method}' against {', '.join(proposal.baselines)} "
            f"on {proposal.dataset} ({proposal.setting or 'default setting'}), "
            f"measuring {proposal.metric} over {n_seeds} seeds.{goal}")


def llm_design(proposal: Proposal, hypothesis: Hypothesis, kind: str, n_seeds: int,
               task: Optional[RevisionTask], client: OllamaClient) -> str:
    """Asks the experimenter model for a short design description."""
    prompt = (
        "Write the experiment design in at most 3 sentences: what is compared, how, and what result "
        "would support the hypothesis. Output only the design, no preamble, no lists.\n"
        f"Hypothesis: {hypothesis.statement}\n"
        f"Experiment type: {kind}\n"
        f"Dataset: {proposal.dataset} ({proposal.setting})\n"
        f"Method: {proposal.method}\n"
        f"Baselines: {', '.join(proposal.baselines)}\n"
        f"Metric: {proposal.metric}, {n_seeds} seeds\n"
        + (f"Reviewer request to address: {task.text}\n" if task else "")
    )
    text = client.generate(user_prompt=prompt, temperature=0, num_predict=200).strip()
    if not text:
        raise ValueError("empty design from LLM")
    return text


def run_experiment(proposal: Proposal,
                   hypothesis: Hypothesis,
                   task: Optional[RevisionTask] = None,
                   sim: Optional[SimParams] = None,
                   n_seeds: int = MIN_SEEDS,
                   mode: Literal["real", "simulated"] = "simulated",
                   use_llm: bool = True,
                   client: Optional[OllamaClient] = None,
                   result_id: str = "E1") -> ExperimentResult:
    """
    Designs and runs one experiment.

    Args:
        proposal: what to test (dataset, method, baselines, metric), from D.CSP.
        hypothesis: the hypothesis the proposal tests, from C.
        task: optional reviewer RevisionTask; its type picks the experiment kind.
        sim: simulator settings (true_effect, noise, p_fail, ...).
        n_seeds: number of seeds; results used as evidence need at least MIN_SEEDS.
        mode: "simulated" only, for now.
        use_llm: write the design text with the LLM (falls back to a template on error).
        client: Ollama client; created with reasoning disabled if not given.
        result_id: ID for the result, e.g. "E1".
    """
    if mode == "real":
        raise NotImplementedError("Real runs need Module A's run_code sandbox and fetch_dataset gate.")
    sim = sim or SimParams()
    kind = experiment_kind(task)

    design = template_design(proposal, kind, n_seeds, task)
    if use_llm:
        try:
            client = client or OllamaClient(model=GENERATOR_MODEL, think=False)
            design = llm_design(proposal, hypothesis, kind, n_seeds, task, client)
        except Exception as e:
            print(f"[Module E] LLM design unavailable ({type(e).__name__}), using template.")

    rng = random.Random(sim.rng_seed)
    seeds = list(range(n_seeds))
    result = ExperimentResult(
        id=result_id, proposal_id=proposal.id, hypothesis_id=hypothesis.id, kind=kind,
        design=design, mode="simulated", n_samples=N_SAMPLES, seeds=seeds, status="ok",
        task_id=task.id if task else None,
    )

    if rng.random() < sim.p_fail:
        result.status = "failed"
        result.error = "simulated crash: out of memory"
        return result

    baseline, method = [], []
    for _ in seeds:
        b = rng.gauss(sim.baseline_mean, sim.noise)
        baseline.append(round(b, 4))
        method.append(round(b + rng.gauss(sim.true_effect, sim.noise), 4))
    result.metrics = {proposal.metric: method}
    result.baseline_metrics = {proposal.metric: baseline}
    return result
