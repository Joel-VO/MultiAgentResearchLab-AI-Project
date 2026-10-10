import random
from pydantic import BaseModel
from typing import List

class ExperimentProposal(BaseModel):
    hypothesis_id: str
    description: str
    target_metric: str = "accuracy"
    baseline_value: float = 0.70

class ExperimentResult(BaseModel):
    hypothesis_id: str
    status: str
    metric_name: str
    baseline_score: float
    achieved_score: float
    seed_runs: List[float]
    logs: str

def run_experiment(proposal: ExperimentProposal) -> ExperimentResult:
    """Executes the test/simulation for Module E."""
    print(f"🔬 [Module E] Running experiment for: {proposal.hypothesis_id}")
    
    # Simulate 3 experimental runs
    seeds = [round(proposal.baseline_value + random.uniform(0.02, 0.08), 3) for _ in range(3)]
    avg_score = round(sum(seeds) / len(seeds), 3)
    
    return ExperimentResult(
        hypothesis_id=proposal.hypothesis_id,
        status="success",
        metric_name=proposal.target_metric,
        baseline_score=proposal.baseline_value,
        achieved_score=avg_score,
        seed_runs=seeds,
        logs=f"Completed 3 seeds. Baseline: {proposal.baseline_value}, Mean: {avg_score}"
    )