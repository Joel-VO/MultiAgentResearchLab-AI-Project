import ollama
from pydantic import BaseModel

class Review(BaseModel):
    hypothesis_id: str
    verdict: str  # "APPROVED", "REJECTED", "NEEDS_REVISION"
    overall_score: float
    critique: str
    recommended_action: str

def review_experiment(exp_result, hypothesis_text: str, model_name: str = "qwen3.5:2b") -> Review:
    """Evaluates Module E's output using Ollama for Module F."""
    print(f"⚖️ [Module F] Reviewing results for: {exp_result.hypothesis_id}")
    
    prompt = f"""
    You are a peer reviewer in an autonomous AI research lab.
    Hypothesis: {hypothesis_text}
    Baseline Score: {exp_result.baseline_score}
    Achieved Score: {exp_result.achieved_score}
    Seed Runs: {exp_result.seed_runs}
    
    Critique these results. Return a score from 0.0 to 1.0 and a verdict (APPROVED, REJECTED, or NEEDS_REVISION).
    """

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format=Review.model_json_schema(),
    )
    
    return Review.model_validate_json(response['message']['content'])