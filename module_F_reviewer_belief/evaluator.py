"""
Module F: Reviewer & Belief Model
---------------------------------
Evaluates experimental results using local LLMs (via Module A's client)
and determines if a hypothesis should be approved, revised, or rejected.
"""

from pydantic import BaseModel
from module_A_tooling.local_llm_inference import OllamaClient

class Review(BaseModel):
    """Data contract for the peer review verdict and feedback."""
    hypothesis_id: str
    verdict: str  # "APPROVED", "REJECTED", "NEEDS_REVISION"
    overall_score: float
    critique: str
    recommended_action: str

def review_experiment(exp_result, hypothesis_statement: str) -> Review:
    """
    Critiques experimental results using Ollama and returns a structured Review object.
    
    Args:
        exp_result (ExperimentResult): The quantitative output from Module E.
        hypothesis_statement (str): The text statement of the hypothesis being tested.
        
    Returns:
        Review: Pydantic model containing verdict, score, and critique.
    """
    print(f"⚖️ [Module F] Reviewing results for: {exp_result.hypothesis_id}")
    
    # Utilize the team's standard client from Module A
    client = OllamaClient()
    
    prompt = f"""
    You are an autonomous peer reviewer in an AI research lab.
    Hypothesis: {hypothesis_statement}
    Baseline Score: {exp_result.baseline_score}
    Achieved Score: {exp_result.achieved_score}
    Seed Runs: {exp_result.seed_runs}
    
    Evaluate these results. You MUST return a valid JSON object matching this exact schema:
    {Review.model_json_schema()}
    """

    try:
        response_text = client.generate(user_prompt=prompt)
        if not response_text or not response_text.strip():
            raise ValueError("Empty response received from LLM client.")
        return Review.model_validate_json(response_text)
        
    except Exception as e:
        print(f"⚠️ [Module F Fallback] Review generation error: {e}")
        return Review(
            hypothesis_id=exp_result.hypothesis_id,
            verdict="NEEDS_REVISION",
            overall_score=0.5,
            critique=f"Automatic fallback review due to LLM error: {e}",
            recommended_action="Inspect model output or re-run evaluation node."
        )