import importlib.util

# 1. Dynamically import Module E
spec_e = importlib.util.spec_from_file_location("runner", "module-E-experimenter/runner.py")
module_e = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(module_e)

# 2. Dynamically import Module F
spec_f = importlib.util.spec_from_file_location("evaluator", "module-F-reviewer_belief/evaluator.py")
module_f = importlib.util.module_from_spec(spec_f)
spec_f.loader.exec_module(module_f)

def test_pipeline():
    # Construct test input
    proposal = module_e.ExperimentProposal(
        hypothesis_id="H-001",
        description="A separate verifier agent improves factual accuracy.",
        target_metric="accuracy",
        baseline_value=0.72
    )
    
    # Run Module E
    result = module_e.run_experiment(proposal)
    print("\n--- Module E Output ---")
    print(result.model_dump_json(indent=2))
    
    # Run Module F
    review = module_f.review_experiment(result, proposal.description)
    print("\n--- Module F Output ---")
    print(review.model_dump_json(indent=2))

if __name__ == "__main__":
    test_pipeline()