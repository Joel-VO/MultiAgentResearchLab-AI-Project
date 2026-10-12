# Use "python3 -m tests.test_ef_standalone" from root directory to run test_ef_standalone

import importlib.util
from pathlib import Path

# Get the project root directory (one level up from tests/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# 1. Dynamically import Module E using absolute paths relative to root
spec_e = importlib.util.spec_from_file_location("runner", ROOT_DIR / "module_E_experimenter" / "runner.py")
module_e = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(module_e)

# 2. Dynamically import Module F
spec_f = importlib.util.spec_from_file_location("evaluator", ROOT_DIR / "module_F_reviewer_belief" / "evaluator.py")
module_f = importlib.util.module_from_spec(spec_f)
spec_f.loader.exec_module(module_f)

def test_pipeline():
    # Construct test input as a dictionary (matches the flexible schema)
    hypothesis_data = {
        "id": "H-001",
        "description": "A separate verifier agent improves factual accuracy.",
        "target_metric": "accuracy",
        "score": 0.72
    }
    
    print(f"\n🔬 [Test] Starting pipeline for hypothesis: {hypothesis_data['id']}")
    
    # Run Module E
    result = module_e.run_experiment(hypothesis_data)
    print("\n--- Module E Output ---")
    print(result.model_dump_json(indent=2))
    
    # Run Module F
    review = module_f.review_experiment(result, hypothesis_data["description"])
    print("\n--- Module F Output ---")
    print(review.model_dump_json(indent=2))

if __name__ == "__main__":
    test_pipeline()