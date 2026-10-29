"""Global constants (DESIGN.md Section 2). Import with `from config import NAME`."""

# ---- models (Ollama). qwen3:8b is the DESIGN choice; llama3.1:8b is what is pulled locally.
GENERATOR_MODEL = "llama3.1:8b"      # E (experiment design), F (literature evidence labels)
REVIEWER_MODEL = "llama3.1:8b"       # F (reviewer)
LLM_RETRIES = 2                      # extra attempts when the output fails schema validation

# ---- Module E
MIN_SEEDS = 3                        # minimum seeds for any result used as evidence

# ---- Module F
REF_SEEDS = 3                        # experiment weight = min(1, n_seeds / REF_SEEDS)
CI_WIDTH_MAX = 0.4                   # preliminary if the 90% CI is wider than this
MIN_WEIGHT = 3.0                     # ... or if total evidence weight is below this
C_REF = 1000                         # citation count that gives full citation reliability
