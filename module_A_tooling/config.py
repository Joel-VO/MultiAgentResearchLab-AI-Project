"""config.py (A0): every setting of the project in one place.

How to use it:      from config import EMBEDDING_MODEL, run_dir, config_hash
How to change it:   edit the value here and commit. Never hard-code numbers inside other files.

Values marked  PLACEHOLDER  were not decided yet. They are sensible starting points; calibrate them
(for the similarity thresholds: from a small low-threshold run, as agreed for B1).
Secrets (API keys) are never written here. They are read from environment variables.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, get_args

# ------------------------------------------------------------------ folders and files
ROOT_DIR = Path(__file__).resolve().parent
CACHE_DB_PATH = ROOT_DIR / "cache.db"      # disposable SQLite notebook, gitignored
RUNS_DIR = ROOT_DIR / "runs"               # one sub-folder per run, e.g. runs/run_001/
DATASETS_DIR = ROOT_DIR / "datasets"
DEFAULT_RUN_ID = "run_001"

# File names inside runs/<run_id>/. interface.py wins: the graph is node-link JSON, not graphml.
RUN_FILES = {
    "papers": "papers.json",               # {paper_id: PaperMeta}
    "ids": "ids.json",                     # paper ids in embedding-row order
    "embeddings": "embeddings.npy",
    "topic_embedding": "topic.npy",
    "citation_edges": "citation_edges.json",
    "reviews": "reviews.json",             # {paper_id: PaperReview}
    "graph": "graph.json",
    "conflicts": "conflicts.json",
    "ledger_entries": "ledger_entries.json",
    "request": "request.json",
    "manifest": "manifest.json",           # per-step status, lets a run resume
    "paper_records": "paper_records.json", # sidecar: provenance + B1 scores per paper
    "seeds": "seeds.json",                 # B0 step 1: resolved seed papers + resolution log
    "references": "references.json",       # B0 step 2: reference stubs + raw citation edges
    "discovery": "discovery.json",         # B1: accepted papers (stubs + edges, same shape as references.json) + every score
    "fulltext": "fulltext",                # B0 step 3: folder, one .txt + one .json per seed, plus index.json
    "canonical": "canonical_papers.json",  # B0 step 4: merged paper records keyed by internal id P1, P2, ...
    "alias_map": "alias_map.json",         # B0 step 4: every raw key -> its internal id
    "dedupe_log": "dedupe_log.json",       # B0 step 4: every merge / candidate with evidence and confidence
    "dedupe_overrides": "dedupe_overrides.json",   # OPTIONAL, written by hand: {"undo": [[keyA, keyB]], "merge": [[keyA, keyB]]}
    "embedding_status": "embedding_status.json",   # B0 step 5: per paper embedded / skipped + flags
    "b0_report": "b0_report.json",         # B0 step 6: counts, warnings, quality-gate results
    "raw_replies": "raw_replies",          # folder: raw API replies saved per run
    "reviews_b2b": "reviews_b2b.json",
    "reviews_b2b_records": "reviews_b2b_records.json",

}


def run_dir(run_id: str = DEFAULT_RUN_ID) -> Path:
    """Folder of one run. Created if missing."""
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_file(name: str, run_id: str = DEFAULT_RUN_ID) -> Path:
    """Full path of a standard file of a run, e.g. run_file("papers")."""
    return run_dir(run_id) / RUN_FILES[name]


# ------------------------------------------------------------------ outside services
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("S2_API_KEY")                  # optional, gives better rate limits
ARXIV_API_URL = "http://export.arxiv.org/api/query"        # backup, only for missing abstracts
AR5IV_BASE_URL = "https://ar5iv.labs.arxiv.org/html/"      # seed full text (HTML), by arXiv ID
REQUEST_TIMEOUT_SECONDS = 30.0
USER_AGENT = "MultiAgentResearchLab/0.1 (course project)"

S2_NOT_FOUND_TTL_SECONDS = 24 * 3600.0     # "paper not found" answers are cached briefly   PLACEHOLDER
S2_MIN_INTERVAL_SECONDS = 1.0              # polite pause between Semantic Scholar calls        PLACEHOLDER
ARXIV_MIN_INTERVAL_SECONDS = 3.0           # arXiv asks for about 3 s between calls
S2_REFERENCES_PAGE_SIZE = 100              # references fetched per page
S2_TITLE_SEARCH_LIMIT = 5                  # candidates looked at when a seed is given as a title

# Retry: 3 tries, growing wait (1 s, then 2 s, then 4 s ...)
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 1.0
RETRY_BACKOFF_FACTOR = 2.0

# ------------------------------------------------------------------ cache (A1)
# Seconds a cached answer stays valid. None = never expires.
CACHE_TTL_SECONDS: dict[str, float | None] = {
    "s2": 30 * 24 * 3600.0,         # Semantic Scholar replies (citation counts drift)   PLACEHOLDER
    "arxiv_api": 30 * 24 * 3600.0,  # arXiv API replies                                  PLACEHOLDER
    "fulltext": None,               # seed paper full texts
    "llm": None,                    # LLM replies (the key already contains model + prompt hash)
    "embedding": None,              # embeddings (the key already contains model + revision)
}

# ------------------------------------------------------------------ B0: embeddings
EMBEDDING_MODEL = "BAAI/bge-m3"    # decided: one model for the topic and all papers
EMBEDDING_REVISION = "main"        # PLACEHOLDER: pin to a commit hash once the model is downloaded
EMBEDDING_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"  # "cuda" or "cpu"
EMBEDDING_BATCH_SIZE = 8           # small batches on CPU
EMBEDDING_INPUT = "title+abstract"
VECTOR_NORM_TOLERANCE = 1e-3       # a vector counts as "unit length" if its length is within this of 1.0
EMBEDDING_MAX_SEQ_LENGTH = 512     # PLACEHOLDER: tokens per text; abstracts are short, and CPU time grows with length

# ------------------------------------------------------------------ B0: seeds
SEED_TITLE_MIN = 0.80              # PLACEHOLDER: title match below this -> seed is unresolved
SEED_TITLE_CONFIRM = 0.92          # PLACEHOLDER: between MIN and this -> accepted, flagged for you to confirm

# ------------------------------------------------------------------ B0: full text (all PLACEHOLDER: calibrate on ~20 real papers)
EXTRACTOR_VERSION = "1"            # bump when the ar5iv HTML extractor changes (forces re-extraction, no re-download)
PDF_EXTRACTOR_VERSION = "1"        # same for the PDF text extractor
MIN_FULLTEXT_CHARS = 3000          # shorter text is never "success"
TITLE_MATCH_MIN = 0.8              # share of the expected title's words that must appear in the extracted title
MIN_COVERAGE_RATIO = 0.5           # HTML: extracted characters / characters the page really contains
MIN_SECTIONS = 3                   # fewer section headings than this is a critical warning
MAX_MINOR_ERROR_SPANS = 5          # HTML: more conversion-error markers than this is a critical warning
MINOR_WARNINGS_FOR_DEGRADED = 2    # this many minor warnings make a text "degraded"
MIN_CHARS_PER_PAGE = 500           # PDF: fewer characters per page means a scan or a failed extraction
MAX_GARBAGE_RATIO = 0.05           # PDF: share of garbage characters (critical above this)
MIN_READABLE_WORD_RATIO = 0.8      # PDF: share of normal-looking words (critical below this)
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf/"
MAX_PDF_BYTES = 30 * 1024 * 1024   # bigger downloads are refused
FAILED_FULLTEXT_RETRY_SECONDS = 24 * 3600.0   # a failed download is remembered this long, then retried
PDF_CACHE_DIR = ROOT_DIR / "pdf_cache"        # downloaded PDFs (disposable, gitignored)

# ------------------------------------------------------------------ B0: dedupe
DEDUPE_TITLE_SIM = 0.95            # PLACEHOLDER: titles at least this similar (but not identical) are only FLAGGED, never merged
DEDUPE_YEAR_TOLERANCE = 1          # preprint vs published version can differ by a year
DEDUPE_MIN_TITLE_CHARS = 15        # PLACEHOLDER: shorter normalised titles ("Introduction") are too generic to match on

# ------------------------------------------------------------------ B0: references
MAX_REFERENCES_PER_PAPER = 10 #for small corpus set to 5     # PLACEHOLDER: cap; a paper that hits it is flagged as partial

# ------------------------------------------------------------------ B1: discovery (v2.2 spec)
SeedSimMode = Literal["centroid", "max", "top_k_mean"]
DiscoveryMode = Literal["classic", "groups"]
GroupThresholdMode = Literal["per_group", "global"]
GroupScoreMode = Literal["centroid", "centroid_or_member"]
GroupSelfTest = Literal["error", "warn"]

# Defaults used when the request carries no budget. They MIRROR schemas.Budget; the request wins.
# max_papers None = "not chosen by the user": B1 then uses  PAPERS_BASE + PAPERS_PER_SEED * number_of_seeds.
B1_DEFAULT_BUDGET = {"max_papers": 10, "max_depth": 1, "fanout_per_node": 10} 
PAPERS_BASE = 40                   # PLACEHOLDER: papers of the formula that do not depend on the seeds (also the floor)
PAPERS_PER_SEED = 50               # PLACEHOLDER: extra papers for every seed (linear on purpose: later steps cost per paper)

# similarity to the seeds: "max" = closest single seed (default). The others stay for offline replay.
SEED_SIM_MODE: SeedSimMode = "max"
SEED_SIM_K = 3                     # used only by "top_k_mean": average of the K highest similarities
TOPIC_TITLE_WEIGHT = 0.4           # PLACEHOLDER: topic_sim = w*sim(topic,title) + (1-w)*sim(topic,title+abstract)
# anchor = blend of topic and seed similarity (no parent term). Used for the relevance door and the depth exemption.
ANCHOR_TOPIC_WEIGHT = 0.5          # PLACEHOLDER: anchor = w*topic_sim + (1-w)*seed_sim

# citation score (fixed rule): min(1, log1p(citations / max(as_of_year - year, 1)) / log1p(CITE_SATURATION))
CITE_SATURATION = 100.0            # PLACEHOLDER: citations per year that count as 1.0
CITE_MISSING_SCORE = 0.3           # PLACEHOLDER: neutral score when citations or year are unknown (flagged "cite_missing")

# priority = w.anchor*anchor + w.cite*cite_score + w.parent*parent_best     ("which accepted paper do I open next?")
# It only orders the EXPANSION queue. Which papers are ACCEPTED is decided by the anchor alone (no citations).
# parent_best = highest priority among the expanded papers that cite it (a seed counts as 1.0); only ever goes up
PRIORITY_WEIGHTS = {"anchor": 0.65, "cite": 0.15, "parent": 0.20}

SIM_THRESHOLD = 0.35               # PLACEHOLDER: door 1, minimum anchor to accept a paper. Calibrate.
SEED_ACCEPT_THRESHOLD = 0.60       # PLACEHOLDER: door 2, minimum seed_sim to accept a paper (protects off-topic seeds' neighbours). Calibrate.
HIGH_SIM = 0.65                    # PLACEHOLDER: papers above this are explored regardless of the soft depth. Calibrate.
ABS_MAX_DEPTH = 4                  # PLACEHOLDER: hard depth cap, even for HIGH_SIM papers
MAX_EXPANSIONS_FACTOR = 1.5        # PLACEHOLDER: cap on expansions = ceil(factor * max_papers). Each accepted paper is read at most once,
                                   # so this is only a safety valve and rarely the reason a run stops.
PHASE0_QUOTA = 20                  # PLACEHOLDER: upper limit of guaranteed children per seed (the real Q is derived from the budget)
GUARANTEE_FRACTION = 1.00         # PLACEHOLDER: share of max_papers the guaranteed children may use     

# exploration (Phase 2). 0 = off: the first run is the baseline.
EXPLORE_FRACTION = 0.0             # share of max_papers reserved for exploration papers (0 .. 0.5)
EXPLORE_MIN_ANCHOR = 0.25          # PLACEHOLDER: relevance floor of an exploration paper (must not exceed SIM_THRESHOLD)
NOVELTY_MIN = 0.10                 # PLACEHOLDER: minimum novelty = 1 - highest similarity to an accepted paper

# ------------------------------------------------------------------ B1: discovery MODE switch
# "classic" = the v2.2 rule (anchor door + seed_sim door, per-seed guaranteed children).
# "groups"  = seeds that look alike are merged into groups, each group is one centroid, a paper is accepted
#             when it is close enough to ANY group centroid (or to the topic, if GROUP_INCLUDE_TOPIC).
# Phase 2 (exploration, EXPLORE_FRACTION) works the same way in both modes.
# Command line: python -m module_B_literature_kg.b0.pipeline ... --discovery-mode groups
DISCOVERY_MODE: DiscoveryMode = "classic"

# --- settings used ONLY in "groups" mode (all PLACEHOLDER: calibrate with replay_group_doors on a real run)
GROUP_LINK_THRESHOLD = 0.70        # two seed groups are merged while their AVERAGE cosine is >= this (average-linkage clustering)
GROUP_ACCEPT_THRESHOLD = 0.50      # a paper is accepted if its similarity to some group centroid reaches the group's threshold (see below)
GROUP_THRESHOLD_MODE: GroupThresholdMode = "per_group"
                                   # "global": every group uses GROUP_ACCEPT_THRESHOLD.
                                   # "per_group": threshold = max(GROUP_MIN_THRESHOLD, GROUP_ACCEPT_THRESHOLD * tightness),
                                   #   tightness = average member-to-centroid similarity of that group (loose groups get a lower bar)
GROUP_MIN_THRESHOLD = 0.30         # floor of a per-group threshold (must not exceed GROUP_ACCEPT_THRESHOLD)
GROUP_SCORE_MODE: GroupScoreMode = "centroid"
                                   # "centroid": similarity to the group centroid.
                                   # "centroid_or_member": the larger of (centroid, closest single member of the group)
GROUP_INCLUDE_TOPIC = True         # the topic sentence acts as one extra centroid (safety net for off-topic seeds)
GROUP_TOPIC_THRESHOLD = 0.45       # bar for the topic "centroid" (topic_sim, the title/full blend)
GROUP_HIGH_SIM = 0.70              # groups-mode twin of HIGH_SIM, on the scale of the group score (bar = GROUP_ACCEPT_THRESHOLD)
GROUP_EXPLORE_MIN_ANCHOR = 0.40    # groups-mode twin of EXPLORE_MIN_ANCHOR (must not exceed GROUP_ACCEPT_THRESHOLD)
GROUP_SELF_TEST: GroupSelfTest = "error"
                                   # every seed must pass its OWN group's bar. "error" stops the run, "warn" only logs a warning.



# ------------------------------------------------------------------ A2: local LLM client and B2 settings
# The LLM is hosted by YOU in a local server program (Ollama, vLLM, llama.cpp server, LM Studio). No API keys anywhere.
LOCAL_LLM_URL = "http://localhost:11434/v1"   # Ollama default. vLLM: http://localhost:8000/v1, llama.cpp: http://localhost:8080/v1
LLM_TIMEOUT_SECONDS = 600.0        # a local model can be slow; this is the wait for ONE answer
LLM_MAX_ATTEMPTS = 3               # tries per question (timeout, busy server, unusable answer)
CHARS_PER_TOKEN_ESTIMATE = 3.0     # to check that an input fits the context window; low on purpose (safe side)
# nickname -> how to reach one model. PLACEHOLDER: replace "SET_ME" with the model name your server shows
# (Ollama: `ollama list`). Two nicknames may point to the same model: they differ in answer and context sizes.
#   max_tokens          longest answer allowed. Models that "think" before answering (printing <think>) need much more.
#   context_tokens      the model's context window AS YOUR SERVER IS SET UP (it must be set in the server too:
#                       Ollama num_ctx / OLLAMA_CONTEXT_LENGTH, vLLM --max-model-len, llama.cpp -c). Longer inputs are refused.
#   structured_output   "json_schema" (server forces the answer to fit the form, best), "json_object" (valid JSON only)
#                       or "none". If the server answers HTTP 400 about the format, use "json_object".
#   temperature, seed   0 and 0: the same question gives the same answer. temperature None = do not send it.


LLM_MODELS: dict[str, dict] = {
    "local":      {"base_url": LOCAL_LLM_URL, "think": False, "model": "qwen3:8b", "temperature": 0.5, "seed": 0,
                   "max_tokens": 2500, "context_tokens": 8192, "structured_output": "json_schema"},
    "local_long": {"base_url": LOCAL_LLM_URL, "think": False, "model": "qwen3:8b", "temperature": 0.5, "seed": 0,
                   "max_tokens": 3000, "context_tokens": 32768, "structured_output": "json_schema"},
}

B2B_PROMPT_VERSION = "1"
B2B_MIN_ROUTED_CHARS = 1500        # below this a call gets the whole paper instead
B2B_MAX_INPUT_CHARS = 40000        # per call. Use about 80000 for a 27B model
REVIEW_MODEL_B2B = "local_long"    # skip if it already exists


REVIEW_MODEL_B2A = "local"         # nickname from LLM_MODELS: abstract reviews (short inputs)
REVIEW_MODEL_B2B = "local_long"    # nickname from LLM_MODELS: seed full-text reviews (long inputs, needs a big context)
B2A_PROMPT_VERSION = "2"           # bump when the B2a prompt changes (part of the cache key)
MAX_ABSTRACT_CHARS = 3000          # longer abstracts are cut
MIN_QUOTE_WORDS = 5                # a supporting quote shorter than this is not evidence
MAX_QUOTE_CHARS = 300
MAX_CONCEPTS = 8
B2A_RETRY_FAIL_FRACTION = 0.5      # if MORE than this share of claims fail the quote check, ask once more



B4_MODEL = "local"            # must be a key of LLM_MODELS
B4_PROMPT_VERSION = "1"
B4_MAX_PAIRS = 50


# ------------------------------------------------------------------ manifest: which settings belong to which step
STEP_SETTINGS: dict[str, list[str]] = {
    "b0_embed": ["EMBEDDING_MODEL", "EMBEDDING_REVISION", "EMBEDDING_INPUT", "EMBEDDING_MAX_SEQ_LENGTH"],
    "b0_dedupe": ["DEDUPE_TITLE_SIM", "DEDUPE_YEAR_TOLERANCE", "DEDUPE_MIN_TITLE_CHARS"],
    "b0_assemble": ["VECTOR_NORM_TOLERANCE"],
    "b0_seeds": ["SEED_TITLE_MIN", "SEED_TITLE_CONFIRM", "S2_TITLE_SEARCH_LIMIT"],
    "b0_references": ["MAX_REFERENCES_PER_PAPER"],
    "b0_fulltext": ["EXTRACTOR_VERSION", "PDF_EXTRACTOR_VERSION", "MIN_FULLTEXT_CHARS", "TITLE_MATCH_MIN",
                    "MIN_COVERAGE_RATIO", "MIN_SECTIONS", "MAX_MINOR_ERROR_SPANS",
                    "MINOR_WARNINGS_FOR_DEGRADED", "MIN_CHARS_PER_PAGE", "MAX_GARBAGE_RATIO",
                    "MIN_READABLE_WORD_RATIO", "ARXIV_PDF_BASE_URL", "MAX_PDF_BYTES"],
    "b1_discovery": ["SEED_SIM_MODE", "SEED_SIM_K", "TOPIC_TITLE_WEIGHT", "ANCHOR_TOPIC_WEIGHT",
                     "CITE_SATURATION", "CITE_MISSING_SCORE", "PRIORITY_WEIGHTS", "PAPERS_BASE", "PAPERS_PER_SEED",
                     "SIM_THRESHOLD", "SEED_ACCEPT_THRESHOLD", "HIGH_SIM", "ABS_MAX_DEPTH",
                     "MAX_EXPANSIONS_FACTOR", "PHASE0_QUOTA", "GUARANTEE_FRACTION", "EXPLORE_FRACTION",
                     "EXPLORE_MIN_ANCHOR", "NOVELTY_MIN", "MAX_REFERENCES_PER_PAPER",
                     "DISCOVERY_MODE", "GROUP_LINK_THRESHOLD", "GROUP_ACCEPT_THRESHOLD", "GROUP_THRESHOLD_MODE",
                     "GROUP_MIN_THRESHOLD", "GROUP_SCORE_MODE", "GROUP_INCLUDE_TOPIC", "GROUP_TOPIC_THRESHOLD",
                     "GROUP_HIGH_SIM", "GROUP_EXPLORE_MIN_ANCHOR", "GROUP_SELF_TEST",
                     "EMBEDDING_MODEL", "EMBEDDING_REVISION", "EMBEDDING_INPUT", "EMBEDDING_MAX_SEQ_LENGTH"],
    
    "b2b_review": [
        "B2B_MIN_ROUTED_CHARS",
        "B2B_MAX_INPUT_CHARS",
        "MAX_CONCEPTS",
    ],

    "b2a_review": [
        "B2A_PROMPT_VERSION",
        "B2A_RETRY_FAIL_FRACTION",
        "MAX_ABSTRACT_CHARS",
        "MAX_CONCEPTS",
    ],

    "b2_merge": [],
    "b3_graph": [],
    "b4_conflicts": ["B4_MODEL", "B4_PROMPT_VERSION", "B4_MAX_PAIRS"],
}



def config_hash(step: str) -> str:
    """Short fingerprint of the settings a step depends on. The manifest stores it per step; if it
    changes, the step is re-run. Unknown steps raise KeyError so typos are not silent."""
    names = STEP_SETTINGS[step]
    values = {name: globals()[name] for name in names}
    blob = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]







# ------------------------------------------------------------------ safety check
def validate() -> None:
    """Fail fast on a typo or an impossible combination. Runs when this file is imported."""
    if SEED_SIM_MODE not in get_args(SeedSimMode):
        raise ValueError(f"SEED_SIM_MODE must be one of {get_args(SeedSimMode)}, got {SEED_SIM_MODE!r}")
    if set(PRIORITY_WEIGHTS) != {"anchor", "cite", "parent"} or abs(sum(PRIORITY_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("PRIORITY_WEIGHTS needs exactly the keys anchor, cite, parent and must add up to 1.0")
    for name, value in (("TOPIC_TITLE_WEIGHT", TOPIC_TITLE_WEIGHT), ("ANCHOR_TOPIC_WEIGHT", ANCHOR_TOPIC_WEIGHT),
                        ("CITE_MISSING_SCORE", CITE_MISSING_SCORE), ("SIM_THRESHOLD", SIM_THRESHOLD),
                        ("SEED_ACCEPT_THRESHOLD", SEED_ACCEPT_THRESHOLD), ("HIGH_SIM", HIGH_SIM),
                        ("GUARANTEE_FRACTION", GUARANTEE_FRACTION), ("EXPLORE_MIN_ANCHOR", EXPLORE_MIN_ANCHOR),
                        ("NOVELTY_MIN", NOVELTY_MIN)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if HIGH_SIM < SIM_THRESHOLD:
        raise ValueError("HIGH_SIM must not be lower than SIM_THRESHOLD")
    if EXPLORE_MIN_ANCHOR > SIM_THRESHOLD:
        raise ValueError("EXPLORE_MIN_ANCHOR must not be higher than SIM_THRESHOLD")
    if not 0.0 <= EXPLORE_FRACTION <= 0.5:
        raise ValueError(f"EXPLORE_FRACTION must be between 0 and 0.5, got {EXPLORE_FRACTION}")
    if CITE_SATURATION <= 0:
        raise ValueError("CITE_SATURATION must be > 0")
    if SEED_SIM_K < 1 or ABS_MAX_DEPTH < 1 or PHASE0_QUOTA < 1:
        raise ValueError("SEED_SIM_K, ABS_MAX_DEPTH and PHASE0_QUOTA must be >= 1")
    if MAX_EXPANSIONS_FACTOR < 1.0:
        raise ValueError("MAX_EXPANSIONS_FACTOR must be >= 1.0 (each accepted paper may be read once)")
    if PAPERS_BASE < 1 or PAPERS_PER_SEED < 0:
        raise ValueError("PAPERS_BASE must be >= 1 and PAPERS_PER_SEED >= 0")
    if set(B1_DEFAULT_BUDGET) != {"max_papers", "max_depth", "fanout_per_node"}:
        raise ValueError("B1_DEFAULT_BUDGET needs exactly max_papers, max_depth, fanout_per_node")
    if B1_DEFAULT_BUDGET["max_depth"] < 1 or B1_DEFAULT_BUDGET["fanout_per_node"] < 1 or \
            (B1_DEFAULT_BUDGET["max_papers"] is not None and B1_DEFAULT_BUDGET["max_papers"] < 1):
        raise ValueError("B1_DEFAULT_BUDGET: max_depth and fanout_per_node must be >= 1; max_papers None or >= 1")
    if not 0.0 <= SEED_TITLE_MIN <= SEED_TITLE_CONFIRM <= 1.0:
        raise ValueError("need 0 <= SEED_TITLE_MIN <= SEED_TITLE_CONFIRM <= 1")
    if S2_REFERENCES_PAGE_SIZE < 1 or S2_TITLE_SEARCH_LIMIT < 1 or MAX_REFERENCES_PER_PAPER < 1:
        raise ValueError("S2_REFERENCES_PAGE_SIZE, S2_TITLE_SEARCH_LIMIT and MAX_REFERENCES_PER_PAPER must be >= 1")
    for name, value in (("TITLE_MATCH_MIN", TITLE_MATCH_MIN), ("MIN_COVERAGE_RATIO", MIN_COVERAGE_RATIO),
                        ("MAX_GARBAGE_RATIO", MAX_GARBAGE_RATIO), ("MIN_READABLE_WORD_RATIO", MIN_READABLE_WORD_RATIO)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if MIN_FULLTEXT_CHARS < 1 or MIN_SECTIONS < 1 or MINOR_WARNINGS_FOR_DEGRADED < 1 or MAX_PDF_BYTES < 1:
        raise ValueError("MIN_FULLTEXT_CHARS, MIN_SECTIONS, MINOR_WARNINGS_FOR_DEGRADED and MAX_PDF_BYTES must be >= 1")
    if not 0.0 <= DEDUPE_TITLE_SIM <= 1.0:
        raise ValueError(f"DEDUPE_TITLE_SIM must be between 0 and 1, got {DEDUPE_TITLE_SIM}")
    if DEDUPE_YEAR_TOLERANCE < 0 or DEDUPE_MIN_TITLE_CHARS < 1 or EMBEDDING_MAX_SEQ_LENGTH < 1 or EMBEDDING_BATCH_SIZE < 1:
        raise ValueError("DEDUPE_YEAR_TOLERANCE >= 0; DEDUPE_MIN_TITLE_CHARS, EMBEDDING_MAX_SEQ_LENGTH, EMBEDDING_BATCH_SIZE >= 1")
    if not 0.0 < VECTOR_NORM_TOLERANCE < 0.5:
        raise ValueError(f"VECTOR_NORM_TOLERANCE must be between 0 and 0.5, got {VECTOR_NORM_TOLERANCE}")
    if RETRY_MAX_ATTEMPTS < 1 or RETRY_BASE_SECONDS <= 0 or RETRY_BACKOFF_FACTOR < 1:
        raise ValueError("retry settings are invalid")
    if DISCOVERY_MODE not in get_args(DiscoveryMode):
        raise ValueError(f"DISCOVERY_MODE must be one of {get_args(DiscoveryMode)}, got {DISCOVERY_MODE!r}")
    if GROUP_THRESHOLD_MODE not in get_args(GroupThresholdMode):
        raise ValueError(f"GROUP_THRESHOLD_MODE must be one of {get_args(GroupThresholdMode)}, got {GROUP_THRESHOLD_MODE!r}")
    if GROUP_SCORE_MODE not in get_args(GroupScoreMode):
        raise ValueError(f"GROUP_SCORE_MODE must be one of {get_args(GroupScoreMode)}, got {GROUP_SCORE_MODE!r}")
    if GROUP_SELF_TEST not in get_args(GroupSelfTest):
        raise ValueError(f"GROUP_SELF_TEST must be one of {get_args(GroupSelfTest)}, got {GROUP_SELF_TEST!r}")
    for name, value in (("GROUP_LINK_THRESHOLD", GROUP_LINK_THRESHOLD), ("GROUP_ACCEPT_THRESHOLD", GROUP_ACCEPT_THRESHOLD),
                        ("GROUP_MIN_THRESHOLD", GROUP_MIN_THRESHOLD), ("GROUP_TOPIC_THRESHOLD", GROUP_TOPIC_THRESHOLD),
                        ("GROUP_HIGH_SIM", GROUP_HIGH_SIM), ("GROUP_EXPLORE_MIN_ANCHOR", GROUP_EXPLORE_MIN_ANCHOR)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if GROUP_MIN_THRESHOLD > GROUP_ACCEPT_THRESHOLD:
        raise ValueError("GROUP_MIN_THRESHOLD must not be higher than GROUP_ACCEPT_THRESHOLD")
    if GROUP_HIGH_SIM < GROUP_ACCEPT_THRESHOLD:
        raise ValueError("GROUP_HIGH_SIM must not be lower than GROUP_ACCEPT_THRESHOLD")
    if GROUP_EXPLORE_MIN_ANCHOR > GROUP_ACCEPT_THRESHOLD:
        raise ValueError("GROUP_EXPLORE_MIN_ANCHOR must not be higher than GROUP_ACCEPT_THRESHOLD")
    for step, names in STEP_SETTINGS.items():
        for name in names:
            if name not in globals():
                raise ValueError(f"STEP_SETTINGS[{step!r}] mentions unknown setting {name}")
    for nick, spec in LLM_MODELS.items():
        if not spec.get("base_url") or not spec.get("model") or set(spec) - {
                "base_url", "model", "temperature", "seed", "max_tokens", "context_tokens", "structured_output","think"}:
            raise ValueError(f"LLM_MODELS[{nick!r}] needs base_url and model, and may have temperature, seed, max_tokens, "
                             "context_tokens, structured_output")
        if spec.get("structured_output", "json_schema") not in ("json_schema", "json_object", "none"):
            raise ValueError(f"LLM_MODELS[{nick!r}]: structured_output must be json_schema, json_object or none")
        if spec.get("context_tokens") is not None and spec["context_tokens"] <= spec.get("max_tokens", 1500):
            raise ValueError(f"LLM_MODELS[{nick!r}]: context_tokens must be larger than max_tokens")
    for name in (REVIEW_MODEL_B2A, REVIEW_MODEL_B2B):
        if name not in LLM_MODELS:
            raise ValueError(f"review model {name!r} is not a key of LLM_MODELS")
    if LLM_MAX_ATTEMPTS < 1 or LLM_TIMEOUT_SECONDS <= 0 or CHARS_PER_TOKEN_ESTIMATE <= 0:
        raise ValueError("LLM_MAX_ATTEMPTS must be >= 1; LLM_TIMEOUT_SECONDS and CHARS_PER_TOKEN_ESTIMATE must be > 0")
    if MAX_ABSTRACT_CHARS < 1 or MIN_QUOTE_WORDS < 1 or MAX_QUOTE_CHARS < 1 or MAX_CONCEPTS < 1:
        raise ValueError("MAX_ABSTRACT_CHARS, MIN_QUOTE_WORDS, MAX_QUOTE_CHARS and MAX_CONCEPTS must be >= 1")
    if not 0.0 <= B2A_RETRY_FAIL_FRACTION <= 1.0:
        raise ValueError("B2A_RETRY_FAIL_FRACTION must be between 0 and 1")
    if B4_MODEL not in LLM_MODELS:
        raise ValueError(f"B4_MODEL {B4_MODEL!r} is not a key of LLM_MODELS")
    if B4_MAX_PAIRS < 0:
        raise ValueError("B4_MAX_PAIRS must be >= 0")


validate()
