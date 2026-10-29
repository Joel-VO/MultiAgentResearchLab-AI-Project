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
    "fulltext": "fulltext",                # B0 step 3: folder, one .txt + one .json per seed, plus index.json
    "canonical": "canonical_papers.json",  # B0 step 4: merged paper records keyed by internal id P1, P2, ...
    "alias_map": "alias_map.json",         # B0 step 4: every raw key -> its internal id
    "dedupe_log": "dedupe_log.json",       # B0 step 4: every merge / candidate with evidence and confidence
    "dedupe_overrides": "dedupe_overrides.json",   # OPTIONAL, written by hand: {"undo": [[keyA, keyB]], "merge": [[keyA, keyB]]}
    "embedding_status": "embedding_status.json",   # B0 step 5: per paper embedded / skipped + flags
    "b0_report": "b0_report.json",         # B0 step 6: counts, warnings, quality-gate results
    "raw_replies": "raw_replies",          # folder: raw API replies saved per run
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
RETRY_MAX_ATTEMPTS = 5
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
EMBEDDING_DEVICE = "cpu"
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
MAX_REFERENCES_PER_PAPER = 200     # PLACEHOLDER: cap; a paper that hits it is flagged as partial

# ------------------------------------------------------------------ B1: discovery
PriorityMode = Literal["v1", "v2"]
SeedSimMode = Literal["centroid", "max", "top_k_mean"]

PRIORITY_MODE: PriorityMode = "v2"
# v2 priority = w_topic*topic_sim + w_seed*seed_sim + w_parent*parent_priority (all terms clamped 0..1)
PRIORITY_WEIGHTS = {"topic": 0.5, "seed": 0.3, "parent": 0.2}
SEED_SIM_MODE: SeedSimMode = "top_k_mean"
SEED_SIM_K = 3                     # "average of the 3 highest similarities to the seed papers"
# anchor = blend of topic and seed similarity (no parent term). Used for acceptance, depth exemption, final trim.
ANCHOR_TOPIC_WEIGHT = 0.5          # PLACEHOLDER: anchor = w*topic_sim + (1-w)*seed_sim

SIM_THRESHOLD = 0.35               # PLACEHOLDER: minimum anchor to accept a paper. Calibrate.
HIGH_SIM = 0.65                    # PLACEHOLDER: papers above this are explored regardless of depth. Calibrate.
ABS_MAX_DEPTH = 4                  # PLACEHOLDER: hard depth cap, even for HIGH_SIM papers
MAX_EXPANSIONS = 300               # PLACEHOLDER: hard cap on papers whose references we fetch
PER_SEED_QUOTA = 20                # PLACEHOLDER: Phase 0, forced expansion of every resolved seed

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
    "b1_discovery": ["PRIORITY_MODE", "PRIORITY_WEIGHTS", "SEED_SIM_MODE", "SEED_SIM_K",
                     "ANCHOR_TOPIC_WEIGHT", "SIM_THRESHOLD", "HIGH_SIM", "ABS_MAX_DEPTH",
                     "MAX_EXPANSIONS", "PER_SEED_QUOTA"],
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
    if PRIORITY_MODE not in get_args(PriorityMode):
        raise ValueError(f"PRIORITY_MODE must be one of {get_args(PriorityMode)}, got {PRIORITY_MODE!r}")
    if SEED_SIM_MODE not in get_args(SeedSimMode):
        raise ValueError(f"SEED_SIM_MODE must be one of {get_args(SeedSimMode)}, got {SEED_SIM_MODE!r}")
    if set(PRIORITY_WEIGHTS) != {"topic", "seed", "parent"}:
        raise ValueError("PRIORITY_WEIGHTS needs exactly the keys topic, seed, parent")
    if abs(sum(PRIORITY_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("PRIORITY_WEIGHTS must add up to 1.0")
    for name, value in (("ANCHOR_TOPIC_WEIGHT", ANCHOR_TOPIC_WEIGHT),
                        ("SIM_THRESHOLD", SIM_THRESHOLD), ("HIGH_SIM", HIGH_SIM)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if HIGH_SIM < SIM_THRESHOLD:
        raise ValueError("HIGH_SIM must not be lower than SIM_THRESHOLD")
    if SEED_SIM_K < 1 or ABS_MAX_DEPTH < 1 or MAX_EXPANSIONS < 1 or PER_SEED_QUOTA < 1:
        raise ValueError("SEED_SIM_K, ABS_MAX_DEPTH, MAX_EXPANSIONS and PER_SEED_QUOTA must be >= 1")
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
    for step, names in STEP_SETTINGS.items():
        for name in names:
            if name not in globals():
                raise ValueError(f"STEP_SETTINGS[{step!r}] mentions unknown setting {name}")


validate()
