"""Shared data types for the research lab graph. Owned by Module A (tooling); every module imports from here.

Precedence rule: this file is the project-wide overview. interface.py is the contract written for
Module B (literature / knowledge base). Where the two disagree about a type Module B owns, interface.py
wins and this file mirrors it. tests/test_schema_sync.py checks that the mirrored classes stay identical.

Conventions
- Every object has a stable string ID with a prefix (below). Objects refer to each other by ID,
  never by embedding one object inside another, so each object has one copy in the state.
- `...Draft` classes are what an LLM returns. The node adds IDs and provenance by building the
  full class from the draft, e.g. `PaperReview(paper_id=pid, title=meta.title, **draft.model_dump())`.
  LLMs never generate IDs, except when returning IDs they were given in the prompt.
- Collections in LabState are dicts keyed by ID and merged with `upsert`, so a node can add a new
  object or replace an existing one by returning only that object. The ledger, reviews and the RL
  log are append-only lists.

ID prefixes
  RQ request · P1, P2, ... paper (internal; the real S2/arXiv/DOI ids live in PaperMeta) · CF conflict · G gap · H hypothesis · PR proposal
  X experiment · D dataset · EV evidence · R review · T revision task · W workflow task · L ledger
"""


"""
This file is a sample on what the types can and would be like. Should make it easier to start working on our modules based on this"""


import operator
import uuid
from typing import Annotated, Literal, Optional, TypedDict, get_args

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def upsert(old: dict | None, new: dict | None) -> dict:
    return {**(old or {}), **(new or {})}


Tag = Literal["fact", "hypothesis", "assumption", "speculation"]
Direction = Literal["support", "contradict", "neutral"]
EvidenceType = Literal["direct", "indirect"]
TaskType = Literal["baseline", "ablation", "analysis", "revise_hypothesis", "other"]
Severity = Literal["minor", "major", "critical"]
ActionType = Literal["read_paper", "run_experiment", "add_baseline", "ablation",
                     "analyse_errors", "revise_hypothesis", "address_comment"]

TASK_TYPES: tuple[str, ...] = get_args(TaskType)
ACTION_TYPES: tuple[str, ...] = get_args(ActionType)


# ---------------------------------------------------------------- request (input to the graph)

class Budget(BaseModel):
    max_papers: int = 150
    max_depth: int = 2
    fanout_per_node: int = 20


class Resources(BaseModel):
    compute_budget_gpu_hours: float
    time_budget_hours: float
    licenses_allowed: list[str]
    run_budget: float = 1.0  # abstract cost units spent by the RL controller's actions


class ResearchRequest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("RQ"))
    topic: str
    seed_papers: list[str]  # DOI, arXiv ID, or title
    budget: Budget = Field(default_factory=Budget)
    resources: Resources


# ---------------------------------------------------------------- ledger (written by every module)

class LedgerEntry(BaseModel):
    id: str = Field(default_factory=lambda: new_id("L"))
    claim: str
    tag: Tag
    source_ids: list[str] = []  # paper, experiment, or review IDs
    hypothesis_id: str | None = None  # lets the reviewer pull the excerpt for one hypothesis
    produced_by: str  # node name, e.g. "B.conflicts"
    confidence: float = 1.0
    round: int = 0


# ---------------------------------------------------------------- Module B (literature, knowledge base)
# KBHandle, PaperMeta, PaperReview and Conflict are MIRRORED from interface.py (interface.py wins).
# Do not edit them here alone: change interface.py first, then copy the change here.

class KBHandle(BaseModel):
    """Lives in LabState.kb. Paths only. Query functions live in interface.py: open_kb(handle)."""
    graph_path: str        # node-link JSON of the MultiDiGraph
    emb_path: str          # .npy, shape (n_papers, dim), L2-normalised
    ids_path: str          # ids.json: ["P1", "P2", ...] in embedding-row order
    meta_path: str         # papers.json: {paper_id: PaperMeta}
    reviews_path: str      # reviews.json: {paper_id: PaperReview}
    topic_emb_path: str    # .npy, shape (dim,), the topic query vector


class PaperMeta(BaseModel):
    paper_id: str                      # internal "P1"
    title: str
    abstract: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    s2_id: Optional[str] = None        # Semantic Scholar paperId
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    is_seed: bool = False
    has_full_text: bool = False


class PaperReviewDraft(BaseModel):
    """What the LLM returns for one paper. The node builds the full PaperReview from it:
    PaperReview(paper_id=pid, title=meta.title, **draft.model_dump())"""
    idea: str = "not stated"
    method: str = "not stated"
    pros: list[str] = Field(default_factory=list)          # [] == not stated
    cons: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class PaperReview(BaseModel):
    paper_id: str
    title: str = ""
    idea: str = "not stated"
    method: str = "not stated"
    pros: list[str] = Field(default_factory=list)          # [] == not stated
    cons: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    # field name -> exact quote from the paper that supports it (from the quote check)
    evidence_quotes: dict[str, str] = Field(default_factory=dict)


class ConflictCheck(BaseModel):  # LLM output for one paper pair
    relation: Literal["contradict", "agree", "unrelated"]
    summary: str
    evidence_text: str
    # optional extras the LLM can fill; copied into Conflict when relation == "contradict"
    shared_dataset: Optional[str] = None
    shared_metric: Optional[str] = None
    claim_a: Optional[str] = None
    claim_b: Optional[str] = None


class Conflict(BaseModel):
    id: str
    type: Literal["finding", "reviewer"]
    paper_ids: list[str]               # 2 papers for "finding", 1 for "reviewer"
    summary: str
    evidence_text: str
    # extras, filled for type == "finding" (None otherwise)
    shared_dataset: Optional[str] = None
    shared_metric: Optional[str] = None
    claim_a: Optional[str] = None
    claim_b: Optional[str] = None


# ---------------------------------------------------------------- Module C

class Gap(BaseModel):
    id: str = Field(default_factory=lambda: new_id("G"))
    source: Literal["sparse_region", "missing_combination", "conflict", "future_work"]
    description: str
    paper_ids: list[str] = []
    conflict_id: str | None = None


class HypothesisDraft(BaseModel):
    statement: str
    testable_prediction: str
    rationale: str
    supporting_paper_ids: list[str]  # the node drops IDs not present in the KB
    assumptions: list[str]


class Hypothesis(HypothesisDraft):
    id: str = Field(default_factory=lambda: new_id("H"))
    gap_ids: list[str] = []
    score_parts: dict[str, float] = {}  # novelty, evidence_support, feasibility, llm_judge
    score: float = 0.0
    status: Literal["candidate", "selected", "rejected", "superseded"] = "candidate"
    parent_id: str | None = None  # set when this hypothesis is a revision of another
    created_round: int = 0


# ---------------------------------------------------------------- Module F.CSP

class ConstraintReport(BaseModel):
    satisfied: list[str] = []
    violated: list[str] = []


class Proposal(BaseModel):
    id: str = Field(default_factory=lambda: new_id("PR"))
    hypothesis_id: str
    status: Literal["active", "rejected", "superseded"] = "active"
    research_question: str = ""
    dataset: str | None = None
    method: str | None = None
    baselines: list[str] = []
    metric: str | None = None
    higher_is_better: bool = True  # E flips the sign of differences when False
    setting: str = ""
    est_compute_gpu_hours: float = 0.0
    constraint_report: ConstraintReport = Field(default_factory=ConstraintReport)
    rejection_reasons: list[str] = []


# ---------------------------------------------------------------- Module D

class ExperimentConfig(BaseModel):  # filled by the experimenter agent
    script: str  # name of one of the hand-written experiment scripts
    dataset: str
    method: str
    baseline: str
    seeds: list[int]
    ablate: str | None = None  # component removed, for ablation runs
    params: dict[str, float | int | str] = {}


class ExperimentResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("X"))
    proposal_id: str
    hypothesis_id: str
    revision_task_id: str | None = None
    kind: Literal["main", "ablation", "baseline"]
    comparison: str  # what baseline_metrics measure, e.g. "self-verification", "without verifier"
    design: str
    config: ExperimentConfig
    mode: Literal["real", "simulated"]
    metrics: dict[str, list[float]] = {}  # per seed, same order as config.seeds
    baseline_metrics: dict[str, list[float]] = {}
    n_samples: int = 0
    status: Literal["ok", "failed"]
    error: str | None = None
    artifacts: list[str] = []
    cost: float = 0.0  # run-budget units consumed
    round: int = 0


# ---------------------------------------------------------------- Module A tool I/O

class DatasetRequest(BaseModel):
    name: str
    source_url: str
    license: str
    size_bytes: int | None = None
    reason: str  # shown to the user at the permission gate
    requested_by: str


class DatasetRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("D"))
    name: str
    url: str
    size_bytes: int | None = None
    license: str
    path: str | None = None
    approved: bool


class CodeResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    files: list[str] = []


class CitationCheck(BaseModel):
    valid: list[str]
    invalid: list[str]  # cited IDs not present in the KB


# ---------------------------------------------------------------- Module E.1 belief

class EvidenceLabel(BaseModel):  # LLM output for one (hypothesis, paper) pair
    direction: Direction
    type: EvidenceType
    quote: str  # must appear verbatim in the abstract, otherwise the label is discarded


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("EV"))
    hypothesis_id: str
    source_id: str  # paper ID or experiment ID
    source_kind: Literal["paper", "experiment"]
    direction: Direction
    type: EvidenceType
    reliability: float = Field(ge=0, le=1)
    similarity: float = Field(ge=0, le=1)
    quote: str | None = None  # papers only
    round: int = 0


class BeliefState(BaseModel):
    hypothesis_id: str
    alpha: float
    beta: float
    mean: float
    ci_low: float  # 90% credible interval
    ci_high: float
    total_weight: float
    n_evidence: int
    preliminary: bool
    round: int = 0


# ---------------------------------------------------------------- Module E.2 reviewer

class Weakness(BaseModel):
    text: str
    why_it_matters: str
    ledger_ids: list[str] = []
    is_assumption: bool = False  # True when no ledger entry backs the weakness


class RevisionTaskDraft(BaseModel):
    text: str
    type: TaskType
    severity: Severity
    est_cost: float  # run-budget units; F may override it from the cost table


class RevisionTask(RevisionTaskDraft):
    id: str = Field(default_factory=lambda: new_id("T"))
    review_id: str
    hypothesis_id: str
    status: Literal["open", "closed", "dropped"] = "open"
    addressed_by: list[str] = []  # experiment, evidence, or hypothesis IDs


class ReviewScores(BaseModel):
    novelty: float = Field(ge=0, le=10)
    soundness: float = Field(ge=0, le=10)
    evidence: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)


class ReviewDraft(BaseModel):
    strengths: list[str]
    weaknesses: list[Weakness]
    questions: list[str]
    revision_tasks: list[RevisionTaskDraft]
    closed_task_ids: list[str] = []  # IDs of open tasks (given in the prompt) judged addressed
    scores: ReviewScores


class Review(ReviewDraft):
    id: str = Field(default_factory=lambda: new_id("R"))
    proposal_id: str
    hypothesis_id: str
    revision_task_ids: list[str] = []  # IDs assigned to revision_tasks, in the same order
    round: int = 0


# ---------------------------------------------------------------- Module F tracker and RL

class Task(BaseModel):
    id: str = Field(default_factory=lambda: new_id("W"))
    node: str  # graph node that executes the task
    preconditions: list[str] = []  # fact names that must be True in LabState.facts
    effects: list[str] = []  # fact names set True on success
    status: Literal["pending", "ready", "running", "done", "failed"] = "pending"
    params: dict = {}  # target IDs, config overrides; changed on replanning
    attempts: int = 0
    max_attempts: int = 2
    error: str | None = None


class Observation(BaseModel):
    ci_width: float
    budget_left: float  # normalized by Resources.run_budget
    last_review_score: float  # overall / 10
    open_tasks_by_type: dict[str, int] = {}

    def to_vector(self) -> list[float]:
        return [self.ci_width, self.budget_left, self.last_review_score,
                *(float(self.open_tasks_by_type.get(t, 0)) for t in TASK_TYPES), 1.0]


class Action(BaseModel):
    type: ActionType
    target_id: str | None = None


class RLStep(BaseModel):
    round: int
    policy: str  # "linucb" or a baseline name
    observation: list[float]
    action: Action
    reward: float


# ---------------------------------------------------------------- evaluation harness (Module D)

class Scenario(BaseModel):
    name: str
    request: ResearchRequest
    kb_fixture: str | None = None  # prebuilt KB to skip live search
    simulator_params: dict[str, float] = {}  # true_effect, noise, p_fail, ...


class ScenarioReport(BaseModel):
    scenario: str
    policy: str
    metrics: dict[str, float]
    traces: list[str] = []


# ---------------------------------------------------------------- LangGraph state

class LabState(TypedDict, total=False):
    # input and loop control (F.RL)
    request: ResearchRequest
    round: int
    budget_left: float
    last_action: Action | None
    stop_reason: str | None
    rl_log: Annotated[list[RLStep], operator.add]
    # workflow tracker (F.1)
    facts: Annotated[dict[str, bool], upsert]
    tasks: Annotated[dict[str, Task], upsert]
    # B (literature)
    kb: KBHandle
    paper_reviews: Annotated[dict[str, PaperReview], upsert]  # keyed by internal paper_id ("P1")
    conflicts: Annotated[dict[str, Conflict], upsert]
    # C
    gaps: Annotated[dict[str, Gap], upsert]
    hypotheses: Annotated[dict[str, Hypothesis], upsert]
    # F.CSP
    proposals: Annotated[dict[str, Proposal], upsert]
    # D
    experiments: Annotated[dict[str, ExperimentResult], upsert]
    datasets: Annotated[dict[str, DatasetRecord], upsert]
    # E
    evidence: Annotated[dict[str, Evidence], upsert]
    beliefs: Annotated[dict[str, BeliefState], upsert]  # keyed by hypothesis_id
    reviews: Annotated[list[Review], operator.add]
    revision_tasks: Annotated[dict[str, RevisionTask], upsert]
    # A (tooling)
    ledger: Annotated[list[LedgerEntry], operator.add]
    report_path: str | None