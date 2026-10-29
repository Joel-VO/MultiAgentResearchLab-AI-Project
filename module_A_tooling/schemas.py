"""
Shared types (DESIGN.md Section 3 and Appendix). Single source of truth.

Only the types used by Modules E and F live here for now; the other modules
move theirs in during integration. Paper types (PaperMeta, PaperReview) are
still defined in module_B_literature_kg/interface.py.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- common


class LedgerEntry(BaseModel):
    id: str
    claim: str
    tag: Literal["fact", "hypothesis", "assumption", "speculation"]
    source_ids: list[str] = Field(default_factory=list)   # paper IDs or experiment IDs
    produced_by: str                                       # module / agent name
    confidence: float = 1.0
    hypothesis_id: Optional[str] = None


class RevisionTask(BaseModel):
    id: str
    text: str
    type: Literal["baseline", "ablation", "analysis", "revise_hypothesis", "other"]
    severity: Literal["minor", "major", "critical"]
    est_cost: float = 0.1


# --------------------------------------------------------------------------- C


class Hypothesis(BaseModel):
    id: str
    gap_ids: list[str] = Field(default_factory=list)
    statement: str
    testable_prediction: str
    rationale: str = ""
    supporting_paper_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    score_parts: dict[str, float] = Field(default_factory=dict)  # novelty, evidence_support, feasibility, llm_judge
    score: float = 0.0
    status: Literal["candidate", "selected", "rejected", "revised"] = "candidate"


# --------------------------------------------------------------------------- D.CSP


class ConstraintReport(BaseModel):
    satisfied: list[str] = Field(default_factory=list)
    violated: list[str] = Field(default_factory=list)


class Proposal(BaseModel):
    id: str
    hypothesis_id: str
    research_question: str
    dataset: str
    method: str
    baselines: list[str]
    metric: str
    setting: str = ""
    constraint_report: ConstraintReport = Field(default_factory=ConstraintReport)


# --------------------------------------------------------------------------- E


class ExperimentResult(BaseModel):
    id: str
    proposal_id: str
    hypothesis_id: str
    kind: Literal["main", "baseline", "ablation", "analysis"] = "main"
    design: str
    mode: Literal["real", "simulated"]
    metrics: dict[str, list[float]] = Field(default_factory=dict)            # per seed
    baseline_metrics: dict[str, list[float]] = Field(default_factory=dict)   # per seed
    higher_is_better: bool = True
    n_samples: int = 0
    seeds: list[int] = Field(default_factory=list)
    status: Literal["ok", "failed"]
    error: Optional[str] = None
    task_id: Optional[str] = None          # RevisionTask this run addresses, if any
    artifacts: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- F


class Evidence(BaseModel):
    id: str                                # "{hypothesis_id}:{source_id}"
    hypothesis_id: str
    source_id: str                         # paper ID or experiment ID
    direction: Literal["support", "contradict", "neutral"]
    type: Literal["direct", "indirect"]
    reliability: float                     # 0-1
    similarity: float                      # 0-1, setting similarity
    quote: Optional[str] = None            # literature evidence only


class EvidenceLabel(BaseModel):
    direction: Literal["support", "contradict", "neutral"]
    type: Literal["direct", "indirect"]
    quote: str                             # sentence from the abstract that justifies the label


class BeliefState(BaseModel):
    hypothesis_id: str
    alpha: float
    beta: float
    mean: float
    ci_low: float                          # 90% credible interval
    ci_high: float
    total_weight: float
    preliminary: bool


class Review(BaseModel):
    id: str
    hypothesis_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)       # each states why the weakness matters
    questions: list[str] = Field(default_factory=list)
    revision_tasks: list[RevisionTask] = Field(default_factory=list)
    closed_task_ids: list[str] = Field(default_factory=list)  # open tasks this review judges addressed
    scores: dict[str, float] = Field(default_factory=dict)    # novelty, soundness, evidence, clarity, overall (0-10)
