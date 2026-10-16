from typing import Literal, Optional
from pydantic import BaseModel, Field


class Gap(BaseModel):
    id: str
    source: Literal["sparse_region",
                    "missing_combination", "conflict", "future_work"]
    description: str
    paper_ids: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str
    gap_ids: list[str] = Field(default_factory=list)
    statement: str
    testable_prediction: str
    rationale: str
    supporting_paper_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    # novelty, evidence_support, feasibility, llm_judge
    score_parts: dict[str, float] = Field(default_factory=dict)
    score: float
    status: Literal["candidate", "selected", "rejected", "revised"]


# Dummy classes of module B

class ResearchRequest:
    topic: str
    seed_papers: list[str]  # DOI, arXiv ID, or title
    # TODO: budget: Budget  # max_papers, max_depth, fanout_per_node
