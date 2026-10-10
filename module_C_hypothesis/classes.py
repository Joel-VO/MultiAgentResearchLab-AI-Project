from typing import Literal


class Gap:
    id: str
    source: Literal["sparse_region",
                    "missing_combination", "conflict", "future_work"]
    description: str
    paper_ids: list[str]


class Hypothesis:
    id: str
    gap_ids: list[str]
    statement: str
    testable_prediction: str
    rationale: str
    supporting_paper_ids: list[str]
    assumptions: list[str]
    # novelty, evidence_support, feasibility, llm_judge
    score_parts: dict[str, float]
    score: float
    status: Literal["candidate", "selected", "rejected", "revised"]


# Dummy classes of module B

class ResearchRequest:
    topic: str
    seed_papers: list[str]  # DOI, arXiv ID, or title
    #TODO: budget: Budget  # max_papers, max_depth, fanout_per_node


class KBHandle:
    graph_path: str
    emb_path: str
    # query functions: neighbors(id), closest_to(text_or_id, k), citation_path(a, b)


class PaperReview:
    paper_id: str
    idea: str
    method: str
    pros: list[str]
    cons: list[str]
    future_work: list[str]  # "not stated" allowed in every text field
    concepts: list[str]
    methods: list[str]
    datasets: list[str]
    metrics: list[str]


class Conflict:
    id: str
    type: Literal["finding", "reviewer"]
    paper_ids: list[str]
    summary: str
    evidence_text: str
