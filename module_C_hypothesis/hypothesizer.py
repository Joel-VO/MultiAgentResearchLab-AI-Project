from module_A_tooling.local_llm_inference import OllamaClient
from classes import Gap, Hypothesis, KBHandle, PaperReview, Conflict, ResearchRequest


def mine_gaps(kb: KBHandle, reviews: list[PaperReview], conflict: list[Conflict], request: ResearchRequest) -> list[Gap]:
    # What format is the KB
    pass


def hypothesise(kb: KBHandle, reviews: list[PaperReview], conflict: list[Conflict], request: ResearchRequest, gaps: list[Gap]) -> list[Hypothesis]:
    pass
