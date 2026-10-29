"""
Module F.1: Belief model
------------------------
Beta-Bernoulli belief per hypothesis (DESIGN.md Module F.1).

    experiment_evidence(result)        ExperimentResult -> Evidence (no LLM)
    literature_evidence(h, paper, ...) paper + LLM label  -> Evidence (quote-checked)
    compute_belief(h_id, evidence)     list[Evidence]     -> BeliefState
    rank(beliefs)                      highest ci_low first
"""
import math
from typing import Literal, Optional

from pydantic import BaseModel
from scipy import stats

from config import C_REF, CI_WIDTH_MAX, GENERATOR_MODEL, LLM_RETRIES, MIN_WEIGHT, REF_SEEDS
from module_A_tooling.local_llm_inference import OllamaClient, llm_structured
from module_A_tooling.schemas import (BeliefState, Evidence, EvidenceLabel, ExperimentResult,
                                      Hypothesis, Proposal)
from module_B_literature_kg.interface import PaperMeta, PaperReview

TYPE_W = {"direct": 1.0, "indirect": 0.3}


# --------------------------------------------------------------------------- belief


def evidence_weight(e: Evidence) -> float:
    return TYPE_W[e.type] * e.reliability * e.similarity


def compute_belief(h_id: str, evidence: list[Evidence]) -> BeliefState:
    """Recomputes the belief for one hypothesis from the full evidence list."""
    a, b = 1.0, 1.0
    for e in evidence:
        if e.hypothesis_id != h_id:
            continue
        if e.direction == "support":
            a += evidence_weight(e)
        elif e.direction == "contradict":
            b += evidence_weight(e)
    lo, hi = stats.beta.ppf([0.05, 0.95], a, b)
    total = a + b - 2.0
    return BeliefState(
        hypothesis_id=h_id, alpha=a, beta=b, mean=a / (a + b),
        ci_low=float(lo), ci_high=float(hi), total_weight=total,
        preliminary=(hi - lo > CI_WIDTH_MAX) or (total < MIN_WEIGHT),
    )


def rank(beliefs: list[BeliefState]) -> list[BeliefState]:
    """Highest lower credible bound first, so thinly evidenced hypotheses don't rank on mean alone."""
    return sorted(beliefs, key=lambda b: b.ci_low, reverse=True)


# --------------------------------------------------------------------------- experiment evidence


def paired_interval(result: ExperimentResult) -> Optional[tuple[float, float, float]]:
    """Mean of per-seed differences (method - baseline, flipped if lower is better)
    and its 90% t-interval. None if there is nothing to compare."""
    if result.status != "ok" or not result.metrics:
        return None
    metric = next(iter(result.metrics))
    method = result.metrics[metric]
    baseline = result.baseline_metrics.get(metric, [])
    if len(method) != len(baseline) or len(method) < 2:
        return None
    sign = 1.0 if result.higher_is_better else -1.0
    d = [sign * (m - b) for m, b in zip(method, baseline)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = stats.t.ppf(0.95, n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def experiment_evidence(result: ExperimentResult) -> Optional[Evidence]:
    """
    One Evidence item from an experiment, or None if the run is not evidence
    (failed run, analysis run, or no paired metrics).

    Interval above 0 -> support, below 0 -> contradict, containing 0 -> neutral.
    For an ablation, metrics are the full method and baseline_metrics the method with the
    component removed, so a drop on removal also reads as support.
    """
    if result.kind == "analysis":
        return None
    interval = paired_interval(result)
    if interval is None:
        return None
    _, lo, hi = interval
    direction = "support" if lo > 0 else "contradict" if hi < 0 else "neutral"
    return Evidence(
        id=f"{result.hypothesis_id}:{result.id}", hypothesis_id=result.hypothesis_id,
        source_id=result.id, direction=direction, type="direct",
        reliability=min(1.0, len(result.seeds) / REF_SEEDS), similarity=1.0,
    )


# --------------------------------------------------------------------------- literature evidence


class LabelDraft(BaseModel):
    """What the LLM fills: a one-line reason first (helps small models), then the EvidenceLabel fields."""
    reason: str
    direction: Literal["support", "contradict", "neutral"]
    type: Literal["direct", "indirect"]
    quote: str


LABEL_SYSTEM = (
    "You decide whether a paper's findings make a hypothesis more likely (support), less likely "
    "(contradict), or say nothing about it (neutral). Judge the hypothesis's exact claim: a paper "
    "showing that the hypothesis's baseline works well is not evidence against the hypothesis. "
    "type is direct only if the paper tests the same intervention with the same kind of outcome; "
    "a related mechanism, a different task or a weaker proxy is indirect. "
    "quote must be one sentence copied exactly, word for word, from the abstract."
)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .\"'")


def quote_in_abstract(quote: str, abstract: str) -> bool:
    q = _normalise(quote)
    return bool(q) and q in _normalise(abstract)


def reliability(paper: PaperMeta) -> float:
    venue = 1.0 if paper.venue else 0.6
    cites = min(1.0, math.log10(1 + (paper.citation_count or 0)) / math.log10(1 + C_REF))
    return 0.5 * venue + 0.5 * cites


def similarity(review: Optional[PaperReview], proposal: Optional[Proposal]) -> float:
    """Jaccard overlap of the paper's setting with the proposal's. 0.5 when either is missing."""
    if review is None or proposal is None:
        return 0.5
    paper = {x.lower() for x in review.datasets + review.metrics + review.methods}
    prop = {x.lower() for x in [proposal.dataset, proposal.metric, proposal.method, *proposal.baselines]}
    union = paper | prop
    return len(paper & prop) / len(union) if union else 0.0


def literature_evidence(hypothesis: Hypothesis,
                        paper: PaperMeta,
                        review: Optional[PaperReview] = None,
                        proposal: Optional[Proposal] = None,
                        client: Optional[OllamaClient] = None) -> Optional[Evidence]:
    """
    Labels one paper against one hypothesis with the generator model.
    Returns None if the LLM fails or its quote is not found in the abstract.
    """
    client = client or OllamaClient(model=GENERATOR_MODEL, think=False)
    prompt = (f"Hypothesis: {hypothesis.statement}\n\n"
              f"Paper title: {paper.title}\n"
              f"Abstract: {paper.abstract}\n"
              + (f"Method: {review.method}\nDatasets: {', '.join(review.datasets)}\n" if review else ""))
    try:
        draft = llm_structured(prompt, LabelDraft, client, system=LABEL_SYSTEM,
                               retries=LLM_RETRIES, temperature=0, num_predict=300)
        label = EvidenceLabel(direction=draft.direction, type=draft.type, quote=draft.quote)
    except Exception as e:
        print(f"[Module F] label failed for {paper.paper_id}: {type(e).__name__}")
        return None
    if not quote_in_abstract(label.quote, paper.abstract):
        print(f"[Module F] label for {paper.paper_id} discarded: quote not in abstract")
        return None
    return Evidence(
        id=f"{hypothesis.id}:{paper.paper_id}", hypothesis_id=hypothesis.id,
        source_id=paper.paper_id, direction=label.direction, type=label.type,
        reliability=reliability(paper), similarity=similarity(review, proposal), quote=label.quote,
    )
