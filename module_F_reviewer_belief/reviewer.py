"""
Module F.2: Reviewer
--------------------
Writes a structured Review of a proposal, its experiment results and the current belief
(DESIGN.md Module F.2). The reviewer model fills a draft; IDs are assigned in code. If the
model fails, a rule-based review is returned instead so the loop never stops here.
"""
from typing import Literal, Optional

from pydantic import BaseModel

from config import LLM_RETRIES, REVIEWER_MODEL
from module_A_tooling.local_llm_inference import OllamaClient, llm_structured
from module_A_tooling.schemas import (BeliefState, ExperimentResult, Hypothesis, Proposal, Review,
                                      RevisionTask)
from module_F_reviewer_belief.belief import paired_interval

SCORE_KEYS = ["novelty", "soundness", "evidence", "clarity", "overall"]


# What the LLM fills. IDs and cost are added in code.
class TaskDraft(BaseModel):
    text: str
    type: Literal["baseline", "ablation", "analysis", "revise_hypothesis", "other"]
    severity: Literal["minor", "major", "critical"]


class Scores(BaseModel):
    novelty: float
    soundness: float
    evidence: float
    clarity: float
    overall: float


class ReviewDraft(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    questions: list[str]
    closed_task_ids: list[str]   # IDs of open tasks that the results now address
    new_tasks: list[TaskDraft]
    scores: Scores


REVIEW_SYSTEM = (
    "You are a strict but fair peer reviewer in an AI research lab. Review the proposal and the "
    "experiment results listed. Give 2-3 strengths and 2-3 weaknesses. Each weakness says why it "
    "matters and refers to a listed experiment ID (e.g. E1) or starts with 'Assumption:'. Never "
    "mention experiments that are not listed. The belief line is the lab's own Bayesian estimate "
    "that the hypothesis holds; use it, do not criticise it.\n"
    "closed_task_ids: IDs of open tasks that the listed results now address.\n"
    "new_tasks: at most 3, only for problems not covered by an open task. Task types: "
    "baseline = add or strengthen a baseline; ablation = remove a component to test its "
    "contribution; analysis = break down existing results (errors, categories); "
    "revise_hypothesis = the claim itself must change; other = anything else. "
    "Severity: critical blocks any conclusion, major weakens it, minor is polish.\n"
    "Scores are 0-10."
)


def summarise(hypothesis: Hypothesis, proposal: Proposal, results: list[ExperimentResult],
              belief: BeliefState, open_tasks: list[RevisionTask]) -> str:
    """Compact text the reviewer model reads. Numbers are pre-computed so the model doesn't do maths."""
    lines = [
        f"Hypothesis {hypothesis.id}: {hypothesis.statement}",
        f"Prediction: {hypothesis.testable_prediction}",
        f"Proposal {proposal.id}: method '{proposal.method}' vs {proposal.baselines} on "
        f"{proposal.dataset} ({proposal.setting}), metric {proposal.metric}.",
        "Experiments:",
    ]
    for r in results:
        if r.status == "failed":
            lines.append(f"- {r.id} [{r.kind}] FAILED: {r.error}")
            continue
        interval = paired_interval(r)
        gain = (f"mean gain {interval[0]:+.4f}, 90% CI [{interval[1]:+.4f}, {interval[2]:+.4f}]"
                if interval else "no paired comparison")
        lines.append(f"- {r.id} [{r.kind}] {len(r.seeds)} seeds, {gain}. Design: {r.design}")
    lines.append(f"Belief: mean {belief.mean:.2f}, 90% CI [{belief.ci_low:.2f}, {belief.ci_high:.2f}], "
                 f"total weight {belief.total_weight:.2f}, "
                 f"{'PRELIMINARY' if belief.preliminary else 'not preliminary'}.")
    lines.append("Open tasks:" if open_tasks else "Open tasks: none")
    lines += [f"- {t.id} [{t.type}, {t.severity}] {t.text}" for t in open_tasks]
    return "\n".join(lines)


def _next_task_id(open_tasks: list[RevisionTask], used: set[str]) -> str:
    n = 1
    while f"T{n}" in used or any(t.id == f"T{n}" for t in open_tasks):
        n += 1
    used.add(f"T{n}")
    return f"T{n}"


def addressed_task_ids(results: list[ExperimentResult], open_tasks: list[RevisionTask]) -> set[str]:
    """Open tasks that a successful run was made for."""
    done = {r.task_id for r in results if r.status == "ok" and r.task_id}
    return {t.id for t in open_tasks if t.id in done}


def fallback_review(hypothesis: Hypothesis, results: list[ExperimentResult], belief: BeliefState,
                    open_tasks: list[RevisionTask], review_id: str) -> Review:
    """Rule-based review used when the LLM is unavailable or keeps returning invalid output."""
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status == "failed"]
    used: set[str] = set()
    tasks, weaknesses = [], []
    if failed:
        weaknesses.append(f"Run {failed[-1].id} failed ({failed[-1].error}); its result is not evidence.")
    if belief.preliminary:
        weaknesses.append("Belief is still preliminary (wide interval or little evidence weight).")
        if not any(t.type == "baseline" for t in open_tasks):
            tasks.append(RevisionTask(id=_next_task_id(open_tasks, used), type="baseline",
                                      severity="major", text="Add a stronger baseline to confirm the gain."))
    overall = round(min(10.0, 3 + 6 * belief.ci_low + 2 * len(ok) / max(1, len(results))), 1)
    return Review(
        id=review_id, hypothesis_id=hypothesis.id,
        strengths=[f"{len(ok)} successful run(s) with paired seeds."] if ok else [],
        weaknesses=weaknesses, questions=[],
        revision_tasks=tasks,
        closed_task_ids=sorted(addressed_task_ids(results, open_tasks)),
        scores={k: overall for k in SCORE_KEYS},
    )


def review(hypothesis: Hypothesis,
           proposal: Proposal,
           results: list[ExperimentResult],
           belief: BeliefState,
           open_tasks: Optional[list[RevisionTask]] = None,
           client: Optional[OllamaClient] = None,
           review_id: str = "R1") -> Review:
    """
    Reviews a proposal and its results.

    Args:
        hypothesis, proposal: what is being tested.
        results: all ExperimentResults for this proposal so far.
        belief: current BeliefState of the hypothesis.
        open_tasks: RevisionTasks from earlier reviews that are not closed yet.
        client: Ollama client; created with reasoning disabled if not given.
        review_id: ID for this review, e.g. "R1".
    """
    open_tasks = open_tasks or []
    client = client or OllamaClient(model=REVIEWER_MODEL, think=False)
    prompt = summarise(hypothesis, proposal, results, belief, open_tasks)
    try:
        draft = llm_structured(prompt, ReviewDraft, client, system=REVIEW_SYSTEM,
                               retries=LLM_RETRIES, temperature=0, num_predict=1024)
    except Exception as e:
        print(f"[Module F] reviewer LLM failed ({type(e).__name__}), using rule-based review.")
        return fallback_review(hypothesis, results, belief, open_tasks, review_id)

    used: set[str] = set()
    open_ids = {t.id for t in open_tasks}
    closed = {i for i in draft.closed_task_ids if i in open_ids} | addressed_task_ids(results, open_tasks)
    new_tasks = [RevisionTask(id=_next_task_id(open_tasks, used), text=t.text, type=t.type,
                              severity=t.severity) for t in draft.new_tasks[:3]]
    return Review(
        id=review_id, hypothesis_id=hypothesis.id,
        strengths=draft.strengths, weaknesses=draft.weaknesses, questions=draft.questions,
        revision_tasks=new_tasks,
        closed_task_ids=sorted(closed),
        scores={k: min(10.0, max(0.0, v)) for k, v in draft.scores.model_dump().items()},
    )
