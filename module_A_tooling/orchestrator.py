"""
LangGraph orchestration of Modules B to F (DESIGN.md Section 6). First version: data passing only.

    START → B → C → D.CSP ─(no active proposal)──────────────────────────────→ A.report → END
                      └──→ F.Belief → E → F.Reviewer → D.controller ─(stop)──→ A.report
                                      ↑                     │
                                      └──(runnable task)────┘

Each node calls one module step through `Modules`, checks the output against its contract
(`conform`, INTEGRATION.md) and returns only the LabState keys it owns. The controller is a
rule-based placeholder for D.RL: it picks the most severe open baseline/ablation/analysis task.

Not wired yet: the D.1 tracker, D.3 RL, re-entering B/C, the ledger, a checkpointer,
human-in-the-loop (interrupt) and the dataset gate.

Run from the repo root:
    python -m module_A_tooling.orchestrator --topic "..." --seed 2305.12345 --seed "Self-Refine"
"""
import argparse
import functools
import importlib
import operator
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from config import DEFAULT_LICENSES, MAX_ROUNDS
from module_A_tooling.config import run_dir
from module_A_tooling.schemas import (BeliefState, Evidence, ExperimentResult, Hypothesis,
                                      LedgerEntry, Proposal, Review, RevisionTask)
from module_B_literature_kg.interface import (Conflict, KBHandle, ModuleBResult, PaperReview,
                                              open_kb)
from schemas import Gap, ResearchRequest, Resources, upsert

SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2}
RUNNABLE_TASK_TYPES = ("baseline", "ablation", "analysis")   # tasks an experiment can address
DOC = "module_A_tooling/INTEGRATION.md"                      # what each module must return


# --------------------------------------------------------------------------- contracts


class ProposalRejection(BaseModel):
    """D.CSP output for a hypothesis it cannot turn into a feasible proposal (DESIGN 10, D.2)."""
    hypothesis_id: str
    reasons: list[str]


class ContractError(RuntimeError):
    """A module's output does not match its contract, or its entry point cannot be loaded."""


def conform(value: Any, model: type[BaseModel], producer: str) -> BaseModel:
    """Checks a module's output at the hand-off, before it goes into LabState.

    Without this, a wrong shape (e.g. D returning `rejection_reasons` instead of `reasons`) is
    stored as is and only crashes later inside another module, pointing at the wrong code.
    Here the run stops at once with a ContractError naming the producer, the expected type and
    the field that failed.

    The output is dumped to a dict and re-validated as `model`, so fields are compared, not
    classes: C's own `classes.Hypothesis` passes and comes back as Module A's `Hypothesis`.
    Extra fields are dropped; missing or invalid ones fail. `conform_list` does the same for
    functions that return a list.
    """
    data = value.model_dump() if isinstance(value, BaseModel) else value
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ContractError(f"{producer} returned {type(value).__name__}, which does not match "
                            f"{model.__name__} (see {DOC}):\n{e}") from e


def conform_list(values: Any, model: type[BaseModel], producer: str) -> list:
    if not isinstance(values, (list, tuple)):
        raise ContractError(f"{producer} must return a list of {model.__name__}, "
                            f"got {type(values).__name__} (see {DOC})")
    return [conform(v, model, producer) for v in values]


# --------------------------------------------------------------------------- state


class LabState(TypedDict, total=False):
    """Graph state. Collections are dicts keyed by ID (merged with upsert); lists are append-only."""
    request: ResearchRequest
    round: int                                                    # finished review rounds
    stop_reason: Optional[str]
    # B
    kb: KBHandle
    paper_reviews: Annotated[dict[str, PaperReview], upsert]      # keyed by paper_id
    conflicts: Annotated[dict[str, Conflict], upsert]
    # C
    gaps: Annotated[dict[str, Gap], upsert]
    hypotheses: Annotated[dict[str, Hypothesis], upsert]
    # D.CSP
    proposals: Annotated[dict[str, Proposal], upsert]
    rejections: Annotated[list[ProposalRejection], operator.add]
    # F.Belief
    evidence: Annotated[dict[str, Evidence], upsert]
    beliefs: Annotated[dict[str, BeliefState], upsert]            # keyed by hypothesis_id
    selected_hypothesis_id: Optional[str]
    selected_proposal_id: Optional[str]
    # E
    experiments: Annotated[dict[str, ExperimentResult], upsert]
    # F.Reviewer
    reviews: Annotated[list[Review], operator.add]
    revision_tasks: Annotated[dict[str, RevisionTask], upsert]
    closed_task_ids: Annotated[list[str], operator.add]
    # D.controller
    next_task_id: Optional[str]
    # A
    ledger: Annotated[list[LedgerEntry], operator.add]
    report_path: Optional[str]


# --------------------------------------------------------------------------- module steps


def _entry(path: str) -> Callable:
    """`package.module:name`, imported on first call so one broken module does not stop the
    graph from being built; the error names the entry point instead."""
    module_name, name = path.split(":")

    def call(*args, **kwargs):
        try:
            fn = getattr(importlib.import_module(module_name), name)
        except Exception as e:
            raise ContractError(f"cannot load {path} ({type(e).__name__}: {e}); see {DOC}") from e
        return fn(*args, **kwargs)
    call.__name__ = path
    return call


@functools.cache
def _embedder():
    from module_B_literature_kg.b0.embed import Embedder
    return Embedder()


def embed_text(text: str):
    """Text -> unit vector with the KB's embedding model (open_kb's embed_fn)."""
    return _embedder().embed_texts([text])[0]


@dataclass
class Modules:
    """One callable per module step; the signatures are the contracts in INTEGRATION.md.
    Defaults are the real entry points. Tests pass stubs."""
    run_module_b: Callable = _entry("module_B_literature_kg.pipeline:run_module_b")
    mine_gaps: Callable = _entry("module_C_hypothesis.hypothesizer:mine_gaps")
    hypothesise: Callable = _entry("module_C_hypothesis.hypothesizer:hypothesise")
    build_proposals: Callable = _entry("module_D_workflow_csp_rl.csp:build_proposals")
    embed_text: Callable = embed_text
    literature_evidence: Callable = _entry("module_F_reviewer_belief.belief:literature_evidence")
    compute_belief: Callable = _entry("module_F_reviewer_belief.belief:compute_belief")
    rank: Callable = _entry("module_F_reviewer_belief.belief:rank")
    experiment_evidence: Callable = _entry("module_F_reviewer_belief.belief:experiment_evidence")
    sim_params: Callable = _entry("module_E_experimenter.runner:SimParams")
    run_experiment: Callable = _entry("module_E_experimenter.runner:run_experiment")
    review: Callable = _entry("module_F_reviewer_belief.reviewer:review")
    generate_report: Callable = lambda state: write_placeholder_report(state)  # noqa: E731


# --------------------------------------------------------------------------- helpers


def _by_id(items: list) -> dict:
    return {x.id: x for x in items}


def _open_tasks(state: LabState) -> list[RevisionTask]:
    closed = set(state.get("closed_task_ids", []))
    return [t for t in state.get("revision_tasks", {}).values() if t.id not in closed]


# --------------------------------------------------------------------------- graph


def build_lab_graph(modules: Optional[Modules] = None, max_rounds: int = MAX_ROUNDS):
    m = modules or Modules()

    def literature(state: LabState) -> dict:
        out = conform(m.run_module_b(state["request"]), ModuleBResult, "Module B run_module_b")
        return {"kb": out.kb,
                "paper_reviews": {r.paper_id: r for r in out.reviews},
                "conflicts": _by_id(out.conflicts),
                "ledger": conform_list(out.ledger_entries, LedgerEntry, "Module B ledger_entries")}

    def hypotheses(state: LabState) -> dict:
        args = (state["kb"], list(state.get("paper_reviews", {}).values()),
                list(state.get("conflicts", {}).values()), state["request"])
        gaps = conform_list(m.mine_gaps(*args), Gap, "Module C mine_gaps")
        hyps = conform_list(m.hypothesise(*args, gaps), Hypothesis, "Module C hypothesise")
        return {"gaps": _by_id(gaps), "hypotheses": _by_id(hyps)}

    def csp(state: LabState) -> dict:
        kb = open_kb(state["kb"], embed_fn=m.embed_text)
        out = m.build_proposals(list(state.get("hypotheses", {}).values()), kb,
                                state["request"].resources)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise ContractError("Module D build_proposals must return "
                                f"(accepted: list[Proposal], rejected: list[ProposalRejection]) (see {DOC})")
        accepted = conform_list(out[0], Proposal, "Module D build_proposals (accepted)")
        rejected = conform_list(out[1], ProposalRejection, "Module D build_proposals (rejected)")
        return {"proposals": _by_id(accepted), "rejections": rejected}

    def belief(state: LabState) -> dict:
        kb = open_kb(state["kb"])
        reviews = state.get("paper_reviews", {})
        proposal_for: dict[str, Proposal] = {}
        for p in state["proposals"].values():
            proposal_for.setdefault(p.hypothesis_id, p)
        evidence = dict(state.get("evidence", {}))
        beliefs = {}
        for hid, proposal in proposal_for.items():
            h = state["hypotheses"][hid]
            for pid in h.supporting_paper_ids:
                try:
                    paper = kb.get_paper(pid)
                except KeyError:
                    continue
                e = m.literature_evidence(h, paper, reviews.get(pid) or kb.get_review(pid), proposal)
                if e is not None:
                    e = conform(e, Evidence, "Module F literature_evidence")
                    evidence[e.id] = e
            beliefs[hid] = conform(m.compute_belief(hid, list(evidence.values())), BeliefState,
                                   "Module F compute_belief")
        top = conform(m.rank(list(beliefs.values()))[0], BeliefState, "Module F rank").hypothesis_id
        return {"evidence": evidence, "beliefs": beliefs,
                "selected_hypothesis_id": top, "selected_proposal_id": proposal_for[top].id,
                "hypotheses": {top: state["hypotheses"][top].model_copy(update={"status": "selected"})}}

    def experiment(state: LabState) -> dict:
        hid, pid = state["selected_hypothesis_id"], state["selected_proposal_id"]
        task_id = state.get("next_task_id")
        n = len(state.get("experiments", {})) + 1
        result = conform(m.run_experiment(state["proposals"][pid], state["hypotheses"][hid],
                                          task=state["revision_tasks"][task_id] if task_id else None,
                                          sim=m.sim_params(rng_seed=n), result_id=f"E{n}"),
                         ExperimentResult, "Module E run_experiment")
        update: dict = {"experiments": {result.id: result}}
        evidence = dict(state.get("evidence", {}))
        e = m.experiment_evidence(result)
        if e is not None:
            e = conform(e, Evidence, "Module F experiment_evidence")
            evidence[e.id] = e
            update["evidence"] = {e.id: e}
        update["beliefs"] = {hid: conform(m.compute_belief(hid, list(evidence.values())),
                                          BeliefState, "Module F compute_belief")}
        return update

    def reviewer(state: LabState) -> dict:
        hid, pid = state["selected_hypothesis_id"], state["selected_proposal_id"]
        results = [r for r in state.get("experiments", {}).values() if r.proposal_id == pid]
        n = len(state.get("reviews", [])) + 1
        rv = conform(m.review(state["hypotheses"][hid], state["proposals"][pid], results,
                              state["beliefs"][hid], open_tasks=_open_tasks(state),
                              review_id=f"R{n}"),
                     Review, "Module F review")
        known = state.get("revision_tasks", {})
        reused = [t.id for t in rv.revision_tasks if t.id in known]
        if reused:
            raise ContractError(f"Module F review {rv.id} reused revision task IDs {reused}; "
                                f"task IDs must be unique across the run (see {DOC})")
        closed = set(state.get("closed_task_ids", []))
        return {"reviews": [rv], "revision_tasks": _by_id(rv.revision_tasks),
                "closed_task_ids": [i for i in rv.closed_task_ids if i not in closed],
                "round": state.get("round", 0) + 1}

    def controller(state: LabState) -> dict:
        """Placeholder for D.RL: run the most severe open task an experiment can address."""
        if state.get("round", 0) >= max_rounds:
            return {"next_task_id": None, "stop_reason": f"reached {max_rounds} rounds"}
        runnable = [t for t in _open_tasks(state) if t.type in RUNNABLE_TASK_TYPES]
        if not runnable:
            return {"next_task_id": None, "stop_reason": "no open task an experiment can address"}
        return {"next_task_id": min(runnable, key=lambda t: SEVERITY_RANK[t.severity]).id}

    def report(state: LabState) -> dict:
        stop = state.get("stop_reason") or ("no hypothesis passed the CSP" if not state.get("proposals") else None)
        return {"report_path": str(m.generate_report({**state, "stop_reason": stop})), "stop_reason": stop}

    g = StateGraph(LabState)
    g.add_node("B_literature", literature)
    g.add_node("C_hypotheses", hypotheses)
    g.add_node("D_csp", csp)
    g.add_node("F_belief", belief)
    g.add_node("E_experiment", experiment)
    g.add_node("F_review", reviewer)
    g.add_node("D_controller", controller)
    g.add_node("A_report", report)
    g.add_edge(START, "B_literature")
    g.add_edge("B_literature", "C_hypotheses")
    g.add_edge("C_hypotheses", "D_csp")
    g.add_conditional_edges("D_csp", lambda s: "F_belief" if s.get("proposals") else "A_report",
                            ["F_belief", "A_report"])
    g.add_edge("F_belief", "E_experiment")
    g.add_edge("E_experiment", "F_review")
    g.add_edge("F_review", "D_controller")
    g.add_conditional_edges("D_controller", lambda s: "E_experiment" if s.get("next_task_id") else "A_report",
                            ["E_experiment", "A_report"])
    g.add_edge("A_report", END)
    return g.compile()


def run_lab(request: ResearchRequest, modules: Optional[Modules] = None,
            max_rounds: int = MAX_ROUNDS) -> LabState:
    return build_lab_graph(modules, max_rounds).invoke({"request": request, "round": 0})


# --------------------------------------------------------------------------- report (placeholder)


def write_placeholder_report(state: LabState, out_dir: Optional[Path] = None) -> Path:
    """Stands in for Module A's generate_report (DESIGN 10, A): a plain dump of the run, not
    built from the ledger yet."""
    request = state["request"]
    beliefs = state.get("beliefs", {})
    lines = [f"# Research report (placeholder): {request.topic}", "",
             f"Request {request.id}. Rounds: {state.get('round', 0)}. "
             f"Stopped: {state.get('stop_reason') or 'n/a'}.", "", "## Hypotheses"]
    for h in state.get("hypotheses", {}).values():
        b = beliefs.get(h.id)
        belief = (f" Belief {b.mean:.2f}, 90% CI [{b.ci_low:.2f}, {b.ci_high:.2f}]"
                  f"{', PRELIMINARY' if b.preliminary else ''}." if b else "")
        lines.append(f"- {h.id} [{h.status}] {h.statement}{belief}")
    lines += ["", "## Proposals"]
    lines += [f"- {p.id} for {p.hypothesis_id}: '{p.method}' vs {p.baselines} on {p.dataset}, "
              f"metric {p.metric}" for p in state.get("proposals", {}).values()]
    lines += [f"- rejected {r.hypothesis_id}: {'; '.join(r.reasons)}" for r in state.get("rejections", [])]
    lines += ["", "## Experiments"]
    lines += [f"- {x.id} [{x.kind}] {x.status}" + (f" for {x.task_id}" if x.task_id else "")
              + (f": {x.error}" if x.error else "") for x in state.get("experiments", {}).values()]
    lines += ["", "## Reviews"]
    for rv in state.get("reviews", []):
        lines.append(f"- {rv.id}: overall {rv.scores.get('overall')}/10"
                     + (f", closed {rv.closed_task_ids}" if rv.closed_task_ids else ""))
        lines += [f"  - {t.id} [{t.type}, {t.severity}] {t.text}" for t in rv.revision_tasks]
    path = Path(out_dir or run_dir(request.id)) / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- command line


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m module_A_tooling.orchestrator",
                                 description="Run the research lab graph (B -> C -> D -> F -> E -> F -> report).")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--seed", action="append", required=True, help="seed paper: arXiv id, DOI or title (repeatable)")
    ap.add_argument("--gpu-hours", type=float, default=10.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--license", action="append", help=f"allowed dataset license (repeatable, default {DEFAULT_LICENSES})")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = ap.parse_args(argv)
    request = ResearchRequest(topic=args.topic, seed_papers=args.seed,
                              resources=Resources(compute_budget_gpu_hours=args.gpu_hours,
                                                  time_budget_hours=args.hours,
                                                  licenses_allowed=args.license or DEFAULT_LICENSES))
    try:
        state = run_lab(request, max_rounds=args.max_rounds)
    except ContractError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Report: {state['report_path']} (stopped: {state.get('stop_reason')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
