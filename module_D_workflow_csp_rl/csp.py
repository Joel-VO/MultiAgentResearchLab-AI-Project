"""D.2: CSP proposal construction. Deterministic, no LLM, no other module's internals.

Input : Hypothesis, a KB (anything with .graph and .closest_to_text, i.e. the object
        returned by module_B_literature_kg.interface.open_kb), Resources.
Output: always a schemas.Proposal. A rejected hypothesis comes back as
        Proposal(status="rejected", rejection_reasons=[...]) so it can be stored in
        LabState.proposals and sent back to Module C.

Variables: dataset, metric, method, baseline.   Domains: KB concept-layer nodes.
Solver   : unary domain pruning, then backtracking with value ordering (best first).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from schemas import ConstraintReport, Hypothesis, Proposal, Resources

# ---- tunables (move to config when the team has one) -----------------------
NOVELTY_MAX_SIM = 0.85        # reject if cosine to any KB paper >= this
MIN_METHOD_RELEVANCE = 0.3    # fraction of a method's name tokens found in the hypothesis
MIN_SEEDS = 3
DEFAULT_DATASET_GPU_H = 1.0   # used only when the catalog has no cost for a dataset
METHOD_RUN_GPU_H = 0.5        # per method run (the method and the baseline each run once)
LOWER_IS_BETTER = {"loss", "error", "perplexity", "latency", "wer", "cer", "mse", "mae", "rmse"}


@dataclass
class DatasetInfo:
    """Facts the KB does not hold. Pass a {normalised dataset name: DatasetInfo} catalog.
    Datasets missing from the catalog get license 'unknown' and therefore fail closed."""
    license: str = "unknown"
    gpu_hours: float = DEFAULT_DATASET_GPU_H
    sensitive: bool = False        # personal data etc.
    ethics_approved: bool = False  # True once the sensitive use has been cleared


# ---- helpers ----------------------------------------------------------------
def _tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def relevance(name: str, text: str) -> float:
    """Fraction of the name's tokens that appear in text (cheap, LLM-free)."""
    a = _tokens(name)
    return len(a & _tokens(text)) / len(a) if a else 0.0


def kb_domains(graph) -> dict[str, dict[str, str]]:
    """{type: {node_id: display_name}} from the concept layer of the KB graph."""
    dom: dict[str, dict[str, str]] = {"dataset": {}, "method": {}, "metric": {}}
    for n, d in graph.nodes(data=True):
        t = d.get("type")
        if t in dom:
            dom[t][n] = d.get("name") or str(n).split(":", 1)[-1]
    return dom


def papers_by_concept(graph) -> dict[str, set[str]]:
    """concept node -> papers that use it (paper --has_*--> concept edges)."""
    out: dict[str, set[str]] = {}
    for u, v, d in graph.edges(data=True):
        if d.get("kind") in ("has_method", "has_dataset", "has_metric"):
            out.setdefault(v, set()).add(u)
    return out


# ---- backtracking solver -----------------------------------------------------
ORDER = ("dataset", "metric", "method", "baseline")


@dataclass
class Constraint:
    name: str
    vars: tuple[str, ...]
    fn: Callable[..., bool]


def _solve(domains: dict[str, list[str]], constraints: list[Constraint],
           fail_count: dict[str, int]) -> Optional[dict[str, str]]:
    def ok(assign: dict[str, str]) -> bool:
        newest = ORDER[len(assign) - 1]
        for c in constraints:
            if newest in c.vars and all(v in assign for v in c.vars):
                if not c.fn(*(assign[v] for v in c.vars)):
                    fail_count[c.name] = fail_count.get(c.name, 0) + 1
                    return False
        return True

    def bt(assign: dict[str, str]) -> Optional[dict[str, str]]:
        if len(assign) == len(ORDER):
            return dict(assign)
        var = ORDER[len(assign)]
        for val in domains[var]:
            assign[var] = val
            if ok(assign):
                res = bt(assign)
                if res:
                    return res
            del assign[var]
        return None

    return bt({})


# ---- main entry --------------------------------------------------------------
def build_proposal(hyp: Hypothesis, kb, resources: Resources,
                   catalog: Optional[dict[str, DatasetInfo]] = None) -> Proposal:
    catalog = catalog or {}
    reasons: list[str] = []
    text = f"{hyp.statement} {hyp.testable_prediction}"

    def reject(rs: list[str]) -> Proposal:
        return Proposal(hypothesis_id=hyp.id, status="rejected", rejection_reasons=rs,
                        research_question=hyp.testable_prediction or hyp.statement)

    # Role hint: "X improves Y over/than/versus Z" -> X is the method, Z the baseline.
    parts = re.split(r"\b(?:over|than|versus|vs\.?|compared (?:to|with)|beats)\b",
                     hyp.statement, maxsplit=1)
    main_text = parts[0]
    base_text = parts[1] if len(parts) > 1 else text

    # 1. Novelty (cheapest check, so it goes first)
    near = kb.closest_to_text(hyp.statement, 1)
    if near and near[0].score >= NOVELTY_MAX_SIM:
        return reject([f"novelty: closest KB paper {near[0].paper_id} has cosine "
                       f"{near[0].score:.2f} >= {NOVELTY_MAX_SIM}"])

    graph = kb.graph
    dom = kb_domains(graph)
    papers = papers_by_concept(graph)
    info = lambda ds: catalog.get(dom["dataset"][ds].lower(), DatasetInfo())  # noqa: E731
    shared = lambda a, b: bool(papers.get(a, set()) & papers.get(b, set()))   # noqa: E731
    budget = resources.compute_budget_gpu_hours

    # 2. Unary pruning (each emptied domain is reported)
    def prune(var, keep, label):
        before = len(dom[var])
        dom[var] = {k: v for k, v in dom[var].items() if keep(k)}
        if before and not dom[var]:
            reasons.append(f"{label}: removed all {before} candidate {var}s")

    allowed = {l.lower() for l in resources.licenses_allowed}
    prune("dataset", lambda d: info(d).license.lower() in allowed, "license")
    prune("dataset", lambda d: info(d).gpu_hours <= budget, "compute budget")
    prune("dataset", lambda d: not info(d).sensitive or info(d).ethics_approved, "ethics")
    prune("method", lambda m: relevance(dom["method"][m], text) >= MIN_METHOD_RELEVANCE,
          "testability (method not related to the hypothesis)")

    for var in ("dataset", "metric", "method"):
        if not dom[var] and not any(var in r for r in reasons):
            reasons.append(f"domain: KB has no {var} nodes")
    if any(not dom[v] for v in ("dataset", "metric", "method")):
        return reject(reasons)

    # 3. Domains with value ordering. A method named in the main clause is the method;
    #    one named after "over X" is the baseline (fall back to all methods if none match).
    def order_dataset():
        return sorted(dom["dataset"], key=lambda k: (-relevance(dom["dataset"][k], text),
                                                     info(k).gpu_hours))

    def order_metric():
        return sorted(dom["metric"], key=lambda k: -relevance(dom["metric"][k], text))

    def mentioned(ref):
        return [m for m in dom["method"] if relevance(dom["method"][m], ref) >= MIN_METHOD_RELEVANCE]

    def order_methods(keys, ref):
        return sorted(keys, key=lambda k: (-relevance(dom["method"][k], ref),
                                           -len(_tokens(dom["method"][k]))))

    method_dom = mentioned(main_text) or list(dom["method"])
    base_dom = (mentioned(base_text) if len(parts) > 1 else []) or list(dom["method"])
    domains = {"dataset": order_dataset(), "metric": order_metric(),
               "method": order_methods(method_dom, main_text),
               "baseline": order_methods(base_dom, base_text)}

    # 4. Binary / n-ary constraints
    run_cost = lambda d: info(d).gpu_hours + 2 * METHOD_RUN_GPU_H  # noqa: E731
    constraints = [
        Constraint("compatibility: metric fits dataset", ("dataset", "metric"), shared),
        Constraint("compatibility: baseline runs on dataset", ("dataset", "baseline"), shared),
        Constraint("compatibility: baseline differs from method", ("method", "baseline"),
                   lambda m, b: m != b),
        Constraint("compute budget (full experiment)", ("dataset",),
                   lambda d: run_cost(d) <= budget),
    ]

    fails: dict[str, int] = {}
    sol = _solve(domains, constraints, fails)
    if sol is None:
        reasons += [f"{k} (failed {n} times)" for k, n in sorted(fails.items(), key=lambda x: -x[1])]
        return reject(reasons or ["no assignment satisfies the constraints"])

    d, m, b, mt = (dom["dataset"][sol["dataset"]], dom["method"][sol["method"]],
                   dom["method"][sol["baseline"]], dom["metric"][sol["metric"]])
    return Proposal(
        hypothesis_id=hyp.id, status="active",
        research_question=hyp.testable_prediction or hyp.statement,
        dataset=d, method=m, baselines=[b], metric=mt,
        higher_is_better=not (_tokens(mt) & LOWER_IS_BETTER),
        setting=f"{MIN_SEEDS} seeds on {d}",
        est_compute_gpu_hours=run_cost(sol["dataset"]),
        constraint_report=ConstraintReport(
            satisfied=["novelty", "license", "compute", "ethics", "compatibility", "testability"],
            violated=[]),
    )


def build_proposals(hyps, kb, resources, catalog=None) -> tuple[list[Proposal], list[Proposal]]:
    """Returns (accepted, rejected). Rejected ones carry rejection_reasons for Module C."""
    out = [build_proposal(h, kb, resources, catalog) for h in hyps]
    return ([p for p in out if p.status == "active"],
            [p for p in out if p.status == "rejected"])
