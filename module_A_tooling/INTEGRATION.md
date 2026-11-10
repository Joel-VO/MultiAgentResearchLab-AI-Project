# Integration contracts (LangGraph orchestrator)

`module_A_tooling/orchestrator.py` connects the modules in one LangGraph `StateGraph`. Each node
calls **one entry point** of a module, checks what comes back against the **contract type** below,
and writes it into `LabState`. If a module returns the wrong shape, the run stops with a
`ContractError` that names the module, the expected type and the field that failed.

This file tells each module owner exactly what to expose. Nothing in your module has to import the
orchestrator. Just make your entry point match the signature and return type below.

```
START → B → C → D.CSP ─(no active proposal)─────────────────────────────→ A.report → END
                  └──→ F.Belief → E → F.Reviewer → D.controller ─(stop)──→ A.report
                                  ↑                     │
                                  └──(runnable task)────┘
```

Run it from the repo root:

```bash
python -m module_A_tooling.orchestrator --topic "..." --seed 2309.11495 --seed "Self-Refine"
pytest tests/test_orchestrator.py      # offline: B, C, D and LLM calls stubbed with the contract types
```

## Where the contract types come from

| Type | Defined in |
|---|---|
| `Hypothesis`, `Proposal`, `ExperimentResult`, `Evidence`, `BeliefState`, `Review`, `RevisionTask`, `LedgerEntry` | `module_A_tooling/schemas.py` (DESIGN §3: source of truth) |
| `KBHandle`, `PaperMeta`, `PaperReview`, `Conflict`, `ModuleBResult` | `module_B_literature_kg/interface.py` |
| `ResearchRequest`, `Resources`, `Gap` | `schemas.py` (root) |
| `ProposalRejection` (`hypothesis_id`, `reasons`) | `module_A_tooling/orchestrator.py` |

You may return your own Pydantic class: the check validates the **fields**, not the class. Extra
fields are ignored (and so are lost). A missing required field, or a value outside a `Literal`,
fails the check.

## Contracts by module

### B: Literature, KB, conflicts. **Not implemented yet**

```python
# module_B_literature_kg/pipeline.py
def run_module_b(request: ResearchRequest) -> ModuleBResult
# ModuleBResult (interface.py):
#   kb: KBHandle                       all six paths must exist on disk
#   reviews: list[PaperReview]         one per paper, written to kb.reviews_path too
#   conflicts: list[Conflict]
#   ledger_entries: list[LedgerEntry]  may be []
```

To do:
- The entry point does not exist yet. Today the orchestrator stops with `cannot load module_B_literature_kg.pipeline:run_module_b`.
- Chain the existing pieces: `run_b0` → paper reviews (B2, not written yet) → `build_graph(papers, embeddings, reviews, citation_edges)` → `save_graph` to `graph.json` → `detect_finding_conflicts`.
- The graph **must contain dataset, method and metric nodes** (they come from the paper reviews). Without them D.CSP rejects every hypothesis with "KB has no dataset nodes".

### C: Gaps and hypotheses. **Cannot be imported today**

```python
# module_C_hypothesis/hypothesizer.py
def mine_gaps(kb: KBHandle, reviews: list[PaperReview], conflicts: list[Conflict],
              request: ResearchRequest) -> list[Gap]
def hypothesise(kb: KBHandle, reviews: list[PaperReview], conflicts: list[Conflict],
                request: ResearchRequest, gaps: list[Gap]) -> list[Hypothesis]
```
The orchestrator passes all five arguments **positionally** and uses your defaults for everything else
(`n_agents`, `k_iterations`, ...). Pick defaults that finish in reasonable time.

To do (current error: `ModuleNotFoundError: No module named 'classes'`):
- `from classes import ...` → `from module_C_hypothesis.classes import ...`
- `Any` and `math` are not imported.
- `mine_gaps` uses undefined names `kbhandle`, `valid_ids` and `conflict`. Open the KB from the `kb` argument with `open_kb(kb)`.
- `_normalize_candidate` is called but never defined.
- `lead.llm.generate(...)` returns a **string**, so `synthesis.get(...)` fails. Parse it first (e.g. `llm_structured` from Module A).
- Return types: `classes.Gap` and `classes.Hypothesis` already match the contract fields. Hypothesis IDs must be unique (`H1`, `H2`, ...) and `supporting_paper_ids` must be KB paper IDs, because F.Belief looks them up there.

### D.CSP: Proposals. **Rejections in the wrong format**

```python
# module_D_workflow_csp_rl/csp.py
def build_proposals(hypotheses: list[Hypothesis], kb: KnowledgeBase, resources: Resources
                    ) -> tuple[list[Proposal], list[ProposalRejection]]
# ProposalRejection: { hypothesis_id: str, reasons: list[str] }      (DESIGN §10, D.2)
```
`kb` is the opened KB (`open_kb(handle, embed_fn)`), so `closest_to_text` works. The embedding
function comes from the orchestrator and uses the same model as B.

To do:
- Accepted proposals already pass.
- Rejected ones come back as `Proposal(status="rejected", rejection_reasons=[...])` and fail the check. Return `ProposalRejection(hypothesis_id=..., reasons=...)` instead.
- The orchestrator calls `build_proposals` **without a `catalog`**, so every dataset gets license `"unknown"` and is pruned. D needs its own catalog source (e.g. a `DATASET_CATALOG` in `config.py`, used as the default).
- `higher_is_better` is dropped, because `module_A_tooling/schemas.Proposal` has no such field. This needs a schema decision (Module A + E): either add it to `Proposal` or have E derive it.

### F.Belief. **Matches**

```python
# module_F_reviewer_belief/belief.py
def literature_evidence(hypothesis: Hypothesis, paper: PaperMeta, review: PaperReview | None,
                        proposal: Proposal) -> Evidence | None
def compute_belief(h_id: str, evidence: list[Evidence]) -> BeliefState
def rank(beliefs: list[BeliefState]) -> list[BeliefState]        # best first
def experiment_evidence(result: ExperimentResult) -> Evidence | None
```
Called once per (hypothesis with an active proposal, supporting paper). The top of `rank` becomes the
selected hypothesis. After every experiment, `compute_belief` is re-run over all evidence.

### E: Experimenter. **Matches**

```python
# module_E_experimenter/runner.py
def run_experiment(proposal: Proposal, hypothesis: Hypothesis, task: RevisionTask | None = None,
                   sim: SimParams = ..., result_id: str = ...) -> ExperimentResult
```
The orchestrator passes `sim=SimParams(rng_seed=n)` and `result_id=f"E{n}"` (n = run number), so
repeated runs draw different noise. `task` is the open revision task chosen by the controller, or
`None` for the first run.

### F.Reviewer. **Matches, one ID issue**

```python
# module_F_reviewer_belief/reviewer.py
def review(hypothesis: Hypothesis, proposal: Proposal, results: list[ExperimentResult],
           belief: BeliefState, open_tasks: list[RevisionTask], review_id: str) -> Review
```
`results` are all runs of the selected proposal so far, `open_tasks` are tasks not yet closed, and
`review_id = f"R{n}"`.

To do:
- New `RevisionTask` IDs must be unique **across the whole run**. `_next_task_id` only avoids the IDs of *open* tasks, so once `T1` is closed a later review can issue a new `T1`. The orchestrator raises `ContractError: ... reused revision task IDs` when that happens, which is possible from round 3 onwards.

### A: Report. **Placeholder**

```python
def generate_report(state: LabState) -> str | Path      # path to report.md (DESIGN §10, A)
```
For now `write_placeholder_report` writes `runs/<request id>/report.md`, a plain dump of
hypotheses, proposals, rejections, experiments and reviews. It is not built from the ledger yet.

## LabState (what each node writes)

| Key | Type | Written by |
|---|---|---|
| `request` | `ResearchRequest` | input |
| `kb` | `KBHandle` | B |
| `paper_reviews`, `conflicts` | `dict[id, PaperReview / Conflict]` | B |
| `gaps`, `hypotheses` | `dict[id, Gap / Hypothesis]` | C (F.Belief sets the selected one's `status`) |
| `proposals` / `rejections` | `dict[id, Proposal]` / `list[ProposalRejection]` | D.CSP |
| `evidence`, `beliefs` | `dict[id, Evidence]`, `dict[hypothesis_id, BeliefState]` | F.Belief, E node |
| `selected_hypothesis_id`, `selected_proposal_id` | `str` | F.Belief |
| `experiments` | `dict[id, ExperimentResult]` | E |
| `reviews` / `revision_tasks` / `closed_task_ids` | `list[Review]` / `dict[id, RevisionTask]` / `list[str]` | F.Reviewer |
| `round` | `int` (finished review rounds) | F.Reviewer |
| `next_task_id`, `stop_reason` | `str \| None` | D.controller |
| `ledger` | `list[LedgerEntry]` | B (others later) |
| `report_path` | `str` | A.report |

Dicts are merged by ID (a node returns only new or changed objects). `reviews`, `rejections`,
`closed_task_ids` and `ledger` are append-only.

## Not wired yet

D.1 tracker gating, D.3 RL (the controller is a rule: most severe open baseline/ablation/analysis
task, stop after `MAX_ROUNDS` in `config.py` or when none is left), re-entering B/C for
`read_paper`/`revise_hypothesis`, the ledger from C–F, SQLite checkpointer, `interrupt` for
human-in-the-loop and the dataset gate, and real (non-simulated) runs.
