# Vishal: Module D1 Workflow Tracker

## Run (Windows PowerShell, Python 3.12 recommended)
Extract this folder, open its terminal, then:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-d1.txt
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe -m pytest -q
```
No environment activation is required with these commands. No Ollama, Groq,
GPU, dataset or API key is required. On Linux/macOS use `.venv/bin/python`.

## Files and repository placement
- `schemas.py`: UNCHANGED copy from the uploaded repository, for standalone use.
  In the team repository reuse its existing root file; do not overwrite newer schemas.
- `module_D_workflow_csp_rl/tracker.py`: new D1 implementation.
- `demo.py`: standalone fake-result demonstration, not a scientific experiment.
- `tests/test_tracker.py`: automated correctness checks.
- `requirements-d1.txt`: minimal standalone dependencies. Team requirements already
  contain Pydantic and pytest; no additional runtime dependency is introduced.

## Hinglish explanation
D1 coordinator hai. Ye experiments/reviews khud nahi karta aur RL action choose
nahi karta. Ye conditions check karke execution allow karta hai.

`preconditions`: pehle kaunsi facts true honi chahiye.
`effects`: successful execution ke baad kaunsi facts true hongi.
`facts`: shared dictionary, e.g. dataset_selected = True.

Functions:
1. `ready_tasks(state)`: eligible task copies; original state change nahi hota.
2. `refresh_tasks(state)`: pending/ready statuses ko latest conditions se update.
3. `start_task(state, id)`: conditions dobara check; running mark; attempts + 1.
4. `finish_task(..., success=...)`: real outcome se done/failed aur effects update.
5. `requeue_task(...)`: retry limit mein failed task pending; explicit config override.
6. `reopen_tasks(...)`: review/resource change par task aur dependent work invalidate.
7. `allowed_actions(state)`: D3 ko eligible action types.
8. `eligible_tasks_for_action(...)`: D3 ko valid targets bhi deta hai.
9. `route_task(...)`: chosen eligible task ka graph node name.

## Integration contract (explicit implementation choices)
PDF requires conditions, effects, retry/replanning and action masking. Existing
schemas are reused; the following conventions make missing implementation details
explicit without changing shared schemas:

- `Task.params.est_cost`: required non-negative finite estimated cost; zero must
  be explicit. Compared against `LabState.budget_left` in the same run-budget units.
- `Task.params.action_type`: one of schemas.ACTION_TYPES for D3-controlled work.
  Bootstrap tasks may omit it; they can still be ready/routed.
- `Task.node`: actual graph node name agreed by the graph integrator.
- Actual cost accounting belongs to orchestrator/Module A. Tracker does NOT deduct
  estimates or actual costs. Caller must update budget before next scheduling step.
- Preconditions must include every execution requirement. Real dataset/code runs
  must include appropriate approval facts from Module A. Simulation need not demand
  real download permission. D1 neither grants approval nor downloads data.
- Use proposal/round-scoped fact names for multiple hypotheses, e.g.
  `PR-1:round-2:baseline_added`. Bare names in demo serve one workflow only.
- Caller translates module results into `success`: current E uses `success`,
  shared ExperimentResult uses `ok`. Exceptions/fallbacks are not scientific success.
  No adapter silently interprets F's fallback NEEDS_REVISION as a completed valid review.
- Critical reviewer tasks must be explicitly mapped to workflow IDs before calling
  reopen_tasks. Task text alone cannot reliably identify affected workflow nodes.
- Retry config overrides and alternative datasets come from executor/team policy,
  never guessed by D1. Exhausted retries stay failed. Caller can supply a new Task
  for an explicit alternative with fresh ID and correct dependencies.
- Resource loss: set resource fact False and call reopen_tasks for affected work
  when old outputs become invalid. It invalidates transitive dependents too.
- Task dictionary keys must equal task.id (existing schema convention).

## LangGraph connection
Functions return partial state updates: `{'tasks': {...}, 'facts': {...}}`.
Existing schemas.upsert merges these dictionaries. Return deltas only, especially
for append-only ledger/review fields. Tracker does not import E/F internals.

The graph owner connects actual nodes and conditional edges. Typical lifecycle:
refresh -> select eligible task -> route_task -> start_task -> execute node ->
finish_task -> optional requeue -> refresh.
D3 chooses only from allowed_actions and eligible_tasks_for_action; validate the
selected target again before execution. A ready node name is not an executed task.

Default dispatch is sequential. Although independent tasks can be ready together,
parallel execution requires budget reservation and conflict handling in the graph
orchestrator; this deliverable does not claim parallel scheduling or checkpoint
integration is tested. Do not route a blocked/empty queue to a report automatically:
distinguish waiting for permission, exhausted retries, and true stop conditions.

## Scope and limits
Standalone D1 logic and fake-result demo are implemented. The uploaded repository
has no full graph, D2/D3 implementations, or common-schema E/F wrappers, so end-to-end
integration cannot be claimed. SQLite persistence, permissions UI, actual experiments,
ledger appending and bandit training belong to their owning components.

This is precondition-based planning/tracking with explicit replanning, as in the
PDF, not a forward-search planner. No unrelated teammate files were modified.

## Viva answer
“Mera D1 workflow tracker dependencies aur resource conditions check karta hai.
Ye task lifecycle aur bounded retries manage karta hai, invalid outputs/dependent
tasks ko reopen karta hai, aur RL controller ko valid actions/targets deta hai.
Research decisions aur actual experiments doosre modules karte hain.”
