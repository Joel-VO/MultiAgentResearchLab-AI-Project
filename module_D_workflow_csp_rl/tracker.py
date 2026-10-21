"""D1: deterministic workflow tracking; no LLM or other module internals.

Functions return partial LabState updates, suitable for the existing upsert
reducers. The caller executes nodes and reports actual success/failure.
"""
from math import isfinite
from schemas import ACTION_TYPES, LabState, Task


def _tasks(state: LabState) -> dict[str, Task]:
    """Accept Pydantic tasks and JSON/checkpoint dictionaries."""
    return {key: Task.model_validate(value).model_copy(deep=True)
            for key, value in state.get('tasks', {}).items()}


def blocked_reasons(state: LabState, task: Task) -> list[str]:
    """Missing facts block execution. Unknown costs are not silently free.

    params.est_cost must be set explicitly (zero allowed). Permission gates
    belong in task.preconditions, e.g. dataset_approved for a real run.
    """
    reasons = []
    facts = state.get('facts', {})
    for fact in task.preconditions:
        if facts.get(fact) is not True:
            reasons.append(f'Missing fact: {fact}')
    if task.attempts >= task.max_attempts:
        reasons.append('Attempt limit reached')
    cost = task.params.get('est_cost')
    budget = state.get('budget_left')
    if (isinstance(cost, bool) or not isinstance(cost, (int, float))
            or not isfinite(cost) or cost < 0):
        reasons.append('Explicit non-negative finite est_cost required')
    elif (isinstance(budget, bool) or not isinstance(budget, (int, float))
          or not isfinite(budget) or budget < cost):
        reasons.append('Insufficient or missing budget')
    return reasons


def ready_tasks(state: LabState) -> list[Task]:
    """Return eligible copies; never mutate the incoming state."""
    return [task.model_copy(update={'status': 'ready'})
            for task in _tasks(state).values()
            if task.status in ('pending', 'ready') and not blocked_reasons(state, task)]


def refresh_tasks(state: LabState) -> dict:
    """Re-check ready tasks too: resources/permissions may have changed."""
    updates = {}
    for key, task in _tasks(state).items():
        if task.status in ('pending', 'ready'):
            task.status = 'pending' if blocked_reasons(state, task) else 'ready'
            updates[key] = task
    return {'tasks': updates}


def start_task(state: LabState, task_id: str) -> dict:
    """Revalidate immediately before dispatch; count one actual attempt."""
    task = _tasks(state)[task_id]
    if task.status not in ('pending', 'ready'):
        raise ValueError('Only pending/ready tasks may start')
    reasons = blocked_reasons(state, task)
    if reasons:
        raise ValueError('; '.join(reasons))
    task.status = 'running'
    task.attempts += 1
    task.error = None
    return {'tasks': {task_id: task}}


def finish_task(state: LabState, task_id: str, *, success: bool,
                error: str | None = None) -> dict:
    """Only actual success asserts effects. Failure clears expected effects.

    Effects must be scoped to a proposal/round when there are multiple runs.
    Resource-loss invalidation of older downstream results is explicit below.
    """
    task = _tasks(state)[task_id]
    if task.status != 'running':
        raise ValueError('Only running tasks may finish')
    task.status = 'done' if success else 'failed'
    task.error = None if success else (error or 'Task failed')
    return {'tasks': {task_id: task},
            'facts': {fact: success for fact in task.effects}}


def requeue_task(state: LabState, task_id: str, *, overrides: dict | None = None) -> dict:
    """Retry a failed task with optional teammate-approved configuration changes.

    D1 does not guess batch sizes or datasets. The caller supplies overrides.
    An exhausted task remains failed; caller may add an explicit alternative.
    """
    task = _tasks(state)[task_id]
    if task.status != 'failed' or task.attempts >= task.max_attempts:
        raise ValueError('Task is not failed or retry limit reached')
    task.params.update(overrides or {})
    task.status = 'pending'
    return {'tasks': {task_id: task}}


def reopen_tasks(state: LabState, task_ids: list[str]) -> dict:
    """Explicit review/resource invalidation, including dependent old results.

    Caller maps reviewer tasks to workflow IDs. Running work must first finish
    or be cancelled by its executor; this function cannot cancel execution.
    Reopened completed work gets a fresh attempt allowance.
    """
    tasks = _tasks(state)
    affected = set(task_ids)
    if not affected <= tasks.keys():
        raise KeyError('Unknown workflow task ID')
    invalid = set()
    while True:
        for key in affected:
            invalid.update(tasks[key].effects)
        dependent = {key for key, task in tasks.items()
                     if set(task.preconditions) & invalid}
        expanded = affected | dependent
        if expanded == affected:
            break
        affected = expanded
    if any(tasks[key].status == 'running' for key in affected):
        raise ValueError('Finish/cancel affected running tasks before reopening')
    updates = {}
    for key in affected:
        task = tasks[key]
        task.status, task.attempts, task.error = 'pending', 0, None
        updates[key] = task
    return {'tasks': updates, 'facts': {fact: False for fact in invalid}}


def allowed_actions(state: LabState) -> list[str]:
    """D3 mask: an action is allowed if at least one matching task is ready.

    D3 must choose its target from eligible_tasks_for_action, not an arbitrary
    task with the same type. Node names and action names are separate concepts.
    """
    kinds = {task.params.get('action_type') for task in ready_tasks(state)}
    return [action for action in ACTION_TYPES if action in kinds]


def eligible_tasks_for_action(state: LabState, action_type: str) -> list[Task]:
    return [task for task in ready_tasks(state)
            if task.params.get('action_type') == action_type]


def route_task(state: LabState, task_id: str) -> str:
    """LangGraph conditional-router helper; caller maps name to real node.

    Use start_task immediately before execution. Dispatch sequentially unless
    the orchestrator reserves budgets and handles conflicting fact writes.
    """
    eligible = {task.id: task for task in ready_tasks(state)}
    if task_id not in eligible:
        raise ValueError('Selected task is not ready')
    return eligible[task_id].node
