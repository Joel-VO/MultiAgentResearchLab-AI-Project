import copy
import pytest
from schemas import Task
from module_D_workflow_csp_rl.tracker import (
    ready_tasks, refresh_tasks, start_task, finish_task, requeue_task,
    allowed_actions, eligible_tasks_for_action, reopen_tasks, route_task,
)


def state():
    t = Task(id='W1', node='experiment', preconditions=['approved'], effects=['result'],
             params={'est_cost': 0.2, 'action_type': 'run_experiment'})
    return {'tasks': {'W1': t}, 'facts': {'approved': True}, 'budget_left': 1.0}


def apply(s, update):
    for key, value in update.items():
        s.setdefault(key, {}).update(value)


def test_permission_and_no_mutation():
    s = state()
    before = copy.deepcopy(s)
    assert ready_tasks(s)[0].status == 'ready'
    assert s == before
    s['facts']['approved'] = False
    assert not ready_tasks(s)
    with pytest.raises(ValueError):
        start_task(s, 'W1')


@pytest.mark.parametrize('cost,budget', [(2, 1), (-1, 1), (None, 1),
                                        (float('nan'), 1), (0.2, float('nan'))])
def test_bad_or_unaffordable_cost(cost, budget):
    s = state()
    s['tasks']['W1'].params['est_cost'] = cost
    s['budget_left'] = budget
    assert not ready_tasks(s)
    assert not allowed_actions(s)


def test_success_transitions_and_json_input():
    s = state()
    s['tasks']['W1'] = s['tasks']['W1'].model_dump()
    assert route_task(s, 'W1') == 'experiment'
    apply(s, start_task(s, 'W1'))
    assert s['tasks']['W1'].attempts == 1
    with pytest.raises(ValueError):
        start_task(s, 'W1')
    apply(s, finish_task(s, 'W1', success=True))
    assert s['facts']['result'] is True
    assert not ready_tasks(s)
    with pytest.raises(ValueError):
        finish_task(s, 'W1', success=True)


def test_failure_retries_bounded():
    s = state()
    for attempt in range(2):
        apply(s, start_task(s, 'W1'))
        apply(s, finish_task(s, 'W1', success=False, error='OOM'))
        assert s['facts']['result'] is False
        if attempt == 0:
            apply(s, requeue_task(s, 'W1', overrides={'batch_size': 16}))
    assert s['tasks']['W1'].params['batch_size'] == 16
    with pytest.raises(ValueError):
        requeue_task(s, 'W1')


def test_stale_ready_and_action_target_filter():
    s = state()
    apply(s, refresh_tasks(s))
    assert allowed_actions(s) == ['run_experiment']
    assert len(eligible_tasks_for_action(s, 'run_experiment')) == 1
    s['budget_left'] = 0
    apply(s, refresh_tasks(s))
    assert s['tasks']['W1'].status == 'pending'
    with pytest.raises(ValueError):
        route_task(s, 'W1')


def test_reopen_invalidates_downstream_and_keeps_unrelated():
    s = state()
    s['tasks']['W1'].status = 'done'
    s['tasks']['W2'] = Task(id='W2', node='review', preconditions=['result'],
                            effects=['reviewed'], status='done')
    s['tasks']['W3'] = Task(id='W3', node='other', status='done')
    s['facts'].update(result=True, reviewed=True)
    apply(s, reopen_tasks(s, ['W1']))
    assert s['tasks']['W2'].status == 'pending'
    assert s['facts']['reviewed'] is False
    assert s['tasks']['W3'].status == 'done'


def test_reopen_refuses_running_dependents():
    s = state()
    s['tasks']['W2'] = Task(id='W2', node='review', preconditions=['result'], status='running')
    with pytest.raises(ValueError):
        reopen_tasks(s, ['W1'])
