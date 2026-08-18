"""TaskManager lifecycle: results must outlive the run that produced them.

The bug these tests pin down: the TTL reaper measured age from ``created_at``,
so any task whose *runtime* exceeded the TTL was deleted at the first
``get_status`` call after it finished — the poller saw 404 "Task not found"
while the results had been applied. Transport-aggregation Localize runs
longer than the TTL at real dataset sizes, so it failed this way every time.
"""
import threading
import time

from xcell.task_manager import TaskManager


def _submit_gated(tm: TaskManager):
    """Submit a task that blocks until the returned gate is set."""
    gate = threading.Event()

    def compute():
        gate.wait(10)
        return {'x': 1}

    def apply(result):
        return {'status': 'completed', **result}

    task_id = tm.submit(compute, apply)
    return task_id, gate


def _wait_terminal(tm: TaskManager, task_id: str, timeout: float = 10.0):
    """Poll get_status until the task leaves 'running' (or vanishes)."""
    deadline = time.time() + timeout
    entry = tm.get_status(task_id)
    while time.time() < deadline:
        entry = tm.get_status(task_id)
        if entry is None or entry.status != 'running':
            return entry
        time.sleep(0.01)
    return entry


def test_basic_completion_exposes_apply_result():
    tm = TaskManager(max_workers=1)
    task_id, gate = _submit_gated(tm)
    gate.set()
    entry = _wait_terminal(tm, task_id)
    assert entry is not None
    assert entry.status == 'completed'
    assert entry.result == {'status': 'completed', 'x': 1}


def test_task_running_longer_than_ttl_still_reports_completed():
    """A long run must not be reaped the instant it finishes.

    Backdating created_at simulates a task that has already been running
    longer than the TTL when it completes.
    """
    tm = TaskManager(max_workers=1)
    task_id, gate = _submit_gated(tm)

    entry = tm.get_status(task_id)
    assert entry is not None and entry.status == 'running'
    entry.created_at = time.time() - tm.TTL_SECONDS - 100

    gate.set()
    entry = _wait_terminal(tm, task_id)
    assert entry is not None, (
        'completed task was reaped the moment it finished — the poller '
        'would see 404 and report the task lost'
    )
    assert entry.status == 'completed'
    assert entry.result == {'status': 'completed', 'x': 1}


def test_finished_tasks_are_reaped_once_ttl_passes_after_completion():
    tm = TaskManager(max_workers=1)
    task_id, gate = _submit_gated(tm)
    gate.set()
    entry = _wait_terminal(tm, task_id)
    assert entry is not None and entry.status == 'completed'

    entry.finished_at = time.time() - tm.TTL_SECONDS - 1
    assert tm.get_status(task_id) is None
