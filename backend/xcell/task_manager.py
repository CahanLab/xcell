"""Background task manager for cancellable analysis operations."""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskEntry:
    """A tracked background task."""
    id: str
    status: str  # 'running', 'completed', 'cancelled', 'error'
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class TaskManager:
    """Manages cancellable background analysis tasks.

    Operations run in a thread pool on data copies. On completion,
    an apply function writes results to the real data -- unless the
    task was cancelled, in which case results are discarded.
    """

    TTL_SECONDS = 300  # 5 minutes

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskEntry] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        compute_fn: Callable[[], dict[str, Any]],
        apply_fn: Callable[[dict[str, Any]], None],
    ) -> str:
        """Submit a cancellable task.

        Args:
            compute_fn: Runs in background thread on copied data.
                Returns a result dict. Must have no side effects on
                the real adata.
            apply_fn: Called with compute_fn's result if not cancelled.
                Writes results into the real adata. Runs in the
                background thread -- must be fast.

        Returns:
            task_id (UUID string)
        """
        task_id = str(uuid.uuid4())
        entry = TaskEntry(id=task_id, status='running')

        with self._lock:
            self._cleanup_expired()
            self._tasks[task_id] = entry

        def _run():
            try:
                result = compute_fn()
                if entry.cancelled.is_set():
                    entry.status = 'cancelled'
                else:
                    apply_fn(result)
                    entry.result = result
                    entry.status = 'completed'
            except Exception as e:
                if entry.cancelled.is_set():
                    entry.status = 'cancelled'
                else:
                    entry.error = str(e)
                    entry.status = 'error'

        self._executor.submit(_run)
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a task. Returns True if it was running."""
        with self._lock:
            entry = self._tasks.get(task_id)
        if entry is None:
            return False
        if entry.status != 'running':
            return False
        entry.cancelled.set()
        # Status will be set to 'cancelled' by _run when it checks the flag
        return True

    def get_status(self, task_id: str) -> TaskEntry | None:
        """Get a task's current state. Returns None if not found."""
        with self._lock:
            self._cleanup_expired()
            return self._tasks.get(task_id)

    def _cleanup_expired(self) -> None:
        """Remove terminal tasks older than TTL_SECONDS. Must hold _lock."""
        now = time.time()
        expired = [
            tid for tid, entry in self._tasks.items()
            if entry.status in ('completed', 'cancelled', 'error')
            and now - entry.created_at > self.TTL_SECONDS
        ]
        for tid in expired:
            del self._tasks[tid]


# Singleton instance used by routes
task_manager = TaskManager(max_workers=2)
