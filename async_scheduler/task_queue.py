"""
task_queue.py
-------------
Priority-aware bounded task queue for Phase 8 Async Execution Scheduler.

Design
------
The TaskQueue is a thin wrapper around Python's ``queue.PriorityQueue`` that
adds:

  1. **Priority levels** (CRITICAL → LOW): tasks are dequeued in priority
     order regardless of submission order.

  2. **Bounded depth**: once ``max_depth`` tasks are pending, ``submit()``
     raises :exc:`QueueFullError` rather than growing unboundedly. This
     creates backpressure on the caller, preventing memory runaway.

  3. **TaskFuture**: every submitted task returns a ``TaskFuture`` that
     allows the caller to optionally block for the result, poll completion,
     or cancel the pending task.

  4. **TaskStats**: live counters for monitoring (submitted, completed,
     cancelled, failed).

Thread safety
-------------
All public methods are thread-safe. The internal PriorityQueue is
thread-safe by design. TaskFuture uses a threading.Event for the
blocking ``result()`` call.

Priority ordering
-----------------
CRITICAL (0) > HIGH (1) > NORMAL (2) > LOW (3)

When two tasks share the same priority, they are ordered by submission
sequence number (FIFO within the same priority band).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Task priority levels. Lower integer = higher priority."""
    CRITICAL = 0
    HIGH     = 1
    NORMAL   = 2
    LOW      = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QueueFullError(RuntimeError):
    """Raised by TaskQueue.submit() when the queue has reached max_depth."""


class TaskCancelledError(RuntimeError):
    """Raised by TaskFuture.result() when the task was cancelled."""


class TaskFailedError(RuntimeError):
    """Raised by TaskFuture.result() when the underlying callable raised."""
    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# TaskFuture
# ---------------------------------------------------------------------------

class TaskFuture(Generic[T]):
    """
    Awaitable result handle for a submitted task.

    The future is in one of four states:

    PENDING → RUNNING → DONE (success)
                      → FAILED
                      → CANCELLED

    Parameters
    ----------
    task_id : int
        Unique sequential task identifier.
    """

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self._event = threading.Event()
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._cancelled = False
        self._done = False
        self._lock = threading.Lock()
        self._submitted_at = time.perf_counter()
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Block until the task completes and return its result.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait. Raises ``TimeoutError`` if exceeded.

        Returns
        -------
        T
            The return value of the submitted callable.

        Raises
        ------
        TaskCancelledError
            If the task was cancelled before it ran.
        TaskFailedError
            If the callable raised an exception.
        TimeoutError
            If the timeout expired before completion.
        """
        finished = self._event.wait(timeout=timeout)
        if not finished:
            raise TimeoutError(f"Task {self.task_id} did not complete within {timeout}s")
        with self._lock:
            if self._cancelled:
                raise TaskCancelledError(f"Task {self.task_id} was cancelled")
            if self._exception is not None:
                raise TaskFailedError(
                    f"Task {self.task_id} failed: {self._exception}",
                    cause=self._exception,
                )
            return self._result  # type: ignore[return-value]

    def cancel(self) -> bool:
        """
        Request cancellation of a pending task.

        Returns True if the task was successfully cancelled (still PENDING).
        Returns False if the task already started or completed.
        """
        with self._lock:
            if self._done or self._started_at is not None:
                return False
            self._cancelled = True
            self._done = True
        self._event.set()
        return True

    @property
    def is_done(self) -> bool:
        """True if the task has completed, failed, or was cancelled."""
        return self._done

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def wall_time_ms(self) -> float:
        """
        Total wall-clock time from submission to completion in milliseconds.
        0.0 if the task has not yet finished.
        """
        if self._finished_at is None:
            return 0.0
        return (self._finished_at - self._submitted_at) * 1000.0

    @property
    def queue_wait_ms(self) -> float:
        """Time spent waiting in the queue before execution started."""
        if self._started_at is None:
            return 0.0
        return (self._started_at - self._submitted_at) * 1000.0

    @property
    def execution_ms(self) -> float:
        """Time spent in active execution (after dequeue, before done)."""
        if self._started_at is None or self._finished_at is None:
            return 0.0
        return (self._finished_at - self._started_at) * 1000.0

    # ------------------------------------------------------------------
    # Internal state setters (called by WorkerPool)
    # ------------------------------------------------------------------

    def _mark_started(self) -> None:
        with self._lock:
            self._started_at = time.perf_counter()

    def _mark_done(self, result: Any) -> None:
        with self._lock:
            self._result = result
            self._done = True
            self._finished_at = time.perf_counter()
        self._event.set()

    def _mark_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._exception = exc
            self._done = True
            self._finished_at = time.perf_counter()
        self._event.set()


# ---------------------------------------------------------------------------
# TaskStats
# ---------------------------------------------------------------------------

@dataclass
class TaskStats:
    """
    Live counters for a TaskQueue.

    All fields are updated atomically by the queue and its workers.
    """
    submitted: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    current_depth: int = 0

    @property
    def in_flight(self) -> int:
        """Tasks submitted but not yet done."""
        return self.submitted - self.completed - self.cancelled - self.failed

    def to_dict(self) -> Dict[str, int]:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "current_depth": self.current_depth,
            "in_flight": self.in_flight,
        }


# ---------------------------------------------------------------------------
# Internal wrapper stored in the PriorityQueue
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _QueueItem:
    priority: int
    seq: int
    future: TaskFuture = field(compare=False)
    fn: Callable = field(compare=False)
    args: tuple = field(compare=False)
    kwargs: dict = field(compare=False)


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------

class TaskQueue:
    """
    Priority-aware bounded task queue.

    Parameters
    ----------
    max_depth : int
        Maximum number of pending tasks. Calls to :meth:`submit` raise
        :exc:`QueueFullError` when this limit is reached.
    name : str, optional
        Human-readable name for logging / metrics.
    """

    def __init__(self, max_depth: int = 8, name: str = "TaskQueue") -> None:
        self.name = name
        self.max_depth = max_depth
        self._q: queue.PriorityQueue[_QueueItem] = queue.PriorityQueue()
        self._stats = TaskStats()
        self._seq = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        priority: Priority = Priority.NORMAL,
        **kwargs: Any,
    ) -> "TaskFuture[T]":
        """
        Submit a callable for async execution.

        Parameters
        ----------
        fn : callable
            The work to execute.
        *args, **kwargs
            Arguments forwarded to ``fn``.
        priority : Priority
            Task priority. CRITICAL tasks are dequeued first. Default NORMAL.

        Returns
        -------
        TaskFuture
            Result handle.

        Raises
        ------
        QueueFullError
            If the queue has reached ``max_depth``.
        """
        with self._lock:
            if self._q.qsize() >= self.max_depth:
                raise QueueFullError(
                    f"[{self.name}] Queue full ({self.max_depth} tasks pending). "
                    "Backpressure applied."
                )
            seq = self._seq
            self._seq += 1
            self._stats.submitted += 1
            self._stats.current_depth = self._q.qsize() + 1

        future: TaskFuture[T] = TaskFuture(task_id=seq)
        item = _QueueItem(
            priority=int(priority),
            seq=seq,
            future=future,
            fn=fn,
            args=args,
            kwargs=kwargs,
        )
        self._q.put(item)
        return future

    def submit_wait(
        self,
        fn: Callable[..., T],
        *args: Any,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Submit and immediately block until the result is available.

        Convenience wrapper around :meth:`submit` + :meth:`TaskFuture.result`.
        """
        future = self.submit(fn, *args, priority=priority, **kwargs)
        return future.result(timeout=timeout)

    def get_nowait(self) -> Optional[_QueueItem]:
        """
        Dequeue the highest-priority item without blocking.
        Returns None if the queue is empty.
        """
        try:
            item = self._q.get_nowait()
            with self._lock:
                self._stats.current_depth = self._q.qsize()
            return item
        except queue.Empty:
            return None

    def get(self, timeout: Optional[float] = None) -> Optional[_QueueItem]:
        """
        Dequeue the highest-priority item, blocking up to ``timeout`` seconds.
        Returns None on timeout.
        """
        try:
            item = self._q.get(timeout=timeout)
            with self._lock:
                self._stats.current_depth = self._q.qsize()
            return item
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Signal that a dequeued task has been fully processed."""
        self._q.task_done()

    def drain(self, timeout: Optional[float] = None) -> None:
        """Block until all currently queued tasks have been dequeued."""
        self._q.join() if timeout is None else None

    @property
    def depth(self) -> int:
        """Current number of pending tasks."""
        return self._q.qsize()

    @property
    def is_empty(self) -> bool:
        return self._q.empty()

    @property
    def stats(self) -> TaskStats:
        """Return a snapshot of the current stats."""
        with self._lock:
            return TaskStats(
                submitted=self._stats.submitted,
                completed=self._stats.completed,
                cancelled=self._stats.cancelled,
                failed=self._stats.failed,
                current_depth=self._q.qsize(),
            )

    def _record_complete(self) -> None:
        with self._lock:
            self._stats.completed += 1

    def _record_cancelled(self) -> None:
        with self._lock:
            self._stats.cancelled += 1

    def _record_failed(self) -> None:
        with self._lock:
            self._stats.failed += 1
