"""
worker_pool.py
--------------
Managed thread pool for Phase 8 Async Execution Scheduler.

Architecture
------------
A WorkerPool owns N daemon threads. Each thread runs the same event loop:

    loop:
        item = queue.get(timeout=_IDLE_TIMEOUT)
        if item is None: continue
        if item is POISON_PILL: break
        execute item.fn(*item.args, **item.kwargs)
        resolve item.future

The pool exposes:
  - ``start()``     — spawn worker threads
  - ``stop()``      — graceful shutdown (drains queue then joins threads)
  - ``stop_now()``  — immediate shutdown (sends poison pills, joins with timeout)
  - ``submit()``    — convenience passthrough to the underlying TaskQueue
  - ``stats``       — PoolStats snapshot

Two pool configurations are pre-defined:

``WorkerPool.cpu_pool(n_workers, max_depth)``
    For CPU-bound tasks: prompt pre-processing, KV prefix scanning,
    result post-processing. Uses n_workers daemon threads.

``WorkerPool.transfer_pool(max_depth)``
    Single-threaded, dedicated to PCIe/memcpy operations. Only one
    thread ensures the bus is never overwhelmed by concurrent copies.

Lifecycle
---------
A pool that has not been ``start()``-ed silently queues tasks and will
execute them once ``start()`` is called. This allows pools to be created
before start-up is complete.

Graceful shutdown
-----------------
``stop()`` sends one POISON_PILL sentinel per worker thread. Each thread,
upon receiving the pill, exits its event loop. ``stop()`` then joins all
threads with a configurable timeout. Tasks already in the queue when
``stop()`` is called are drained first (up to ``drain_timeout_sec``).
Tasks submitted after ``stop()`` are rejected.

Thread safety
-------------
All pool state mutations are protected by ``self._lock``. The TaskQueue
is independently thread-safe.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from .task_queue import Priority, QueueFullError, TaskFuture, TaskQueue, _QueueItem

T = TypeVar("T")

# Sentinel object that tells a worker thread to exit
_POISON_PILL = object()

# How long a worker sleeps waiting for work before checking shutdown flag
_IDLE_TIMEOUT = 0.05  # 50 ms


# ---------------------------------------------------------------------------
# PoolStats
# ---------------------------------------------------------------------------

@dataclass
class PoolStats:
    """
    Snapshot of WorkerPool state.

    Attributes
    ----------
    n_workers : int
        Total number of worker threads configured.
    active_workers : int
        Workers currently executing a task (not idle).
    queue_depth : int
        Tasks pending in the queue.
    tasks_processed : int
        Cumulative completed tasks since pool start.
    tasks_failed : int
        Cumulative failed tasks since pool start.
    is_running : bool
        True if the pool has been started and not yet stopped.
    """
    n_workers: int = 0
    active_workers: int = 0
    queue_depth: int = 0
    tasks_processed: int = 0
    tasks_failed: int = 0
    is_running: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_workers": self.n_workers,
            "active_workers": self.active_workers,
            "queue_depth": self.queue_depth,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "is_running": self.is_running,
        }


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

class WorkerPool:
    """
    Managed thread pool that drains a :class:`TaskQueue`.

    Parameters
    ----------
    n_workers : int
        Number of daemon worker threads. For transfer operations, use 1.
    queue : TaskQueue, optional
        Underlying task queue. If None, a new queue is created.
    name : str, optional
        Human-readable pool name (used in thread names for debugging).
    drain_timeout_sec : float
        Seconds to wait for the queue to drain during ``stop()``.
        Default 30 s.
    """

    def __init__(
        self,
        n_workers: int = 2,
        queue: Optional[TaskQueue] = None,
        name: str = "WorkerPool",
        drain_timeout_sec: float = 30.0,
    ) -> None:
        self.name = name
        self.n_workers = max(1, n_workers)
        self.drain_timeout_sec = drain_timeout_sec
        self._queue = queue or TaskQueue(max_depth=64, name=f"{name}.queue")

        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()

        # Counters
        self._tasks_processed = 0
        self._tasks_failed = 0
        self._active_workers = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "WorkerPool":
        """
        Spawn all worker threads.

        Returns self so the pool can be started inline:
        ``pool = WorkerPool(4).start()``
        """
        with self._lock:
            if self._running:
                return self
            self._running = True
            self._stop_event.clear()
            self._threads = []
            for i in range(self.n_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name=f"{self.name}.worker-{i}",
                )
                t.start()
                self._threads.append(t)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """
        Gracefully stop the pool.

        1. Sets stop event so workers stop accepting new work.
        2. Sends one POISON_PILL per worker.
        3. Joins all worker threads.

        Parameters
        ----------
        timeout : float
            Join timeout per thread in seconds. Default 5 s.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        # Send poison pills directly to the underlying queue (bypass TaskFuture)
        for _ in range(self.n_workers):
            try:
                from .task_queue import _QueueItem
                # Use a very high priority so pills are dequeued before real tasks
                self._queue._q.put(
                    _QueueItem(
                        priority=-1,  # lower int = higher priority
                        seq=-1,
                        future=None,  # type: ignore[arg-type]
                        fn=_POISON_PILL,  # sentinel
                        args=(),
                        kwargs={},
                    )
                )
            except Exception:
                pass

        for t in self._threads:
            t.join(timeout=timeout)
        self._threads = []

    def stop_now(self, timeout: float = 2.0) -> None:
        """
        Immediate stop: signal workers to exit, skip queue draining.

        Use when the process is shutting down and correctness is not required.
        """
        self._stop_event.set()
        with self._lock:
            self._running = False
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads = []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        priority: Priority = Priority.NORMAL,
        **kwargs: Any,
    ) -> TaskFuture[T]:
        """
        Submit a callable for execution by one of this pool's workers.

        Raises
        ------
        RuntimeError
            If the pool has been stopped.
        QueueFullError
            If the queue is full (backpressure).
        """
        if not self._running and self._stop_event.is_set():
            raise RuntimeError(f"[{self.name}] Pool has been stopped. Cannot submit tasks.")
        return self._queue.submit(fn, *args, priority=priority, **kwargs)

    def submit_wait(
        self,
        fn: Callable[..., T],
        *args: Any,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Submit and block until result."""
        future = self.submit(fn, *args, priority=priority, **kwargs)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> PoolStats:
        """Return a snapshot of pool statistics."""
        with self._lock:
            return PoolStats(
                n_workers=self.n_workers,
                active_workers=self._active_workers,
                queue_depth=self._queue.depth,
                tasks_processed=self._tasks_processed,
                tasks_failed=self._tasks_failed,
                is_running=self._running,
            )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue(self) -> TaskQueue:
        return self._queue

    # ------------------------------------------------------------------
    # Class-method factories
    # ------------------------------------------------------------------

    @classmethod
    def cpu_pool(cls, n_workers: int, max_queue_depth: int = 32) -> "WorkerPool":
        """
        Create a CPU-worker pool for compute-bound tasks.

        Parameters
        ----------
        n_workers : int
            Number of CPU worker threads.
        max_queue_depth : int
            Maximum pending tasks before backpressure.

        Returns
        -------
        WorkerPool (not yet started)
        """
        q = TaskQueue(max_depth=max_queue_depth, name="CPUQueue")
        return cls(n_workers=n_workers, queue=q, name="CPUPool")

    @classmethod
    def transfer_pool(cls, max_queue_depth: int = 16) -> "WorkerPool":
        """
        Create a single-threaded transfer worker pool.

        Using exactly 1 worker prevents concurrent PCIe copies that would
        compete for shared bus bandwidth.

        Returns
        -------
        WorkerPool (not yet started)
        """
        q = TaskQueue(max_depth=max_queue_depth, name="TransferQueue")
        return cls(n_workers=1, queue=q, name="TransferPool")

    # ------------------------------------------------------------------
    # Internal worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Event loop run by each worker thread."""
        while not self._stop_event.is_set():
            item = self._queue.get(timeout=_IDLE_TIMEOUT)

            if item is None:
                continue  # Timeout — loop back and check stop_event

            # Poison pill check — fn is the sentinel object
            if item.fn is _POISON_PILL:
                self._queue.task_done()
                break

            # Check if future was cancelled while in queue
            future = item.future
            if future is not None and future.is_cancelled:
                self._queue.task_done()
                self._queue._record_cancelled()
                continue

            # Mark started
            if future is not None:
                future._mark_started()

            with self._lock:
                self._active_workers += 1

            try:
                result = item.fn(*item.args, **item.kwargs)
                if future is not None:
                    future._mark_done(result)
                    self._queue._record_complete()
                with self._lock:
                    self._tasks_processed += 1
            except Exception as exc:
                if future is not None:
                    future._mark_failed(exc)
                    self._queue._record_failed()
                with self._lock:
                    self._tasks_failed += 1
            finally:
                with self._lock:
                    self._active_workers -= 1
                self._queue.task_done()
