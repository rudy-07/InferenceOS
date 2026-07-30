"""
request_queue.py
----------------
Asynchronous request queue manager for InferenceOS Server.
Decouples HTTP endpoint ingestion from inference execution and future schedulers.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Awaitable
from server.config import ServerConfig


@dataclass
class QueuedRequest:
    request_id: str
    model_name: str
    func: Callable[[], Awaitable[Any]]
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)
    session_id: Optional[str] = None


class RequestQueueManager:
    """
    Manages pending requests and controls concurrency via an async task queue.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self.queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start() -> None:
        """Start background queue processing worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())

    async def stop() -> None:
        """Stop background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue_and_execute(
        self,
        model_name: str,
        execution_func: Callable[[], Awaitable[Any]],
        session_id: Optional[str] = None,
    ) -> Any:
        """
        Submit an inference job, wait for semaphore/scheduler slot, and execute.
        """
        async with self.semaphore:
            req_id = f"req_{uuid.uuid4().hex[:12]}"
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            q_req = QueuedRequest(
                request_id=req_id,
                model_name=model_name,
                func=execution_func,
                future=fut,
                session_id=session_id,
            )
            self.active_requests[req_id] = q_req
            try:
                res = await execution_func()
                fut.set_result(res)
                return res
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)
                raise
            finally:
                self.active_requests.pop(req_id, None)

    async def _process_queue(self) -> None:
        """Worker loop processing enqueued requests."""
        while self._running:
            try:
                q_req = await self.queue.get()
                async with self.semaphore:
                    try:
                        res = await q_req.func()
                        if not q_req.future.done():
                            q_req.future.set_result(res)
                    except Exception as e:
                        if not q_req.future.done():
                            q_req.future.set_exception(e)
                    finally:
                        self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.01)

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def active_count(self) -> int:
        return len(self.active_requests)


_queue_manager_instance: Optional[RequestQueueManager] = None


def get_queue_manager(config: Optional[ServerConfig] = None) -> RequestQueueManager:
    global _queue_manager_instance
    if _queue_manager_instance is None:
        _queue_manager_instance = RequestQueueManager(config=config)
    return _queue_manager_instance
