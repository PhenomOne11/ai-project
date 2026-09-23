"""
Thread-based Worker Pool for NeuralStream-Core.

Provides a lightweight, dependency-free task executor with:
    - Priority queue
    - Retry on failure
    - Per-task timeout (best-effort)
    - Live statistics

Note: Runs tasks in *threads* rather than processes so torch tensors
already in memory can be dispatched without pickling. For heavier
parallelism, swap the executor thread for a multiprocessing worker.
"""

import threading
import queue
import time
from typing import List, Optional, Dict, Callable, Any
from collections import defaultdict, deque

from core.workers.task import Task, TaskResult, TaskState
from core.logger import get_logger

logger = get_logger("neuralstream.workers")


class WorkerPool:
    """
    Priority-based thread pool for background task execution.

    Args:
        num_workers: Number of worker threads
        queue_size: Max queued tasks before submit blocks
        default_timeout_s: Default task timeout (None = no timeout)
        on_result: Optional callback fired when a task completes
                   (called from the worker thread; keep it fast)
    """

    def __init__(
        self,
        num_workers: int = 2,
        max_queue_size: int = 200,
        default_timeout_s: Optional[float] = 30.0,
        on_result: Optional[Callable[[TaskResult], None]] = None
    ):
        self.num_workers = max(1, num_workers)
        self._max_queue_size = max_queue_size
        self.default_timeout_s = default_timeout_s
        self.on_result = on_result

        # Priority queue wrapped in a lock for thread-safety
        self._queue: "queue.PriorityQueue[Task]" = queue.PriorityQueue(maxsize=max_queue_size)
        self._results: deque[TaskResult] = deque(maxlen=500)
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Stats
        self._stats = defaultdict(int)
        self._results_by_id: Dict[str, TaskResult] = {}

        self._started = False

    # ---- Lifecycle ----

    def start(self) -> None:
        """Launch worker threads (idempotent)."""
        if self._started:
            return
        self._stop_event.clear()
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"nw-worker-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)
        self._started = True
        logger.info(f"WorkerPool started with {self.num_workers} threads")

    def stop(self, graceful: bool = True, wait_timeout: float = 5.0) -> None:
        """Signal workers to stop and join them."""
        if not self._started:
            return
        self._stop_event.set()

        if graceful:
            for t in self._threads:
                t.join(timeout=wait_timeout)

        self._threads.clear()
        self._started = False
        logger.info("WorkerPool stopped")

    # ---- Submission ----

    def submit(self, task: Task, block: bool = True) -> str:
        """
        Submit a task. Returns the task_id.

        Args:
            task: Task instance
            block: If True, wait for queue space; else raise on full queue
        """
        if not self._started:
            self.start()

        try:
            self._queue.put(task, block=block)
            with self._lock:
                self._stats['submitted'] += 1
            return task.task_id
        except queue.Full:
            with self._lock:
                self._stats['rejected'] += 1
            raise

    def submit_fn(
        self,
        fn: Callable,
        *args,
        kind: str = "generic",
        priority: int = 0,
        max_retries: int = 0,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> str:
        """Convenience: build a Task from fn/args and submit it."""
        task = Task(
            fn=fn,
            args=args,
            kwargs=kwargs,
            kind=kind,
            priority=priority,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        return self.submit(task)

    # ---- Waiting ----

    def wait_for(self, task_id: str, timeout_s: Optional[float] = None) -> Optional[TaskResult]:
        """Block until the given task's result is available."""
        deadline = None if timeout_s is None else time.time() + timeout_s
        while True:
            with self._lock:
                if task_id in self._results_by_id:
                    return self._results_by_id[task_id]
            if deadline is not None and time.time() > deadline:
                return None
            time.sleep(0.005)

    def wait_all(self, task_ids: List[str], timeout_s: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple tasks and return their results in order."""
        results = []
        for tid in task_ids:
            r = self.wait_for(tid, timeout_s=timeout_s)
            if r is None:
                results.append(TaskResult(
                    task_id=tid, state=TaskState.TIMEOUT, error="wait_all timeout"
                ))
            else:
                results.append(r)
        return results

    def drain_results(self, max_items: Optional[int] = None) -> List[TaskResult]:
        """Pop completed results from the inbox."""
        with self._lock:
            n = len(self._results) if max_items is None else min(max_items, len(self._results))
            out = [self._results.popleft() for _ in range(n)]
        return out

    # ---- Introspection ----

    @property
    def queue_size(self) -> int:
        """Current number of queued tasks."""
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        """Configured maximum queue capacity."""
        return self._max_queue_size

    @property
    def active_workers(self) -> int:
        return sum(1 for t in self._threads if t.is_alive())

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def status(self) -> Dict[str, Any]:
        return {
            'num_workers': self.num_workers,
            'active_workers': self.active_workers,
            'queue_size': self.queue_size,
            'max_queue_size': self.max_queue_size,
            'started': self._started,
            'stats': self.stats(),
            'recent_results': len(self._results),
        }

    # ---- Internal ----

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._execute(task)
            self._queue.task_done()

    def _execute(self, task: Task) -> None:
        """Execute a single task with retries and timeout wrapper."""
        task.attempts += 1
        task.started_at = time.time()
        task.state = TaskState.RUNNING

        timeout = task.timeout_s if task.timeout_s is not None else self.default_timeout_s

        # Run the work in a small inner thread so timeout can be enforced
        result_holder: Dict[str, Any] = {'value': None, 'error': None}

        def _run():
            try:
                result_holder['value'] = task.fn(*task.args, **task.kwargs)
            except Exception as e:
                result_holder['error'] = e

        inner = threading.Thread(target=_run, daemon=True)
        t0 = time.perf_counter()
        inner.start()
        inner.join(timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if inner.is_alive():
            task.state = TaskState.TIMEOUT
            error_msg = f"Task {task.task_id} exceeded timeout {timeout}s"
        elif result_holder['error'] is not None:
            task.state = TaskState.FAILED
            error_msg = f"{type(result_holder['error']).__name__}: {result_holder['error']}"
        else:
            task.state = TaskState.COMPLETED
            error_msg = None

        task.finished_at = time.time()

        # Retry?
        if task.state == TaskState.FAILED and task.attempts <= task.max_retries:
            task.state = TaskState.PENDING
            try:
                self._queue.put(task, block=False)
                with self._lock:
                    self._stats['retried'] += 1
                return
            except queue.Full:
                pass  # fall through to record failure

        # Record result
        result = TaskResult(
            task_id=task.task_id,
            state=task.state,
            value=result_holder['value'] if task.state == TaskState.COMPLETED else None,
            error=error_msg,
            elapsed_ms=elapsed_ms,
            attempts=task.attempts,
        )

        with self._lock:
            self._results.append(result)
            self._results_by_id[task.task_id] = result
            self._stats[task.state.value] += 1
            # Trim results_by_id to avoid unbounded growth
            if len(self._results_by_id) > 1000:
                for k in list(self._results_by_id.keys())[:200]:
                    del self._results_by_id[k]

        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception as e:
                logger.warning(f"on_result callback failed: {e}")