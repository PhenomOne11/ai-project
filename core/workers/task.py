"""
Task and Result primitives for the worker pool.

A Task wraps a callable + arguments + metadata. A TaskResult wraps
the outcome (success/failure) plus timing and error details.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Dict


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """A unit of work scheduled on the pool."""
    fn: Callable[..., Any] = field(repr=False)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    # Metadata
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str = "generic"
    priority: int = 0                   # higher = more important
    max_retries: int = 0
    timeout_s: Optional[float] = None

    # Runtime (filled by pool)
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed_ms(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return (end - self.started_at) * 1000.0

    def __lt__(self, other: "Task") -> bool:
        # Higher priority first, then FIFO by creation time
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


@dataclass
class TaskResult:
    """Outcome of executing a task."""
    task_id: str
    state: TaskState
    value: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    attempts: int = 0
    finished_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.state == TaskState.COMPLETED

    def unwrap(self) -> Any:
        """Return the value or raise if failed."""
        if not self.ok:
            raise RuntimeError(f"Task {self.task_id} failed: {self.error}")
        return self.value