"""
Core metrics primitives: Counter, Gauge, Histogram, Timer.

All classes are thread-safe and designed for high-frequency updates
from a single or few writer threads with many readers.
"""

import threading
import time
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque


class Counter:
    """
    Monotonically increasing counter. Thread-safe.
    Use `inc()` to add, `value` to read.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0

    def to_dict(self) -> dict:
        return {'name': self.name, 'value': self.value, 'type': 'counter'}


class Gauge:
    """
    Point-in-time value that can go up or down. Thread-safe.
    Use `set()` to update, `value` to read.
    """

    def __init__(self, name: str, description: str = "", initial: float = 0.0):
        self.name = name
        self.description = description
        self._value: float = float(initial)
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0

    def to_dict(self) -> dict:
        return {'name': self.name, 'value': self.value, 'type': 'gauge'}


class Histogram:
    """
    Value distribution with configurable bins and quantile queries.

    Stores a bounded rolling window of samples for efficient percentile
    computation. Bins are used for dashboard histogram rendering.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        window_size: int = 1000,
        bins: int = 50,
        range_min: float = 0.0,
        range_max: float = 10.0
    ):
        self.name = name
        self.description = description
        self.window_size = window_size
        self.bins = bins
        self.range_min = range_min
        self.range_max = range_max

        self._samples: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._total_count: int = 0
        self._sum: float = 0.0

    def observe(self, value: float) -> None:
        """Record a new observation."""
        value = float(value)
        with self._lock:
            self._samples.append(value)
            self._total_count += 1
            self._sum += value

    @property
    def count(self) -> int:
        with self._lock:
            return self._total_count

    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum

    @property
    def mean(self) -> float:
        with self._lock:
            n = len(self._samples)
            if n == 0:
                return 0.0
            return float(sum(self._samples)) / n

    def percentile(self, q: float) -> float:
        """Return the q-th percentile of the current window (q in [0, 100])."""
        with self._lock:
            if not self._samples:
                return 0.0
            return float(np.percentile(list(self._samples), q))

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def min(self) -> float:
        with self._lock:
            return float(min(self._samples)) if self._samples else 0.0

    @property
    def max(self) -> float:
        with self._lock:
            return float(max(self._samples)) if self._samples else 0.0

    def bin_counts(self) -> np.ndarray:
        """Return histogram bin counts for rendering."""
        with self._lock:
            if not self._samples:
                return np.zeros(self.bins, dtype=np.int64)
            edges = np.linspace(self.range_min, self.range_max, self.bins + 1)
            counts, _ = np.histogram(list(self._samples), bins=edges)
            return counts.astype(np.int64)

    def samples(self) -> List[float]:
        """Return a copy of the current window."""
        with self._lock:
            return list(self._samples)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._total_count = 0
            self._sum = 0.0

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': 'histogram',
            'count': self.count,
            'mean': self.mean,
            'p50': self.p50,
            'p95': self.p95,
            'p99': self.p99,
            'min': self.min,
            'max': self.max,
        }


class Timer:
    """
    Context manager and decorator for measuring durations.
    Feeds observations into an underlying Histogram.
    """

    def __init__(self, histogram: Histogram):
        self._hist = histogram
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            self._hist.observe(elapsed_ms)
        self._start = None

    @property
    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (time.perf_counter() - self._start) * 1000.0