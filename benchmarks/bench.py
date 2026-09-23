"""
Core benchmark primitives.
Provides `Bencher`, a memory-and-latency-aware wrapper that measures
a callable over N iterations and produces a `BenchResult`.
"""

import time
import gc
import numpy as np
import psutil
import os
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field


# =============================================================================
# RESULT RECORDS
# =============================================================================

@dataclass
class BenchResult:
    """Outcome of a single benchmark."""
    name: str
    category: str = "general"

    # Latency
    latencies_ms: List[float] = field(default_factory=list)

    # Memory
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_delta_mb: float = 0.0

    # Metadata
    iterations: int = 0
    warmup: int = 0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- Statistics ----

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.latencies_ms)) if self.latencies_ms else 0.0

    def percentile(self, q: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, q))

    @property
    def p50_ms(self) -> float: return self.percentile(50)
    @property
    def p95_ms(self) -> float: return self.percentile(95)
    @property
    def p99_ms(self) -> float: return self.percentile(99)

    @property
    def throughput_per_s(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category,
            'iterations': self.iterations,
            'warmup': self.warmup,
            'mean_ms': self.mean_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'std_ms': self.std_ms,
            'p50_ms': self.p50_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'throughput_per_s': self.throughput_per_s,
            'rss_before_mb': self.rss_before_mb,
            'rss_after_mb': self.rss_after_mb,
            'rss_peak_mb': self.rss_peak_mb,
            'rss_delta_mb': self.rss_delta_mb,
            'error': self.error,
            'extra': dict(self.extra),
        }


# =============================================================================
# BENCHER
# =============================================================================

class Bencher:
    """
    Memory-and-latency-aware benchmark runner.

    Usage:
        bencher = Bencher(warmup=5, iterations=50)
        result = bencher.run("forward_pass", lambda: model(x))
    """

    def __init__(
        self,
        warmup: int = 5,
        iterations: int = 50,
        track_memory: bool = True,
        pause_ms: float = 0.0,
        gc_between: bool = False,
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.track_memory = track_memory
        self.pause_ms = pause_ms
        self.gc_between = gc_between

        self._process = psutil.Process(os.getpid())

    def _rss_mb(self) -> float:
        try:
            return self._process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def run(
        self,
        name: str,
        fn: Callable[[], Any],
        category: str = "general",
        extra: Optional[Dict[str, Any]] = None,
    ) -> BenchResult:
        """
        Run a benchmark over `iterations` and return aggregated metrics.
        """
        result = BenchResult(
            name=name, category=category,
            iterations=self.iterations, warmup=self.warmup,
            extra=dict(extra or {}),
        )

        try:
            # Warmup
            for _ in range(self.warmup):
                fn()

            if self.gc_between:
                gc.collect()

            rss_before = self._rss_mb() if self.track_memory else 0.0
            rss_peak = rss_before

            latencies = []
            for _ in range(self.iterations):
                t0 = time.perf_counter()
                fn()
                latencies.append((time.perf_counter() - t0) * 1000.0)

                if self.track_memory:
                    rss_peak = max(rss_peak, self._rss_mb())

                if self.pause_ms > 0:
                    time.sleep(self.pause_ms / 1000.0)

            result.latencies_ms = latencies

            if self.track_memory:
                rss_after = self._rss_mb()
                result.rss_before_mb = rss_before
                result.rss_after_mb = rss_after
                result.rss_peak_mb = rss_peak
                result.rss_delta_mb = rss_after - rss_before

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"

        return result


# =============================================================================
# RICHER METRICS (optional helper)
# =============================================================================

def latency_summary(latencies_ms: List[float]) -> Dict[str, float]:
    """Utility: compute summary stats from a list of latencies."""
    if not latencies_ms:
        return {'n': 0, 'mean': 0.0, 'min': 0.0, 'max': 0.0,
                'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'std': 0.0}
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        'n': int(arr.size),
        'mean': float(arr.mean()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'std': float(arr.std()),
    }