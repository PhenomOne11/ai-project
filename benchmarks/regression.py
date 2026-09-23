"""
Baseline persistence and regression detection.

A `BenchmarkBaseline` maps benchmark name → {metric: value}.
Regression is detected by comparing current result to baseline
and flagging when a metric crosses a tolerance band.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict

from benchmarks.bench import BenchResult


# =============================================================================
# BASELINE
# =============================================================================

@dataclass
class BaselineEntry:
    """One saved baseline snapshot for a benchmark."""
    name: str
    category: str = "general"
    mean_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    rss_peak_mb: float = 0.0
    throughput_per_s: float = 0.0
    timestamp: float = 0.0

    @classmethod
    def from_result(cls, result: BenchResult) -> "BaselineEntry":
        return cls(
            name=result.name,
            category=result.category,
            mean_ms=result.mean_ms,
            p95_ms=result.p95_ms,
            p99_ms=result.p99_ms,
            rss_peak_mb=result.rss_peak_mb,
            throughput_per_s=result.throughput_per_s,
            timestamp=time.time(),
        )


class BenchmarkBaseline:
    """
    Persistent baseline storage for regression detection.

    Stores one baseline per benchmark name and supports
    save/load to JSON.
    """

    def __init__(self):
        self._entries: Dict[str, BaselineEntry] = {}

    # ---- Mutate ----

    def set(self, entry: BaselineEntry) -> None:
        self._entries[entry.name] = entry

    def update_from_result(self, result: BenchResult) -> None:
        """Overwrite the baseline entry for this benchmark from a fresh result."""
        self.set(BaselineEntry.from_result(result))

    def get(self, name: str) -> Optional[BaselineEntry]:
        return self._entries.get(name)

    def names(self) -> List[str]:
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    # ---- Persistence ----

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            'version': 1,
            'saved_at': time.time(),
            'entries': {
                name: asdict(entry)
                for name, entry in self._entries.items()
            },
        }
        p.write_text(json.dumps(obj, indent=2))

    @classmethod
    def load(cls, path: str) -> "BenchmarkBaseline":
        p = Path(path)
        baseline = cls()
        if not p.exists():
            return baseline

        try:
            obj = json.loads(p.read_text())
        except Exception:
            return baseline

        entries = obj.get('entries', {})
        for name, data in entries.items():
            try:
                baseline.set(BaselineEntry(**data))
            except TypeError:
                continue

        return baseline


# =============================================================================
# REGRESSION DETECTION
# =============================================================================

@dataclass
class RegressionVerdict:
    """Outcome of comparing a current result to its baseline."""
    name: str
    passed: bool
    reason: str
    deltas: Dict[str, float] = field(default_factory=dict)
    levels: Dict[str, str] = field(default_factory=dict)   # metric → nominal|warning|critical


def compare_to_baseline(
    result: BenchResult,
    baseline: Optional[BaselineEntry],
    latency_tolerance_pct: float = 15.0,
    memory_tolerance_pct: float = 20.0,
    absolute_latency_floor_ms: float = 0.5,
    absolute_memory_floor_mb: float = 10.0,
) -> RegressionVerdict:
    """
    Compare a current benchmark result to its baseline.

    Verdict is PASS if no metric exceeds its tolerance band.
    Warning band: between 1× and 2× the tolerance.
    Critical band: greater than 2× the tolerance.

    Sub-floor guard: if the absolute delta is under `absolute_latency_floor_ms`
    (or `absolute_memory_floor_mb`), the metric is treated as nominal
    regardless of percentage change. This prevents microbenchmark
    noise from triggering false regressions.
    """
    if baseline is None:
        return RegressionVerdict(
            name=result.name, passed=True,
            reason="no baseline yet",
            deltas={}, levels={},
        )

    deltas: Dict[str, float] = {}
    levels: Dict[str, str] = {}
    any_critical = False
    any_warning = False

    def pct_change(new: float, old: float) -> float:
        if old == 0:
            return 0.0 if new == 0 else float('inf')
        return (new - old) / old * 100.0

    def classify(pct: float, tolerance: float, absolute_delta: float, absolute_floor: float) -> str:
        # Absolute floor: microbenchmarks with tiny absolute deltas are
        # treated as nominal regardless of percentage.
        if abs(absolute_delta) <= absolute_floor:
            return "nominal"
        if pct <= tolerance:
            return "nominal"
        if pct <= tolerance * 2:
            return "warning"
        return "critical"

    # Mean latency
    mean_delta_ms = result.mean_ms - baseline.mean_ms
    mean_pct = pct_change(result.mean_ms, baseline.mean_ms)
    deltas['mean_ms'] = mean_pct
    levels['mean_ms'] = classify(mean_pct, latency_tolerance_pct,
                                 mean_delta_ms, absolute_latency_floor_ms)

    # p95 latency
    p95_delta_ms = result.p95_ms - baseline.p95_ms
    p95_pct = pct_change(result.p95_ms, baseline.p95_ms)
    deltas['p95_ms'] = p95_pct
    levels['p95_ms'] = classify(p95_pct, latency_tolerance_pct,
                                p95_delta_ms, absolute_latency_floor_ms)

    # Peak RSS
    mem_delta_mb = result.rss_peak_mb - baseline.rss_peak_mb
    mem_pct = pct_change(result.rss_peak_mb, baseline.rss_peak_mb)
    deltas['rss_peak_mb'] = mem_pct
    levels['rss_peak_mb'] = classify(mem_pct, memory_tolerance_pct,
                                     mem_delta_mb, absolute_memory_floor_mb)

    for level in levels.values():
        if level == "critical":
            any_critical = True
        elif level == "warning":
            any_warning = True

    if any_critical:
        passed = False
        reason = "critical regression detected"
    elif any_warning:
        passed = True
        reason = "warning-level regression"
    else:
        passed = True
        reason = "nominal"

    return RegressionVerdict(
        name=result.name,
        passed=passed,
        reason=reason,
        deltas=deltas,
        levels=levels,
    )