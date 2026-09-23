"""
Central Metrics Registry
Holds all named counters/gauges/histograms and provides bulk operations.
"""

import time
import json
import threading
from typing import Dict, Optional, Any, List, Callable
from core.metrics.collector import Counter, Gauge, Histogram, Timer
from core.metrics.system import snapshot_all


class MetricsRegistry:
    """
    Central registry for all system metrics.

    Usage:
        reg = MetricsRegistry()
        reg.counter("batches_processed").inc()
        reg.gauge("active_precision").set(1)
        with reg.timer("forward_latency_ms"):
            do_forward_pass()
    """

    def __init__(self, name: str = "neuralstream"):
        self.name = name
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self.created_at = time.time()

    # ---- Named metric accessors (create-if-missing) ----

    def counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "", initial: float = 0.0) -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, initial)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        window_size: int = 1000,
        bins: int = 50,
        range_min: float = 0.0,
        range_max: float = 100.0
    ) -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, description, window_size, bins, range_min, range_max
                )
            return self._histograms[name]

    def timer(self, name: str, description: str = "") -> Timer:
        return Timer(self.histogram(name, description))

    # ---- Bulk operations ----

    def snapshot(self) -> Dict[str, Any]:
        """Full snapshot of all counter/gauge/histogram values."""
        with self._lock:
            return {
                'registry': self.name,
                'timestamp': time.time(),
                'uptime_s': time.time() - self.created_at,
                'counters': {n: c.value for n, c in self._counters.items()},
                'gauges': {n: g.value for n, g in self._gauges.items()},
                'histograms': {
                    n: h.to_dict() for n, h in self._histograms.items()
                },
            }

    def snapshot_system(self) -> Dict[str, Any]:
        """Extended snapshot including system telemetry."""
        data = self.snapshot()
        data['system'] = snapshot_all()
        return data

    def export_json(self, path: Optional[str] = None) -> str:
        """Serialize the snapshot to JSON, optionally writing to disk."""
        data = self.snapshot_system()
        # Convert any non-serializable values
        text = json.dumps(data, default=str, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(text)
        return text

    def list_names(self) -> Dict[str, List[str]]:
        with self._lock:
            return {
                'counters': list(self._counters.keys()),
                'gauges': list(self._gauges.keys()),
                'histograms': list(self._histograms.keys()),
            }

    def reset(self) -> None:
        with self._lock:
            for c in self._counters.values():
                c.reset()
            for g in self._gauges.values():
                g.reset()
            for h in self._histograms.values():
                h.reset()

    def __len__(self) -> int:
        with self._lock:
            return len(self._counters) + len(self._gauges) + len(self._histograms)


# ---- Module-level default registry ----

_DEFAULT_REGISTRY: Optional[MetricsRegistry] = None


def get_default_registry() -> MetricsRegistry:
    """Return the module-level default registry (lazy-init)."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = MetricsRegistry()
    return _DEFAULT_REGISTRY