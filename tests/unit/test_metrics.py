"""
Unit Tests for core.metrics
"""

import time
import json
import tempfile
from pathlib import Path
import pytest


class TestCounter:
    def test_initial_zero(self):
        from core.metrics.collector import Counter
        c = Counter("test")
        assert c.value == 0.0

    def test_increment(self):
        from core.metrics.collector import Counter
        c = Counter("test")
        c.inc()
        c.inc(5)
        assert c.value == 6.0

    def test_reset(self):
        from core.metrics.collector import Counter
        c = Counter("test")
        c.inc(10)
        c.reset()
        assert c.value == 0.0

    def test_to_dict(self):
        from core.metrics.collector import Counter
        c = Counter("test", "my counter")
        c.inc(3)
        d = c.to_dict()
        assert d['name'] == 'test'
        assert d['value'] == 3.0
        assert d['type'] == 'counter'


class TestGauge:
    def test_set_and_read(self):
        from core.metrics.collector import Gauge
        g = Gauge("g")
        g.set(42.0)
        assert g.value == 42.0

    def test_inc_dec(self):
        from core.metrics.collector import Gauge
        g = Gauge("g", initial=10)
        g.inc(5)
        g.dec(3)
        assert g.value == 12.0

    def test_reset(self):
        from core.metrics.collector import Gauge
        g = Gauge("g", initial=5)
        g.reset()
        assert g.value == 0.0


class TestHistogram:
    def test_empty_stats(self):
        from core.metrics.collector import Histogram
        h = Histogram("test")
        assert h.mean == 0.0
        assert h.p95 == 0.0
        assert h.count == 0

    def test_observe_and_stats(self):
        from core.metrics.collector import Histogram
        h = Histogram("test", window_size=100)
        for v in range(100):
            h.observe(float(v))
        assert h.count == 100
        assert 45 <= h.mean <= 55
        assert 90 <= h.p95 <= 99
        assert h.min == 0.0
        assert h.max == 99.0

    def test_window_rollover(self):
        from core.metrics.collector import Histogram
        h = Histogram("test", window_size=10)
        for v in range(100):
            h.observe(float(v))
        # Window should keep only last 10
        assert len(h.samples()) == 10
        assert h.min == 90.0
        # But total count is preserved
        assert h.count == 100

    def test_percentile_ordering(self):
        from core.metrics.collector import Histogram
        h = Histogram("test")
        for v in range(1000):
            h.observe(float(v))
        assert h.p95 > h.p50
        assert h.p99 >= h.p95

    def test_bin_counts(self):
        from core.metrics.collector import Histogram
        import numpy as np
        h = Histogram("test", bins=10, range_min=0.0, range_max=10.0)
        for v in range(100):
            h.observe(float(v % 10))
        counts = h.bin_counts()
        assert len(counts) == 10
        assert int(counts.sum()) == 100

    def test_reset(self):
        from core.metrics.collector import Histogram
        h = Histogram("test")
        for _ in range(50):
            h.observe(1.0)
        h.reset()
        assert h.count == 0
        assert len(h.samples()) == 0


class TestTimer:
    def test_context_manager(self):
        from core.metrics.collector import Histogram, Timer
        h = Histogram("test")
        with Timer(h):
            time.sleep(0.01)
        assert h.count == 1
        assert h.samples()[0] > 5.0  # > 5ms

    def test_elapsed_property(self):
        from core.metrics.collector import Histogram, Timer
        h = Histogram("test")
        t = Timer(h)
        with t:
            time.sleep(0.01)
        # After exit, elapsed_ms should still be readable
        assert h.count == 1


class TestSystemProbes:
    def test_probe_process(self):
        from core.metrics.system import probe_process
        p = probe_process()
        assert 'rss_mb' in p
        assert p['rss_mb'] > 0
        assert 'cpu_percent' in p

    def test_probe_system(self):
        from core.metrics.system import probe_system
        s = probe_system()
        assert 'cpu_percent' in s
        assert 'mem_percent' in s
        assert s['mem_total_mb'] > 0
        assert s['cpu_count'] >= 1

    def test_snapshot_all(self):
        from core.metrics.system import snapshot_all
        snap = snapshot_all()
        assert 'process' in snap
        assert 'system' in snap


class TestMetricsRegistry:
    def test_create_named_metrics(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        c = reg.counter("batches")
        g = reg.gauge("precision_id")
        h = reg.histogram("latency_ms")

        c.inc(3)
        g.set(1.0)
        h.observe(5.0)
        h.observe(10.0)

        # Idempotent access
        assert reg.counter("batches").value == 3.0
        assert reg.gauge("precision_id").value == 1.0
        assert reg.histogram("latency_ms").count == 2

    def test_timer_helper(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        with reg.timer("op_latency"):
            time.sleep(0.005)
        assert reg.histogram("op_latency").count == 1

    def test_snapshot(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        reg.counter("a").inc(10)
        reg.gauge("b").set(2.5)
        reg.histogram("c").observe(1.0)

        snap = reg.snapshot()
        assert snap['counters']['a'] == 10.0
        assert snap['gauges']['b'] == 2.5
        assert 'c' in snap['histograms']

    def test_snapshot_system(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        snap = reg.snapshot_system()
        assert 'system' in snap
        assert 'process' in snap['system']

    def test_export_json(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        reg.counter("x").inc(1)
        text = reg.export_json()
        parsed = json.loads(text)
        assert parsed['counters']['x'] == 1.0

    def test_export_json_to_file(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        reg.counter("y").inc(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            reg.export_json(str(path))
            assert path.exists()
            content = json.loads(path.read_text())
            assert content['counters']['y'] == 5.0

    def test_list_names(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        reg.counter("c1")
        reg.gauge("g1")
        reg.histogram("h1")
        names = reg.list_names()
        assert 'c1' in names['counters']
        assert 'g1' in names['gauges']
        assert 'h1' in names['histograms']

    def test_reset(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        reg.counter("a").inc(5)
        reg.gauge("b").set(3)
        reg.histogram("c").observe(1.0)
        reg.reset()
        assert reg.counter("a").value == 0.0
        assert reg.gauge("b").value == 0.0
        assert reg.histogram("c").count == 0

    def test_default_registry_singleton(self):
        from core.metrics.registry import get_default_registry
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2

    def test_len(self):
        from core.metrics.registry import MetricsRegistry
        reg = MetricsRegistry()
        reg.counter("c")
        reg.gauge("g")
        reg.histogram("h")
        assert len(reg) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])