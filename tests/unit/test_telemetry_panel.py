"""
Unit Tests for dashboard.panels.telemetry
"""

import numpy as np
import pytest


class TestTelemetrySample:
    def test_construction(self):
        from dashboard.panels.telemetry import TelemetrySample
        s = TelemetrySample(timestamp=1.0, tick=0, cpu_percent=42.0, rss_mb=512.0)
        assert s.cpu_percent == 42.0
        assert s.rss_mb == 512.0

    def test_to_dict(self):
        from dashboard.panels.telemetry import TelemetrySample
        s = TelemetrySample(timestamp=1.0, tick=3, cpu_percent=10.0)
        d = s.to_dict()
        assert d['tick'] == 3
        assert d['cpu_percent'] == 10.0


class TestTelemetryState:
    def test_empty_init(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        assert s.is_empty
        assert s.size == 0
        assert s.total_pushed == 0

    def test_append_and_size(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        for i in range(10):
            s.append(cpu_percent=10.0 + i)
        assert s.size == 10
        assert s.total_pushed == 10

    def test_buffer_bound(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState(history_length=5)
        for i in range(20):
            s.append(cpu_percent=float(i))
        assert s.size == 5
        assert s.total_pushed == 20
        latest = s.latest()
        assert latest.cpu_percent == 19.0

    def test_to_arrays(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        for i in range(8):
            s.append(cpu_percent=float(i), rss_mb=100.0 + i, latency_ms=1.0 + i * 0.1)
        cols = s.to_arrays()
        assert 'tick' in cols
        assert cols['cpu_percent'].shape == (8,)
        assert cols['rss_mb'][0] == 100.0
        assert cols['latency_ms'][-1] == pytest.approx(1.7, abs=1e-6)

    def test_stats_empty(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        stats = s.stats()
        assert stats['n_samples'] == 0
        assert stats['overall'] == 'nominal'

    def test_stats_basic(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState(history_length=100)
        for i in range(50):
            s.append(cpu_percent=50.0, rss_mb=500.0, latency_ms=2.0)
        stats = s.stats()
        assert stats['n_samples'] == 50
        assert stats['cpu']['mean'] == pytest.approx(50.0, abs=1e-6)
        assert stats['latency']['p95'] == pytest.approx(2.0, abs=1e-6)

    def test_health_nominal(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState(history_length=100)
        for _ in range(30):
            s.append(cpu_percent=30.0, rss_mb=300.0, latency_ms=1.0)
        stats = s.stats()
        assert stats['overall'] == 'nominal'
        assert stats['health']['cpu'] == 'nominal'
        assert stats['health']['rss'] == 'nominal'
        assert stats['health']['latency'] == 'nominal'

    def test_health_warning(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState(history_length=100, cpu_warning=40.0, cpu_critical=80.0)
        for _ in range(30):
            s.append(cpu_percent=55.0, rss_mb=300.0, latency_ms=1.0)
        stats = s.stats()
        assert stats['health']['cpu'] == 'warning'
        assert stats['overall'] == 'warning'

    def test_health_critical(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState(
            history_length=100,
            latency_warning_ms=3.0, latency_critical_ms=6.0,
        )
        for _ in range(30):
            s.append(latency_ms=8.0)
        stats = s.stats()
        assert stats['health']['latency'] == 'critical'
        assert stats['overall'] == 'critical'

    def test_latency_histogram(self):
        from dashboard.panels.telemetry import TelemetryState
        # Large enough buffer to hold all 500 samples
        s = TelemetryState(history_length=500)
        rng = np.random.default_rng(7)
        for _ in range(500):
            s.append(latency_ms=float(abs(rng.normal(2.0, 0.5))))
        counts, edges = s.latency_histogram(bins=20)
        assert counts.shape == (20,)
        assert edges.shape == (21,)
        assert counts.sum() == 500

    def test_latency_histogram_empty(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        counts, edges = s.latency_histogram(10)
        assert counts.shape == (10,)
        assert counts.sum() == 0

    def test_seed_demo(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        s.seed_demo(150)
        assert s.size == 150
        stats = s.stats()
        # Demo includes a latency spike → should trigger warning or critical
        assert stats['latency']['max'] > 3.0

    def test_seed_demo_shows_warmup_and_spike(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        s.seed_demo(200, seed=1)
        cols = s.to_arrays()
        # First CPU should be lower than mid-run plateau
        first = float(np.mean(cols['cpu_percent'][:10]))
        mid = float(np.mean(cols['cpu_percent'][80:120]))
        assert mid > first

    def test_reset(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        s.seed_demo(50)
        assert not s.is_empty
        s.reset()
        assert s.is_empty
        assert s.total_pushed == 0

    def test_auto_probe(self):
        from dashboard.panels.telemetry import TelemetryState
        s = TelemetryState()
        sample = s.append_auto(latency_ms=2.5)
        assert sample is not None
        # RSS should always be > 0 for a running process
        assert sample.rss_mb > 0
        assert s.size == 1


class TestFigureBuilders:
    def test_multi_metric_empty(self):
        from dashboard.panels.telemetry import build_multi_metric_chart, TelemetryState
        from dashboard.theme import get_palette
        s = TelemetryState()
        fig = build_multi_metric_chart(s.to_arrays(), get_palette("cyber_dark"))
        assert fig is not None

    def test_multi_metric_basic(self):
        from dashboard.panels.telemetry import build_multi_metric_chart, TelemetryState
        from dashboard.theme import get_palette
        s = TelemetryState()
        s.seed_demo(50)
        fig = build_multi_metric_chart(s.to_arrays(), get_palette("cyber_dark"), height=400)
        assert fig.layout.height == 400
        # Should have at least 5 traces (2 CPU + 2 mem + 1 latency)
        assert len(fig.data) >= 5

    def test_gauge_nominal(self):
        from dashboard.panels.telemetry import build_gauge
        from dashboard.theme import get_palette
        fig = build_gauge(
            title="CPU", value=30.0, max_value=100.0,
            warning=70.0, critical=90.0,
            palette=get_palette("cyber_dark"),
        )
        assert fig is not None
        # Single indicator trace
        assert len(fig.data) >= 1

    def test_gauge_critical(self):
        from dashboard.panels.telemetry import build_gauge
        from dashboard.theme import get_palette
        fig = build_gauge(
            title="CPU", value=95.0, max_value=100.0,
            warning=70.0, critical=90.0,
            palette=get_palette("cyber_dark"),
        )
        assert fig is not None

    def test_histogram_basic(self):
        from dashboard.panels.telemetry import build_latency_histogram, TelemetryState
        from dashboard.theme import get_palette
        s = TelemetryState()
        s.seed_demo(100)
        counts, edges = s.latency_histogram(20)
        fig = build_latency_histogram(
            counts, edges, get_palette("cyber_dark"), height=300,
        )
        assert len(fig.data) >= 1
        assert fig.layout.height == 300

    def test_histogram_empty(self):
        from dashboard.panels.telemetry import build_latency_histogram
        from dashboard.theme import get_palette
        fig = build_latency_histogram(
            np.zeros(10, dtype=np.int64), np.linspace(0, 1, 11),
            get_palette("cyber_dark"),
        )
        assert fig is not None

    def test_all_palettes(self):
        from dashboard.panels.telemetry import (
            build_multi_metric_chart, build_gauge,
            build_latency_histogram, TelemetryState,
        )
        from dashboard.theme import get_palette
        s = TelemetryState()
        s.seed_demo(80)
        counts, edges = s.latency_histogram(20)
        for pname in ["cyber_dark", "plasma", "mono", "solar"]:
            pal = get_palette(pname)
            build_multi_metric_chart(s.to_arrays(), pal)
            build_gauge("CPU", 40.0, 100.0, 70.0, 90.0, pal)
            build_latency_histogram(counts, edges, pal)


class TestPackageExports:
    def test_panels_exports(self):
        from dashboard.panels import (
            TelemetryState, render_telemetry,
            build_multi_metric_chart, build_gauge, build_latency_histogram,
        )
        assert TelemetryState is not None
        assert callable(render_telemetry)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])