"""
Unit Tests for dashboard.panels.anomaly_console
"""

import numpy as np
import pytest


class TestAnomalyRecord:
    def test_construction(self):
        from dashboard.panels.anomaly_console import AnomalyRecord
        r = AnomalyRecord(
            tick=0, timestamp=1.0, fused_score=0.42,
            recon_component=0.2, iso_component=0.1, maha_component=0.12,
        )
        assert r.fused_score == 0.42
        assert r.severity == "nominal"

    def test_to_dict(self):
        from dashboard.panels.anomaly_console import AnomalyRecord
        r = AnomalyRecord(tick=1, timestamp=2.0, fused_score=0.4)
        d = r.to_dict()
        assert d['tick'] == 1
        assert d['fused_score'] == 0.4


class TestAdaptationRecord:
    def test_delta(self):
        from dashboard.panels.anomaly_console import AdaptationRecord
        a = AdaptationRecord(tick=0, timestamp=1.0, loss_before=0.5, loss_after=0.3)
        assert abs(a.delta - 0.2) < 1e-9

    def test_negative_delta(self):
        from dashboard.panels.anomaly_console import AdaptationRecord
        a = AdaptationRecord(tick=0, timestamp=1.0, loss_before=0.3, loss_after=0.5)
        assert a.delta < 0


class TestAnomalyConsoleState:
    def test_empty_init(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        assert s.is_empty
        assert s.size == 0
        assert s.ood_count == 0

    def test_push_score_classification(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState(
            warning_threshold=0.5, critical_threshold=0.75, emergency_threshold=0.9,
        )
        r1 = s.push_score(0.3)
        r2 = s.push_score(0.6)
        r3 = s.push_score(0.8)
        r4 = s.push_score(0.95)
        assert r1.severity == "nominal"
        assert r2.severity == "warning"
        assert r3.severity == "critical"
        assert r4.severity == "emergency"

    def test_buffer_bound(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState(history_length=10)
        for i in range(50):
            s.push_score(0.1)
        assert s.size == 10

    def test_to_arrays(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        for i in range(8):
            s.push_score(0.1 * i, recon_component=0.05 * i)
        cols = s.to_arrays()
        assert cols['fused_score'].shape == (8,)
        assert cols['recon_component'][-1] == pytest.approx(0.35, abs=1e-6)

    def test_ood_events(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        for i in range(5):
            s.push_ood_event(priority=0.4 + i * 0.1, label=f"batch-{i}")
        assert s.ood_count == 5
        priorities = s.ood_priorities()
        assert priorities.shape == (5,)
        assert priorities.max() == pytest.approx(0.8, abs=1e-6)

    def test_adaptation_events(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        for i in range(3):
            s.push_adaptation(loss_before=0.5, loss_after=0.4 - i * 0.02,
                              ewc_enabled=True)
        assert s.adaptation_count == 3
        arrs = s.adaptation_arrays()
        assert arrs['delta'].shape == (3,)

    def test_circuit_breaker_trip(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        s.push_score(0.6)
        s.mark_circuit_breaker_trip()
        s.mark_circuit_breaker_trip(tick=42)
        assert len(s._circuit_breaker_trips) == 2

    def test_stats_empty(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        stats = s.stats()
        assert stats['n_samples'] == 0
        assert stats['severity_counts']['nominal'] == 0

    def test_stats_basic(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        for _ in range(20):
            s.push_score(0.3)
        for _ in range(5):
            s.push_score(0.7)  # warning by default
        stats = s.stats()
        assert stats['n_samples'] == 25
        assert stats['severity_counts']['nominal'] == 20
        assert stats['severity_counts']['warning'] == 5

    def test_stats_ema_exists(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        for _ in range(30):
            s.push_score(0.5)
        stats = s.stats()
        # EMA should converge to ~0.5
        assert 0.4 < stats['score_ema'] < 0.6

    def test_ingest_anomaly_result(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        import torch
        s = AnomalyConsoleState()
        result = {
            'score': torch.tensor([0.1, 0.3, 0.8]),
            'recon_component': torch.tensor([0.05, 0.15, 0.4]),
            'isolation_component': torch.tensor([0.03, 0.09, 0.25]),
            'mahalanobis_component': torch.tensor([0.02, 0.06, 0.15]),
        }
        s.ingest_anomaly_result(result)
        assert s.size == 3
        cols = s.to_arrays()
        assert cols['fused_score'][2] == pytest.approx(0.8, abs=1e-6)

    def test_seed_demo(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        s.seed_demo(200)
        assert s.size == 200
        stats = s.stats()
        # Demo has warning + critical clusters
        assert stats['severity_counts']['warning'] > 0
        assert stats['severity_counts']['critical'] > 0
        # Demo has OOD events + adaptations
        assert stats['ood_count'] > 0
        assert stats['adaptation_count'] > 0
        # Demo has a circuit breaker trip
        assert stats['circuit_breaker_trips'] == 1

    def test_seed_demo_adaptation_improves(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        s.seed_demo(300)
        stats = s.stats()
        # Sum of deltas should be positive (net improvement)
        assert stats['adaptation_total_improvement'] > 0

    def test_reset(self):
        from dashboard.panels.anomaly_console import AnomalyConsoleState
        s = AnomalyConsoleState()
        s.seed_demo(50)
        assert not s.is_empty
        s.reset()
        assert s.is_empty
        assert s.ood_count == 0
        assert s.adaptation_count == 0


class TestFigureBuilders:
    def test_score_timeline_empty(self):
        from dashboard.panels.anomaly_console import build_score_timeline, AnomalyConsoleState
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        fig = build_score_timeline(s.to_arrays(), get_palette("cyber_dark"))
        assert fig is not None

    def test_score_timeline_basic(self):
        from dashboard.panels.anomaly_console import build_score_timeline, AnomalyConsoleState
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        s.seed_demo(80)
        fig = build_score_timeline(s.to_arrays(), get_palette("cyber_dark"), height=400)
        assert len(fig.data) >= 1
        assert fig.layout.height == 400

    def test_score_timeline_with_cb(self):
        from dashboard.panels.anomaly_console import build_score_timeline, AnomalyConsoleState
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        s.seed_demo(80)
        fig = build_score_timeline(
            s.to_arrays(), get_palette("cyber_dark"),
            cb_trip_ticks=s._circuit_breaker_trips,
        )
        assert fig is not None

    def test_component_breakdown(self):
        from dashboard.panels.anomaly_console import build_component_breakdown, AnomalyConsoleState
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        s.seed_demo(60)
        fig = build_component_breakdown(s.to_arrays(), get_palette("cyber_dark"))
        assert len(fig.data) == 3

    def test_score_histogram(self):
        from dashboard.panels.anomaly_console import build_score_histogram, AnomalyConsoleState
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        s.seed_demo(100)
        cols = s.to_arrays()
        fig = build_score_histogram(cols['fused_score'], get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_ood_buffer_view(self):
        from dashboard.panels.anomaly_console import build_ood_buffer_view
        from dashboard.theme import get_palette
        priorities = np.random.rand(30)
        fig = build_ood_buffer_view(priorities, get_palette("cyber_dark"), top_n=10)
        assert len(fig.data) == 1
        # Should have top-10 bars
        assert len(fig.data[0].y) == 10

    def test_ood_buffer_view_empty(self):
        from dashboard.panels.anomaly_console import build_ood_buffer_view
        from dashboard.theme import get_palette
        fig = build_ood_buffer_view(np.array([]), get_palette("cyber_dark"))
        assert fig is not None

    def test_adaptation_history(self):
        from dashboard.panels.anomaly_console import build_adaptation_history, AnomalyConsoleState
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        s.seed_demo(100)
        fig = build_adaptation_history(s.adaptation_arrays(), get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_all_palettes(self):
        from dashboard.panels.anomaly_console import (
            AnomalyConsoleState,
            build_score_timeline, build_component_breakdown,
            build_score_histogram, build_ood_buffer_view, build_adaptation_history,
        )
        from dashboard.theme import get_palette
        s = AnomalyConsoleState()
        s.seed_demo(100)
        cols = s.to_arrays()

        for pname in ["cyber_dark", "plasma", "mono", "solar"]:
            pal = get_palette(pname)
            build_score_timeline(cols, pal)
            build_component_breakdown(cols, pal)
            build_score_histogram(cols['fused_score'], pal)
            build_ood_buffer_view(s.ood_priorities(), pal)
            build_adaptation_history(s.adaptation_arrays(), pal)


class TestPackageExports:
    def test_exports(self):
        from dashboard.panels import (
            AnomalyConsoleState, render_anomaly_console,
            build_score_timeline, build_component_breakdown,
            build_score_histogram, build_ood_buffer_view, build_adaptation_history,
        )
        assert callable(render_anomaly_console)
        assert callable(build_score_timeline)
        assert AnomalyConsoleState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])