"""
Unit Tests for dashboard.panels.drift_heatmap
"""

import numpy as np
import pytest


class TestDriftHeatmapState:
    def test_empty_init(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        assert s.is_empty
        assert s.num_layers == 0

    def test_push_snapshot_inserts_layers(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        s.push_snapshot({"layer_a": 0.1, "layer_b": 0.3})
        assert s.num_layers == 2
        assert not s.is_empty

    def test_push_snapshot_missing_layers_become_nan(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState(layer_names=["a", "b", "c"])
        s.push_snapshot({"a": 0.1, "b": 0.2})
        matrix, names = s.as_matrix()
        assert matrix.shape == (3, 1)
        assert np.isnan(matrix[2, 0])

    def test_matrix_shape(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        for i in range(10):
            s.push_snapshot({"a": 0.1 * i, "b": 0.05 * i})
        matrix, names = s.as_matrix()
        assert matrix.shape == (2, 10)
        assert "a" in names and "b" in names

    def test_buffer_bound(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState(layer_names=["a"], history_length=10)
        for i in range(50):
            s.push_snapshot({"a": float(i) * 0.01})
        matrix, _ = s.as_matrix()
        assert matrix.shape[1] == 10
        # Last value should be 49 * 0.01 = 0.49
        assert abs(matrix[0, -1] - 0.49) < 1e-6

    def test_layer_stats_basic(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        for i in range(20):
            s.push_snapshot({"a": 0.1})
        stats = s.layer_stats("a")
        assert abs(stats['latest'] - 0.1) < 1e-9
        assert abs(stats['mean'] - 0.1) < 1e-9
        assert abs(stats['max'] - 0.1) < 1e-9

    def test_layer_stats_trend_rising(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        for i in range(40):
            s.push_snapshot({"a": i * 0.01})  # steadily rising
        stats = s.layer_stats("a")
        assert stats['trend'] == 'rising'

    def test_layer_stats_trend_falling(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        for i in range(40):
            s.push_snapshot({"a": (40 - i) * 0.01})  # steadily falling
        stats = s.layer_stats("a")
        assert stats['trend'] == 'falling'

    def test_layer_stats_trend_stable(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        for i in range(40):
            s.push_snapshot({"a": 0.1})
        stats = s.layer_stats("a")
        assert stats['trend'] == 'stable'

    def test_global_stats(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState(warning_threshold=0.15, critical_threshold=0.35)
        for i in range(10):
            s.push_snapshot({"a": 0.1, "b": 0.4})
        gs = s.global_stats()
        assert gs['num_layers'] == 2
        assert gs['num_ticks'] == 10
        assert abs(gs['max_score'] - 0.4) < 1e-5
        # Only layer 'b' exceeds warning (0.4 > 0.15)
        assert gs['total_alerts'] == 10
        # Both warnings so far since only b triggers
        assert gs['critical_alerts'] == 10

    def test_alert_events(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState(warning_threshold=0.15, critical_threshold=0.35)
        s.push_snapshot({"a": 0.05})     # nominal
        s.push_snapshot({"a": 0.20})     # warning
        s.push_snapshot({"a": 0.50})     # critical
        s.push_snapshot({"a": 0.10})     # nominal

        events = s.alert_events()
        assert len(events) == 2
        # Newest first
        assert events[0]['level'] == 'critical'
        assert events[1]['level'] == 'warning'

    def test_seed_demo(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        s.seed_demo(80)
        assert s.num_layers == 4
        matrix, names = s.as_matrix()
        assert matrix.shape == (4, 80)

    def test_seed_demo_creates_drift_events(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState(warning_threshold=0.15, critical_threshold=0.35)
        s.seed_demo(120)
        gs = s.global_stats()
        assert gs['total_alerts'] > 0
        assert gs['critical_alerts'] > 0

    def test_reset(self):
        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        s.seed_demo(20)
        assert not s.is_empty
        s.reset()
        assert s.is_empty
        assert s.num_layers == 0

    def test_import_from_radar(self):
        # Simulate a DriftRadar with the expected attribute
        class FakeRadar:
            drift_history = {
                "layer_a": [
                    {"composite_score": 0.05},
                    {"composite_score": 0.10},
                ],
                "layer_b": [
                    {"composite_score": 0.20},
                    {"composite_score": 0.40},
                ],
            }

        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        imported = s.import_from_radar(FakeRadar())
        assert imported == 2
        matrix, names = s.as_matrix()
        assert matrix.shape[0] == 2
        assert matrix.shape[1] == 2
        assert "layer_a" in names

    def test_import_from_empty_radar(self):
        class EmptyRadar:
            drift_history = {}

        from dashboard.panels.drift_heatmap import DriftHeatmapState
        s = DriftHeatmapState()
        imported = s.import_from_radar(EmptyRadar())
        assert imported == 0


class TestFigureBuilders:
    def test_heatmap_empty(self):
        from dashboard.panels.drift_heatmap import build_heatmap_figure
        from dashboard.theme import get_palette
        fig = build_heatmap_figure(
            np.empty((0, 0)), [], get_palette("cyber_dark"),
        )
        assert fig is not None

    def test_heatmap_basic(self):
        from dashboard.panels.drift_heatmap import build_heatmap_figure
        from dashboard.theme import get_palette
        matrix = np.random.rand(4, 50) * 0.5
        fig = build_heatmap_figure(
            matrix=matrix,
            layer_names=["a", "b", "c", "d"],
            palette=get_palette("cyber_dark"),
            height=400,
        )
        assert len(fig.data) == 1
        assert fig.layout.height == 400

    def test_trajectories_basic(self):
        from dashboard.panels.drift_heatmap import build_layer_trajectories
        from dashboard.theme import get_palette
        matrix = np.random.rand(3, 40)
        fig = build_layer_trajectories(
            matrix=matrix,
            layer_names=["x", "y", "z"],
            palette=get_palette("cyber_dark"),
        )
        # 3 traces
        assert len(fig.data) == 3

    def test_trajectories_empty(self):
        from dashboard.panels.drift_heatmap import build_layer_trajectories
        from dashboard.theme import get_palette
        fig = build_layer_trajectories(
            np.empty((0, 0)), [], get_palette("cyber_dark"),
        )
        assert fig is not None

    def test_current_scores_bar(self):
        from dashboard.panels.drift_heatmap import build_current_scores_bar
        from dashboard.theme import get_palette
        fig = build_current_scores_bar(
            layer_names=["a", "b", "c"],
            latest_scores=[0.05, 0.20, 0.45],
            palette=get_palette("cyber_dark"),
        )
        assert len(fig.data) == 1

    def test_current_scores_empty(self):
        from dashboard.panels.drift_heatmap import build_current_scores_bar
        from dashboard.theme import get_palette
        fig = build_current_scores_bar([], [], get_palette("cyber_dark"))
        assert fig is not None

    def test_figures_all_palettes(self):
        from dashboard.panels.drift_heatmap import (
            build_heatmap_figure, build_layer_trajectories, build_current_scores_bar,
        )
        from dashboard.theme import get_palette
        matrix = np.random.rand(3, 30)
        for pname in ["cyber_dark", "plasma", "mono", "solar"]:
            pal = get_palette(pname)
            build_heatmap_figure(matrix, ["a", "b", "c"], pal)
            build_layer_trajectories(matrix, ["a", "b", "c"], pal)
            build_current_scores_bar(["a", "b", "c"], [0.1, 0.2, 0.3], pal)


class TestPackageExports:
    def test_panels_exports(self):
        from dashboard.panels import (
            DriftHeatmapState, render_drift_heatmap,
            build_heatmap_figure, build_layer_trajectories,
            build_current_scores_bar,
        )
        assert DriftHeatmapState is not None
        assert callable(render_drift_heatmap)
        assert callable(build_heatmap_figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])