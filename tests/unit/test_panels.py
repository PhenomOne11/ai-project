"""
Unit Tests for dashboard.panels
"""

import numpy as np
import pytest


class TestLatentExplorerState:
    def test_empty_init(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=16, buffer_size=100)
        assert s.is_empty
        assert s.size == 0

    def test_push_single(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        s.push(np.random.randn(8))
        assert s.size == 1
        assert not s.is_empty

    def test_push_batch(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        s.push_batch(np.random.randn(10, 8))
        assert s.size == 10

    def test_buffer_overflow(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=4, buffer_size=10)
        for i in range(50):
            s.push(np.random.randn(4))
        assert s.size == 10

    def test_auto_dims_on_mismatch(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        s.push(np.random.randn(12))
        assert s.latent_dim == 12

    def test_project_empty(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        coords, ratios = s.project()
        assert coords.shape[0] == 0

    def test_project_basic(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=16, n_components=3)
        s.push_batch(np.random.randn(100, 16))
        coords, ratios = s.project()
        assert coords.shape == (100, 3)
        assert len(ratios) == 3
        assert all(0 <= r <= 1 for r in ratios)

    def test_stats(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        s.push_batch(np.random.randn(50, 8) + 5.0)
        stats = s.stats()
        assert stats['n_samples'] == 50
        assert stats['n_pushed_total'] == 50
        assert stats['mean_norm'] > 0

    def test_cluster_assignments(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        # Two well-separated clusters
        cluster_a = np.random.randn(50, 8) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
        cluster_b = np.random.randn(50, 8) - np.array([5, 0, 0, 0, 0, 0, 0, 0])
        s.push_batch(np.vstack([cluster_a, cluster_b]))
        labels = s.cluster(n_clusters=2, n_iter=30)
        # Should cleanly separate into 2 clusters
        first_half_labels = set(labels[:50])
        second_half_labels = set(labels[50:])
        assert len(first_half_labels) == 1
        assert len(second_half_labels) == 1
        assert first_half_labels != second_half_labels

    def test_reset(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=8)
        s.push_batch(np.random.randn(30, 8))
        assert s.size == 30
        s.reset()
        assert s.is_empty
        assert s._n_pushed == 0

    def test_seed_demo(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=32)
        s.seed_demo(n=150)
        assert s.size == 150

    def test_seed_demo_produces_projectable_data(self):
        from dashboard.panels.latent_explorer import LatentExplorerState
        s = LatentExplorerState(latent_dim=32, buffer_size=500, n_components=3)
        s.seed_demo(n=200)
        coords, ratios = s.project()
        assert coords.shape == (200, 3)
        assert ratios.sum() > 0.1  # Should explain some variance


class TestFigureBuilders:
    def test_trajectory_figure_empty(self):
        from dashboard.panels.latent_explorer import build_trajectory_figure
        from dashboard.theme import get_palette
        fig = build_trajectory_figure(
            coords=np.empty((0, 3)),
            palette=get_palette("cyber_dark"),
        )
        assert fig is not None
        assert len(fig.data) == 0

    def test_trajectory_figure_basic(self):
        from dashboard.panels.latent_explorer import build_trajectory_figure
        from dashboard.theme import get_palette
        coords = np.random.randn(50, 3)
        fig = build_trajectory_figure(
            coords=coords,
            palette=get_palette("cyber_dark"),
            color_mode="time",
            height=400,
        )
        # 3 traces: trajectory + latest + centroid
        assert len(fig.data) == 3
        assert fig.layout.height == 400

    def test_trajectory_figure_cluster_mode(self):
        from dashboard.panels.latent_explorer import build_trajectory_figure
        from dashboard.theme import get_palette
        coords = np.random.randn(60, 3)
        labels = np.array([i % 3 for i in range(60)])
        fig = build_trajectory_figure(
            coords=coords,
            labels=labels,
            palette=get_palette("cyber_dark"),
            color_mode="cluster",
        )
        assert len(fig.data) >= 1

    def test_trajectory_figure_novelty_mode(self):
        from dashboard.panels.latent_explorer import build_trajectory_figure
        from dashboard.theme import get_palette
        coords = np.random.randn(40, 3)
        fig = build_trajectory_figure(
            coords=coords,
            palette=get_palette("cyber_dark"),
            color_mode="novelty",
        )
        assert len(fig.data) >= 1

    def test_projection_panel(self):
        from dashboard.panels.latent_explorer import build_projection_panel
        from dashboard.theme import get_palette
        coords = np.random.randn(80, 3)
        fig = build_projection_panel(coords=coords, palette=get_palette("cyber_dark"))
        # 3 subplot traces
        assert len(fig.data) == 3

    def test_projection_panel_empty(self):
        from dashboard.panels.latent_explorer import build_projection_panel
        from dashboard.theme import get_palette
        fig = build_projection_panel(coords=np.empty((0, 3)), palette=get_palette("cyber_dark"))
        assert fig is not None


class TestPanelRegistry:
    def test_panel_exports(self):
        from dashboard.panels import (
            LatentExplorerState, build_trajectory_figure,
            build_projection_panel, render_latent_explorer,
        )
        assert callable(render_latent_explorer)
        assert LatentExplorerState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])