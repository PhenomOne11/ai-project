"""
Unit Tests for dashboard.panels.model_comparison
"""

import numpy as np
import pytest


class TestModelProfile:
    def test_construction(self):
        from dashboard.panels.model_comparison import ModelProfile
        m = ModelProfile(name="edge", architecture="EdgeAE",
                         parameters=150_000, memory_mb=0.55)
        assert m.name == "edge"
        assert m.parameters == 150_000
        assert m.enabled is True

    def test_latency_stats_empty(self):
        from dashboard.panels.model_comparison import ModelProfile
        m = ModelProfile(name="edge")
        s = m.latency_stats()
        assert s['mean'] == 0.0
        assert s['n'] == 0

    def test_latency_stats_basic(self):
        from dashboard.panels.model_comparison import ModelProfile
        m = ModelProfile(name="edge")
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.latency_samples.append(v)
        s = m.latency_stats()
        assert s['mean'] == 3.0
        assert s['p50'] == 3.0
        assert s['n'] == 5

    def test_recon_stats(self):
        from dashboard.panels.model_comparison import ModelProfile
        m = ModelProfile(name="edge")
        for v in [0.001, 0.002, 0.003]:
            m.recon_samples.append(v)
        s = m.recon_stats()
        assert abs(s['mean'] - 0.002) < 1e-9

    def test_to_dict(self):
        from dashboard.panels.model_comparison import ModelProfile
        m = ModelProfile(name="edge", architecture="EdgeAE", parameters=100)
        d = m.to_dict()
        assert d['name'] == 'edge'
        assert d['architecture'] == 'EdgeAE'
        assert 'latency' in d
        assert 'recon' in d


class TestModelComparisonState:
    def test_empty_init(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        assert s.is_empty
        assert s.size == 0

    def test_add_model(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("edge", architecture="EdgeAE", parameters=100_000)
        assert s.size == 1
        m = s.get_model("edge")
        assert m is not None
        assert m.parameters == 100_000

    def test_add_model_idempotent(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("edge", architecture="EdgeAE", parameters=100_000)
        s.add_model("edge", architecture="EdgeAE-updated", parameters=200_000)
        assert s.size == 1
        assert s.get_model("edge").parameters == 200_000

    def test_remove_model(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("edge")
        assert s.remove_model("edge") is True
        assert s.size == 0
        assert s.remove_model("nonexistent") is False

    def test_record_latency(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        for i in range(10):
            s.record_latency("edge", 2.0 + i * 0.1)
        m = s.get_model("edge")
        assert m.latency_stats()['n'] == 10

    def test_record_recon(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        for i in range(5):
            s.record_recon("edge", 0.001 + i * 0.0001)
        assert s.get_model("edge").recon_stats()['n'] == 5

    def test_record_batch(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.record_batch("edge", [1.0, 2.0, 3.0], [0.01, 0.02, 0.03])
        m = s.get_model("edge")
        assert m.latency_stats()['n'] == 3
        assert m.recon_stats()['n'] == 3

    def test_auto_creates_model_in_record(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.record_latency("implicit", 1.0)
        assert s.size == 1
        assert s.get_model("implicit") is not None

    def test_enable_disable(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("a")
        s.add_model("b")
        s.enable_model("a", False)
        assert "a" not in s.model_names(enabled_only=True)
        assert "b" in s.model_names(enabled_only=True)

    def test_to_dataframe_like(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("a", parameters=1000)
        s.add_model("b", parameters=2000)
        rows = s.to_dataframe_like()
        assert len(rows) == 2
        assert any(r['Model'] == 'a' for r in rows)
        assert 'Params' in rows[0]

    def test_rank_by_latency(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("fast")
        s.add_model("slow")
        for v in [1.0, 1.1, 1.2]:
            s.record_latency("fast", v)
        for v in [4.0, 4.5, 5.0]:
            s.record_latency("slow", v)
        ranked = s.rank_by('latency_mean')
        assert ranked[0][0] == "fast"

    def test_rank_by_recon(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("good")
        s.add_model("bad")
        for v in [0.001, 0.0011]:
            s.record_recon("good", v)
        for v in [0.01, 0.011]:
            s.record_recon("bad", v)
        ranked = s.rank_by('recon_mean')
        assert ranked[0][0] == "good"

    def test_rank_by_throughput(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("fast", throughput_fps=1000)
        s.add_model("slow", throughput_fps=100)
        ranked = s.rank_by('throughput')
        # Highest throughput should rank first regardless of inversion
        assert ranked[0][0] == "fast"

    def test_winner_for(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.add_model("fast")
        s.add_model("slow")
        for v in [1.0, 1.1]:
            s.record_latency("fast", v)
        for v in [5.0, 5.5]:
            s.record_latency("slow", v)
        assert s.winner_for('latency_mean') == "fast"

    def test_summary(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.seed_demo()
        summary = s.summary()
        assert summary['n_models'] == 4
        assert summary['best_latency'] is not None
        assert summary['best_recon'] is not None
        assert summary['best_efficiency'] is not None

    def test_seed_demo(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.seed_demo()
        assert s.size == 4
        names = s.model_names()
        assert "edge_ae" in names
        assert "vae" in names
        assert "unet_ae" in names
        assert "transformer_ae" in names

    def test_seed_demo_edge_fastest(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.seed_demo()
        # Edge AE has lowest latency_mean of the demo set
        winner = s.winner_for('latency_mean')
        assert winner == "edge_ae"

    def test_seed_demo_unet_best_recon(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.seed_demo()
        # UNet has lowest recon_mean in demo
        winner = s.winner_for('recon_mean')
        assert winner == "unet_ae"

    def test_reset(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        s.seed_demo()
        assert s.size == 4
        s.reset()
        assert s.is_empty
        assert s.size == 0

    def test_import_from_registry_missing(self):
        from dashboard.panels.model_comparison import ModelComparisonState
        s = ModelComparisonState()
        # Fake registry that raises
        class BrokenRegistry:
            def compare_architectures(self, **kwargs):
                raise ValueError("simulated failure")
        count = s.import_from_registry(BrokenRegistry())
        assert count == 0

    def test_import_from_registry(self):
        from dashboard.panels.model_comparison import ModelComparisonState

        class FakeRegistry:
            def compare_architectures(self, **kwargs):
                return {
                    'edge_ae': {
                        'parameters': 149_024, 'memory_mb': 0.57,
                        'latency_mean_ms': 2.4, 'latency_p95_ms': 3.1,
                    },
                    'vae': {
                        'parameters': 168_448, 'memory_mb': 0.64,
                        'latency_mean_ms': 3.1, 'latency_p95_ms': 4.2,
                    },
                }

        s = ModelComparisonState()
        count = s.import_from_registry(FakeRegistry())
        assert count == 2
        assert s.size == 2
        edge = s.get_model("edge_ae")
        assert edge.parameters == 149_024


class TestFigureBuilders:
    def test_params_bar_empty(self):
        from dashboard.panels.model_comparison import build_params_bar
        from dashboard.theme import get_palette
        fig = build_params_bar([], get_palette("cyber_dark"))
        assert fig is not None

    def test_params_bar_basic(self):
        from dashboard.panels.model_comparison import (
            build_params_bar, ModelComparisonState,
        )
        from dashboard.theme import get_palette
        s = ModelComparisonState()
        s.seed_demo()
        fig = build_params_bar(s.to_dataframe_like(), get_palette("cyber_dark"))
        # 2 traces: params + memory
        assert len(fig.data) == 2

    def test_latency_box_basic(self):
        from dashboard.panels.model_comparison import build_latency_box
        from dashboard.theme import get_palette
        fig = build_latency_box(
            [("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0])],
            get_palette("cyber_dark"),
        )
        assert len(fig.data) == 2

    def test_latency_percentile_grouped(self):
        from dashboard.panels.model_comparison import (
            build_latency_percentile_grouped, ModelComparisonState,
        )
        from dashboard.theme import get_palette
        s = ModelComparisonState()
        s.seed_demo()
        fig = build_latency_percentile_grouped(s.to_dataframe_like(), get_palette("cyber_dark"))
        assert len(fig.data) == 3  # p50, p95, p99

    def test_recon_bar_basic(self):
        from dashboard.panels.model_comparison import (
            build_recon_bar, ModelComparisonState,
        )
        from dashboard.theme import get_palette
        s = ModelComparisonState()
        s.seed_demo()
        fig = build_recon_bar(s.to_dataframe_like(), get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_radar_basic(self):
        from dashboard.panels.model_comparison import (
            build_radar, ModelComparisonState,
        )
        from dashboard.theme import get_palette
        s = ModelComparisonState()
        s.seed_demo()
        fig = build_radar(s.to_dataframe_like(), get_palette("cyber_dark"))
        # One radar trace per model
        assert len(fig.data) == 4

    def test_throughput_fps(self):
        from dashboard.panels.model_comparison import (
            build_throughput_fps, ModelComparisonState,
        )
        from dashboard.theme import get_palette
        s = ModelComparisonState()
        s.seed_demo()
        fig = build_throughput_fps(s.to_dataframe_like(), get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_all_palettes(self):
        from dashboard.panels.model_comparison import (
            ModelComparisonState,
            build_params_bar, build_latency_box,
            build_latency_percentile_grouped, build_recon_bar,
            build_radar, build_throughput_fps,
        )
        from dashboard.theme import get_palette
        s = ModelComparisonState()
        s.seed_demo()
        rows = s.to_dataframe_like()
        samples = [(n, list(s.get_model(n).latency_samples)) for n in s.model_names()]

        for pname in ["cyber_dark", "plasma", "mono", "solar"]:
            pal = get_palette(pname)
            build_params_bar(rows, pal)
            build_latency_box(samples, pal)
            build_latency_percentile_grouped(rows, pal)
            build_recon_bar(rows, pal)
            build_radar(rows, pal)
            build_throughput_fps(rows, pal)


class TestPackageExports:
    def test_exports(self):
        from dashboard.panels import (
            ModelComparisonState, ModelProfile,
            build_params_bar, build_latency_box,
            build_latency_percentile_grouped, build_recon_bar,
            build_radar, build_throughput_fps,
            render_model_comparison,
        )
        assert callable(render_model_comparison)
        assert ModelComparisonState is not None
        assert ModelProfile is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])