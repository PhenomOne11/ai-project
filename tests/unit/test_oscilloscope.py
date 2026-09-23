"""
Unit Tests for dashboard.panels.oscilloscope
"""

import numpy as np
import pytest


class TestOscilloscopeState:
    def test_empty_init(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a", "b"])
        assert s.is_empty
        assert s.num_channels == 2
        assert s.size == 0

    def test_push_waveform(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"], samples_per_view=100)
        s.push_waveform("a", np.random.randn(200))
        assert not s.is_empty
        assert s.size == 1
        w = s.get_waveform("a")
        assert w.size == 100  # truncated to samples_per_view

    def test_push_new_channel(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"], max_channels=3)
        s.push_waveform("b", np.random.randn(50))
        assert s.num_channels == 2
        assert "b" in s.channel_names

    def test_max_channels_limit(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a", "b"], max_channels=2)
        s.push_waveform("c", np.random.randn(50))
        # Should not add a third channel
        assert s.num_channels == 2

    def test_nan_handling(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"])
        arr = np.array([1.0, float('nan'), float('inf'), -float('inf'), 2.0])
        s.push_waveform("a", arr)
        w = s.get_waveform("a")
        assert np.all(np.isfinite(w))

    def test_waterfall_accumulation(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"], samples_per_view=128, trail_length=10)
        for _ in range(5):
            s.push_waveform("a", np.random.randn(128))
        wf = s.get_waterfall("a")
        assert wf.shape[0] >= 1
        assert wf.shape[1] == s.waterfall_rebin

    def test_channel_stats_on_empty(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"])
        stats = s.channel_stats("a")
        assert stats['samples'] == 0
        assert stats['rms'] == 0.0

    def test_channel_stats_pure_sine(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"], samples_per_view=1024)
        t = np.linspace(0, 1, 1024)
        wave = np.sin(2 * np.pi * 5 * t)
        s.push_waveform("a", wave)
        stats = s.channel_stats("a")
        assert stats['rms'] > 0.6
        assert stats['rms'] < 0.75  # Pure sine RMS = 1/√2 ≈ 0.707
        assert stats['spectral_entropy'] < 0.5  # Tonal signal
        assert stats['num_peaks'] >= 4

    def test_channel_stats_noise(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"], samples_per_view=1024)
        rng = np.random.default_rng(7)
        s.push_waveform("a", rng.standard_normal(1024))
        stats = s.channel_stats("a")
        # Noise has higher spectral entropy than a sine
        assert stats['spectral_entropy'] > 0.8

    def test_global_stats(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a", "b"])
        for _ in range(3):
            s.push_waveform("a", np.random.randn(100))
            s.push_waveform("b", np.random.randn(100) * 2)
        g = s.global_stats()
        assert g['num_channels'] == 2
        assert g['active_channels'] == 2
        assert g['total_pushed'] == 6

    def test_seed_demo(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState()
        s.seed_demo(30)
        assert s.num_channels == 3
        assert s.size == 90  # 3 channels × 30 frames

    def test_seed_demo_creates_waterfall(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(trail_length=20)
        s.seed_demo(30)
        for name in s.channel_names:
            wf = s.get_waterfall(name)
            # Waterfall is capped at trail_length
            assert wf.shape[0] <= 20

    def test_reset(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        s = OscilloscopeState(channel_names=["a"])
        s.push_waveform("a", np.random.randn(100))
        assert not s.is_empty
        s.reset()
        assert s.is_empty
        assert s.size == 0

    def test_rebin_spectrum(self):
        from dashboard.panels.oscilloscope import OscilloscopeState
        arr = np.arange(100).astype(np.float64)
        rebinned = OscilloscopeState._rebin_spectrum(arr, 20)
        assert rebinned.shape == (20,)
        # Should preserve average roughly
        assert abs(rebinned.mean() - arr.mean()) < 5.0


class TestFigureBuilders:
    def test_waveform_empty(self):
        from dashboard.panels.oscilloscope import build_waveform_traces
        from dashboard.theme import get_palette
        fig = build_waveform_traces({}, get_palette("cyber_dark"))
        assert fig is not None

    def test_waveform_basic(self):
        from dashboard.panels.oscilloscope import build_waveform_traces
        from dashboard.theme import get_palette
        t = np.linspace(0, 1, 200)
        waveforms = {
            "a": np.sin(2 * np.pi * 3 * t),
            "b": np.sin(2 * np.pi * 5 * t),
        }
        fig = build_waveform_traces(waveforms, get_palette("cyber_dark"), height=400)
        assert fig.layout.height == 400
        # 2 waveforms + at least 2 peak traces
        assert len(fig.data) >= 2

    def test_waveform_with_zero_crossings(self):
        from dashboard.panels.oscilloscope import build_waveform_traces
        from dashboard.theme import get_palette
        t = np.linspace(0, 1, 300)
        waveforms = {"a": np.sin(2 * np.pi * 4 * t)}
        fig = build_waveform_traces(
            waveforms, get_palette("cyber_dark"),
            show_peaks=True, show_zero_crossings=True,
        )
        # Waveform + peaks + zero crossings
        assert len(fig.data) >= 3

    def test_fft_empty(self):
        from dashboard.panels.oscilloscope import build_fft_spectrum
        from dashboard.theme import get_palette
        fig = build_fft_spectrum({}, get_palette("cyber_dark"))
        assert fig is not None

    def test_fft_basic(self):
        from dashboard.panels.oscilloscope import build_fft_spectrum
        from dashboard.theme import get_palette
        t = np.linspace(0, 1, 1024)
        waveforms = {"a": np.sin(2 * np.pi * 50 * t)}
        fig = build_fft_spectrum(
            waveforms, get_palette("cyber_dark"), sample_rate=1000.0,
        )
        assert len(fig.data) == 1

    def test_waterfall_empty(self):
        from dashboard.panels.oscilloscope import build_waterfall
        from dashboard.theme import get_palette
        fig = build_waterfall(np.zeros((0, 32)), get_palette("cyber_dark"))
        assert fig is not None

    def test_waterfall_basic(self):
        from dashboard.panels.oscilloscope import build_waterfall
        from dashboard.theme import get_palette
        wf = np.random.rand(20, 64)
        fig = build_waterfall(wf, get_palette("cyber_dark"), height=320)
        assert len(fig.data) == 1
        assert fig.layout.height == 320

    def test_band_energy_bar(self):
        from dashboard.panels.oscilloscope import build_band_energy_bar
        from dashboard.theme import get_palette
        stats = {'low_band': 0.5, 'mid_band': 0.3, 'high_band': 0.2}
        fig = build_band_energy_bar(stats, get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_all_palettes(self):
        from dashboard.panels.oscilloscope import (
            build_waveform_traces, build_fft_spectrum,
            build_waterfall, build_band_energy_bar,
        )
        from dashboard.theme import get_palette
        t = np.linspace(0, 1, 300)
        waveforms = {"a": np.sin(2 * np.pi * 4 * t)}
        wf = np.random.rand(20, 64)
        stats = {'low_band': 0.5, 'mid_band': 0.3, 'high_band': 0.2}

        for pname in ["cyber_dark", "plasma", "mono", "solar"]:
            pal = get_palette(pname)
            build_waveform_traces(waveforms, pal)
            build_fft_spectrum(waveforms, pal, sample_rate=1000.0)
            build_waterfall(wf, pal)
            build_band_energy_bar(stats, pal)


class TestPackageExports:
    def test_exports(self):
        from dashboard.panels import (
            OscilloscopeState, render_oscilloscope,
            build_waveform_traces, build_fft_spectrum,
            build_waterfall, build_band_energy_bar,
        )
        assert callable(render_oscilloscope)
        assert OscilloscopeState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])