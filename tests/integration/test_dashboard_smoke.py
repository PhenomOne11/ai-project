"""
Dashboard smoke tests.

Instantiates each panel's state, seeds demo data, and calls render()
with a Streamlit stub to ensure no exceptions are raised.
"""

import pytest
from tests.integration.helpers import MockST


class TestPanelRenderSmoke:
    """Each panel renders without raising exceptions."""

    def test_latent_explorer_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import LatentExplorerState, render_latent_explorer

        st = MockST()
        state = LatentExplorerState(latent_dim=16, buffer_size=100)
        state.seed_demo(50)
        render_latent_explorer(st, get_palette("cyber_dark"), state,
                               key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 1
        assert len(st.markdown_calls) > 0

    def test_drift_heatmap_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import DriftHeatmapState, render_drift_heatmap

        st = MockST()
        state = DriftHeatmapState()
        state.seed_demo(60)
        render_drift_heatmap(st, get_palette("cyber_dark"), state,
                             key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 2

    def test_telemetry_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import TelemetryState, render_telemetry

        st = MockST()
        state = TelemetryState()
        state.seed_demo(80)
        render_telemetry(st, get_palette("cyber_dark"), state,
                         key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 3

    def test_anomaly_console_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import AnomalyConsoleState, render_anomaly_console

        st = MockST()
        state = AnomalyConsoleState()
        state.seed_demo(100)
        render_anomaly_console(st, get_palette("cyber_dark"), state,
                               key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 3

    def test_model_comparison_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import ModelComparisonState, render_model_comparison

        st = MockST()
        state = ModelComparisonState()
        state.seed_demo()
        render_model_comparison(st, get_palette("cyber_dark"), state,
                                key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 3

    def test_oscilloscope_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import OscilloscopeState, render_oscilloscope

        st = MockST()
        state = OscilloscopeState(samples_per_view=256)
        state.seed_demo(20)
        render_oscilloscope(st, get_palette("cyber_dark"), state,
                            key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 3

    def test_topology_smoke(self):
        from dashboard.theme import get_palette
        from dashboard.panels import (
            TopologyState, seed_default_topology, render_topology,
        )

        st = MockST()
        state = TopologyState()
        seed_default_topology(state)
        render_topology(st, get_palette("cyber_dark"), state,
                        key_prefix="test", show_demo_button=False)
        assert len(st.plotly_calls) >= 2


class TestAllPalettesRender:
    """Every panel renders cleanly across all 4 palettes."""

    @pytest.mark.parametrize("palette_name",
                             ["cyber_dark", "plasma", "mono", "solar"])
    def test_latent_all_palettes(self, palette_name):
        from dashboard.theme import get_palette
        from dashboard.panels import LatentExplorerState, render_latent_explorer

        st = MockST()
        state = LatentExplorerState(latent_dim=16, buffer_size=50)
        state.seed_demo(30)
        render_latent_explorer(st, get_palette(palette_name), state,
                               key_prefix="t", show_demo_button=False)
        assert len(st.plotly_calls) >= 1

    @pytest.mark.parametrize("palette_name",
                             ["cyber_dark", "plasma", "mono", "solar"])
    def test_topology_all_palettes(self, palette_name):
        from dashboard.theme import get_palette
        from dashboard.panels import (
            TopologyState, seed_default_topology, render_topology,
        )
        st = MockST()
        state = TopologyState()
        seed_default_topology(state)
        render_topology(st, get_palette(palette_name), state,
                        key_prefix="t", show_demo_button=False)
        assert len(st.plotly_calls) >= 2


class TestThemeConsistency:
    """Theme functions produce well-formed output for every palette."""

    @pytest.mark.parametrize("palette_name",
                             ["cyber_dark", "plasma", "mono", "solar"])
    def test_css_generates(self, palette_name):
        from dashboard.theme import generate_css, get_palette
        css = generate_css(get_palette(palette_name))
        assert "<style>" in css
        assert len(css) > 1000

    @pytest.mark.parametrize("palette_name",
                             ["cyber_dark", "plasma", "mono", "solar"])
    def test_component_library(self, palette_name):
        from dashboard.theme import get_palette
        # Components don't depend on palette directly, but they should
        # always return strings
        from dashboard.theme import (
            status_badge, metric_card, section_header, divider,
            chip, kv_row, alert_box, sparkline_card, progress_bar,
        )
        for fn_result in [
            status_badge("X", "active", "●"),
            metric_card("X", "1.0"),
            section_header("⚡", "X"),
            divider(),
            chip("y"),
            kv_row("k", "v"),
            alert_box("t", "m", "info"),
            sparkline_card("t", "1", [1, 2, 3]),
            progress_bar("p", 0.5),
        ]:
            assert isinstance(fn_result, str)
            assert len(fn_result) > 0