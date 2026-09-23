"""
Standalone panel preview app.
Run with: streamlit run dashboard/preview_panels.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.theme import get_palette, list_palettes, inject_theme
from dashboard.layouts import render_header, sidebar_section, sidebar_divider
from dashboard.panels import (
    LatentExplorerState, render_latent_explorer,
    DriftHeatmapState, render_drift_heatmap,
    TelemetryState, render_telemetry,
    AnomalyConsoleState, render_anomaly_console,
    ModelComparisonState, render_model_comparison,
    OscilloscopeState, render_oscilloscope,
    TopologyState, render_topology, seed_default_topology,
)


st.set_page_config(
    page_title="NeuralStream · Panel Preview",
    page_icon="🔬",
    layout="wide",
)

sidebar_section(st, "🎨", "Palette")
palette_name = st.sidebar.selectbox(
    "Active palette", options=list_palettes(), index=0,
    key="preview_panels_palette",
)
palette = get_palette(palette_name)
inject_theme(st, palette)

sidebar_divider(st)
sidebar_section(st, "🧭", "Panel")
panel_choice = st.sidebar.radio(
    "Choose panel",
    options=[
        "Latent Explorer",
        "Drift Heatmap",
        "Hardware Telemetry",
        "Anomaly Console",
        "Model Comparison",
        "Oscilloscope",
        "Topology View",
    ],
    index=0,
)

render_header(
    st,
    title="🔬 Panel Preview",
    subtitle=f"individual panel preview · palette={palette_name}",
    state_text="PREVIEW",
    state_level="info",
)

# ---- Session state ----
if "latent_state_v1" not in st.session_state:
    st.session_state.latent_state_v1 = LatentExplorerState(
        latent_dim=32, buffer_size=500, n_components=3,
    )
if "drift_state_v1" not in st.session_state:
    st.session_state.drift_state_v1 = DriftHeatmapState(
        history_length=200, warning_threshold=0.15, critical_threshold=0.35,
    )
if "telemetry_state_v1" not in st.session_state:
    st.session_state.telemetry_state_v1 = TelemetryState(history_length=300)
if "anomaly_state_v1" not in st.session_state:
    st.session_state.anomaly_state_v1 = AnomalyConsoleState(history_length=500)
if "comparison_state_v1" not in st.session_state:
    st.session_state.comparison_state_v1 = ModelComparisonState()
if "oscilloscope_state_v1" not in st.session_state:
    st.session_state.oscilloscope_state_v1 = OscilloscopeState(
        samples_per_view=512, trail_length=64,
    )
if "topology_state_v1" not in st.session_state:
    topo = TopologyState()
    seed_default_topology(topo)
    st.session_state.topology_state_v1 = topo

# ---- Dispatch ----
if panel_choice == "Latent Explorer":
    render_latent_explorer(
        st, palette=palette, state=st.session_state.latent_state_v1,
        key_prefix="preview_latent", show_demo_button=True,
    )
elif panel_choice == "Drift Heatmap":
    render_drift_heatmap(
        st, palette=palette, state=st.session_state.drift_state_v1,
        key_prefix="preview_drift", show_demo_button=True,
    )
elif panel_choice == "Hardware Telemetry":
    render_telemetry(
        st, palette=palette, state=st.session_state.telemetry_state_v1,
        key_prefix="preview_telemetry", show_demo_button=True,
    )
elif panel_choice == "Anomaly Console":
    render_anomaly_console(
        st, palette=palette, state=st.session_state.anomaly_state_v1,
        key_prefix="preview_anomaly", show_demo_button=True,
    )
elif panel_choice == "Model Comparison":
    render_model_comparison(
        st, palette=palette, state=st.session_state.comparison_state_v1,
        key_prefix="preview_comparison", show_demo_button=True,
    )
elif panel_choice == "Oscilloscope":
    render_oscilloscope(
        st, palette=palette, state=st.session_state.oscilloscope_state_v1,
        key_prefix="preview_osc", show_demo_button=True,
    )
elif panel_choice == "Topology View":
    render_topology(
        st, palette=palette, state=st.session_state.topology_state_v1,
        key_prefix="preview_topo", show_demo_button=True,
    )