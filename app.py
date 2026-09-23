"""
NeuralStream-Core — Mission Control HUD

The unified dashboard. Renders all seven interactive panels in a
single themed Streamlit application, using every subsystem built
in Phases 1–24.

Run with:
    streamlit run app.py
    python -m cli demo
    streamlit run app.py --server.port 8502

Profile is loaded from (highest priority first):
    1. NEURALSTREAM_PROFILE environment variable
    2. Sidebar selector (defaults to "default")
"""

import os
import sys
import time
from pathlib import Path

# ---- Path bootstrap ----
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np
import torch


# =============================================================================
# PAGE CONFIGURATION (must run before any st.* call)
# =============================================================================

st.set_page_config(
    page_title="NeuralStream-Core · Mission Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CORE IMPORTS
# =============================================================================

from config.loader import load_config, list_profiles
from dashboard.theme import (
    get_palette, list_palettes, inject_theme,
    metric_card, section_header, divider, status_badge,
)
from dashboard.layouts import (
    render_header, sidebar_section, sidebar_divider,
    sync_engine_state,
)
from dashboard.panels import (
    # Latent explorer
    LatentExplorerState, render_latent_explorer,
    # Drift heatmap
    DriftHeatmapState, render_drift_heatmap,
    # Telemetry
    TelemetryState, render_telemetry,
    # Anomaly console
    AnomalyConsoleState, render_anomaly_console,
    # Model comparison
    ModelComparisonState, render_model_comparison,
    # Oscilloscope
    OscilloscopeState, render_oscilloscope,
    # Topology
    TopologyState, render_topology, seed_default_topology,
)
from core.metrics.system import probe_process, probe_system

# =============================================================================
# CUSTOM UPLOAD PROCESSING
# =============================================================================

def _process_uploaded_file(uploaded_file, cfg, palette):
    """
    Load uploaded file → run through model → populate all 7 panels.
    Returns number of samples processed.
    """
    import time
    from core.streams import (
        load_npy_to_tensor, load_pt_to_tensor,
        load_images_to_tensor, load_csv_to_tensor,
    )
    from core.models import EdgeStreamingAE
    from core.drift import DriftRadar
    from core.anomaly import AnomalyScorer

    # --- 1. Load file → tensor ---
    name = uploaded_file.name.lower()
    size = cfg.model.input_size

    if name.endswith(".npy"):
        tensor = load_npy_to_tensor(uploaded_file, target_size=size)
    elif name.endswith((".pt", ".pth")):
        tensor = load_pt_to_tensor(uploaded_file, target_size=size)
    elif name.endswith((".png", ".jpg", ".jpeg")):
        tensor = load_images_to_tensor(uploaded_file, target_size=size)
    elif name.endswith(".csv"):
        tensor = load_csv_to_tensor(uploaded_file, target_size=size)
    else:
        raise ValueError(f"Unsupported: {name}")

    n_samples = len(tensor)

    # --- 2. Build model ---
    model = EdgeStreamingAE(
        in_channels=tensor.shape[1],
        input_size=size,
        latent_dim=cfg.model.latent_dim,
        base_channels=8,
    ).eval()

    # --- 3. Clear old panel data ---
    st.session_state.latent_state_v2.reset()
    st.session_state.drift_state_v2.reset()
    st.session_state.telemetry_state_v2.reset()
    st.session_state.anomaly_state_v2.reset()

    # --- 4. Setup drift radar ---
    ref_batch = tensor[:min(16, n_samples)]
    drift = DriftRadar(
        methods=['wasserstein'],
        evaluation_interval=1,
        layer_names=["encoder_conv.0", "encoder_conv.2", "encoder_fc.0"],
    )
    drift.capture_reference(model, ref_batch)

    # --- 5. First pass: collect latents for anomaly fit ---
    batch_size = 16
    latents_buf = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            z = model.encode(tensor[i:i+batch_size])
            latents_buf.append(z.cpu().numpy())

    anomaly = AnomalyScorer(
        n_estimators=30,
        fit_contamination=cfg.anomaly.contamination,
    )
    if latents_buf:
        all_latents = np.concatenate(latents_buf, axis=0)
        anomaly.fit(all_latents)

    # --- 6. Second pass: populate all panels ---
    import psutil, os
    proc = psutil.Process(os.getpid())

    progress = st.progress(0, text="Processing batches...")
    n_batches = len(range(0, n_samples, batch_size))
    for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
        batch = tensor[i:i+batch_size]

        t0 = time.perf_counter()
        with torch.no_grad():
            z = model.encode(batch)
            x_hat = model.decode(z)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Latent explorer
        st.session_state.latent_state_v2.push_batch(z.cpu().numpy())

        # Drift heatmap
        reports = drift.evaluate_from_tensor(model, batch)
        for layer, report in reports.items():
            st.session_state.drift_state_v2.push_snapshot(
                {layer: report.composite_score}
            )

        # Anomaly console
        result = anomaly.score(z, x_hat, batch)
        st.session_state.anomaly_state_v2.ingest_anomaly_result(result)

        # Telemetry
        st.session_state.telemetry_state_v2.append(
            cpu_percent=proc.cpu_percent(interval=None),
            rss_mb=proc.memory_info().rss / (1024 * 1024),
            latency_ms=latency_ms,
            tick=i // batch_size,
        )
        
        progress.progress(
            (batch_idx + 1) / n_batches,
            text=f"Processing batch {batch_idx+1}/{n_batches}",
        )
        
        progress.empty()
        
    return n_samples

# =============================================================================
# SIDEBAR: PALETTE + PROFILE
# =============================================================================

env_profile = os.environ.get("NEURALSTREAM_PROFILE", None)
available_profiles = list_profiles() or ["default"]

if env_profile and env_profile in available_profiles:
    default_profile_idx = available_profiles.index(env_profile)
else:
    default_profile_idx = 0


with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 4px 0;">
        <div style="font-size:1.35rem;font-weight:800;
                    background: linear-gradient(90deg, #38bdf8, #a78bfa);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing:-0.02em;">
            ⚡ NeuralStream-Core
        </div>
        <div style="color:var(--text-muted);font-size:0.72rem;
                    letter-spacing:0.12em;text-transform:uppercase;
                    font-family:'JetBrains Mono',monospace;margin-top:2px;">
            Mission Control · v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Custom Dataset Upload ---
    sidebar_section(st, "📁", "Your Dataset")
    
    uploaded = st.file_uploader(
        "Upload dataset file",
        type=["npy", "pt", "pth", "png", "jpg", "jpeg", "csv"],
        key="custom_upload",
        help="Supported: .npy, .pt, .pth, .png, .jpg, .csv",
    )
    
    if uploaded is not None:
        if st.button("▶️ Process & Load", type="primary",
                     use_container_width=True, key="btn_process"):
            with st.spinner(f"Loading {uploaded.name}..."):
                try:
                    n = _process_uploaded_file(uploaded, cfg, palette)
                    st.success(f"✅ Loaded {n} samples from {uploaded.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {type(e).__name__}: {e}")
                    
    sidebar_divider(st)

    # --- Palette selector ---
    sidebar_section(st, "🎨", "Appearance")
    palette_name = st.selectbox(
        "Theme palette",
        options=list_palettes(),
        index=0,
        key="global_palette",
    )
    palette = get_palette(palette_name)
    inject_theme(st, palette)

    sidebar_divider(st)

    # --- Profile selector ---
    sidebar_section(st, "⚙️", "Configuration")
    profile_name = st.selectbox(
        "Config profile",
        options=available_profiles,
        index=default_profile_idx,
        key="global_profile",
    )
    cfg = load_config(profile=profile_name)

    st.caption(
        f"**{cfg.model.architecture}** · "
        f"{cfg.model.input_size}×{cfg.model.input_size} · "
        f"latent {cfg.model.latent_dim}"
    )

    sidebar_divider(st)

    # --- Demo controls ---
    sidebar_section(st, "🎬", "Demo Mode")

    demo_active = st.session_state.get("demo_active", False)

    if not demo_active:
        if st.button("🌱 Load All Demo Data",
                     type="primary", use_container_width=True,
                     key="btn_load_demo"):
            st.session_state.demo_active = True
            st.rerun()
    else:
        st.markdown(status_badge("DEMO LOADED", "active", "●"),
                    unsafe_allow_html=True)
        if st.button("🧹 Clear Demo",
                     use_container_width=True,
                     key="btn_clear_demo"):
            st.session_state.demo_active = False
            for k in list(st.session_state.keys()):
                if k.endswith(("_state_v1", "_state_v2")):
                    del st.session_state[k]
            st.rerun()

    sidebar_divider(st)

    # --- System snapshot (live) ---
    sidebar_section(st, "💻", "Host")

    proc = probe_process()
    sysinfo = probe_system()

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;
                font-size:0.75rem; color:var(--text-secondary);
                line-height:1.7;">
        <div>RSS: <span style="color:var(--accent);">
            {proc.get('rss_mb', 0):.0f} MB</span></div>
        <div>CPU: <span style="color:var(--accent);">
            {sysinfo.get('cpu_percent', 0):.1f}%</span></div>
        <div>RAM: <span style="color:var(--accent);">
            {sysinfo.get('mem_percent', 0):.1f}%</span></div>
        <div>Cores: <span style="color:var(--accent);">
            {sysinfo.get('cpu_count', 1)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_divider(st)

    st.caption(
        f"`{cfg.project_name}` · `v{cfg.version}` · "
        f"`profile={profile_name}`"
    )


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

demo_active = st.session_state.get("demo_active", False)

if "latent_state_v2" not in st.session_state:
    st.session_state.latent_state_v2 = LatentExplorerState(
        latent_dim=cfg.model.latent_dim,
        buffer_size=500, n_components=3,
    )
if "drift_state_v2" not in st.session_state:
    st.session_state.drift_state_v2 = DriftHeatmapState(
        layer_names=list(cfg.drift.layer_names),
        history_length=cfg.drift.history_length,
        warning_threshold=cfg.drift.alert_threshold,
        critical_threshold=cfg.drift.critical_threshold,
    )
if "telemetry_state_v2" not in st.session_state:
    st.session_state.telemetry_state_v2 = TelemetryState(
        history_length=cfg.telemetry.history_length,
    )
if "anomaly_state_v2" not in st.session_state:
    st.session_state.anomaly_state_v2 = AnomalyConsoleState(
        history_length=500,
    )
if "comparison_state_v2" not in st.session_state:
    st.session_state.comparison_state_v2 = ModelComparisonState()
if "oscilloscope_state_v2" not in st.session_state:
    st.session_state.oscilloscope_state_v2 = OscilloscopeState(
        samples_per_view=512, trail_length=cfg.dashboard.max_sparkline_points,
    )
if "topology_state_v2" not in st.session_state:
    topo = TopologyState(activity_window_s=30.0)
    seed_default_topology(topo)
    st.session_state.topology_state_v2 = topo


# ---- Seed demo data if requested ----
if demo_active:
    # Bump per-panel counters only once when transitioning into demo mode
    if not st.session_state.get("_demo_seeded", False):
        st.session_state.latent_state_v2.seed_demo(200)
        st.session_state.drift_state_v2.seed_demo(120)
        st.session_state.telemetry_state_v2.seed_demo(200)
        st.session_state.anomaly_state_v2.seed_demo(300)
        st.session_state.comparison_state_v2.seed_demo()
        st.session_state.oscilloscope_state_v2.seed_demo(40)

        # Simulate activity across topology nodes
        from dashboard.panels.topology import _simulate_activity
        _simulate_activity(st.session_state.topology_state_v2, duration_s=30.0)

        st.session_state._demo_seeded = True
else:
    st.session_state._demo_seeded = False


# =============================================================================
# HEADER
# =============================================================================

# Compute a live engine-state indicator from subsystem health
topo_state = st.session_state.topology_state_v2
topo_health = topo_state.overall_health()
topo_level, topo_text = sync_engine_state({
    'nominal': 'STREAMING',
    'warning': 'ADAPTING',
    'critical': 'FAULT',
    'offline': 'OFFLINE',
}.get(topo_health, 'STREAMING'))

render_header(
    st,
    title="⚡ NeuralStream-Core · Mission Control",
    subtitle=(
        f"profile: **{profile_name}** · palette: **{palette_name}** · "
        f"model: **{cfg.model.architecture}** · "
        f"input: **{cfg.model.input_size}×{cfg.model.input_size}**"
    ),
    state_text=topo_text,
    state_level=topo_level,
)


# =============================================================================
# TOP METRIC STRIP (cross-panel snapshot)
# =============================================================================

# Snapshot numbers from every subsystem
latest_telemetry = st.session_state.telemetry_state_v2.latest()
telemetry_stats = st.session_state.telemetry_state_v2.stats()
drift_stats = st.session_state.drift_state_v2.global_stats()
anomaly_stats = st.session_state.anomaly_state_v2.stats()

m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    v = f"{telemetry_stats['cpu']['p95']:.1f}%" if latest_telemetry else "—"
    st.markdown(metric_card(
        "CPU p95",
        v,
        f"{len(st.session_state.telemetry_state_v2.samples if hasattr(st.session_state.telemetry_state_v2, 'samples') else getattr(st.session_state.telemetry_state_v2, 'history', []))} samples",
    ), unsafe_allow_html=True)

with m2:
    v = f"{telemetry_stats['latency']['p95']:.2f} ms" if latest_telemetry else "—"
    st.markdown(metric_card(
        "Latency p95",
        v,
        f"target {cfg.precision.target_latency_ms:.1f} ms",
    ), unsafe_allow_html=True)

with m3:
    st.markdown(metric_card(
        "Drift Peak",
        f"{drift_stats['max_score']:.4f}",
        f"{drift_stats['critical_alerts']} critical",
        accent=(palette.danger if drift_stats['critical_alerts'] > 0
                else palette.warning if drift_stats['total_alerts'] > 0
                else palette.success),
    ), unsafe_allow_html=True)

with m4:
    st.markdown(metric_card(
        "Anomaly EMA",
        f"{anomaly_stats['score_ema']:.4f}",
        f"{anomaly_stats['severity_counts'].get('critical', 0)} critical",
        accent=(palette.danger
                if anomaly_stats['severity_counts'].get('critical', 0) > 0
                else palette.success),
    ), unsafe_allow_html=True)

with m5:
    st.markdown(metric_card(
        "OOD Queue",
        f"{anomaly_stats['ood_count']}",
        f"{anomaly_stats['adaptation_count']} adaptations",
        accent=palette.warning if anomaly_stats['ood_count'] > 0 else palette.success,
    ), unsafe_allow_html=True)

with m6:
    active = len(topo_state.active_nodes(min_rate=0.02))
    st.markdown(metric_card(
        "System Nodes",
        f"{len(topo_state.nodes)}",
        f"{active} active",
        accent=palette.accent_primary,
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# PANEL TABS
# =============================================================================

tabs = st.tabs([
    "🌐 Overview",
    "📡 Latent",
    "🧠 Drift",
    "📈 Telemetry",
    "🚨 Anomaly",
    "⚖️ Compare",
    "📻 Oscilloscope",
    "🗺️ Topology",
])


# ---- Tab 0: Overview (topology + telemetry summary) ----
with tabs[0]:
    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        render_topology(
            st,
            palette=palette,
            state=st.session_state.topology_state_v2,
            key_prefix="overview_topo",
            show_demo_button=False,
        )

    with col_b:
        render_telemetry(
            st,
            palette=palette,
            state=st.session_state.telemetry_state_v2,
            key_prefix="overview_tele",
            show_demo_button=False,
        )


# ---- Tab 1: Latent explorer ----
with tabs[1]:
    render_latent_explorer(
        st,
        palette=palette,
        state=st.session_state.latent_state_v2,
        key_prefix="main_latent",
        show_demo_button=False,
    )


# ---- Tab 2: Drift heatmap ----
with tabs[2]:
    render_drift_heatmap(
        st,
        palette=palette,
        state=st.session_state.drift_state_v2,
        key_prefix="main_drift",
        show_demo_button=False,
    )


# ---- Tab 3: Telemetry ----
with tabs[3]:
    render_telemetry(
        st,
        palette=palette,
        state=st.session_state.telemetry_state_v2,
        key_prefix="main_telemetry",
        show_demo_button=False,
    )


# ---- Tab 4: Anomaly console ----
with tabs[4]:
    render_anomaly_console(
        st,
        palette=palette,
        state=st.session_state.anomaly_state_v2,
        key_prefix="main_anomaly",
        show_demo_button=False,
    )


# ---- Tab 5: Model comparison ----
with tabs[5]:
    render_model_comparison(
        st,
        palette=palette,
        state=st.session_state.comparison_state_v2,
        key_prefix="main_compare",
        show_demo_button=False,
    )


# ---- Tab 6: Oscilloscope ----
with tabs[6]:
    render_oscilloscope(
        st,
        palette=palette,
        state=st.session_state.oscilloscope_state_v2,
        key_prefix="main_osc",
        show_demo_button=False,
    )


# ---- Tab 7: Full topology view ----
with tabs[7]:
    render_topology(
        st,
        palette=palette,
        state=st.session_state.topology_state_v2,
        key_prefix="main_topology",
        show_demo_button=False,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(divider(), unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
with col_f1:
    st.caption(
        f"**{cfg.project_name}** v{cfg.version} · "
        f"profile `{profile_name}` · "
        f"theme `{palette_name}` · "
        f"panels 7 · nodes {len(topo_state.nodes)}"
    )
with col_f2:
    st.caption(f"Debug: `{cfg.debug}` · Log: `{cfg.log_level}`")
with col_f3:
    st.caption(f"Refresh: `{cfg.dashboard.refresh_rate_ms}ms`")
    
