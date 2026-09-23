"""
Multi-Layer Drift Heatmap Panel

Visualizes Wasserstein drift scores across monitored layers as:
    - A time × layer heatmap matrix (severity-graded color)
    - Per-layer line overlay with trend arrows
    - Threshold bands (warning / critical)
    - An event timeline listing all above-threshold moments

Designed to accept either a real `DriftRadar` (Phase 5) or synthetic
demo data, with a state container that is Streamlit-agnostic.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import (
    metric_card, section_header, divider, status_badge,
    apply_theme, accent_color, severity_color, progress_bar,
    kv_row, alert_box,
)
from dashboard.theme.palette import Palette


# =============================================================================
# STATE
# =============================================================================

class DriftHeatmapState:
    """
    Rolling buffer of drift scores per layer.

    Ingestion is either:
        - Direct API: `push_snapshot({layer: score}, tick)` from a live engine
        - Bulk import: `import_from_radar(radar)` from a Phase-5 DriftRadar
        - Synthetic: `seed_demo(...)` for presentation

    Args:
        layer_names: Names of monitored layers
        history_length: Max ticks retained
        warning_threshold: Score above which a warning is flagged
        critical_threshold: Score above which a critical is flagged
    """

    def __init__(
        self,
        layer_names: Optional[List[str]] = None,
        history_length: int = 200,
        warning_threshold: float = 0.15,
        critical_threshold: float = 0.35,
    ):
        self.layer_names: List[str] = list(layer_names) if layer_names else []
        self.history_length = history_length
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # layer_name → deque of scores
        # Pre-reserve buffers for all declared layers so push_snapshot
        # does not treat them as newly discovered.
        self._history: Dict[str, deque] = {
            name: deque(maxlen=history_length) for name in self.layer_names
        }
        # layer_name → deque of tick indices
        self._ticks: deque = deque(maxlen=history_length)
        self._n_ticks: int = 0

    # ---- Ingestion ----

    def push_snapshot(self, scores: Dict[str, float], tick: Optional[int] = None) -> None:
        """
        Add one time-slice of drift scores.
        Missing layers are filled with 0.0.
        """
        # Ensure every layer has a buffer
        for name in scores.keys():
            if name not in self.layer_names:
                self.layer_names.append(name)
            if name not in self._history:
                self._history[name] = deque(maxlen=self.history_length)

        # Insert NaN for absent layers, actual score otherwise
        for name in self.layer_names:
            if name not in self._history:
                self._history[name] = deque(maxlen=self.history_length)
            score = scores.get(name, np.nan)
            self._history[name].append(float(score))

        self._n_ticks += 1
        self._ticks.append(tick if tick is not None else self._n_ticks)

    def import_from_radar(self, radar) -> int:
        """
        Import historical drift data from a Phase-5 DriftRadar instance.
        Returns the number of layers imported.
        """
        if not hasattr(radar, 'drift_history'):
            return 0

        imported = 0
        radar_history = getattr(radar, 'drift_history', {})
        if not radar_history:
            return 0

        # Reset our own state
        self.layer_names = sorted(radar_history.keys())
        self._history = {}
        self._ticks = deque(maxlen=self.history_length)
        self._n_ticks = 0

        # Find the longest history to align ticks
        max_len = max(len(h) for h in radar_history.values())

        for name in self.layer_names:
            self._history[name] = deque(maxlen=self.history_length)

        for i in range(max_len):
            self._n_ticks += 1
            self._ticks.append(i)
            for name in self.layer_names:
                entries = radar_history[name]
                if i < len(entries):
                    entry = entries[i]
                    if isinstance(entry, dict):
                        score = entry.get('composite_score', 0.0)
                    else:
                        score = float(entry)
                else:
                    score = np.nan
                self._history[name].append(float(score))

        imported = len(self.layer_names)
        return imported

    def seed_demo(self, n_ticks: int = 120, seed: int = 42) -> None:
        """
        Generate synthetic drift data that tells a visual story:
            - Layer 1 stays nominal
            - Layer 2 shows a gentle rise mid-way
            - Layer 3 shows a strong critical spike near the end
            - Layer 4 oscillates
        """
        rng = np.random.default_rng(seed)
        demo_layers = [
            "encoder_conv.0",
            "encoder_conv.2",
            "encoder_fc.0",
            "decoder_conv.0",
        ]

        self.layer_names = demo_layers
        self._history = {name: deque(maxlen=self.history_length) for name in demo_layers}
        self._ticks = deque(maxlen=self.history_length)
        self._n_ticks = 0

        t = np.linspace(0, 1, n_ticks)

        # Layer 0: stays low with tiny noise
        layer0 = np.abs(rng.normal(0.04, 0.015, n_ticks))

        # Layer 1: steady rise from 0.05 → 0.22
        layer1 = 0.05 + t * 0.17 + np.abs(rng.normal(0, 0.02, n_ticks))

        # Layer 2: mostly quiet, then a spike in the last 25%
        layer2 = np.abs(rng.normal(0.06, 0.02, n_ticks))
        spike_start = int(n_ticks * 0.75)
        spike_len = n_ticks - spike_start
        layer2[spike_start:] += np.linspace(0, 0.55, spike_len)

        # Layer 3: slow oscillation
        layer3 = 0.12 + 0.08 * np.abs(np.sin(t * 6 * np.pi)) + np.abs(rng.normal(0, 0.02, n_ticks))

        series = {
            demo_layers[0]: layer0,
            demo_layers[1]: layer1,
            demo_layers[2]: layer2,
            demo_layers[3]: layer3,
        }

        for i in range(n_ticks):
            self._n_ticks += 1
            self._ticks.append(i)
            for name in demo_layers:
                self._history[name].append(float(series[name][i]))

    # ---- Retrieval ----

    def as_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return the full (layer × tick) matrix with NaN for missing data.

        Returns:
            (matrix, layer_names)
            matrix shape: (num_layers, num_ticks)
        """
        if not self.layer_names or not any(len(h) > 0 for h in self._history.values()):
            return np.empty((0, 0)), []

        max_len = max((len(h) for h in self._history.values()), default=0)
        if max_len == 0:
            return np.empty((0, 0)), []

        matrix = np.full((len(self.layer_names), max_len), np.nan)
        for i, name in enumerate(self.layer_names):
            history = list(self._history.get(name, []))
            if history:
                matrix[i, -len(history):] = np.array(history)

        return matrix, list(self.layer_names)

    def layer_stats(self, layer: str) -> Dict[str, float]:
        """Quick stats for a specific layer."""
        if layer not in self._history or not self._history[layer]:
            return {'latest': 0.0, 'mean': 0.0, 'max': 0.0, 'alerts': 0, 'trend': 'stable'}

        arr = np.array(self._history[layer])
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {'latest': 0.0, 'mean': 0.0, 'max': 0.0, 'alerts': 0, 'trend': 'stable'}

        latest = float(arr[-1])
        mean = float(arr.mean())
        max_val = float(arr.max())
        alerts = int((arr >= self.warning_threshold).sum())

        # Trend: compare second half mean vs first half mean
        if len(arr) >= 3:                              # ← was >= 4
            half = len(arr) // 2
            first = float(arr[:half].mean())
            second = float(arr[half:].mean())
            delta = second - first
            if delta > 0.02:
                trend = 'rising'
            elif delta < -0.02:
                trend = 'falling'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        return {
            'latest': latest,
            'mean': mean,
            'max': max_val,
            'alerts': alerts,
            'trend': trend,
        }

    def global_stats(self) -> Dict[str, Any]:
        """Aggregate stats across all layers."""
        matrix, _ = self.as_matrix()
        if matrix.size == 0:
            return {
                'num_layers': 0,
                'num_ticks': 0,
                'max_score': 0.0,
                'mean_score': 0.0,
                'total_alerts': 0,
                'critical_alerts': 0,
                'trending_up': 0,
                'trending_down': 0,
            }

        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            finite = np.array([0.0])

        all_stats = [self.layer_stats(name) for name in self.layer_names]
        total_alerts = sum(s['alerts'] for s in all_stats)
        critical_alerts = sum(1 for name in self.layer_names
                              for v in self._history.get(name, [])
                              if v >= self.critical_threshold)
        trending_up = sum(1 for s in all_stats if s['trend'] == 'rising')
        trending_down = sum(1 for s in all_stats if s['trend'] == 'falling')

        return {
            'num_layers': len(self.layer_names),
            'num_ticks': matrix.shape[1],
            'max_score': float(finite.max()),
            'mean_score': float(finite.mean()),
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'trending_up': trending_up,
            'trending_down': trending_down,
        }

    def alert_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Chronological list of every above-threshold moment.
        Returns newest first.
        """
        events = []
        ticks = list(self._ticks)
        for name in self.layer_names:
            history = list(self._history.get(name, []))
            for i, score in enumerate(history):
                if not np.isfinite(score):
                    continue
                if score >= self.critical_threshold:
                    level = 'critical'
                elif score >= self.warning_threshold:
                    level = 'warning'
                else:
                    continue
                tick = ticks[i] if i < len(ticks) else i
                events.append({
                    'tick': tick,
                    'layer': name,
                    'score': float(score),
                    'level': level,
                })

        # Sort newest first
        events.sort(key=lambda e: e['tick'], reverse=True)
        return events[:limit]

    @property
    def is_empty(self) -> bool:
        return all(len(h) == 0 for h in self._history.values()) if self._history else True

    @property
    def num_layers(self) -> int:
        return len(self.layer_names)

    def reset(self) -> None:
        self._history.clear()
        self._ticks.clear()
        self.layer_names = []
        self._n_ticks = 0


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_heatmap_figure(
    matrix: np.ndarray,
    layer_names: List[str],
    palette: Palette = None,
    warning_threshold: float = 0.15,
    critical_threshold: float = 0.35,
    height: int = 380,
) -> go.Figure:
    """
    Build the layer × tick drift heatmap.

    The colorscale is deliberately non-linear:
        0 → success green (nominal)
        warning → warning amber
        critical → danger red
    Spaced so that small deviations remain visually distinct.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()

    if matrix.size == 0:
        fig.update_layout(title="Multi-Layer Drift Heatmap")
        apply_theme(fig, p, height=height)
        return fig

    # Build severity-graded colorscale
    max_val = max(float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0,
                  1e-6)
    crit_norm = critical_threshold / max_val
    warn_norm = warning_threshold / max_val

    colorscale = [
        [0.0, _hex_to_rgb(p.success, 0.35)],
        [min(warn_norm * 0.5, 0.3), _hex_to_rgb(p.success, 0.9)],
        [min(warn_norm, 0.5), p.warning],
        [min(crit_norm, 0.75), _hex_to_rgb(p.danger, 0.85)],
        [1.0, p.danger],
    ]

    fig.add_trace(go.Heatmap(
        z=matrix,
        y=layer_names,
        colorscale=colorscale,
        zmin=0.0,
        zmax=max_val,
        colorbar=dict(
            title=dict(text="Drift Score", font=dict(color=p.text_secondary, size=10)),
            tickfont=dict(color=p.text_muted, size=9),
            bgcolor="rgba(20, 28, 46, 0.6)",
            bordercolor=p.border,
            borderwidth=1,
            thickness=14,
            len=0.75,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tick %{x}<br>"
            "Drift: %{z:.4f}<br>"
            "<extra></extra>"
        ),
        xgap=1, ygap=2,
    ))

    fig.update_layout(
        title="Multi-Layer Drift Heatmap · Wasserstein-1",
        xaxis_title="Evaluation Tick",
        yaxis_title=None,
        yaxis=dict(autorange="reversed"),
    )
    apply_theme(fig, p, height=height)

    # Extend y-axis to give more room for layer labels
    fig.update_yaxes(tickfont=dict(size=10, family="JetBrains Mono, monospace"))

    return fig


def build_layer_trajectories(
    matrix: np.ndarray,
    layer_names: List[str],
    palette: Palette = None,
    warning_threshold: float = 0.15,
    critical_threshold: float = 0.35,
    height: int = 320,
) -> go.Figure:
    """
    Per-layer line chart of drift over time with threshold bands.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()

    if matrix.size == 0:
        fig.update_layout(title="Per-Layer Drift Trajectories")
        apply_theme(fig, p, height=height)
        return fig

    n_layers, n_ticks = matrix.shape
    x = list(range(n_ticks))

    for i, name in enumerate(layer_names):
        y = matrix[i]
        color = accent_color(p, i)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Tick %{x}<br>"
                "Drift: %{y:.4f}<extra></extra>"
            ),
        ))

    # Warning threshold band
    fig.add_hline(
        y=warning_threshold,
        line=dict(color=p.warning, width=1, dash="dot"),
        annotation_text=f"warn {warning_threshold:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=p.warning, size=9),
    )
    # Critical threshold band
    fig.add_hline(
        y=critical_threshold,
        line=dict(color=p.danger, width=1, dash="dot"),
        annotation_text=f"critical {critical_threshold:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=p.danger, size=9),
    )

    fig.update_layout(
        title="Per-Layer Drift Trajectories",
        xaxis_title="Evaluation Tick",
        yaxis_title="Wasserstein-1 Score",
    )
    apply_theme(fig, p, height=height)
    return fig


def build_current_scores_bar(
    layer_names: List[str],
    latest_scores: List[float],
    palette: Palette = None,
    warning_threshold: float = 0.15,
    critical_threshold: float = 0.35,
    height: int = 260,
) -> go.Figure:
    """
    Horizontal bar chart of the CURRENT drift score per layer,
    with severity-based coloring.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()

    if not layer_names:
        apply_theme(fig, p, height=height)
        return fig

    # Assign colors by severity
    bar_colors = []
    for score in latest_scores:
        if score >= critical_threshold:
            bar_colors.append(p.danger)
        elif score >= warning_threshold:
            bar_colors.append(p.warning)
        else:
            bar_colors.append(p.success)

    fig.add_trace(go.Bar(
        y=layer_names,
        x=latest_scores,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        hovertemplate="<b>%{y}</b><br>Current: %{x:.4f}<extra></extra>",
        text=[f"{v:.3f}" for v in latest_scores],
        textposition="outside",
        textfont=dict(color=p.text_secondary, family="JetBrains Mono, monospace", size=10),
    ))

    fig.add_vline(
        x=warning_threshold,
        line=dict(color=p.warning, width=1, dash="dot"),
    )
    fig.add_vline(
        x=critical_threshold,
        line=dict(color=p.danger, width=1, dash="dot"),
    )

    fig.update_layout(
        title="Current Drift Score by Layer",
        xaxis_title="Wasserstein-1",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def _hex_to_rgb(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB → 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    if alpha < 1.0:
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgb({r},{g},{b})"


# =============================================================================
# STREAMLIT RENDER
# =============================================================================

def render(
    st,
    palette: Palette,
    state: DriftHeatmapState,
    key_prefix: str = "drift_heatmap",
    show_demo_button: bool = True,
) -> None:
    """
    Render the drift heatmap panel into a Streamlit context.
    """
    # ---- Header ----
    st.markdown(section_header("🧠", "Wasserstein-1 Drift Radar"), unsafe_allow_html=True)

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        warn_th = st.number_input(
            "Warning threshold",
            min_value=0.01, max_value=2.0,
            value=float(state.warning_threshold), step=0.01,
            key=f"{key_prefix}_warn",
            format="%.2f",
        )
        state.warning_threshold = warn_th

    with ctrl2:
        crit_th = st.number_input(
            "Critical threshold",
            min_value=0.05, max_value=5.0,
            value=float(state.critical_threshold), step=0.05,
            key=f"{key_prefix}_crit",
            format="%.2f",
        )
        state.critical_threshold = crit_th

    with ctrl3:
        heat_height = st.slider(
            "Heatmap height",
            min_value=240, max_value=600, value=380, step=20,
            key=f"{key_prefix}_height",
        )

    with ctrl4:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key=f"{key_prefix}_reset", use_container_width=True):
            state.reset()
            st.toast("Drift history cleared.", icon="🧹")

    if show_demo_button and state.is_empty:
        if st.button("🌱 Seed demo drift data (120 ticks)",
                     key=f"{key_prefix}_seed"):
            state.seed_demo(120)
            st.rerun()

    if state.is_empty:
        st.info(
            "No drift data captured yet. Run the engine, or click "
            "**Seed demo drift data** above to preview the panel."
        )
        return

    # ---- Data ----
    matrix, layer_names = state.as_matrix()
    gstats = state.global_stats()

    # ---- Top Metric Row ----
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(metric_card(
            "Layers Monitored",
            f"{gstats['num_layers']}",
            f"{gstats['num_ticks']} ticks",
        ), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_card(
            "Peak Drift",
            f"{gstats['max_score']:.4f}",
            "across all layers",
            accent=(palette.danger
                    if gstats['max_score'] >= state.critical_threshold
                    else palette.warning
                    if gstats['max_score'] >= state.warning_threshold
                    else palette.success),
        ), unsafe_allow_html=True)

    with m3:
        st.markdown(metric_card(
            "Total Alerts",
            f"{gstats['total_alerts']:,}",
            f"{gstats['critical_alerts']} critical",
            accent=(palette.danger if gstats['critical_alerts'] > 0
                    else palette.warning if gstats['total_alerts'] > 0
                    else palette.success),
        ), unsafe_allow_html=True)

    with m4:
        arrow = "↑" if gstats['trending_up'] > 0 else "↓" if gstats['trending_down'] > 0 else "→"
        st.markdown(metric_card(
            "Trending Layers",
            f"{arrow} {gstats['trending_up']} / {gstats['trending_down']}",
            "rising / falling",
            accent=(palette.warning if gstats['trending_up'] > 0 else palette.success),
        ), unsafe_allow_html=True)

    with m5:
        # Overall health badge
        if gstats['critical_alerts'] > 0:
            badge_html = status_badge("CRITICAL", "critical", "⚠")
            badge_level = "critical"
        elif gstats['total_alerts'] > 0:
            badge_html = status_badge("WARNING", "paused", "●")
            badge_level = "warning"
        else:
            badge_html = status_badge("NOMINAL", "active", "●")
            badge_level = "nominal"

        st.markdown(f"""
        <div class="hud-card hud-fade-in" style="height:100%; display:flex;
                    flex-direction:column; justify-content:center; align-items:center;">
            <div class="hud-card-title" style="margin-bottom:12px;">Overall Health</div>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Main Heatmap ----
    fig_heat = build_heatmap_figure(
        matrix=matrix,
        layer_names=layer_names,
        palette=palette,
        warning_threshold=state.warning_threshold,
        critical_threshold=state.critical_threshold,
        height=heat_height,
    )
    st.plotly_chart(fig_heat, use_container_width=True, key=f"{key_prefix}_heat")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Split: Line trajectories | Current-score bar ----
    left, right = st.columns([2, 1])

    with left:
        fig_lines = build_layer_trajectories(
            matrix=matrix,
            layer_names=layer_names,
            palette=palette,
            warning_threshold=state.warning_threshold,
            critical_threshold=state.critical_threshold,
            height=320,
        )
        st.plotly_chart(fig_lines, use_container_width=True, key=f"{key_prefix}_lines")

    with right:
        latest_scores = [state.layer_stats(name)['latest'] for name in layer_names]
        fig_bar = build_current_scores_bar(
            layer_names=layer_names,
            latest_scores=latest_scores,
            palette=palette,
            warning_threshold=state.warning_threshold,
            critical_threshold=state.critical_threshold,
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_bar")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Layer Severity Table | Alert Timeline ----
    tleft, tright = st.columns([1, 1])

    with tleft:
        st.markdown(section_header("📊", "Per-Layer Status"), unsafe_allow_html=True)

        for name in layer_names:
            s = state.layer_stats(name)
            trend_icon = {'rising': '↑', 'falling': '↓', 'stable': '→'}[s['trend']]
            trend_color = {
                'rising': palette.warning,
                'falling': palette.success,
                'stable': palette.text_muted,
            }[s['trend']]

            # Severity tone
            if s['latest'] >= state.critical_threshold:
                tone = palette.danger
            elif s['latest'] >= state.warning_threshold:
                tone = palette.warning
            else:
                tone = palette.success

            st.markdown(f"""
            <div class="hud-card" style="padding:10px 14px; margin:6px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-family:var(--mono); font-size:0.85rem; color:var(--text-primary);">
                        {name}
                    </div>
                    <div style="display:flex; gap:14px; align-items:center;">
                        <span style="font-family:var(--mono); color:{trend_color};">
                            {trend_icon}
                        </span>
                        <span style="font-family:var(--mono); color:{tone}; font-weight:700;">
                            {s['latest']:.4f}
                        </span>
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between;
                            margin-top:6px; font-size:0.72rem;
                            color:var(--text-muted); font-family:var(--mono);">
                    <span>mean {s['mean']:.4f}</span>
                    <span>max {s['max']:.4f}</span>
                    <span>{s['alerts']} alerts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tright:
        st.markdown(section_header("🚨", "Alert Timeline"), unsafe_allow_html=True)

        events = state.alert_events(limit=10)
        if not events:
            st.markdown(alert_box(
                "All Clear",
                "No drift events have crossed the warning threshold yet.",
                level="success", icon="✅",
            ), unsafe_allow_html=True)
        else:
            for e in events:
                level = e['level']
                icon = "🚨" if level == "critical" else "⚠️"
                color = palette.danger if level == "critical" else palette.warning
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px;
                            padding:8px 12px; margin:4px 0;
                            background:rgba(20,28,46,0.5);
                            border-left:3px solid {color};
                            border-radius:8px; font-family:var(--mono);
                            font-size:0.78rem;">
                    <span>{icon}</span>
                    <span style="color:var(--text-muted);">tick {e['tick']}</span>
                    <span style="color:var(--text-primary); flex:1;">{e['layer']}</span>
                    <span style="color:{color}; font-weight:700;">{e['score']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)