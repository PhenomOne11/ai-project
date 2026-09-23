"""
Hardware Telemetry Dashboard Panel

Real-time visualization of system resource usage:
    - Rolling multi-axis line chart (CPU% / RAM MB / Latency ms)
    - Radial gauges for current resource usage vs soft limits
    - Latency histogram with warning/critical bands
    - Memory breakdown (RSS + VMS)
    - Per-resource health indicators

Data ingestion is via `TelemetrySample` snapshots. The panel can also
auto-probe the live process using `probe_live()` (from Phase 12).
"""

import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from dataclasses import dataclass, field

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import (
    metric_card, section_header, divider, status_badge,
    apply_theme, accent_color, progress_bar, kv_row, alert_box,
)
from dashboard.theme.palette import Palette
from core.metrics.system import probe_process, probe_system


# =============================================================================
# STATE
# =============================================================================

@dataclass
class TelemetrySample:
    """One time-slice of hardware telemetry."""
    timestamp: float
    tick: int = 0
    cpu_percent: float = 0.0           # Process CPU%
    system_cpu_percent: float = 0.0    # System CPU%
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    system_mem_percent: float = 0.0
    system_mem_available_mb: float = 0.0
    latency_ms: float = 0.0            # Last forward-pass latency
    num_threads: int = 0

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'tick': self.tick,
            'cpu_percent': self.cpu_percent,
            'system_cpu_percent': self.system_cpu_percent,
            'rss_mb': self.rss_mb,
            'vms_mb': self.vms_mb,
            'system_mem_percent': self.system_mem_percent,
            'system_mem_available_mb': self.system_mem_available_mb,
            'latency_ms': self.latency_ms,
            'num_threads': self.num_threads,
        }


class TelemetryState:
    """
    Rolling buffer of hardware telemetry samples.

    Bounded at `history_length` records. All queries preserve write order.

    Args:
        history_length: Max samples retained
        cpu_warning: Warning threshold for CPU% (0–100)
        cpu_critical: Critical threshold for CPU%
        mem_warning_mb: Warning threshold for RSS in MB
        mem_critical_mb: Critical threshold for RSS in MB
        latency_warning_ms: Warning threshold for latency
        latency_critical_ms: Critical threshold for latency
        auto_probe: If True, `append_auto()` populates from live psutil
    """

    def __init__(
        self,
        history_length: int = 300,
        cpu_warning: float = 70.0,
        cpu_critical: float = 90.0,
        mem_warning_mb: float = 1024.0,
        mem_critical_mb: float = 2048.0,
        latency_warning_ms: float = 3.0,
        latency_critical_ms: float = 6.0,
        auto_probe: bool = True,
    ):
        self.history_length = history_length
        self.cpu_warning = cpu_warning
        self.cpu_critical = cpu_critical
        self.mem_warning_mb = mem_warning_mb
        self.mem_critical_mb = mem_critical_mb
        self.latency_warning_ms = latency_warning_ms
        self.latency_critical_ms = latency_critical_ms
        self.auto_probe = auto_probe

        self._samples: deque = deque(maxlen=history_length)
        self._n_pushed: int = 0
        self._tick: int = 0

    # ---- Ingestion ----

    def append(
        self,
        cpu_percent: float = 0.0,
        rss_mb: float = 0.0,
        vms_mb: float = 0.0,
        latency_ms: float = 0.0,
        system_cpu_percent: float = 0.0,
        system_mem_percent: float = 0.0,
        system_mem_available_mb: float = 0.0,
        num_threads: int = 0,
        tick: Optional[int] = None,
    ) -> TelemetrySample:
        """Append a fully-specified sample."""
        self._n_pushed += 1
        self._tick += 1
        sample = TelemetrySample(
            timestamp=time.time(),
            tick=tick if tick is not None else self._tick,
            cpu_percent=float(cpu_percent),
            system_cpu_percent=float(system_cpu_percent),
            rss_mb=float(rss_mb),
            vms_mb=float(vms_mb),
            system_mem_percent=float(system_mem_percent),
            system_mem_available_mb=float(system_mem_available_mb),
            latency_ms=float(latency_ms),
            num_threads=int(num_threads),
        )
        self._samples.append(sample)
        return sample

    def append_auto(self, latency_ms: float = 0.0) -> TelemetrySample:
        """Auto-probe the live process and append the sample."""
        proc = probe_process()
        sysinfo = probe_system()
        return self.append(
            cpu_percent=proc.get('cpu_percent', 0.0),
            rss_mb=proc.get('rss_mb', 0.0),
            vms_mb=proc.get('vms_mb', 0.0),
            latency_ms=latency_ms,
            system_cpu_percent=sysinfo.get('cpu_percent', 0.0),
            system_mem_percent=sysinfo.get('mem_percent', 0.0),
            system_mem_available_mb=sysinfo.get('mem_available_mb', 0.0),
            num_threads=proc.get('num_threads', 0),
        )

    # ---- Retrieval ----

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """
        Return column-oriented view of the buffer.

        Keys: ticks, timestamps, cpu_percent, system_cpu_percent,
              rss_mb, vms_mb, latency_ms, system_mem_percent,
              system_mem_available_mb, num_threads
        """
        n = len(self._samples)
        keys = [
            'tick', 'timestamp', 'cpu_percent', 'system_cpu_percent',
            'rss_mb', 'vms_mb', 'latency_ms', 'system_mem_percent',
            'system_mem_available_mb', 'num_threads',
        ]
        out = {}
        for k in keys:
            out[k] = np.array([getattr(s, k) for s in self._samples], dtype=np.float64)
        return out

    def latest(self) -> Optional[TelemetrySample]:
        return self._samples[-1] if self._samples else None

    @property
    def size(self) -> int:
        return len(self._samples)

    @property
    def is_empty(self) -> bool:
        return len(self._samples) == 0

    @property
    def total_pushed(self) -> int:
        return self._n_pushed

    # ---- Statistics ----

    def stats(self) -> Dict[str, Any]:
        """Compute a summary of the current buffer."""
        if self.is_empty:
            return self._empty_stats()

        cols = self.to_arrays()

        def summary(arr):
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {'mean': 0.0, 'min': 0.0, 'max': 0.0,
                        'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'std': 0.0}
            return {
                'mean': float(arr.mean()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'std': float(arr.std()),
            }

        cpu = summary(cols['cpu_percent'])
        sys_cpu = summary(cols['system_cpu_percent'])
        rss = summary(cols['rss_mb'])
        vms = summary(cols['vms_mb'])
        lat = summary(cols['latency_ms'])
        sys_mem = summary(cols['system_mem_percent'])

        # Health decisions
        health = {
            'cpu': self._classify(cpu['p95'], self.cpu_warning, self.cpu_critical),
            'rss': self._classify(rss['p95'], self.mem_warning_mb, self.mem_critical_mb),
            'latency': self._classify(lat['p95'], self.latency_warning_ms, self.latency_critical_ms),
        }

        overall = 'nominal'
        for level in health.values():
            if level == 'critical':
                overall = 'critical'
                break
            if level == 'warning' and overall != 'critical':
                overall = 'warning'

        return {
            'n_samples': self.size,
            'total_pushed': self._n_pushed,
            'cpu': cpu,
            'system_cpu': sys_cpu,
            'rss': rss,
            'vms': vms,
            'latency': lat,
            'system_mem': sys_mem,
            'health': health,
            'overall': overall,
        }

    def _empty_stats(self) -> Dict[str, Any]:
        empty = {'mean': 0.0, 'min': 0.0, 'max': 0.0,
                 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'std': 0.0}
        return {
            'n_samples': 0,
            'total_pushed': self._n_pushed,
            'cpu': dict(empty),
            'system_cpu': dict(empty),
            'rss': dict(empty),
            'vms': dict(empty),
            'latency': dict(empty),
            'system_mem': dict(empty),
            'health': {'cpu': 'nominal', 'rss': 'nominal', 'latency': 'nominal'},
            'overall': 'nominal',
        }

    def _classify(self, value: float, warn: float, crit: float) -> str:
        if value >= crit:
            return 'critical'
        if value >= warn:
            return 'warning'
        return 'nominal'

    def latency_histogram(self, bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Return (counts, bin_edges) for histogram of latency values."""
        if self.is_empty:
            return np.zeros(bins, dtype=np.int64), np.linspace(0, 1, bins + 1)
        cols = self.to_arrays()
        lat = cols['latency_ms'][np.isfinite(cols['latency_ms'])]
        if lat.size == 0:
            return np.zeros(bins, dtype=np.int64), np.linspace(0, 1, bins + 1)
        lo = max(0.0, float(lat.min()))
        hi = float(lat.max())
        if hi <= lo:
            hi = lo + 1e-3
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(lat, bins=edges)
        return counts.astype(np.int64), edges

    # ---- Synthetic demo ----

    def seed_demo(self, n: int = 200, seed: int = 42) -> None:
        """
        Generate plausible synthetic telemetry.
        Produces a warm-up curve, then a modest latency spike mid-run,
        then a return to baseline so lines show visible structure.
        """
        rng = np.random.default_rng(seed)
        self._samples.clear()
        self._n_pushed = 0
        self._tick = 0

        t = np.linspace(0, 1, n)

        # CPU: starts at ~30, warmup ramp, plateau at ~55
        cpu = 30.0 + 25.0 * (1 - np.exp(-t * 8)) + rng.normal(0, 2.0, n)
        cpu = np.clip(cpu, 0, 100)

        # RSS: gradual growth from 380 → 620 MB with a bump
        rss = 380.0 + 240.0 * t + rng.normal(0, 5.0, n)
        rss[int(n * 0.7):] += 40.0

        # Latency: baseline ~2.2, spike around 0.55–0.65
        lat = 2.2 + 0.3 * rng.standard_normal(n)
        spike_a = int(n * 0.55)
        spike_b = int(n * 0.65)
        if spike_b > spike_a:
            lat[spike_a:spike_b] += 3.5 + 0.5 * rng.standard_normal(spike_b - spike_a)
        lat = np.clip(lat, 0.1, None)

        # VMS
        vms = rss * 2.3 + rng.normal(0, 20, n)

        # System metrics
        sys_cpu = 20.0 + 30.0 * t + rng.normal(0, 3.0, n)
        sys_mem = 40.0 + 15.0 * t + rng.normal(0, 1.0, n)
        sys_mem_avail = np.maximum(200.0, 4000.0 - 2000.0 * t + rng.normal(0, 30, n))

        for i in range(n):
            self.append(
                cpu_percent=cpu[i],
                rss_mb=rss[i],
                vms_mb=vms[i],
                latency_ms=lat[i],
                system_cpu_percent=sys_cpu[i],
                system_mem_percent=sys_mem[i],
                system_mem_available_mb=sys_mem_avail[i],
                num_threads=int(6 + rng.integers(0, 4)),
                tick=i,
            )

    def reset(self) -> None:
        self._samples.clear()
        self._n_pushed = 0
        self._tick = 0


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_multi_metric_chart(
    columns: Dict[str, np.ndarray],
    palette: Palette = None,
    height: int = 340,
    latency_warning_ms: float = 3.0,
    latency_critical_ms: float = 6.0,
) -> go.Figure:
    """
    3-row subplot overlay:
        Row 1: Process CPU% + System CPU%
        Row 2: RSS / VMS (MB)
        Row 3: Latency (ms) with warning/critical bands
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.33, 0.33, 0.34],
        subplot_titles=("CPU Utilization", "Memory (MB)", "Forward Latency (ms)"),
    )

    if 'tick' not in columns or len(columns['tick']) == 0:
        apply_theme(fig, p, height=height)
        return fig

    x = columns['tick']

    # Row 1 — CPU
    fig.add_trace(go.Scatter(
        x=x, y=columns['cpu_percent'],
        mode="lines", name="Process CPU%",
        line=dict(color=p.accent_primary, width=2, shape="spline", smoothing=0.6),
        hovertemplate="tick %{x}<br>Process CPU: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=columns['system_cpu_percent'],
        mode="lines", name="System CPU%",
        line=dict(color=p.accent_secondary, width=1.5, dash="dot"),
        hovertemplate="tick %{x}<br>System CPU: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # Row 2 — Memory
    fig.add_trace(go.Scatter(
        x=x, y=columns['rss_mb'],
        mode="lines", name="RSS (MB)",
        line=dict(color=p.success, width=2, shape="spline", smoothing=0.6),
        hovertemplate="tick %{x}<br>RSS: %{y:.1f} MB<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=columns['vms_mb'],
        mode="lines", name="VMS (MB)",
        line=dict(color=p.info, width=1.5, dash="dot"),
        hovertemplate="tick %{x}<br>VMS: %{y:.1f} MB<extra></extra>",
    ), row=2, col=1)

    # Row 3 — Latency
    fig.add_trace(go.Scatter(
        x=x, y=columns['latency_ms'],
        mode="lines", name="Latency (ms)",
        line=dict(color=p.warning, width=2, shape="spline", smoothing=0.6),
        hovertemplate="tick %{x}<br>Latency: %{y:.3f} ms<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(
        y=latency_warning_ms, line=dict(color=p.warning, width=1, dash="dot"),
        row=3, col=1,
        annotation_text=f"warn {latency_warning_ms:.1f}ms",
        annotation_position="top right",
        annotation_font=dict(color=p.warning, size=9),
    )
    fig.add_hline(
        y=latency_critical_ms, line=dict(color=p.danger, width=1, dash="dot"),
        row=3, col=1,
        annotation_text=f"crit {latency_critical_ms:.1f}ms",
        annotation_position="top right",
        annotation_font=dict(color=p.danger, size=9),
    )

    fig.update_layout(title="Rolling Hardware Telemetry")
    apply_theme(fig, p, height=height)
    fig.update_xaxes(range=[float(x.min()), float(x.max())])
    return fig


def build_gauge(
    title: str,
    value: float,
    max_value: float,
    warning: float,
    critical: float,
    palette: Palette = None,
    suffix: str = "",
    height: int = 220,
) -> go.Figure:
    """
    Radial gauge with severity zones (green → amber → red).
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    # Normalize thresholds to [0, 1]
    warn_norm = min(warning / max_value, 0.99) if max_value > 0 else 0.6
    crit_norm = min(critical / max_value, 1.0) if max_value > 0 else 0.85
    if crit_norm <= warn_norm:
        crit_norm = min(warn_norm + 0.1, 1.0)

    value_norm = min(max(value / max_value, 0.0), 1.0) if max_value > 0 else 0.0

    # Bar color by severity
    if value >= critical:
        bar_color = p.danger
    elif value >= warning:
        bar_color = p.warning
    else:
        bar_color = p.success

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(
            suffix=suffix,
            font=dict(color=bar_color, size=20, family="JetBrains Mono, monospace"),
        ),
        title=dict(
            text=title,
            font=dict(color=p.text_secondary, size=12),
        ),
        gauge=dict(
            axis=dict(
                range=[0, max_value],
                tickcolor=p.text_muted,
                tickfont=dict(color=p.text_muted, size=9, family="JetBrains Mono"),
            ),
            bar=dict(
                color=bar_color,
                thickness=0.28,
            ),
            bgcolor="rgba(20, 28, 46, 0.5)",
            borderwidth=0,
            steps=[
                dict(range=[0, warning], color="rgba(16, 185, 129, 0.12)"),
                dict(range=[warning, critical], color="rgba(245, 158, 11, 0.12)"),
                dict(range=[critical, max_value], color="rgba(239, 68, 68, 0.14)"),
            ],
            threshold=dict(
                line=dict(color=p.danger, width=3),
                thickness=0.85,
                value=critical,
            ),
        ),
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=10))
    apply_theme(fig, p, height=height)
    return fig


def build_latency_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    palette: Palette = None,
    warning_ms: float = 3.0,
    critical_ms: float = 6.0,
    height: int = 280,
) -> go.Figure:
    """
    Latency histogram with vertical warning/critical lines.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if counts.size == 0:
        apply_theme(fig, p, height=height)
        return fig

    centers = (edges[:-1] + edges[1:]) / 2.0
    # Color bars by band
    bar_colors = []
    for c in centers:
        if c >= critical_ms:
            bar_colors.append(p.danger)
        elif c >= warning_ms:
            bar_colors.append(p.warning)
        else:
            bar_colors.append(p.success)

    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="Latency %{x:.2f} ms<br>Count %{y}<extra></extra>",
    ))

    fig.add_vline(x=warning_ms, line=dict(color=p.warning, width=1, dash="dash"))
    fig.add_vline(x=critical_ms, line=dict(color=p.danger, width=1, dash="dash"))

    fig.update_layout(
        title="Latency Distribution",
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


# =============================================================================
# STREAMLIT RENDER
# =============================================================================

_HEALTH_STYLE = {
    'nominal': ('active', '●'),
    'warning': ('paused', '●'),
    'critical': ('critical', '⚠'),
}


def render(
    st,
    palette: Palette,
    state: TelemetryState,
    key_prefix: str = "telemetry",
    show_demo_button: bool = True,
) -> None:
    """Render the hardware telemetry panel into a Streamlit context."""
    st.markdown(section_header("📈", "Hardware Telemetry"), unsafe_allow_html=True)

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        hist_len = st.number_input(
            "History length",
            min_value=50, max_value=2000,
            value=int(state.history_length), step=50,
            key=f"{key_prefix}_histlen",
        )
        if hist_len != state.history_length and hist_len >= state.size:
            old = list(state._samples)
            state.history_length = int(hist_len)
            state._samples = deque(old[-int(hist_len):], maxlen=int(hist_len))

    with ctrl2:
        auto_probe = st.checkbox(
            "Auto-Poll (psutil)",
            value=False,
            key=f"{key_prefix}_auto_probe",
            help="Sample the live process when the page refreshes.",
        )

    with ctrl3:
        chart_h = st.slider(
            "Chart height",
            min_value=260, max_value=700, value=340, step=40,
            key=f"{key_prefix}_height",
        )

    with ctrl4:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key=f"{key_prefix}_reset", use_container_width=True):
            state.reset()
            st.toast("Telemetry history cleared.", icon="🧹")

    if show_demo_button and state.is_empty:
        if st.button("🌱 Seed demo telemetry (200 samples)",
                     key=f"{key_prefix}_seed"):
            state.seed_demo(200)
            st.rerun()

    # Auto-probe on each render
    if auto_probe and not state.is_empty:
        state.append_auto()

    if state.is_empty:
        st.info(
            "No telemetry samples collected yet. Boot the engine, "
            "enable auto-poll, or click **Seed demo telemetry** above."
        )
        return

    # ---- Data ----
    cols = state.to_arrays()
    stats = state.stats()

    # ---- Top Metric Row ----
    latest = state.latest()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        health = stats['health']['cpu']
        accent = (palette.danger if health == 'critical'
                  else palette.warning if health == 'warning'
                  else palette.success)
        st.markdown(metric_card(
            "CPU (process)",
            f"{latest.cpu_percent:.1f}%",
            f"p95 {stats['cpu']['p95']:.1f}%",
            accent=accent,
        ), unsafe_allow_html=True)

    with m2:
        health = stats['health']['rss']
        accent = (palette.danger if health == 'critical'
                  else palette.warning if health == 'warning'
                  else palette.success)
        st.markdown(metric_card(
            "RSS Memory",
            f"{latest.rss_mb:.0f} MB",
            f"p95 {stats['rss']['p95']:.0f} MB",
            accent=accent,
        ), unsafe_allow_html=True)

    with m3:
        health = stats['health']['latency']
        accent = (palette.danger if health == 'critical'
                  else palette.warning if health == 'warning'
                  else palette.success)
        st.markdown(metric_card(
            "Latency (last)",
            f"{latest.latency_ms:.3f} ms",
            f"p95 {stats['latency']['p95']:.3f} ms",
            accent=accent,
        ), unsafe_allow_html=True)

    with m4:
        st.markdown(metric_card(
            "System CPU",
            f"{latest.system_cpu_percent:.1f}%",
            f"threads {latest.num_threads}",
        ), unsafe_allow_html=True)

    with m5:
        level, icon = _HEALTH_STYLE.get(stats['overall'], _HEALTH_STYLE['nominal'])
        st.markdown(f"""
        <div class="hud-card hud-fade-in" style="height:100%; display:flex;
                    flex-direction:column; justify-content:center; align-items:center;">
            <div class="hud-card-title" style="margin-bottom:12px;">Overall Health</div>
            {status_badge(stats['overall'].upper(), level, icon)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Main chart ----
    fig_main = build_multi_metric_chart(
        columns=cols,
        palette=palette,
        height=chart_h,
        latency_warning_ms=state.latency_warning_ms,
        latency_critical_ms=state.latency_critical_ms,
    )
    st.plotly_chart(fig_main, use_container_width=True, key=f"{key_prefix}_main")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Gauges row ----
    g1, g2, g3 = st.columns(3)
    with g1:
        fig_g1 = build_gauge(
            title="CPU Load",
            value=latest.cpu_percent,
            max_value=100.0,
            warning=state.cpu_warning,
            critical=state.cpu_critical,
            palette=palette,
            suffix="%",
        )
        st.plotly_chart(fig_g1, use_container_width=True, key=f"{key_prefix}_g1")

    with g2:
        # Memory gauge uses RSS with a nominal max
        mem_max = max(state.mem_critical_mb * 1.3, latest.rss_mb * 1.15, 1.0)
        fig_g2 = build_gauge(
            title="RSS Memory",
            value=latest.rss_mb,
            max_value=mem_max,
            warning=state.mem_warning_mb,
            critical=state.mem_critical_mb,
            palette=palette,
            suffix=" MB",
        )
        st.plotly_chart(fig_g2, use_container_width=True, key=f"{key_prefix}_g2")

    with g3:
        lat_max = max(state.latency_critical_ms * 1.5, latest.latency_ms * 1.15, 1.0)
        fig_g3 = build_gauge(
            title="Forward Latency",
            value=latest.latency_ms,
            max_value=lat_max,
            warning=state.latency_warning_ms,
            critical=state.latency_critical_ms,
            palette=palette,
            suffix=" ms",
        )
        st.plotly_chart(fig_g3, use_container_width=True, key=f"{key_prefix}_g3")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Histogram + stats table ----
    left, right = st.columns([1.3, 1])

    with left:
        counts, edges = state.latency_histogram(bins=30)
        fig_hist = build_latency_histogram(
            counts=counts,
            edges=edges,
            palette=palette,
            warning_ms=state.latency_warning_ms,
            critical_ms=state.latency_critical_ms,
            height=280,
        )
        st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")

    with right:
        st.markdown(section_header("📊", "Rolling Statistics"), unsafe_allow_html=True)

        for key, label in [('cpu', 'CPU %'), ('rss', 'RSS MB'),
                            ('vms', 'VMS MB'), ('latency', 'Latency ms')]:
            s = stats[key]
            st.markdown(
                kv_row(
                    label,
                    f"μ {s['mean']:.2f} · p95 {s['p95']:.2f} · max {s['max']:.2f}",
                ),
                unsafe_allow_html=True,
            )

        # Health status per dimension
        st.markdown("<br>", unsafe_allow_html=True)
        for dim, level in stats['health'].items():
            lvl, icon = _HEALTH_STYLE.get(level, _HEALTH_STYLE['nominal'])
            label_text = {'cpu': 'CPU', 'rss': 'Memory', 'latency': 'Latency'}.get(dim, dim)
            st.markdown(
                f"<div style='margin:4px 0;'>"
                f"<span style='color:var(--text-muted);font-size:0.75rem;"
                f"letter-spacing:0.08em;text-transform:uppercase;'>{label_text}</span> "
                f"&nbsp;{status_badge(level.upper(), lvl, icon)}</div>",
                unsafe_allow_html=True,
            )

        # Buffer fill
        st.markdown("<br>", unsafe_allow_html=True)
        fill = min(1.0, state.size / max(state.history_length, 1))
        st.markdown(
            progress_bar("Buffer fill", fill, 1.0),
            unsafe_allow_html=True,
        )

        st.caption(f"Total samples pushed: **{state.total_pushed:,}**")