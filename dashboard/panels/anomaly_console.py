"""
Anomaly & OOD Console Panel

Visualizes:
    - Fused anomaly score timeline with severity bands
    - Per-component breakdown (recon / isolation / mahalanobis)
    - Anomaly score histogram with threshold zones
    - OOD replay buffer inspector (top-N priorities)
    - Adaptation step history (loss deltas)
    - Circuit breaker trip markers

Accepts data from:
    - Direct API: `push_score(...)` per tick
    - Phase 9 integration: `ingest_anomaly_result(result_dict)`
    - Phase 8 integration: `push_adaptation(...)` per step
    - Synthetic: `seed_demo(...)` for demonstration
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


# =============================================================================
# RECORDS
# =============================================================================

@dataclass
class AnomalyRecord:
    """One tick of anomaly scoring."""
    tick: int
    timestamp: float
    fused_score: float
    recon_component: float = 0.0
    iso_component: float = 0.0
    maha_component: float = 0.0
    severity: str = "nominal"

    def to_dict(self) -> dict:
        return {
            'tick': self.tick, 'timestamp': self.timestamp,
            'fused_score': self.fused_score,
            'recon_component': self.recon_component,
            'iso_component': self.iso_component,
            'maha_component': self.maha_component,
            'severity': self.severity,
        }


@dataclass
class OODEvent:
    """A single OOD detection event."""
    tick: int
    timestamp: float
    priority: float
    label: str = ""

    def to_dict(self) -> dict:
        return {'tick': self.tick, 'timestamp': self.timestamp,
                'priority': self.priority, 'label': self.label}


@dataclass
class AdaptationRecord:
    """A single background micro-adaptation step."""
    tick: int
    timestamp: float
    loss_before: float
    loss_after: float
    ewc_enabled: bool = False
    batch_size: int = 16

    @property
    def delta(self) -> float:
        return self.loss_before - self.loss_after


# =============================================================================
# STATE
# =============================================================================

class AnomalyConsoleState:
    """
    Rolling buffer for the entire anomaly/adaptation pipeline.

    Args:
        history_length: Max anomaly score records to retain
        ood_history_length: Max OOD events to retain
        adaptation_history_length: Max adaptation steps to retain
        warning_threshold: Score above which severity = warning
        critical_threshold: Score above which severity = critical
        emergency_threshold: Score above which severity = emergency
    """

    def __init__(
        self,
        history_length: int = 500,
        ood_history_length: int = 200,
        adaptation_history_length: int = 200,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.75,
        emergency_threshold: float = 0.9,
    ):
        self.history_length = history_length
        self.ood_history_length = ood_history_length
        self.adaptation_history_length = adaptation_history_length

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold

        self._records: deque = deque(maxlen=history_length)
        self._ood_events: deque = deque(maxlen=ood_history_length)
        self._adaptations: deque = deque(maxlen=adaptation_history_length)

        self._tick: int = 0
        self._circuit_breaker_trips: List[int] = []

    # ---- Ingestion ----

    def _classify(self, score: float) -> str:
        if score >= self.emergency_threshold:
            return "emergency"
        if score >= self.critical_threshold:
            return "critical"
        if score >= self.warning_threshold:
            return "warning"
        return "nominal"

    def push_score(
        self,
        fused_score: float,
        recon_component: float = 0.0,
        iso_component: float = 0.0,
        maha_component: float = 0.0,
        tick: Optional[int] = None,
    ) -> AnomalyRecord:
        """Append a single anomaly score record."""
        self._tick += 1
        ts = time.time()
        rec = AnomalyRecord(
            tick=tick if tick is not None else self._tick,
            timestamp=ts,
            fused_score=float(fused_score),
            recon_component=float(recon_component),
            iso_component=float(iso_component),
            maha_component=float(maha_component),
            severity=self._classify(float(fused_score)),
        )
        self._records.append(rec)
        return rec

    def push_ood_event(
        self,
        priority: float,
        label: str = "",
        tick: Optional[int] = None,
    ) -> OODEvent:
        """Register an OOD sample entering the replay buffer."""
        ev = OODEvent(
            tick=tick if tick is not None else self._tick,
            timestamp=time.time(),
            priority=float(priority),
            label=label,
        )
        self._ood_events.append(ev)
        return ev

    def push_adaptation(
        self,
        loss_before: float,
        loss_after: float,
        ewc_enabled: bool = False,
        batch_size: int = 16,
        tick: Optional[int] = None,
    ) -> AdaptationRecord:
        """Register a completed micro-adaptation step."""
        rec = AdaptationRecord(
            tick=tick if tick is not None else self._tick,
            timestamp=time.time(),
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            ewc_enabled=bool(ewc_enabled),
            batch_size=int(batch_size),
        )
        self._adaptations.append(rec)
        return rec

    def mark_circuit_breaker_trip(self, tick: Optional[int] = None) -> None:
        """Log a circuit breaker trip at the given tick."""
        self._circuit_breaker_trips.append(
            tick if tick is not None else self._tick
        )

    # ---- Phase 9 Integration ----

    def ingest_anomaly_result(self, result: Dict[str, Any]) -> None:
        """
        Accept the dict-form output of `AnomalyScorer.score()`.

        Expected keys:
            'score' (tensor/array/list of per-sample floats)
            'recon_component', 'isolation_component', 'mahalanobis_component'
        Each is a tensor/array/list of same length.
        """
        def _to_list(x):
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            return np.asarray(x).flatten().tolist()

        scores = _to_list(result.get('score', []))
        recons = _to_list(result.get('recon_component', [0.0] * len(scores)))
        isos = _to_list(result.get('isolation_component', [0.0] * len(scores)))
        mahas = _to_list(result.get('mahalanobis_component', [0.0] * len(scores)))

        n = min(len(scores), len(recons), len(isos), len(mahas))
        for i in range(n):
            self.push_score(
                fused_score=scores[i],
                recon_component=recons[i],
                iso_component=isos[i],
                maha_component=mahas[i],
            )

    # ---- Demo Data ----

    def seed_demo(self, n_ticks: int = 300, seed: int = 42) -> None:
        """
        Generate plausible anomaly data with a story:
            - Steady baseline (score ~0.1-0.2)
            - Warning cluster around 35% of the run
            - Critical cluster around 65% of the run
            - Adaptation steps fire after each cluster
        """
        rng = np.random.default_rng(seed)

        self._records.clear()
        self._ood_events.clear()
        self._adaptations.clear()
        self._circuit_breaker_trips.clear()
        self._tick = 0

        w1 = int(n_ticks * 0.35)
        w2 = int(n_ticks * 0.65)

        for i in range(n_ticks):
            base = 0.12 + rng.normal(0, 0.03)

            # Warning cluster
            if w1 - 15 < i < w1 + 15:
                base += 0.4 + rng.normal(0, 0.05)

            # Critical cluster
            if w2 - 12 < i < w2 + 12:
                base += 0.65 + rng.normal(0, 0.06)

            fused = float(np.clip(base, 0.0, 1.0))

            # Component split
            recon = float(np.clip(rng.uniform(0.3, 0.5) * fused
                                  + rng.normal(0, 0.01), 0, 1))
            iso = float(np.clip(rng.uniform(0.25, 0.4) * fused
                                + rng.normal(0, 0.01), 0, 1))
            maha = float(np.clip(rng.uniform(0.15, 0.3) * fused
                                 + rng.normal(0, 0.01), 0, 1))

            rec = self.push_score(fused, recon, iso, maha, tick=i)

            # Spawn OOD events near high-score ticks
            if rec.severity in ("critical", "emergency") and rng.random() < 0.35:
                self.push_ood_event(
                    priority=float(fused * rng.uniform(0.9, 1.0)),
                    label=f"batch-{i // 16}",
                    tick=i,
                )

        # Adaptation steps shortly after each cluster
        for center in [w1 + 20, w2 + 20]:
            if 0 < center < n_ticks:
                for _ in range(3):
                    before = float(rng.uniform(0.15, 0.35))
                    after = before - float(rng.uniform(0.02, 0.08))
                    self.push_adaptation(
                        loss_before=before,
                        loss_after=after,
                        ewc_enabled=True,
                        batch_size=16,
                        tick=center,
                    )

        # Synthetic circuit breaker trip on the critical cluster
        if 0 < w2 < n_ticks:
            self.mark_circuit_breaker_trip(tick=w2)

    # ---- Retrieval ----

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Column-oriented view of the anomaly records."""
        n = len(self._records)
        keys = ['tick', 'timestamp', 'fused_score',
                'recon_component', 'iso_component', 'maha_component']
        out = {}
        for k in keys:
            out[k] = np.array([getattr(r, k) for r in self._records], dtype=np.float64)
        return out

    def ood_priorities(self) -> np.ndarray:
        return np.array([e.priority for e in self._ood_events], dtype=np.float64)

    def adaptation_arrays(self) -> Dict[str, np.ndarray]:
        n = len(self._adaptations)
        return {
            'tick': np.array([a.tick for a in self._adaptations], dtype=np.float64),
            'loss_before': np.array([a.loss_before for a in self._adaptations], dtype=np.float64),
            'loss_after': np.array([a.loss_after for a in self._adaptations], dtype=np.float64),
            'delta': np.array([a.delta for a in self._adaptations], dtype=np.float64),
        }

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def ood_count(self) -> int:
        return len(self._ood_events)

    @property
    def adaptation_count(self) -> int:
        return len(self._adaptations)

    @property
    def is_empty(self) -> bool:
        return len(self._records) == 0

    def latest(self) -> Optional[AnomalyRecord]:
        return self._records[-1] if self._records else None

    # ---- Statistics ----

    def stats(self) -> Dict[str, Any]:
        if self.is_empty:
            return self._empty_stats()

        cols = self.to_arrays()
        score = cols['fused_score']

        severity_counts = {"nominal": 0, "warning": 0, "critical": 0, "emergency": 0}
        for r in self._records:
            severity_counts[r.severity] += 1

        # Delta metrics from adaptations
        if self._adaptations:
            deltas = np.array([a.delta for a in self._adaptations])
            mean_delta = float(deltas.mean())
            total_improvement = float(deltas.sum())
        else:
            mean_delta = 0.0
            total_improvement = 0.0

        # Current EMA score
        if len(score) > 0:
            alpha = 0.1
            ema = score[0]
            for v in score[1:]:
                ema = alpha * v + (1 - alpha) * ema
        else:
            ema = 0.0

        return {
            'n_samples': self.size,
            'score_mean': float(score.mean()),
            'score_max': float(score.max()),
            'score_p95': float(np.percentile(score, 95)),
            'score_p99': float(np.percentile(score, 99)),
            'score_ema': float(ema),
            'latest': float(score[-1]),
            'severity_counts': severity_counts,
            'ood_count': self.ood_count,
            'adaptation_count': self.adaptation_count,
            'adaptation_mean_delta': mean_delta,
            'adaptation_total_improvement': total_improvement,
            'circuit_breaker_trips': len(self._circuit_breaker_trips),
        }

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            'n_samples': 0,
            'score_mean': 0.0, 'score_max': 0.0, 'score_p95': 0.0,
            'score_p99': 0.0, 'score_ema': 0.0, 'latest': 0.0,
            'severity_counts': {"nominal": 0, "warning": 0,
                                "critical": 0, "emergency": 0},
            'ood_count': 0, 'adaptation_count': 0,
            'adaptation_mean_delta': 0.0,
            'adaptation_total_improvement': 0.0,
            'circuit_breaker_trips': 0,
        }

    def reset(self) -> None:
        self._records.clear()
        self._ood_events.clear()
        self._adaptations.clear()
        self._circuit_breaker_trips.clear()
        self._tick = 0


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_score_timeline(
    cols: Dict[str, np.ndarray],
    palette: Palette = None,
    warning: float = 0.5,
    critical: float = 0.75,
    emergency: float = 0.9,
    height: int = 340,
    cb_trip_ticks: Optional[List[int]] = None,
) -> go.Figure:
    """
    Fused anomaly score over time with severity bands + circuit breaker markers.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if 'tick' not in cols or len(cols['tick']) == 0:
        apply_theme(fig, p, height=height)
        return fig

    x = cols['tick']
    y = cols['fused_score']

    # Severity zone shading
    fig.add_hrect(y0=warning, y1=critical, fillcolor="rgba(245, 158, 11, 0.07)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=critical, y1=emergency, fillcolor="rgba(239, 68, 68, 0.08)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=emergency, y1=1.0, fillcolor="rgba(239, 68, 68, 0.16)",
                  line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="Fused Score",
        line=dict(color=p.accent_primary, width=2, shape="spline", smoothing=0.5),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(p.accent_primary, 0.08),
        hovertemplate="tick %{x}<br>Fused: %{y:.4f}<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(y=warning, line=dict(color=p.warning, width=1, dash="dash"),
                  annotation_text=f"warn {warning:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(color=p.warning, size=9))
    fig.add_hline(y=critical, line=dict(color=p.danger, width=1, dash="dash"),
                  annotation_text=f"crit {critical:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(color=p.danger, size=9))

    # Circuit breaker trip markers
    if cb_trip_ticks:
        for t in cb_trip_ticks:
            fig.add_vline(x=t, line=dict(color=p.danger, width=2, dash="dot"))
            fig.add_annotation(
                x=t, y=1.0,
                text="⚡ CB",
                showarrow=False,
                font=dict(color=p.danger, size=11),
                yshift=8,
            )

    fig.update_layout(
        title="Anomaly Score Timeline",
        xaxis_title="Tick",
        yaxis_title="Fused Anomaly Score",
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_component_breakdown(
    cols: Dict[str, np.ndarray],
    palette: Palette = None,
    height: int = 280,
) -> go.Figure:
    """
    Stacked area plot of the three scoring components over time.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if 'tick' not in cols or len(cols['tick']) == 0:
        apply_theme(fig, p, height=height)
        return fig

    x = cols['tick']

    fig.add_trace(go.Scatter(
        x=x, y=cols['recon_component'], mode="lines",
        stackgroup="one", name="Reconstruction",
        line=dict(width=0.5),
        fillcolor=_hex_to_rgba(p.accent_primary, 0.55),
        hovertemplate="Recon: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cols['iso_component'], mode="lines",
        stackgroup="one", name="Isolation Forest",
        line=dict(width=0.5),
        fillcolor=_hex_to_rgba(p.warning, 0.55),
        hovertemplate="Isolation: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cols['maha_component'], mode="lines",
        stackgroup="one", name="Mahalanobis",
        line=dict(width=0.5),
        fillcolor=_hex_to_rgba(p.danger, 0.5),
        hovertemplate="Mahalanobis: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Score Component Contributions",
        xaxis_title="Tick",
        yaxis_title="Contribution",
    )
    apply_theme(fig, p, height=height)
    return fig


def build_score_histogram(
    score_values: np.ndarray,
    palette: Palette = None,
    warning: float = 0.5,
    critical: float = 0.75,
    emergency: float = 0.9,
    bins: int = 40,
    height: int = 280,
) -> go.Figure:
    """
    Distribution of anomaly scores with severity-colored bars.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if score_values.size == 0:
        apply_theme(fig, p, height=height)
        return fig

    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(score_values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0

    bar_colors = []
    for c in centers:
        if c >= emergency:
            bar_colors.append(p.danger)
        elif c >= critical:
            bar_colors.append(_hex_to_rgba(p.danger, 0.7))
        elif c >= warning:
            bar_colors.append(p.warning)
        else:
            bar_colors.append(p.success)

    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="Score %{x:.3f}<br>Count %{y}<extra></extra>",
    ))

    fig.add_vline(x=warning, line=dict(color=p.warning, width=1, dash="dot"))
    fig.add_vline(x=critical, line=dict(color=p.danger, width=1, dash="dot"))

    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Fused Score",
        yaxis_title="Count",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_ood_buffer_view(
    priorities: np.ndarray,
    palette: Palette = None,
    top_n: int = 20,
    height: int = 280,
) -> go.Figure:
    """
    Horizontal bar chart of the top-N OOD priorities.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if priorities.size == 0:
        apply_theme(fig, p, height=height)
        return fig

    top = np.sort(priorities)[-top_n:][::-1]
    labels = [f"#{i+1}" for i in range(len(top))]

    # Color by priority band
    threshold = 0.75
    bar_colors = [p.danger if v >= threshold else p.warning for v in top]

    fig.add_trace(go.Bar(
        y=labels, x=top, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="Priority: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Top {len(top)} OOD Priorities",
        xaxis_title="Priority",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_adaptation_history(
    adict: Dict[str, np.ndarray],
    palette: Palette = None,
    height: int = 280,
) -> go.Figure:
    """
    Bar chart of adaptation loss deltas (positive = improvement).
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if 'delta' not in adict or len(adict['delta']) == 0:
        apply_theme(fig, p, height=height)
        return fig

    deltas = adict['delta']
    ticks = adict['tick']
    colors = [p.success if d > 0 else p.danger for d in deltas]

    fig.add_trace(go.Bar(
        x=[f"t{int(t)}" for t in ticks],
        y=deltas,
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="Tick %{x}<br>Δloss %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0.0, line=dict(color=p.text_muted, width=1))

    fig.update_layout(
        title="Adaptation Loss Improvement per Step",
        xaxis_title="Step",
        yaxis_title="Δ Loss",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# STREAMLIT RENDER
# =============================================================================

_SEVERITY_STYLE = {
    'nominal':   ('active', '●'),
    'warning':   ('paused', '●'),
    'critical':  ('critical', '⚠'),
    'emergency': ('critical', '🚨'),
}


def render(
    st,
    palette: Palette,
    state: AnomalyConsoleState,
    key_prefix: str = "anomaly_console",
    show_demo_button: bool = True,
) -> None:
    """Render the anomaly console panel inside a Streamlit context."""
    st.markdown(section_header("🚨", "Anomaly & OOD Console"), unsafe_allow_html=True)

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        warn = st.number_input(
            "Warning threshold",
            min_value=0.05, max_value=1.0,
            value=float(state.warning_threshold), step=0.05,
            key=f"{key_prefix}_warn", format="%.2f",
        )
        state.warning_threshold = warn

    with ctrl2:
        crit = st.number_input(
            "Critical threshold",
            min_value=warn + 0.01, max_value=1.2,
            value=max(state.critical_threshold, warn + 0.05), step=0.05,
            key=f"{key_prefix}_crit", format="%.2f",
        )
        state.critical_threshold = crit

    with ctrl3:
        emg = st.number_input(
            "Emergency threshold",
            min_value=crit + 0.01, max_value=1.5,
            value=max(state.emergency_threshold, crit + 0.05), step=0.05,
            key=f"{key_prefix}_emg", format="%.2f",
        )
        state.emergency_threshold = emg

    with ctrl4:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key=f"{key_prefix}_reset", use_container_width=True):
            state.reset()
            st.toast("Anomaly console cleared.", icon="🧹")

    if show_demo_button and state.is_empty:
        if st.button("🌱 Seed demo anomaly data (300 ticks)",
                     key=f"{key_prefix}_seed"):
            state.seed_demo(300)
            st.rerun()

    if state.is_empty:
        st.info(
            "No anomaly data captured yet. Run the engine, or click "
            "**Seed demo anomaly data** above."
        )
        return

    # ---- Data ----
    cols = state.to_arrays()
    stats = state.stats()

    # ---- Top Metric Row ----
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        latest = state.latest()
        sev = latest.severity if latest else 'nominal'
        level, icon = _SEVERITY_STYLE.get(sev, _SEVERITY_STYLE['nominal'])
        accent = (
            palette.danger if sev in ('critical', 'emergency')
            else palette.warning if sev == 'warning'
            else palette.success
        )
        st.markdown(metric_card(
            "Latest Score",
            f"{stats['latest']:.4f}",
            f"severity: {sev}",
            accent=accent,
        ), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_card(
            "Score EMA",
            f"{stats['score_ema']:.4f}",
            f"p95 {stats['score_p95']:.3f}",
        ), unsafe_allow_html=True)

    with m3:
        st.markdown(metric_card(
            "OOD Events",
            f"{stats['ood_count']:,}",
            "quarantined samples",
            accent=palette.warning if stats['ood_count'] > 0 else palette.success,
        ), unsafe_allow_html=True)

    with m4:
        st.markdown(metric_card(
            "Adaptations",
            f"{stats['adaptation_count']:,}",
            f"mean Δ {stats['adaptation_mean_delta']:+.4f}",
            accent=palette.success if stats['adaptation_total_improvement'] > 0 else palette.text_muted,
        ), unsafe_allow_html=True)

    with m5:
        cb_trips = stats['circuit_breaker_trips']
        st.markdown(metric_card(
            "CB Trips",
            f"{cb_trips}",
            "circuit breaks",
            accent=palette.danger if cb_trips > 0 else palette.success,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Score Timeline ----
    fig_time = build_score_timeline(
        cols=cols, palette=palette,
        warning=state.warning_threshold,
        critical=state.critical_threshold,
        emergency=state.emergency_threshold,
        height=340,
        cb_trip_ticks=state._circuit_breaker_trips,
    )
    st.plotly_chart(fig_time, use_container_width=True, key=f"{key_prefix}_timeline")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Component breakdown | histogram ----
    left, right = st.columns([1.3, 1])

    with left:
        fig_comp = build_component_breakdown(cols, palette, height=280)
        st.plotly_chart(fig_comp, use_container_width=True, key=f"{key_prefix}_comp")

    with right:
        fig_hist = build_score_histogram(
            cols['fused_score'], palette,
            warning=state.warning_threshold,
            critical=state.critical_threshold,
            emergency=state.emergency_threshold,
            height=280,
        )
        st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- OOD buffer | Adaptation history ----
    left2, right2 = st.columns(2)

    with left2:
        fig_ood = build_ood_buffer_view(
            state.ood_priorities(), palette, top_n=20, height=280,
        )
        st.plotly_chart(fig_ood, use_container_width=True, key=f"{key_prefix}_ood")

    with right2:
        adict = state.adaptation_arrays()
        fig_adapt = build_adaptation_history(adict, palette, height=280)
        st.plotly_chart(fig_adapt, use_container_width=True, key=f"{key_prefix}_adapt")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Severity breakdown + summary ----
    sleft, sright = st.columns([1, 1])

    with sleft:
        st.markdown(section_header("📊", "Severity Breakdown"), unsafe_allow_html=True)

        total = max(stats['n_samples'], 1)
        breakdown = stats['severity_counts']

        for sev in ['nominal', 'warning', 'critical', 'emergency']:
            count = breakdown.get(sev, 0)
            frac = count / total
            color = {
                'nominal': palette.success,
                'warning': palette.warning,
                'critical': palette.danger,
                'emergency': palette.danger,
            }[sev]
            st.markdown(
                progress_bar(f"{sev} ({count})", frac, 1.0, color=color),
                unsafe_allow_html=True,
            )

    with sright:
        st.markdown(section_header("📋", "Rolling Summary"), unsafe_allow_html=True)

        st.markdown(kv_row("Samples", f"{stats['n_samples']:,}"), unsafe_allow_html=True)
        st.markdown(kv_row("Score mean", f"{stats['score_mean']:.4f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Score p95", f"{stats['score_p95']:.4f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Score p99", f"{stats['score_p99']:.4f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Score max", f"{stats['score_max']:.4f}"), unsafe_allow_html=True)

        if stats['adaptation_count'] > 0:
            st.markdown(
                kv_row("Total loss improvement",
                       f"{stats['adaptation_total_improvement']:+.4f}"),
                unsafe_allow_html=True,
            )

        # Health badge
        if stats['severity_counts'].get('emergency', 0) > 0:
            level, icon = _SEVERITY_STYLE['emergency']
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(status_badge("EMERGENCY DETECTED", level, icon),
                        unsafe_allow_html=True)
        elif stats['severity_counts'].get('critical', 0) > 0:
            level, icon = _SEVERITY_STYLE['critical']
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(status_badge("CRITICAL LEVEL", level, icon),
                        unsafe_allow_html=True)
        elif stats['severity_counts'].get('warning', 0) > 0:
            level, icon = _SEVERITY_STYLE['warning']
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(status_badge("WARNING LEVEL", level, icon),
                        unsafe_allow_html=True)
        else:
            level, icon = _SEVERITY_STYLE['nominal']
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(status_badge("NOMINAL", level, icon),
                        unsafe_allow_html=True)