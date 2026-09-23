"""
System Topology Visualizer Panel

Renders the entire NeuralStream-Core architecture as an interactive
node graph with:
    - Layered left-to-right flow: sources → engine → models → analysis
      → infrastructure → dashboard panels
    - Live status coloring per node (nominal / warning / critical / offline)
    - Category-coded node borders
    - Activity-driven node sizing (recently touched nodes glow)
    - Hoverable nodes with phase origin + metric summary
    - Click-through inspector showing per-node metrics
    - Activity timeline: rolling histogram of events per category

The panel is fully driven by TopologyState, which is Streamlit-agnostic
and testable in isolation.
"""

import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Set, Iterable
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import (
    metric_card, section_header, divider, status_badge,
    apply_theme, accent_color, progress_bar, kv_row, alert_box,
)
from dashboard.theme.palette import Palette


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class TopologyNode:
    """A single node in the system topology graph."""
    id: str
    label: str
    category: str           # 'source' | 'engine' | 'model' | 'analysis' | 'infra' | 'panel'
    x: float = 0.0
    y: float = 0.0
    description: str = ""
    phase: str = ""
    status: str = "nominal"  # nominal | warning | critical | offline
    metrics: Dict[str, str] = field(default_factory=dict)
    importance: float = 1.0   # 0.5 lazy, 1.0 normal, 1.5 primary

    def to_dict(self) -> dict:
        return {
            'id': self.id, 'label': self.label, 'category': self.category,
            'x': self.x, 'y': self.y, 'status': self.status,
            'description': self.description, 'phase': self.phase,
            'metrics': dict(self.metrics), 'importance': self.importance,
        }


@dataclass
class TopologyEdge:
    """A directed data-flow edge between two nodes."""
    src: str
    dst: str
    label: str = ""
    dashed: bool = False      # use dashed for feedback loops

    def to_dict(self) -> dict:
        return {'src': self.src, 'dst': self.dst,
                'label': self.label, 'dashed': self.dashed}


# =============================================================================
# STATE
# =============================================================================

class TopologyState:
    """
    Manages the topology graph + live activity tracking.

    Args:
        activity_window_s: Seconds of history retained for activity rate
        max_activity_events: Backing event buffer size
        activity_bins: Number of bins in the activity timeline
    """

    STATUS_LEVELS = ('nominal', 'warning', 'critical', 'offline')

    def __init__(
        self,
        activity_window_s: float = 30.0,
        max_activity_events: int = 4000,
        activity_bins: int = 60,
    ):
        self.activity_window_s = activity_window_s
        self.activity_bins = activity_bins

        self.nodes: Dict[str, TopologyNode] = {}
        self.edges: List[TopologyEdge] = []

        self._activity_events: deque = deque(maxlen=max_activity_events)
        self._category_totals: Counter = Counter()

    # ---- Graph construction ----

    def add_node(self, node: TopologyNode) -> None:
        """Add or replace a node."""
        self.nodes[node.id] = node

    def add_edge(self, edge: TopologyEdge) -> None:
        """Add a data-flow edge. Silently ignores duplicates."""
        for e in self.edges:
            if e.src == edge.src and e.dst == edge.dst:
                return
        self.edges.append(edge)

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges
                      if e.src != node_id and e.dst != node_id]

    # ---- Status & metrics ----

    def update_status(self, node_id: str, status: str) -> None:
        if node_id in self.nodes and status in self.STATUS_LEVELS:
            self.nodes[node_id].status = status

    def update_metric(self, node_id: str, key: str, value: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].metrics[key] = str(value)

    def set_all_status(self, status: str) -> None:
        for n in self.nodes.values():
            n.status = status

    # ---- Activity tracking ----

    def mark_active(self, node_id: str, count: int = 1) -> None:
        """Record a burst of activity for the given node."""
        if node_id not in self.nodes:
            return
        now = time.time()
        for _ in range(max(1, count)):
            self._activity_events.append((now, node_id))

    def _prune_activity(self) -> None:
        cutoff = time.time() - self.activity_window_s
        while self._activity_events and self._activity_events[0][0] < cutoff:
            self._activity_events.popleft()

    def activity_rate(self, node_id: str, window_s: Optional[float] = None) -> float:
        """Return events per second for a node over the recent window."""
        self._prune_activity()
        cutoff = time.time() - (window_s or self.activity_window_s)
        count = sum(1 for t, nid in self._activity_events
                    if nid == node_id and t >= cutoff)
        return count / max(window_s or self.activity_window_s, 1e-6)

    def active_nodes(self, min_rate: float = 0.05) -> List[str]:
        """Return node IDs with recent activity above the threshold."""
        self._prune_activity()
        counts: Counter = Counter()
        for _, nid in self._activity_events:
            counts[nid] += 1
        window = max(self.activity_window_s, 1e-6)
        return [nid for nid, c in counts.items() if (c / window) >= min_rate]

    def category_activity(self) -> Dict[str, float]:
        """Total activity rate per category."""
        self._prune_activity()
        by_cat: Counter = Counter()
        for _, nid in self._activity_events:
            node = self.nodes.get(nid)
            if node is not None:
                by_cat[node.category] += 1
        window = max(self.activity_window_s, 1e-6)
        return {c: by_cat.get(c, 0) / window
                for c in self.categories()}

    def activity_histogram(self, bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (counts, bin_edges) showing event density over the recent window.
        """
        self._prune_activity()
        bins = bins or self.activity_bins
        now = time.time()
        edges = np.linspace(now - self.activity_window_s, now, bins + 1)
        times = np.array([t for t, _ in self._activity_events], dtype=np.float64)
        if times.size == 0:
            return np.zeros(bins, dtype=np.int64), edges
        counts, _ = np.histogram(times, bins=edges)
        return counts.astype(np.int64), edges

    # ---- Query helpers ----

    def categories(self) -> List[str]:
        return sorted({n.category for n in self.nodes.values()})

    def nodes_by_category(self) -> Dict[str, List[TopologyNode]]:
        out: Dict[str, List[TopologyNode]] = defaultdict(list)
        for n in self.nodes.values():
            out[n.category].append(n)
        for k in out:
            out[k].sort(key=lambda x: x.id)
        return dict(out)

    def status_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in self.STATUS_LEVELS}
        for n in self.nodes.values():
            counts[n.status] = counts.get(n.status, 0) + 1
        return counts

    def overall_health(self) -> str:
        counts = self.status_counts()
        if counts.get('critical', 0) > 0:
            return 'critical'
        if counts.get('warning', 0) > 0:
            return 'warning'
        if counts.get('offline', 0) > 0:
            return 'warning'
        return 'nominal'

    def to_dataframe_like(self) -> List[Dict[str, Any]]:
        rows = []
        for n in self.nodes.values():
            rows.append({
                'ID': n.id,
                'Label': n.label,
                'Category': n.category,
                'Status': n.status,
                'Phase': n.phase,
                'Activity (Hz)': round(self.activity_rate(n.id), 3),
            })
        return rows

    # ---- Lifecycle ----

    def reset_activity(self) -> None:
        self._activity_events.clear()

    def reset(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self._activity_events.clear()


# =============================================================================
# DEFAULT TOPOLOGY
# =============================================================================

def seed_default_topology(state: TopologyState) -> None:
    """
    Populate the state with the complete NeuralStream-Core architecture.

    Layout is a 6-tier left-to-right flow:
        Tier 0 (x=-4): Sources
        Tier 1 (x=-2): Engine
        Tier 2 (x= 0): Models
        Tier 3 (x= 2): Analysis
        Tier 4 (x= 4): Infrastructure
        Tier 5 (x= 6): Dashboard panels
    """
    state.reset()

    # ---- Tier 0: Sources ----
    sources = [
        TopologyNode("src_synthetic", "Synthetic", "source", -4.0, -3.0,
                     "Gaussian noise generator", "P10"),
        TopologyNode("src_files", "File Loaders", "source", -4.0, -1.5,
                     ".npy / .pt tensor loaders", "P10"),
        TopologyNode("src_images", "Image Array", "source", -4.0, 0.0,
                     "PIL image stream", "P10"),
        TopologyNode("src_audio", "Audio / Signals", "source", -4.0, 1.5,
                     "Waveform + spectrogram", "P10"),
        TopologyNode("src_csv", "CSV / Time-Series", "source", -4.0, 3.0,
                     "Tabular streaming loader", "P10"),
    ]
    for n in sources:
        state.add_node(n)

    # ---- Tier 1: Engine ----
    engine = [
        TopologyNode("engine_bootstrap", "Bootstrap", "engine", -2.0, -1.0,
                     "Model instantiation + LSUV calib", "P14"),
        TopologyNode("engine_learner", "Learner Loop", "engine", -2.0, 1.0,
                     "Main iteration driver", "P14", importance=1.5),
    ]
    for n in engine:
        state.add_node(n)

    # ---- Tier 2: Models ----
    models = [
        TopologyNode("model_edge", "EdgeStreamingAE", "model", 0.0, -3.0,
                     "Fast conv autoencoder (32-D latent)", "P3", importance=1.5),
        TopologyNode("model_vae", "ConvVAE", "model", 0.0, -1.5,
                     "Variational with KL loss", "P3"),
        TopologyNode("model_unet", "U-Net AE", "model", 0.0, 0.0,
                     "Multi-scale skip connections", "P4", importance=1.2),
        TopologyNode("model_transformer", "Transformer AE", "model", 0.0, 1.5,
                     "Patch-based attention encoder", "P3"),
        TopologyNode("model_ensemble", "Ensemble", "model", 0.0, 3.0,
                     "Weighted fusion of 3 architectures", "P4"),
    ]
    for n in models:
        state.add_node(n)

    # ---- Tier 3: Analysis ----
    analysis = [
        TopologyNode("analysis_drift", "Drift Radar", "analysis", 2.0, -3.0,
                     "Wasserstein-1 across layers", "P5"),
        TopologyNode("analysis_precision", "Precision Engine", "analysis", 2.0, -1.5,
                     "FP32→FP16→INT8 dynamic scaling", "P7"),
        TopologyNode("analysis_anomaly", "Anomaly Scorer", "analysis", 2.0, 0.0,
                     "Recon + IF + Mahalanobis fusion", "P9"),
        TopologyNode("analysis_adapt", "Adaptation Engine", "analysis", 2.0, 1.5,
                     "EWC-protected micro-tuning", "P8"),
        TopologyNode("analysis_breaker", "Circuit Breaker", "analysis", 2.0, 3.0,
                     "NaN/Inf quarantine + recovery", "P14"),
    ]
    for n in analysis:
        state.add_node(n)

    # ---- Tier 4: Infrastructure ----
    infra = [
        TopologyNode("infra_ipc", "IPC Bus", "infra", 4.0, -2.0,
                     "Zero-copy shared memory channels", "P6", importance=1.3),
        TopologyNode("infra_workers", "Worker Pool", "infra", 4.0, -0.5,
                     "Priority queue + retry + timeout", "P11"),
        TopologyNode("infra_metrics", "Metrics Registry", "infra", 4.0, 1.0,
                     "Counter/Gauge/Histogram tracking", "P12"),
        TopologyNode("infra_events", "Event Bus", "infra", 4.0, 2.5,
                     "Wildcard pub/sub with priority", "P13"),
    ]
    for n in infra:
        state.add_node(n)

    # ---- Tier 5: Dashboard panels ----
    panels = [
        TopologyNode("panel_latent", "Latent Explorer", "panel", 6.0, -4.5,
                     "3D PCA trajectory", "P15"),
        TopologyNode("panel_drift", "Drift Heatmap", "panel", 6.0, -3.0,
                     "Layer × tick severity matrix", "P16"),
        TopologyNode("panel_telemetry", "Hardware Telemetry", "panel", 6.0, -1.5,
                     "CPU / RAM / latency gauges", "P17"),
        TopologyNode("panel_anomaly", "Anomaly Console", "panel", 6.0, 0.0,
                     "Score timeline + OOD buffer", "P18"),
        TopologyNode("panel_compare", "Model Comparison", "panel", 6.0, 1.5,
                     "A/B radar + leaderboards", "P19"),
        TopologyNode("panel_osc", "Oscilloscope", "panel", 6.0, 3.0,
                     "Waveform + FFT + waterfall", "P20"),
        TopologyNode("panel_topology", "Topology View", "panel", 6.0, 4.5,
                     "This panel (self-referential)", "P21"),
    ]
    for n in panels:
        state.add_node(n)

    # ---- Edges: sources → bootstrap ----
    for src in ["src_synthetic", "src_files", "src_images", "src_audio", "src_csv"]:
        state.add_edge(TopologyEdge(src, "engine_bootstrap", "stream"))

    # ---- bootstrap → learner ----
    state.add_edge(TopologyEdge("engine_bootstrap", "engine_learner", "ready"))

    # ---- learner → models ----
    for m in ["model_edge", "model_vae", "model_unet",
              "model_transformer", "model_ensemble"]:
        state.add_edge(TopologyEdge("engine_learner", m, "batch"))
        state.add_edge(TopologyEdge(m, "analysis_drift", "activations"))
        state.add_edge(TopologyEdge(m, "analysis_precision", "latency"))
        state.add_edge(TopologyEdge(m, "analysis_anomaly", "recon"))
        state.add_edge(TopologyEdge(m, "analysis_breaker", "output"))

    # ---- Analysis interactions ----
    state.add_edge(TopologyEdge("analysis_anomaly", "analysis_adapt", "OOD samples"))
    state.add_edge(TopologyEdge("analysis_adapt", "model_edge",
                                "weight update", dashed=True))
    state.add_edge(TopologyEdge("analysis_breaker", "engine_learner",
                                "recovery", dashed=True))

    # ---- Analysis → IPC ----
    for a in ["analysis_drift", "analysis_precision",
              "analysis_anomaly", "analysis_adapt"]:
        state.add_edge(TopologyEdge(a, "infra_ipc", "channel write"))

    # ---- Infra cross-links ----
    state.add_edge(TopologyEdge("infra_ipc", "infra_metrics", "sample"))
    state.add_edge(TopologyEdge("analysis_drift", "infra_events", "alert"))
    state.add_edge(TopologyEdge("analysis_precision", "infra_events", "transition"))
    state.add_edge(TopologyEdge("analysis_adapt", "infra_events", "step"))
    state.add_edge(TopologyEdge("analysis_breaker", "infra_events", "trip"))
    state.add_edge(TopologyEdge("infra_ipc", "infra_workers", "offload"))

    # ---- Infra → panels ----
    for p in ["panel_latent", "panel_drift", "panel_telemetry",
              "panel_anomaly", "panel_compare", "panel_osc", "panel_topology"]:
        state.add_edge(TopologyEdge("infra_ipc", p, "read"))
        state.add_edge(TopologyEdge("infra_events", p, "subscribe"))

    # ---- Metric source for telemetry ----
    state.add_edge(TopologyEdge("infra_metrics", "panel_telemetry", "scrape"))


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

_CATEGORY_COLORS = {
    'source':   lambda p: p.info,
    'engine':   lambda p: p.accent_secondary,
    'model':    lambda p: p.accent_primary,
    'analysis': lambda p: p.warning,
    'infra':    lambda p: p.success,
    'panel':    lambda p: p.danger,
}

_STATUS_COLORS = {
    'nominal':  lambda p: p.success,
    'warning':  lambda p: p.warning,
    'critical': lambda p: p.danger,
    'offline':  lambda p: p.text_muted,
}


def _node_display_color(node: TopologyNode, palette: Palette) -> str:
    """Fill color = category; outline = status."""
    cat_fn = _CATEGORY_COLORS.get(node.category, lambda p: p.accent_primary)
    return cat_fn(palette)


def _node_outline_color(node: TopologyNode, palette: Palette) -> str:
    status_fn = _STATUS_COLORS.get(node.status, lambda p: p.success)
    return status_fn(palette)


def build_topology_figure(
    state: TopologyState,
    palette: Palette = None,
    height: int = 620,
    show_active_glow: bool = True,
) -> go.Figure:
    """
    Render the full topology node graph with edges and status outlines.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not state.nodes:
        apply_theme(fig, p, height=height)
        return fig

    # ---- Edges as line segments ----
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_colors: List[str] = []
    edge_hover: List[str] = []

    active = set(state.active_nodes(min_rate=0.02))

    for e in state.edges:
        src = state.nodes.get(e.src)
        dst = state.nodes.get(e.dst)
        if src is None or dst is None:
            continue

        # Edge highlight if source is active
        if e.src in active:
            edge_color = _hex_to_rgba(p.accent_primary, 0.55)
        elif src.status == 'critical':
            edge_color = _hex_to_rgba(p.danger, 0.5)
        elif src.status == 'warning':
            edge_color = _hex_to_rgba(p.warning, 0.35)
        else:
            edge_color = _hex_to_rgba(p.text_muted, 0.18)

        edge_x += [src.x, dst.x, None]
        edge_y += [src.y, dst.y, None]
        edge_colors.append(edge_color)
        edge_hover.append(f"{e.src} → {e.dst}")

    if edge_x:
        # Edges are drawn as one big trace with per-point colors
        # Plotly doesn't support per-segment colors in a single trace,
        # so we use a middle gray and rely on node colors for emphasis.
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color=_hex_to_rgba(p.text_muted, 0.25), width=1.2),
            hoverinfo="skip",
            showlegend=False,
            name="Data flow",
        ))

        # Overlay highlighted edges (from active sources)
        hi_x: List[Optional[float]] = []
        hi_y: List[Optional[float]] = []
        for e in state.edges:
            if e.src not in active:
                continue
            src = state.nodes.get(e.src)
            dst = state.nodes.get(e.dst)
            if src is None or dst is None:
                continue
            hi_x += [src.x, dst.x, None]
            hi_y += [src.y, dst.y, None]
        if hi_x:
            fig.add_trace(go.Scatter(
                x=hi_x, y=hi_y,
                mode="lines",
                line=dict(color=_hex_to_rgba(p.accent_primary, 0.55), width=2.2),
                hoverinfo="skip",
                showlegend=False,
                name="Active flow",
            ))

    # ---- Node glow for active nodes ----
    if show_active_glow and active:
        gx, gy, gsizes, gcolors = [], [], [], []
        for nid in active:
            n = state.nodes.get(nid)
            if n is None:
                continue
            gx.append(n.x)
            gy.append(n.y)
            gsizes.append(38 + 12 * n.importance)
            gcolors.append(_node_display_color(n, p))
        fig.add_trace(go.Scatter(
            x=gx, y=gy,
            mode="markers",
            marker=dict(
                size=gsizes,
                color=gcolors,
                opacity=0.18,
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Active glow",
        ))

    # ---- Nodes ----
    node_x, node_y, node_fill, node_outline, node_ovr, node_size = [], [], [], [], [], []
    hover_texts = []

    for n in state.nodes.values():
        node_x.append(n.x)
        node_y.append(n.y)
        node_fill.append(_node_display_color(n, p))

        # Outline by status
        status_fn = _STATUS_COLORS.get(n.status, lambda p: p.success)
        node_outline.append(status_fn(p))

        # Size by importance
        node_ovr.append(20 + 8 * n.importance)
        node_size.append(n.importance)

        # Hover text
        metrics_str = " · ".join(f"{k}={v}" for k, v in n.metrics.items())
        hover_texts.append(
            f"<b>{n.label}</b><br>"
            f"<span style='color:{p.text_muted};'>{n.id} · {n.category} · {n.phase}</span><br>"
            f"<span style='color:{p.text_secondary};'>{n.description}</span><br>"
            f"<span style='color:{_node_outline_color(n, p)};'>status: {n.status}</span>"
            + (f"<br>{metrics_str}" if metrics_str else "")
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_ovr,
            color=node_fill,
            line=dict(color=node_outline, width=2.5),
            symbol="circle",
        ),
        text=[n.label for n in state.nodes.values()],
        textposition="bottom center",
        textfont=dict(
            color=p.text_secondary,
            family="JetBrains Mono, monospace",
            size=10,
        ),
        hovertext=hover_texts,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
        name="Nodes",
    ))

    # ---- Layout ----
    fig.update_xaxes(
        showgrid=True, gridcolor=p.grid_line,
        zeroline=False, showticklabels=False,
        range=[-5.5, 7.5],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=p.grid_line,
        zeroline=False, showticklabels=False,
        range=[-6.0, 6.0],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        title="NeuralStream-Core · System Topology",
        hovermode="closest",
    )
    apply_theme(fig, p, height=height)
    return fig


def build_activity_timeline(
    state: TopologyState,
    palette: Palette = None,
    height: int = 220,
) -> go.Figure:
    """Histogram of recent activity events over time."""
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    counts, edges = state.activity_histogram()
    if counts.size == 0 or counts.sum() == 0:
        apply_theme(fig, p, height=height)
        return fig

    centers = (edges[:-1] + edges[1:]) / 2.0
    # Convert absolute times → relative seconds-ago
    now = time.time()
    x_ago = (centers - now).tolist()

    fig.add_trace(go.Bar(
        x=x_ago, y=counts,
        marker=dict(color=_hex_to_rgba(p.accent_primary, 0.75),
                    line=dict(width=0)),
        hovertemplate="%{x:.1f} s ago<br>events: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Activity Timeline (last {state.activity_window_s:.0f}s)",
        xaxis_title="Seconds ago",
        yaxis_title="Events",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_category_activity_bar(
    state: TopologyState,
    palette: Palette = None,
    height: int = 240,
) -> go.Figure:
    """Horizontal bar of activity rate per category."""
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    cat_rates = state.category_activity()
    if not cat_rates:
        apply_theme(fig, p, height=height)
        return fig

    cats = list(cat_rates.keys())
    rates = [cat_rates[c] for c in cats]
    colors = [
        _CATEGORY_COLORS.get(c, lambda p: p.accent_primary)(p)
        for c in cats
    ]

    # Sort descending
    pairs = sorted(zip(cats, rates, colors), key=lambda x: x[1], reverse=True)
    cats = [c for c, _, _ in pairs]
    rates = [r for _, r, _ in pairs]
    colors = [col for _, _, col in pairs]

    fig.add_trace(go.Bar(
        y=cats, x=rates, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{r:.2f} Hz" for r in rates],
        textposition="outside",
        textfont=dict(color=p.text_secondary,
                      family="JetBrains Mono, monospace", size=10),
        hovertemplate="%{y}<br>%{x:.3f} Hz<extra></extra>",
    ))

    fig.update_layout(
        title="Activity by Category",
        xaxis_title="Events per second",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    fig.update_yaxes(autorange="reversed")
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

_STATUS_STYLE = {
    'nominal':  ('active', '●'),
    'warning':  ('paused', '●'),
    'critical': ('critical', '⚠'),
    'offline':  ('neutral', '○'),
}


def render(
    st,
    palette: Palette,
    state: TopologyState,
    key_prefix: str = "topology",
    show_demo_button: bool = True,
) -> None:
    """Render the topology visualizer inside a Streamlit context."""
    st.markdown(section_header("🗺️", "System Topology · Data Flow"), unsafe_allow_html=True)

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        height = st.slider(
            "Graph height",
            min_value=380, max_value=900, value=620, step=40,
            key=f"{key_prefix}_height",
        )

    with ctrl2:
        show_glow = st.checkbox(
            "Active node glow",
            value=True,
            key=f"{key_prefix}_glow",
        )

    with ctrl3:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset activity", key=f"{key_prefix}_reset_act",
                     use_container_width=True):
            state.reset_activity()
            st.toast("Activity buffer cleared.", icon="🧹")

    with ctrl4:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset graph", key=f"{key_prefix}_reset_all",
                     use_container_width=True):
            state.reset()
            st.toast("Topology cleared.", icon="🧹")

    # ---- Empty state ----
    if not state.nodes:
        st.info(
            "Topology graph is empty. Click **Load NeuralStream topology** "
            "below to populate the entire architecture."
        )
        if st.button("🌐 Load NeuralStream topology",
                     key=f"{key_prefix}_seed_default"):
            seed_default_topology(state)
            st.rerun()
        return

    # ---- Demo activity simulation ----
    if show_demo_button:
        d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
        with d1:
            if st.button("🎬 Simulate 30s of activity",
                         key=f"{key_prefix}_sim_act",
                         use_container_width=True):
                _simulate_activity(state, duration_s=30.0)
                st.rerun()
        with d2:
            if st.button("🎨 Randomize statuses",
                         key=f"{key_prefix}_random_status",
                         use_container_width=True):
                _randomize_statuses(state)
                st.rerun()
        with d3:
            if st.button("✅ All nominal",
                         key=f"{key_prefix}_all_nominal",
                         use_container_width=True):
                state.set_all_status("nominal")
                st.rerun()
        with d4:
            if st.button("🌐 Reload topology",
                         key=f"{key_prefix}_reload",
                         use_container_width=True):
                seed_default_topology(state)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Metric row ----
    total_nodes = len(state.nodes)
    total_edges = len(state.edges)
    active = len(state.active_nodes(min_rate=0.02))
    statuses = state.status_counts()
    overall = state.overall_health()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(metric_card(
            "Nodes",
            f"{total_nodes}",
            f"{total_edges} edges",
        ), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_card(
            "Active Nodes",
            f"{active}",
            f"of {total_nodes} regs",
            accent=palette.accent_primary,
        ), unsafe_allow_html=True)

    with m3:
        degraded = statuses.get('warning', 0) + statuses.get('critical', 0)
        accent = (palette.danger if statuses.get('critical', 0) > 0
                  else palette.warning if degraded > 0
                  else palette.success)
        st.markdown(metric_card(
            "Degraded",
            f"{degraded}",
            f"{statuses.get('critical', 0)} critical",
            accent=accent,
        ), unsafe_allow_html=True)

    with m4:
        cat_counts = {c: len(v) for c, v in state.nodes_by_category().items()}
        st.markdown(metric_card(
            "Categories",
            f"{len(cat_counts)}",
            " · ".join(f"{c}:{n}" for c, n in sorted(cat_counts.items())),
        ), unsafe_allow_html=True)

    with m5:
        level, icon = _STATUS_STYLE.get(overall, _STATUS_STYLE['nominal'])
        st.markdown(f"""
        <div class="hud-card hud-fade-in" style="height:100%; display:flex;
                    flex-direction:column; justify-content:center; align-items:center;">
            <div class="hud-card-title" style="margin-bottom:12px;">Overall Health</div>
            {status_badge(overall.upper(), level, icon)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Main graph ----
    fig_graph = build_topology_figure(
        state=state,
        palette=palette,
        height=height,
        show_active_glow=show_glow,
    )
    st.plotly_chart(fig_graph, use_container_width=True, key=f"{key_prefix}_graph")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Timeline + Category summary ----
    left, right = st.columns([1.4, 1])

    with left:
        fig_time = build_activity_timeline(state, palette, height=240)
        st.plotly_chart(fig_time, use_container_width=True, key=f"{key_prefix}_timeline")

    with right:
        fig_cat = build_category_activity_bar(state, palette, height=240)
        st.plotly_chart(fig_cat, use_container_width=True, key=f"{key_prefix}_cat")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Node inspector ----
    st.markdown(section_header("🔍", "Node Inspector"), unsafe_allow_html=True)

    node_ids = sorted(state.nodes.keys())
    selected_id = st.selectbox(
        "Select a node",
        options=node_ids,
        index=0,
        key=f"{key_prefix}_select_node",
        format_func=lambda nid: f"{nid} · {state.nodes[nid].label}",
    )

    selected = state.nodes.get(selected_id)
    if selected is not None:
        # Left column: metadata
        insp_l, insp_r = st.columns([2, 1])

        with insp_l:
            st.markdown(f"""
            <div class="hud-card hud-fade-in" style="padding:18px 22px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-family:var(--mono); font-size:1.1rem;
                                    font-weight:700; color:var(--text-primary);">
                            {selected.label}
                        </div>
                        <div style="font-family:var(--mono); font-size:0.75rem;
                                    color:var(--text-muted); margin-top:2px;">
                            id={selected.id} · category={selected.category}
                            · phase={selected.phase}
                        </div>
                    </div>
                    {status_badge(selected.status.upper(),
                                  *_STATUS_STYLE.get(selected.status, _STATUS_STYLE['nominal']))}
                </div>
                <div style="margin-top:14px; color:var(--text-secondary);
                            font-size:0.9rem; line-height:1.55;">
                    {selected.description}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics table
            if selected.metrics:
                st.markdown("<br>", unsafe_allow_html=True)
                for k, v in selected.metrics.items():
                    st.markdown(kv_row(k, v), unsafe_allow_html=True)
            else:
                st.caption("No per-node metrics registered yet.")

        with insp_r:
            rate = state.activity_rate(selected.id)
            st.markdown(metric_card(
                "Activity Rate",
                f"{rate:.3f} Hz",
                "recent events / sec",
                accent=palette.accent_primary if rate > 0 else palette.text_muted,
            ), unsafe_allow_html=True)

            # Outgoing edges count
            out_edges = [e for e in state.edges if e.src == selected.id]
            in_edges = [e for e in state.edges if e.dst == selected.id]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(metric_card(
                "Connections",
                f"{len(in_edges)} → {len(out_edges)}",
                "incoming → outgoing",
            ), unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Full node table ----
    st.markdown(section_header("📋", "All Nodes"), unsafe_allow_html=True)
    st.dataframe(
        state.to_dataframe_like(),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# DEMO HELPERS
# =============================================================================

def _simulate_activity(state: TopologyState, duration_s: float = 30.0) -> None:
    """
    Distribute synthetic activity events across the past `duration_s` seconds.
    Emphasizes hot-path nodes (learner, models, drift, IPC).
    """
    rng = np.random.default_rng()
    now = time.time()

    # Hot path nodes with weighted frequencies
    weights = {
        "engine_learner":     20.0,
        "model_edge":         18.0,
        "model_vae":          10.0,
        "model_unet":         8.0,
        "model_transformer":  6.0,
        "model_ensemble":     4.0,
        "analysis_drift":     8.0,
        "analysis_precision": 6.0,
        "analysis_anomaly":   7.0,
        "analysis_adapt":     2.0,
        "infra_ipc":          15.0,
        "infra_metrics":      6.0,
        "infra_events":       5.0,
        "infra_workers":      4.0,
        "panel_latent":       3.0,
        "panel_drift":        3.0,
        "panel_telemetry":    4.0,
        "panel_anomaly":      3.0,
        "panel_compare":      1.0,
        "panel_osc":          2.0,
        "panel_topology":     1.5,
    }

    # Filter to actual nodes present in this graph
    weights = {k: v for k, v in weights.items() if k in state.nodes}
    if not weights:
        return

    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()

    # Generate ~500 events spread over the window
    n_events = 500
    event_times_ago = np.sort(rng.uniform(0, duration_s, n_events))

    # Overwrite the internal event buffer
    state._activity_events.clear()
    for t_ago, nid in zip(event_times_ago,
                          rng.choice(keys, size=n_events, p=probs)):
        state._activity_events.append((now - t_ago, str(nid)))

    # Poke a couple of statuses to make the demo visually rich
    for nid, status in [
        ("analysis_adapt", "warning"),
        ("model_transformer", "warning"),
    ]:
        state.update_status(nid, status)


def _randomize_statuses(state: TopologyState) -> None:
    """Random statuses — mostly nominal with a few warnings/critical."""
    rng = np.random.default_rng()
    for nid, node in state.nodes.items():
        r = rng.random()
        if r < 0.06:
            node.status = "critical"
        elif r < 0.18:
            node.status = "warning"
        elif r < 0.20:
            node.status = "offline"
        else:
            node.status = "nominal"