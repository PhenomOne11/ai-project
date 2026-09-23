"""
Model Comparison & A/B Workspace Panel

Side-by-side comparison of multiple model architectures:
    - Parameter count & memory footprint bar charts
    - Latency distribution (box plot) + percentile breakdown
    - Reconstruction error comparison
    - Multi-axis radar chart for trade-off analysis
    - Automatic "winner" suggestion per metric
    - Trained-model profiles can be imported from Phase 3's ModelRegistry

Accepts data from:
    - Direct API: `add_model(...)` then `record_latency(...)` / `record_recon(...)`
    - Phase 3 integration: `import_from_registry(ModelRegistry)`
    - Synthetic: `seed_demo(...)` for demonstration
"""

import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from collections import deque, defaultdict
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
class ModelProfile:
    """
    A single model's profile in the comparison workspace.

    Latency and reconstruction values are tracked as rolling samples,
    supporting per-model distribution analysis.
    """
    name: str
    architecture: str = "unknown"
    parameters: int = 0
    memory_mb: float = 0.0
    latent_dim: int = 32
    throughput_fps: float = 0.0           # Optional, if known

    # Rolling samples
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=200))
    recon_samples: deque = field(default_factory=lambda: deque(maxlen=200))

    # Meta
    color_idx: int = 0
    enabled: bool = True
    notes: str = ""

    def latency_stats(self) -> Dict[str, float]:
        if not self.latency_samples:
            return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0,
                    'min': 0.0, 'max': 0.0, 'std': 0.0, 'n': 0}
        arr = np.array(self.latency_samples, dtype=np.float64)
        return {
            'mean': float(arr.mean()),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'std': float(arr.std()),
            'n': int(arr.size),
        }

    def recon_stats(self) -> Dict[str, float]:
        if not self.recon_samples:
            return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0,
                    'min': 0.0, 'max': 0.0, 'std': 0.0, 'n': 0}
        arr = np.array(self.recon_samples, dtype=np.float64)
        return {
            'mean': float(arr.mean()),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'std': float(arr.std()),
            'n': int(arr.size),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'architecture': self.architecture,
            'parameters': self.parameters,
            'memory_mb': self.memory_mb,
            'latent_dim': self.latent_dim,
            'throughput_fps': self.throughput_fps,
            'latency': self.latency_stats(),
            'recon': self.recon_stats(),
            'enabled': self.enabled,
        }


# =============================================================================
# STATE
# =============================================================================

class ModelComparisonState:
    """
    Workspace for comparing multiple model profiles.

    Args:
        latency_window: Max latency samples retained per model
        recon_window: Max recon-error samples retained per model
    """

    def __init__(
        self,
        latency_window: int = 200,
        recon_window: int = 200,
    ):
        self.latency_window = latency_window
        self.recon_window = recon_window

        self._models: Dict[str, ModelProfile] = {}
        self._color_counter: int = 0

    # ---- Setup ----

    def add_model(
        self,
        name: str,
        architecture: str = "unknown",
        parameters: int = 0,
        memory_mb: float = 0.0,
        latent_dim: int = 32,
        throughput_fps: float = 0.0,
        notes: str = "",
    ) -> ModelProfile:
        """Add or replace a model profile."""
        if name in self._models:
            # Preserve existing samples if replacing metadata
            existing = self._models[name]
            existing.architecture = architecture
            existing.parameters = parameters
            existing.memory_mb = memory_mb
            existing.latent_dim = latent_dim
            existing.throughput_fps = throughput_fps
            existing.notes = notes
            return existing

        profile = ModelProfile(
            name=name,
            architecture=architecture,
            parameters=int(parameters),
            memory_mb=float(memory_mb),
            latent_dim=int(latent_dim),
            throughput_fps=float(throughput_fps),
            notes=notes,
            color_idx=self._color_counter,
            latency_samples=deque(maxlen=self.latency_window),
            recon_samples=deque(maxlen=self.recon_window),
        )
        self._models[name] = profile
        self._color_counter += 1
        return profile

    def remove_model(self, name: str) -> bool:
        if name in self._models:
            del self._models[name]
            return True
        return False

    def enable_model(self, name: str, enabled: bool = True) -> None:
        if name in self._models:
            self._models[name].enabled = bool(enabled)

    # ---- Ingestion ----

    def record_latency(self, name: str, latency_ms: float) -> None:
        if name not in self._models:
            self.add_model(name)
        self._models[name].latency_samples.append(float(latency_ms))

    def record_recon(self, name: str, recon_error: float) -> None:
        if name not in self._models:
            self.add_model(name)
        self._models[name].recon_samples.append(float(recon_error))

    def record_batch(
        self,
        name: str,
        latency_samples: List[float],
        recon_samples: Optional[List[float]] = None,
    ) -> None:
        if name not in self._models:
            self.add_model(name)
        m = self._models[name]
        for v in latency_samples:
            m.latency_samples.append(float(v))
        if recon_samples is not None:
            for v in recon_samples:
                m.recon_samples.append(float(v))

    # ---- Phase 3 Integration ----

    def import_from_registry(self, registry, **kwargs) -> int:
        """
        Populate profiles from a Phase-3 ModelRegistry benchmark result
        (as produced by `ModelRegistry.compare_architectures(...)`).

        Args:
            registry: A ModelRegistry instance
            kwargs: Forwarded to `compare_architectures()`

        Returns:
            Number of models imported
        """
        try:
            results = registry.compare_architectures(**kwargs)
        except Exception:
            return 0

        count = 0
        for name, metrics in results.items():
            if 'error' in metrics:
                continue
            self.add_model(
                name=name,
                architecture=name,
                parameters=int(metrics.get('parameters', 0)),
                memory_mb=float(metrics.get('memory_mb', 0.0)),
            )
            m = self._models[name]
            # Inject benchmark latencies
            if 'latency_mean_ms' in metrics:
                for _ in range(20):
                    self.record_latency(name, metrics['latency_mean_ms'])
            if 'latency_p95_ms' in metrics:
                self.record_latency(name, metrics['latency_p95_ms'])
            count += 1

        return count

    # ---- Demo ----

    def seed_demo(self) -> None:
        """
        Populate the workspace with the four Phase-3 architectures,
        using realistic values from a typical 64×64 single-channel run.
        """
        self._models.clear()
        self._color_counter = 0

        demo = [
            dict(
                name="edge_ae", architecture="EdgeStreamingAE",
                parameters=149_024, memory_mb=0.57, latent_dim=32,
                throughput_fps=1180,
                latency_mean=2.4, latency_std=0.35,
                recon_mean=0.0034, recon_std=0.0008,
            ),
            dict(
                name="vae", architecture="ConvVAE",
                parameters=168_448, memory_mb=0.64, latent_dim=32,
                throughput_fps=920,
                latency_mean=3.1, latency_std=0.55,
                recon_mean=0.0058, recon_std=0.0014,
            ),
            dict(
                name="unet_ae", architecture="UNetAutoencoder",
                parameters=209_617, memory_mb=0.80, latent_dim=32,
                throughput_fps=580,
                latency_mean=5.2, latency_std=1.1,
                recon_mean=0.0021, recon_std=0.0005,
            ),
            dict(
                name="transformer_ae", architecture="TransformerAutoencoder",
                parameters=182_144, memory_mb=0.70, latent_dim=32,
                throughput_fps=380,
                latency_mean=6.8, latency_std=1.4,
                recon_mean=0.0029, recon_std=0.0009,
            ),
        ]

        rng = np.random.default_rng(42)
        for spec in demo:
            self.add_model(
                name=spec['name'],
                architecture=spec['architecture'],
                parameters=spec['parameters'],
                memory_mb=spec['memory_mb'],
                latent_dim=spec['latent_dim'],
                throughput_fps=spec['throughput_fps'],
            )
            # Generate 80 samples per metric
            for _ in range(80):
                self.record_latency(spec['name'],
                                    max(0.5, spec['latency_mean']
                                        + rng.normal(0, spec['latency_std'])))
                self.record_recon(spec['name'],
                                  max(0.0, spec['recon_mean']
                                      + rng.normal(0, spec['recon_std'])))

    # ---- Retrieval ----

    def model_names(self, enabled_only: bool = False) -> List[str]:
        if enabled_only:
            return [n for n, m in self._models.items() if m.enabled]
        return list(self._models.keys())

    def get_model(self, name: str) -> Optional[ModelProfile]:
        return self._models.get(name)

    @property
    def size(self) -> int:
        return len(self._models)

    @property
    def is_empty(self) -> bool:
        return len(self._models) == 0

    def to_dataframe_like(self) -> List[Dict[str, Any]]:
        """Return list-of-dicts suitable for Streamlit table rendering."""
        rows = []
        for name, m in self._models.items():
            ls = m.latency_stats()
            rs = m.recon_stats()
            rows.append({
                'Model': name,
                'Architecture': m.architecture,
                'Params': m.parameters,
                'Memory (MB)': m.memory_mb,
                'Latent': m.latent_dim,
                'FPS': m.throughput_fps,
                'Latency p50': ls['p50'],
                'Latency p95': ls['p95'],
                'Recon mean': rs['mean'],
                'Samples': ls['n'],
            })
        return rows

    # ---- Ranking ----

    def rank_by(self, metric: str) -> List[Tuple[str, float]]:
        """
        Rank models by a metric (ascending = better, except throughput).

        Returns list of (name, value) sorted best-first.
        """
        models = [m for m in self._models.values() if m.enabled]
        if not models:
            return []

        def get_value(m: ModelProfile) -> float:
            if metric == 'latency_p95':
                return m.latency_stats()['p95']
            if metric == 'latency_mean':
                return m.latency_stats()['mean']
            if metric == 'recon_mean':
                return m.recon_stats()['mean']
            if metric == 'parameters':
                return float(m.parameters)
            if metric == 'memory':
                return m.memory_mb
            if metric == 'throughput':
                return -m.throughput_fps  # higher is better → invert for sort
            return 0.0

        pairs = [(m.name, get_value(m)) for m in models]
        pairs.sort(key=lambda x: x[1])
        return pairs

    def winner_for(self, metric: str) -> Optional[str]:
        ranked = self.rank_by(metric)
        return ranked[0][0] if ranked else None

    def summary(self) -> Dict[str, Any]:
        """High-level summary of the workspace."""
        if self.is_empty:
            return {'n_models': 0, 'best_latency': None,
                    'best_recon': None, 'best_efficiency': None}

        latency_winner = self.winner_for('latency_p95')
        recon_winner = self.winner_for('recon_mean')

        # Efficiency score = quality-per-latency
        # Combine recon (lower is better) and p95 latency (lower is better)
        scores = {}
        for name, m in self._models.items():
            if not m.enabled:
                continue
            ls = m.latency_stats()
            rs = m.recon_stats()
            # Avoid div-by-zero
            eff = 1.0 / (ls['p95'] * rs['mean'] * 1000.0 + 1e-6)
            scores[name] = eff
        best_eff = max(scores.items(), key=lambda x: x[1])[0] if scores else None

        return {
            'n_models': self.size,
            'best_latency': latency_winner,
            'best_recon': recon_winner,
            'best_efficiency': best_eff,
        }

    def reset(self) -> None:
        self._models.clear()
        self._color_counter = 0


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_params_bar(
    rows: List[Dict[str, Any]],
    palette: Palette = None,
    height: int = 260,
) -> go.Figure:
    """Horizontal bar of parameters + memory per model."""
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Parameters", "Memory (MB)"),
        horizontal_spacing=0.18,
    )

    if not rows:
        apply_theme(fig, p, height=height)
        return fig

    names = [r['Model'] for r in rows]
    params = [r['Params'] for r in rows]
    memory = [r['Memory (MB)'] for r in rows]

    colors = [accent_color(p, i) for i in range(len(rows))]

    fig.add_trace(go.Bar(
        y=names, x=params, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,}" for v in params],
        textposition="outside",
        textfont=dict(color=p.text_secondary, family="JetBrains Mono, monospace", size=9),
        hovertemplate="%{y}<br>Params: %{x:,}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=names, x=memory, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in memory],
        textposition="outside",
        textfont=dict(color=p.text_secondary, family="JetBrains Mono, monospace", size=9),
        hovertemplate="%{y}<br>Memory: %{x:.3f} MB<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(title="Model Complexity")
    apply_theme(fig, p, height=height)
    fig.update_yaxes(autorange="reversed")
    return fig


def build_latency_box(
    rows_with_samples: List[Tuple[str, List[float]]],
    palette: Palette = None,
    height: int = 320,
) -> go.Figure:
    """
    Latency box plots per model — shows distribution shape at a glance.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not rows_with_samples:
        apply_theme(fig, p, height=height)
        return fig

    for i, (name, samples) in enumerate(rows_with_samples):
        if not samples:
            continue
        color = accent_color(p, i)
        fig.add_trace(go.Box(
            y=samples,
            name=name,
            marker=dict(color=color, size=3),
            line=dict(color=color, width=2),
            boxpoints="outliers",
            jitter=0.3,
            hovertemplate=f"<b>{name}</b><br>%{{y:.3f}} ms<extra></extra>",
        ))

    fig.update_layout(
        title="Latency Distribution",
        yaxis_title="Latency (ms)",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_latency_percentile_grouped(
    rows: List[Dict[str, Any]],
    palette: Palette = None,
    height: int = 300,
) -> go.Figure:
    """
    Grouped bar chart of p50/p95/p99 latencies for each model.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not rows:
        apply_theme(fig, p, height=height)
        return fig

    names = [r['Model'] for r in rows]
    p50 = [r['Latency p50'] for r in rows]
    p95 = [r['Latency p95'] for r in rows]
    # We don't have p99 stored per row, generate it from p95 × 1.15 as estimate
    p99 = [v * 1.15 for v in p95]

    fig.add_trace(go.Bar(
        name="p50", x=names, y=p50,
        marker=dict(color=p.success, line=dict(width=0)),
        hovertemplate="%{x}<br>p50: %{y:.3f} ms<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="p95", x=names, y=p95,
        marker=dict(color=p.warning, line=dict(width=0)),
        hovertemplate="%{x}<br>p95: %{y:.3f} ms<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="p99", x=names, y=p99,
        marker=dict(color=p.danger, line=dict(width=0)),
        hovertemplate="%{x}<br>p99 (est): %{y:.3f} ms<extra></extra>",
    ))

    fig.update_layout(
        title="Latency Percentiles",
        yaxis_title="Latency (ms)",
        barmode="group",
        xaxis_title="Model",
    )
    apply_theme(fig, p, height=height)
    return fig


def build_recon_bar(
    rows: List[Dict[str, Any]],
    palette: Palette = None,
    height: int = 280,
) -> go.Figure:
    """Reconstruction error comparison (lower is better)."""
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not rows:
        apply_theme(fig, p, height=height)
        return fig

    names = [r['Model'] for r in rows]
    values = [r['Recon mean'] for r in rows]
    best = min(values) if values else 0.0
    # Color best model green, others accent
    colors = [p.success if v == best else accent_color(p, i)
              for i, v in enumerate(values)]

    fig.add_trace(go.Bar(
        x=names, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.5f}" for v in values],
        textposition="outside",
        textfont=dict(color=p.text_secondary, family="JetBrains Mono, monospace", size=10),
        hovertemplate="%{x}<br>Recon MSE: %{y:.6f}<extra></extra>",
    ))

    fig.update_layout(
        title="Reconstruction Error (lower is better)",
        yaxis_title="MSE",
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_radar(
    rows: List[Dict[str, Any]],
    palette: Palette = None,
    height: int = 400,
) -> go.Figure:
    """
    Multi-axis radar showing trade-offs.

    Normalized axes (all mapped to [0, 1] where 1 = best):
        Speed      = inverse of p95 latency
        Quality    = inverse of recon error
        Compact    = inverse of parameters
        Efficiency = inverse of (p95 × recon)
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not rows:
        apply_theme(fig, p, height=height)
        return fig

    # Extract raw values
    lats = np.array([r['Latency p95'] for r in rows], dtype=np.float64)
    recons = np.array([r['Recon mean'] for r in rows], dtype=np.float64)
    params = np.array([r['Params'] for r in rows], dtype=np.float64)

    def normalize_inverse(arr):
        # Lower-is-better metric → map to [0,1] where 1 = best (lowest)
        safe = np.maximum(arr, 1e-9)
        inv = 1.0 / safe
        lo, hi = inv.min(), inv.max()
        if hi - lo < 1e-12:
            return np.ones_like(inv) * 0.5
        return (inv - lo) / (hi - lo)

    speed_axis = normalize_inverse(lats)
    quality_axis = normalize_inverse(recons)
    compact_axis = normalize_inverse(params)
    efficiency_axis = normalize_inverse(lats * recons)

    axes = ['Speed', 'Quality', 'Compact', 'Efficiency']

    for i, r in enumerate(rows):
        name = r['Model']
        vals = [
            float(speed_axis[i]),
            float(quality_axis[i]),
            float(compact_axis[i]),
            float(efficiency_axis[i]),
        ]
        # Close the loop
        vals_closed = vals + [vals[0]]
        theta = axes + [axes[0]]

        color = accent_color(p, i)
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=theta,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.12),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Multi-Axis Trade-off Radar (1.0 = best)",
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=p.grid_line, linecolor=p.border,
                tickfont=dict(color=p.text_muted, size=9),
            ),
            angularaxis=dict(
                gridcolor=p.grid_line, linecolor=p.border,
                tickfont=dict(color=p.text_secondary, size=11),
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5),
    )
    apply_theme(fig, p, height=height)
    return fig


def build_throughput_fps(
    rows: List[Dict[str, Any]],
    palette: Palette = None,
    height: int = 260,
) -> go.Figure:
    """Horizontal bars of estimated throughput (FPS)."""
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not rows:
        apply_theme(fig, p, height=height)
        return fig

    names = [r['Model'] for r in rows]
    fps = [r['FPS'] for r in rows]
    # Sort descending
    pairs = sorted(zip(names, fps), key=lambda x: x[1], reverse=True)
    names = [n for n, _ in pairs]
    fps = [f for _, f in pairs]

    colors = [accent_color(p, i) for i in range(len(names))]

    fig.add_trace(go.Bar(
        y=names, x=fps, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f} fps" for v in fps],
        textposition="outside",
        textfont=dict(color=p.text_secondary, family="JetBrains Mono, monospace", size=10),
        hovertemplate="%{y}<br>FPS: %{x:.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Throughput (higher is better)",
        xaxis_title="Frames per second (est.)",
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

def render(
    st,
    palette: Palette,
    state: ModelComparisonState,
    key_prefix: str = "model_comparison",
    show_demo_button: bool = True,
) -> None:
    """Render the model comparison workspace inside a Streamlit context."""
    st.markdown(section_header("⚖️", "Model Comparison · A/B Workspace"), unsafe_allow_html=True)

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

    with ctrl1:
        model_names = state.model_names()
        selected = st.multiselect(
            "Models in view",
            options=model_names,
            default=model_names,
            key=f"{key_prefix}_selected",
        )
        # Toggle enabled based on selection
        for name in model_names:
            state.enable_model(name, name in selected)

    with ctrl2:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key=f"{key_prefix}_reset", use_container_width=True):
            state.reset()
            st.toast("Comparison workspace cleared.", icon="🧹")

    with ctrl3:
        if show_demo_button:
            st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
            if st.button("🌱 Load demo models (4 architectures)",
                         key=f"{key_prefix}_seed"):
                state.seed_demo()
                st.rerun()

    if state.is_empty:
        st.info(
            "No models in the workspace. Load demo models above, call "
            "`state.add_model(...)`, or use `state.import_from_registry(...)`."
        )
        return

    # ---- Data ----
    rows = [r for r in state.to_dataframe_like() if state.get_model(r['Model']).enabled]
    summary = state.summary()

    if not rows:
        st.warning("No models selected. Re-enable at least one from the sidebar above.")
        return

    # ---- Top Metric Row ----
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(metric_card(
            "Models in View",
            f"{len(rows)}",
            f"of {state.size} total",
        ), unsafe_allow_html=True)

    with m2:
        winner = summary['best_latency']
        winner_p95 = ""
        if winner:
            winner_p95 = f"p95 {state.get_model(winner).latency_stats()['p95']:.2f} ms"
        st.markdown(metric_card(
            "Fastest (p95)",
            winner or "—",
            winner_p95,
            accent=palette.success,
        ), unsafe_allow_html=True)

    with m3:
        winner = summary['best_recon']
        winner_recon = ""
        if winner:
            winner_recon = f"MSE {state.get_model(winner).recon_stats()['mean']:.5f}"
        st.markdown(metric_card(
            "Best Reconstruction",
            winner or "—",
            winner_recon,
            accent=palette.success,
        ), unsafe_allow_html=True)

    with m4:
        winner = summary['best_efficiency']
        st.markdown(metric_card(
            "Best Quality-per-Latency",
            winner or "—",
            "efficiency winner",
            accent=palette.accent_secondary,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Comparison Table ----
    st.markdown(section_header("📋", "Metric Summary"), unsafe_allow_html=True)
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Parameters + Memory ----
    fig_params = build_params_bar(rows, palette, height=280)
    st.plotly_chart(fig_params, use_container_width=True, key=f"{key_prefix}_params")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Latency: box plot + grouped percentiles ----
    left, right = st.columns(2)

    with left:
        samples_list = []
        for r in rows:
            m = state.get_model(r['Model'])
            if m is not None:
                samples_list.append((m.name, list(m.latency_samples)))
        fig_box = build_latency_box(samples_list, palette, height=320)
        st.plotly_chart(fig_box, use_container_width=True, key=f"{key_prefix}_box")

    with right:
        fig_pct = build_latency_percentile_grouped(rows, palette, height=320)
        st.plotly_chart(fig_pct, use_container_width=True, key=f"{key_prefix}_pct")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Reconstruction + throughput ----
    left2, right2 = st.columns(2)

    with left2:
        fig_recon = build_recon_bar(rows, palette, height=280)
        st.plotly_chart(fig_recon, use_container_width=True, key=f"{key_prefix}_recon")

    with right2:
        fig_fps = build_throughput_fps(rows, palette, height=280)
        st.plotly_chart(fig_fps, use_container_width=True, key=f"{key_prefix}_fps")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Radar ----
    st.markdown(section_header("🧭", "Multi-Axis Trade-off Radar"), unsafe_allow_html=True)
    fig_radar = build_radar(rows, palette, height=440)
    st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Per-model cards ----
    st.markdown(section_header("🏁", "Per-Model Deep Dive"), unsafe_allow_html=True)

    cols = st.columns(min(len(rows), 4))
    for i, r in enumerate(rows):
        col = cols[i % len(cols)]
        m = state.get_model(r['Model'])
        if m is None:
            continue

        ls = m.latency_stats()
        rs = m.recon_stats()

        # Winner marks
        marks = []
        if summary['best_latency'] == m.name:
            marks.append("⚡ Fastest")
        if summary['best_recon'] == m.name:
            marks.append("🎯 Sharpest")
        if summary['best_efficiency'] == m.name:
            marks.append("🏆 Best trade-off")

        marks_html = "".join(
            f'<span style="display:inline-block; margin-right:6px; '
            f'padding:2px 8px; border-radius:6px; font-size:0.7rem; '
            f'background:rgba(56,189,248,0.15); color:{palette.accent_primary};">'
            f'{mk}</span>'
            for mk in marks
        )

        with col:
            st.markdown(f"""
            <div class="hud-card hud-fade-in" style="padding:14px 16px; margin-bottom:10px;">
                <div style="font-family:var(--mono); font-size:1rem;
                            color:var(--text-primary); font-weight:700;
                            margin-bottom:2px;">
                    {m.name}
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted);
                            margin-bottom:8px;">
                    {m.architecture}
                </div>
                <div style="margin-bottom:10px;">{marks_html}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(kv_row("Params", f"{m.parameters:,}"), unsafe_allow_html=True)
            st.markdown(kv_row("Memory", f"{m.memory_mb:.3f} MB"), unsafe_allow_html=True)
            st.markdown(kv_row("Latency p50", f"{ls['p50']:.3f} ms"), unsafe_allow_html=True)
            st.markdown(kv_row("Latency p95", f"{ls['p95']:.3f} ms"), unsafe_allow_html=True)
            st.markdown(kv_row("Recon mean", f"{rs['mean']:.5f}"), unsafe_allow_html=True)
            st.markdown(kv_row("Samples", f"{ls['n']}"), unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Ranking tables ----
    left3, right3 = st.columns(2)

    with left3:
        st.markdown(section_header("⚡", "Latency Leaderboard"), unsafe_allow_html=True)
        for i, (name, val) in enumerate(state.rank_by('latency_p95')):
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:8px 12px; margin:4px 0;
                        background:rgba(20,28,46,0.5);
                        border-radius:8px; font-family:var(--mono);
                        font-size:0.85rem;">
                <span>{marker} <b>{name}</b></span>
                <span style="color:{palette.warning};">{val:.3f} ms (p95)</span>
            </div>
            """, unsafe_allow_html=True)

    with right3:
        st.markdown(section_header("🎯", "Reconstruction Leaderboard"), unsafe_allow_html=True)
        for i, (name, val) in enumerate(state.rank_by('recon_mean')):
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:8px 12px; margin:4px 0;
                        background:rgba(20,28,46,0.5);
                        border-radius:8px; font-family:var(--mono);
                        font-size:0.85rem;">
                <span>{marker} <b>{name}</b></span>
                <span style="color:{palette.success};">{val:.6f} MSE</span>
            </div>
            """, unsafe_allow_html=True)