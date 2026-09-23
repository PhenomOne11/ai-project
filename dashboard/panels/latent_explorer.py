"""
3D Latent Space Explorer Panel

Wraps the 32-D latent bottleneck into an interactive 3D PCA plot with:
    - Trail-fading trajectory (older samples more transparent)
    - Newest sample highlighted with a larger marker
    - Three coloring modes: time, cluster (k-means), novelty
    - Live statistics sidebar (variance explained, spread, density)
    - Reset & reseed controls

The state container (`LatentExplorerState`) has zero Streamlit dependencies
and can be tested in isolation.
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import (
    metric_card, section_header, divider, status_badge,
    apply_theme, accent_color, progress_bar,
)
from dashboard.theme.palette import Palette
from core.math_utils.linear_algebra import batch_svd_project


# =============================================================================
# STATE
# =============================================================================

class LatentExplorerState:
    """
    Manages the rolling latent-vector buffer and PCA projection.

    Thread-agnostic, Streamlit-agnostic, testable in isolation.

    Args:
        latent_dim: Expected latent dimensionality
        buffer_size: Maximum retained samples (rolling window)
        n_components: PCA components to keep (2 or 3 typical)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        buffer_size: int = 500,
        n_components: int = 3,
    ):
        self.latent_dim = latent_dim
        self.buffer_size = buffer_size
        self.n_components = n_components

        self._buffer: deque = deque(maxlen=buffer_size)
        self._n_pushed: int = 0

    # ---- Ingestion ----

    def push(self, latent: np.ndarray) -> None:
        """Add a single latent vector to the buffer."""
        v = np.asarray(latent).flatten().astype(np.float64)
        if v.size != self.latent_dim:
            # Auto-expand if actual dim differs (robustness)
            self.latent_dim = v.size
        self._buffer.append(v)
        self._n_pushed += 1

    def push_batch(self, latents: np.ndarray) -> None:
        """Add a batch of latent vectors (N, D)."""
        arr = np.asarray(latents, dtype=np.float64)
        if arr.ndim == 1:
            self.push(arr)
            return
        for row in arr:
            self.push(row)

    # ---- Projection ----

    def project(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the buffer into n_components-dimensional PCA space.

        Returns:
            (coords, explained_variance_ratio)
            coords shape: (N, n_components)
        """
        if len(self._buffer) < max(self.n_components, 3):
            return np.empty((0, self.n_components)), np.array([])

        data = np.stack(self._buffer, axis=0)
        projected, ratios, _ = batch_svd_project(data, n_components=self.n_components)
        return projected, ratios

    # ---- Statistics ----

    def stats(self) -> Dict[str, Any]:
        """Compute quick stats over the current buffer."""
        if len(self._buffer) < 3:
            return {
                'n_samples': len(self._buffer),
                'n_pushed_total': self._n_pushed,
                'mean_norm': 0.0,
                'std_norm': 0.0,
                'spread': 0.0,
                'variance_explained': 0.0,
                'buffer_full': False,
            }

        data = np.stack(self._buffer, axis=0)
        norms = np.linalg.norm(data, axis=1)

        coords, ratios = self.project()
        if coords.shape[0] >= 2:
            centroid = coords.mean(axis=0)
            spread = float(np.linalg.norm(coords - centroid, axis=1).mean())
        else:
            spread = 0.0

        return {
            'n_samples': len(self._buffer),
            'n_pushed_total': self._n_pushed,
            'mean_norm': float(norms.mean()),
            'std_norm': float(norms.std()),
            'spread': spread,
            'variance_explained': float(ratios.sum()) if len(ratios) else 0.0,
            'buffer_full': len(self._buffer) >= self.buffer_size,
        }

    # ---- Clustering (lightweight k-means) ----

    def cluster(
        self,
        n_clusters: int = 4,
        n_iter: int = 20,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simple k-means on the PCA-projected coordinates.
        Returns cluster labels of shape (N,).
        """
        coords, _ = self.project()
        if coords.shape[0] < n_clusters:
            return np.zeros(coords.shape[0], dtype=int)

        rng = np.random.default_rng(seed)
        n, d = coords.shape
        # Initialize with k random points
        init_idx = rng.choice(n, size=n_clusters, replace=False)
        centers = coords[init_idx].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(n_iter):
            # Assign
            dists = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Update
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    centers[k] = coords[mask].mean(axis=0)

        return labels

    # ---- Lifecycle ----

    def reset(self) -> None:
        self._buffer.clear()
        self._n_pushed = 0

    def seed_demo(self, n: int = 200) -> None:
        """
        Seed the buffer with synthetic helix-like latents for demo mode.
        Produces three interleaved clusters for visual interest.
        """
        rng = np.random.default_rng(123)
        t = np.linspace(0, 4 * np.pi, n)
        data = np.zeros((n, self.latent_dim), dtype=np.float64)

        # Helix in first 3 latent dims
        data[:, 0] = np.cos(t) * 1.5
        data[:, 1] = np.sin(t) * 1.5
        data[:, 2] = t / 4.0

        # Add 2 clusters in dims 3-5 offset from helix
        half = n // 2
        data[:half, 3] += 2.0 + rng.normal(0, 0.2, half)
        data[half:, 3] -= 2.0 + rng.normal(0, 0.2, n - half)

        # Noise on remaining dims
        data[:, 6:] += rng.normal(0, 0.15, (n, self.latent_dim - 6))

        for row in data:
            self.push(row)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0


# =============================================================================
# FIGURE BUILDERS (pure — no Streamlit)
# =============================================================================

def build_trajectory_figure(
    coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    palette: Palette = None,
    color_mode: str = "time",
    height: int = 520,
    title: str = "32-D Latent Manifold · 3D PCA Projection",
) -> go.Figure:
    """
    Build the main 3D trajectory figure.

    Args:
        coords: (N, 3) PCA coordinates
        labels: Optional cluster labels (N,) for color_mode='cluster'
        palette: Active theme palette
        color_mode: 'time' | 'cluster' | 'novelty'
        height: Plot height in px
        title: Plot title
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()

    if coords.shape[0] == 0:
        fig.update_layout(title=title)
        apply_theme(fig, p, height=height)
        return fig

    n = coords.shape[0]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Determine per-point colors
    if color_mode == "cluster" and labels is not None and len(labels) == n:
        n_clusters = int(labels.max()) + 1
        color_values = labels.astype(float)
        colorscale = [[i / max(n_clusters - 1, 1), p.chart_sequence[i % len(p.chart_sequence)]]
                      for i in range(n_clusters)]
        marker_color = color_values
        marker_colorscale = colorscale
        colorbar = dict(
            title=dict(text="Cluster", font=dict(color=p.text_secondary, size=10)),
            tickfont=dict(color=p.text_muted, size=9),
            bgcolor="rgba(20, 28, 46, 0.6)",
            bordercolor=p.border,
            borderwidth=1,
            thickness=12,
            len=0.6,
        )
    elif color_mode == "novelty":
        # Novelty = distance from the centroid
        centroid = coords.mean(axis=0)
        novelty = np.linalg.norm(coords - centroid, axis=1)
        marker_color = novelty
        marker_colorscale = [
            [0.0, p.success],
            [0.5, p.warning],
            [1.0, p.danger],
        ]
        colorbar = dict(
            title=dict(text="Novelty", font=dict(color=p.text_secondary, size=10)),
            tickfont=dict(color=p.text_muted, size=9),
            bgcolor="rgba(20, 28, 46, 0.6)",
            bordercolor=p.border,
            borderwidth=1,
            thickness=12,
            len=0.6,
        )
    else:  # time
        marker_color = np.arange(n)
        marker_colorscale = [
            [0.0, "rgba(56, 189, 248, 0.15)"],
            [0.5, p.accent_primary],
            [1.0, p.accent_secondary],
        ]
        colorbar = None

    # Main trail scatter
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        line=dict(
            color="rgba(56, 189, 248, 0.35)",
            width=2,
        ),
        marker=dict(
            size=4,
            color=marker_color,
            colorscale=marker_colorscale,
            colorbar=colorbar,
            opacity=0.85,
            showscale=(color_mode in ("cluster", "novelty")),
            line=dict(width=0),
        ),
        name="Trajectory",
        hovertemplate=(
            "PCA-X: %{x:.3f}<br>"
            "PCA-Y: %{y:.3f}<br>"
            "PCA-Z: %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Highlight the newest point
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(
            size=11,
            color=p.warning,
            symbol="diamond",
            line=dict(color=p.warning, width=2),
            opacity=1.0,
        ),
        name="Latest",
        hovertemplate="<b>LATEST</b><br>"
                      "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
    ))

    # Centroid ghost
    centroid = coords.mean(axis=0)
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
        mode="markers",
        marker=dict(
            size=8,
            color=p.text_muted,
            symbol="x",
        ),
        name="Centroid",
        hoverinfo="skip",
    ))

    fig.update_layout(title=title)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="PC1", font=dict(size=10)), gridcolor=p.grid_line),
            yaxis=dict(title=dict(text="PC2", font=dict(size=10)), gridcolor=p.grid_line),
            zaxis=dict(title=dict(text="PC3", font=dict(size=10)), gridcolor=p.grid_line),
        ),
        showlegend=False,
    )
    apply_theme(fig, p, height=height)
    return fig


def build_projection_panel(
    coords: np.ndarray,
    palette: Palette = None,
    height: int = 280,
) -> go.Figure:
    """
    2×2 panel of 2D projections: XY, XZ, YZ, and PCA variance explained.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PC1 × PC2", "PC1 × PC3", "PC2 × PC3"),
    )

    if coords.shape[0] == 0:
        apply_theme(fig, p, height=height)
        return fig

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    n = len(x)
    color = np.arange(n)

    colorscale = [[0.0, "rgba(56, 189, 248, 0.35)"],
                  [1.0, p.accent_secondary]]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=4, color=color, colorscale=colorscale, opacity=0.75),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=z, mode="markers",
        marker=dict(size=4, color=color, colorscale=colorscale, opacity=0.75),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=y, y=z, mode="markers",
        marker=dict(size=4, color=color, colorscale=colorscale, opacity=0.75),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=3)

    fig.update_layout(title="2D Marginal Projections")
    apply_theme(fig, p, height=height)

    # Contained axis ticks
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=9, family="JetBrains Mono"))
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=9, family="JetBrains Mono"))

    return fig


# =============================================================================
# STREAMLIT RENDER
# =============================================================================

def render(
    st,
    palette: Palette,
    state: LatentExplorerState,
    key_prefix: str = "latent_explorer",
    show_demo_button: bool = True,
) -> None:
    """
    Render the full latent explorer panel into a Streamlit context.

    Args:
        st: Reference to the streamlit module
        palette: Active palette
        state: LatentExplorerState instance (persisted by caller)
        key_prefix: Unique prefix for widget keys
        show_demo_button: Whether to offer the "seed demo data" button
    """
    # ---- Section Header ----
    st.markdown(section_header("📡", "32-D Latent Manifold Explorer"), unsafe_allow_html=True)

    # ---- Controls Row ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        color_mode = st.selectbox(
            "Color mode",
            options=["time", "cluster", "novelty"],
            index=0,
            key=f"{key_prefix}_color_mode",
        )

    with ctrl2:
        n_clusters = st.slider(
            "K-means clusters",
            min_value=2, max_value=8, value=4, step=1,
            key=f"{key_prefix}_n_clusters",
            disabled=(color_mode != "cluster"),
        )

    with ctrl3:
        height = st.slider(
            "3D height (px)",
            min_value=360, max_value=800, value=520, step=40,
            key=f"{key_prefix}_height",
        )

    with ctrl4:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        reset_clicked = st.button(
            "🔄 Reset Buffer",
            key=f"{key_prefix}_reset",
            use_container_width=True,
        )
        if reset_clicked:
            state.reset()
            st.toast("Latent buffer cleared.", icon="🧹")

    if show_demo_button and state.is_empty:
        if st.button("🌱 Seed demo data (200 samples)",
                     key=f"{key_prefix}_seed", use_container_width=False):
            state.seed_demo(200)
            st.rerun()

    # ---- Early return if buffer empty ----
    if state.is_empty:
        st.info(
            "No latents captured yet. Boot the engine to stream vectors, "
            "or click **Seed demo data** above."
        )
        return

    # ---- Compute Projection & Stats ----
    coords, ratios = state.project()
    stats = state.stats()

    labels = None
    if color_mode == "cluster" and coords.shape[0] >= n_clusters:
        labels = state.cluster(n_clusters=n_clusters)

    # ---- Top Statistics Row ----
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(metric_card(
            "Samples Buffered",
            f"{stats['n_samples']:,}",
            f"Total pushed: {stats['n_pushed_total']:,}",
        ), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_card(
            "Variance Explained",
            f"{stats['variance_explained'] * 100:.1f}%",
            "by top-3 PCs",
            accent=palette.success if stats['variance_explained'] > 0.7 else palette.warning,
        ), unsafe_allow_html=True)

    with m3:
        st.markdown(metric_card(
            "Mean ‖z‖₂",
            f"{stats['mean_norm']:.3f}",
            f"σ = {stats['std_norm']:.3f}",
        ), unsafe_allow_html=True)

    with m4:
        st.markdown(metric_card(
            "Manifold Spread",
            f"{stats['spread']:.3f}",
            "mean distance from centroid",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Main Layout: 3D + Side Panel ----
    left, right = st.columns([3, 1])

    with left:
        fig = build_trajectory_figure(
            coords=coords,
            labels=labels,
            palette=palette,
            color_mode=color_mode,
            height=height,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_main")

    with right:
        st.markdown(section_header("📊", "Live Stats"), unsafe_allow_html=True)

        # Buffer fill progress
        fill_pct = min(1.0, stats['n_samples'] / max(state.buffer_size, 1))
        st.markdown(
            progress_bar("Buffer fill", fill_pct, 1.0),
            unsafe_allow_html=True,
        )

        if len(ratios) >= 3:
            for i, r in enumerate(ratios[:3]):
                st.markdown(
                    progress_bar(
                        f"PC{i+1} variance",
                        float(r), 1.0,
                        color=accent_color(palette, i),
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown(divider(), unsafe_allow_html=True)

        # Status badges
        if stats['buffer_full']:
            st.markdown(status_badge("Buffer Saturated", "info", "●"),
                        unsafe_allow_html=True)
        else:
            st.markdown(status_badge("Accumulating", "active", "●"),
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"Color mode: **{color_mode}**")
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            st.caption("**Cluster breakdown**")
            for k, c in zip(unique, counts):
                st.caption(f"  • Cluster {int(k)}: {int(c)} pts")

    # ---- 2D Projection Panel ----
    st.markdown(divider(), unsafe_allow_html=True)
    fig_2d = build_projection_panel(coords=coords, palette=palette)
    st.plotly_chart(fig_2d, use_container_width=True, key=f"{key_prefix}_2d")