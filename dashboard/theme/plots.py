"""
Plotly theming utilities for consistent visuals across the HUD.
"""

import plotly.graph_objects as go
from dashboard.theme.palette import Palette


def apply_theme(fig: go.Figure, palette: Palette, height: int = 350) -> go.Figure:
    """
    Apply the HUD theme to a Plotly figure.

    Modifies layout: colors, fonts, grids, background, modebar.
    """
    p = palette

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=f"rgba({_hex_to_rgb(p.bg_surface, alpha=0.45)})",
        font=dict(
            color=p.text_secondary,
            family="Inter, system-ui, sans-serif",
            size=12,
        ),
        title=dict(
            font=dict(color=p.text_primary, size=13, family="Inter, sans-serif"),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        colorway=p.chart_sequence,
        hoverlabel=dict(
            bgcolor=p.bg_elevated,
            bordercolor=p.accent_primary,
            font=dict(color=p.text_primary, family="JetBrains Mono, monospace", size=11),
        ),
        legend=dict(
            bgcolor="rgba(20, 28, 46, 0.6)",
            bordercolor=p.border,
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )

    # Axis styling
    axis_common = dict(
        gridcolor=p.grid_line,
        zerolinecolor=p.grid_line,
        linecolor=p.border,
        tickfont=dict(color=p.text_muted, size=10, family="JetBrains Mono, monospace"),
        title_font=dict(color=p.text_secondary, size=11),
        showline=True,
        mirror=False,
    )

    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)

    # 3D scene
    fig.update_layout(
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                gridcolor=p.grid_line, zerolinecolor=p.grid_line,
                backgroundcolor="rgba(20, 28, 46, 0.2)",
                showbackground=True,
                tickfont=dict(color=p.text_muted, size=9),
                title_font=dict(color=p.text_secondary, size=10),
            ),
            yaxis=dict(
                gridcolor=p.grid_line, zerolinecolor=p.grid_line,
                backgroundcolor="rgba(20, 28, 46, 0.2)",
                showbackground=True,
                tickfont=dict(color=p.text_muted, size=9),
                title_font=dict(color=p.text_secondary, size=10),
            ),
            zaxis=dict(
                gridcolor=p.grid_line, zerolinecolor=p.grid_line,
                backgroundcolor="rgba(20, 28, 46, 0.2)",
                showbackground=True,
                tickfont=dict(color=p.text_muted, size=9),
                title_font=dict(color=p.text_secondary, size=10),
            ),
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.1),
            up=dict(x=0, y=0, z=1),
        ),
    )

    # Polar styling
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(20, 28, 46, 0.25)",
            radialaxis=dict(
                gridcolor=p.grid_line, linecolor=p.border,
                tickfont=dict(color=p.text_muted, size=9),
            ),
            angularaxis=dict(
                gridcolor=p.grid_line, linecolor=p.border,
                tickfont=dict(color=p.text_muted, size=9),
            ),
        ),
    )

    return fig


def _hex_to_rgb(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB to 'r, g, b, a' string."""
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    if alpha < 1.0:
        return f"{r}, {g}, {b}, {alpha}"
    return f"rgb({r}, {g}, {b})"


def accent_color(palette: Palette, index: int = 0) -> str:
    """Retrieve the nth accent color from the palette chart sequence."""
    seq = palette.chart_sequence
    return seq[index % len(seq)]


def severity_color(palette: Palette, level: str) -> str:
    """Map severity string → palette accent."""
    return {
        'nominal': palette.success,
        'info': palette.info,
        'warning': palette.warning,
        'critical': palette.danger,
        'emergency': palette.danger,
    }.get(level.lower(), palette.accent_primary)