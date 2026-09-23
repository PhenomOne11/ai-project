"""
Reusable HTML components styled with the HUD CSS.

All functions return raw HTML strings to be rendered with
`st.markdown(html, unsafe_allow_html=True)`.
"""

from typing import Optional, List, Tuple


def status_badge(text: str, state: str = "neutral", icon: str = "") -> str:
    """
    Inline status pill.

    States: 'active', 'paused', 'critical', 'info', 'neutral'
    """
    icon_html = f'<span style="font-size:0.9rem">{icon}</span>' if icon else ''
    return f'<span class="hud-badge {state}">{icon_html}{text}</span>'


def metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    accent: Optional[str] = None
) -> str:
    """Glassmorphic metric card (larger than st.metric)."""
    sub_html = f'<div class="hud-card-sub">{subtitle}</div>' if subtitle else ''
    accent_style = f'style="color:{accent};text-shadow:0 0 20px {accent}66;"' if accent else ''
    return f"""
    <div class="hud-card hud-fade-in">
        <div class="hud-card-title">{title}</div>
        <div class="hud-card-value" {accent_style}>{value}</div>
        {sub_html}
    </div>
    """


def section_header(icon: str, title: str) -> str:
    """Section header with icon and accent-colored title."""
    return f"""
    <div class="hud-section">
        <span class="hud-section-icon">{icon}</span>
        <span class="hud-section-title">{title}</span>
    </div>
    """


def divider() -> str:
    return '<div class="hud-divider"></div>'


def chip(text: str) -> str:
    """Small inline kbd-style chip."""
    return f'<span class="hud-chip">{text}</span>'


def kv_row(label: str, value: str, mono: bool = True) -> str:
    """Two-column key/value row with subtle styling."""
    value_style = 'font-family:var(--mono);color:var(--accent);' if mono else ''
    return f"""
    <div style="display:flex;justify-content:space-between;padding:6px 0;
                border-bottom:1px solid var(--border);font-size:0.82rem;">
        <span style="color:var(--text-muted);font-weight:600;
                     letter-spacing:0.04em;text-transform:uppercase;font-size:0.72rem;">
            {label}
        </span>
        <span style="{value_style}">{value}</span>
    </div>
    """


def alert_box(
    title: str,
    message: str,
    level: str = "info",
    icon: str = "ℹ️"
) -> str:
    """
    Styled alert box.

    Levels: 'info', 'success', 'warning', 'error'
    """
    colors = {
        'info': ('rgba(56, 189, 248, 0.12)', '#38bdf8'),
        'success': ('rgba(16, 185, 129, 0.12)', '#10b981'),
        'warning': ('rgba(245, 158, 11, 0.14)', '#f59e0b'),
        'error': ('rgba(239, 68, 68, 0.14)', '#ef4444'),
    }
    bg, border_color = colors.get(level, colors['info'])
    return f"""
    <div style="background:{bg}; border:1px solid {border_color};
                border-left:3px solid {border_color};
                border-radius:10px; padding:12px 16px; margin:8px 0;
                backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);">
        <div style="font-weight:700; color:{border_color};
                    font-size:0.85rem; letter-spacing:0.05em;
                    text-transform:uppercase; margin-bottom:4px;">
            {icon} {title}
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem;
                    line-height:1.5;">{message}</div>
    </div>
    """


def sparkline_card(
    title: str,
    value: str,
    points: List[float],
    color: Optional[str] = None,
    height: int = 56,
    width: int = 300
) -> str:
    """
    Card with an embedded inline SVG sparkline.
    Lightweight — no Plotly dependency.
    """
    color = color or 'var(--accent)'
    if not points or len(points) < 2:
        path_d = ""
    else:
        n = len(points)
        pmin, pmax = min(points), max(points)
        span = max(pmax - pmin, 1e-6)
        step_x = width / (n - 1)
        path_points = []
        for i, v in enumerate(points):
            x = i * step_x
            y = height - ((v - pmin) / span) * (height - 8) - 4
            path_points.append((x, y))
        path_d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in path_points)

    return f"""
    <div class="hud-card hud-fade-in" style="padding:14px 18px;">
        <div class="hud-card-title">{title}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
            <div class="hud-card-value" style="font-size:1.35rem;">{value}</div>
            <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
                 style="overflow:visible; flex:0 1 auto; max-width:55%;">
                <defs>
                    <linearGradient id="sg-{abs(hash(title))}" x1="0" x2="1" y1="0" y2="0">
                        <stop offset="0%" stop-color="{color}" stop-opacity="0.3"/>
                        <stop offset="100%" stop-color="{color}" stop-opacity="1"/>
                    </linearGradient>
                </defs>
                <path d="{path_d}" fill="none"
                      stroke="url(#sg-{abs(hash(title))})"
                      stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                      style="filter:drop-shadow(0 0 6px {color})"/>
            </svg>
        </div>
    </div>
    """


def progress_bar(
    label: str,
    value: float,
    max_value: float = 1.0,
    color: Optional[str] = None,
    show_percent: bool = True
) -> str:
    """Horizontal progress bar with label."""
    color = color or 'var(--accent)'
    pct = max(0.0, min(100.0, (value / max(max_value, 1e-9)) * 100))
    pct_str = f"{pct:.1f}%" if show_percent else f"{value:.2f}"
    return f"""
    <div style="margin:8px 0;">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.72rem;color:var(--text-muted);
                    text-transform:uppercase;letter-spacing:0.08em;
                    margin-bottom:6px;">
            <span>{label}</span>
            <span style="font-family:var(--mono);color:{color};">{pct_str}</span>
        </div>
        <div style="background:rgba(255,255,255,0.05); height:6px;
                    border-radius:3px; overflow:hidden;">
            <div style="width:{pct}%; height:100%;
                        background:linear-gradient(90deg, {color}, {color}cc);
                        border-radius:3px;
                        box-shadow:0 0 12px {color}66;
                        transition: width 0.3s ease;"></div>
        </div>
    </div>
    """