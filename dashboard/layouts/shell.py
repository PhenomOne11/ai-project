"""
App shell builder.
Renders the top-level page structure: header, sidebar, main content tabs.
"""

from typing import Optional, Callable, List, Tuple
from dashboard.theme.palette import Palette
from dashboard.theme.css import inject_theme
from dashboard.theme.components import (
    section_header, status_badge, divider, alert_box
)


def init_page(st_module, title: str, icon: str = "⚡", palette: Palette = None):
    """
    Configure page settings and inject the theme CSS.

    Call ONCE at the top of app.py.
    """
    from dashboard.theme.palette import get_palette

    if palette is None:
        palette = get_palette("cyber_dark")

    st_module.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme(st_module, palette)
    return palette


def render_header(st_module, title: str, subtitle: str = "", state_text: str = "READY", state_level: str = "info"):
    """
    Render the top-of-page HUD header.

    Args:
        title: Main title string
        subtitle: Optional smaller caption
        state_text: Right-aligned status text
        state_level: 'info' | 'active' | 'paused' | 'critical'
    """
    badge = status_badge(state_text, state_level, icon="●")
    subtitle_html = (
        f'<div style="color:var(--text-secondary);font-size:0.85rem;margin-top:4px;">'
        f'{subtitle}</div>' if subtitle else ''
    )

    html = f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;
                padding: 8px 4px 20px 4px;border-bottom: 1px solid var(--border);">
        <div>
            <h1 style="margin:0; font-size:1.85rem; letter-spacing:-0.02em;
                       background: linear-gradient(90deg,
                                  var(--accent) 0%,
                                  var(--accent-2) 100%);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       background-clip: text;">
                {title}
            </h1>
            {subtitle_html}
        </div>
        <div style="padding-top: 8px;">
            {badge}
        </div>
    </div>
    """
    st_module.markdown(html, unsafe_allow_html=True)


def build_shell(
    st_module,
    title: str,
    icon: str = "⚡",
    subtitle: str = "",
    palette: Palette = None
) -> Palette:
    """
    High-level: init page + render header. Returns the active palette.
    """
    pal = init_page(st_module, title, icon=icon, palette=palette)
    render_header(st_module, title, subtitle=subtitle)
    return pal


def sidebar_section(st_module, icon: str, title: str):
    """Render a styled section header inside the sidebar."""
    st_module.markdown(
        section_header(icon, title),
        unsafe_allow_html=True
    )


def sidebar_divider(st_module):
    st_module.markdown(divider(), unsafe_allow_html=True)


def sync_engine_state(state: str) -> Tuple[str, str]:
    """
    Map an engine state string → (level, display_text) for badges.

    Engine states (from constants.EngineState):
        UNINITIALIZED, BOOTING, CALIBRATING, IDLE, STREAMING,
        PAUSED, ADAPTING, FAULT, SHUTDOWN
    """
    mapping = {
        'UNINITIALIZED': ('neutral', 'OFFLINE'),
        'BOOTING': ('info', 'BOOTING'),
        'CALIBRATING': ('info', 'CALIBRATING'),
        'IDLE': ('info', 'IDLE'),
        'STREAMING': ('active', 'STREAMING'),
        'PAUSED': ('paused', 'PAUSED'),
        'ADAPTING': ('info', 'ADAPTING'),
        'FAULT': ('critical', 'FAULT'),
        'SHUTDOWN': ('neutral', 'SHUTDOWN'),
    }
    return mapping.get(state.upper(), ('neutral', state.upper()))