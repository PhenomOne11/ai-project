"""
NeuralStream Dashboard Theme Package
"""

from dashboard.theme.palette import (
    Palette, PALETTES, CYBER_DARK, PLASMA, MONO, SOLAR,
    get_palette, list_palettes,
)
from dashboard.theme.css import generate_css, inject_theme
from dashboard.theme.components import (
    status_badge, metric_card, section_header, divider,
    chip, kv_row, alert_box, sparkline_card, progress_bar,
)
from dashboard.theme.plots import (
    apply_theme, accent_color, severity_color,
)

__all__ = [
    # Palette
    'Palette', 'PALETTES', 'CYBER_DARK', 'PLASMA', 'MONO', 'SOLAR',
    'get_palette', 'list_palettes',
    # CSS
    'generate_css', 'inject_theme',
    # Components
    'status_badge', 'metric_card', 'section_header', 'divider',
    'chip', 'kv_row', 'alert_box', 'sparkline_card', 'progress_bar',
    # Plots
    'apply_theme', 'accent_color', 'severity_color',
]