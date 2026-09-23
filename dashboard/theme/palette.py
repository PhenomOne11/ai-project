"""
Color palettes for the NeuralStream HUD.
Multiple themes provide visual variety while maintaining a cohesive design language.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Palette:
    """A named color palette."""
    name: str
    # Backgrounds (darkest → lightest)
    bg_root: str
    bg_surface: str
    bg_elevated: str
    # Text
    text_primary: str
    text_secondary: str
    text_muted: str
    # Accents
    accent_primary: str       # Main brand color
    accent_secondary: str     # Complementary
    success: str
    warning: str
    danger: str
    info: str
    # Effects
    border: str
    border_active: str
    glow_primary: str
    glow_secondary: str
    grid_line: str
    # Charts
    chart_sequence: List[str]

    def get_accent_by_name(self, name: str) -> str:
        return {
            'primary': self.accent_primary,
            'secondary': self.accent_secondary,
            'success': self.success,
            'warning': self.warning,
            'danger': self.danger,
            'info': self.info,
        }.get(name, self.accent_primary)


CYBER_DARK = Palette(
    name="cyber_dark",
    bg_root="#050810",
    bg_surface="#0a0f1e",
    bg_elevated="#141c2e",
    text_primary="#e2e8f0",
    text_secondary="#94a3b8",
    text_muted="#64748b",
    accent_primary="#38bdf8",       # Cyan
    accent_secondary="#a78bfa",     # Violet
    success="#10b981",
    warning="#f59e0b",
    danger="#ef4444",
    info="#06b6d4",
    border="rgba(56, 189, 248, 0.22)",
    border_active="rgba(56, 189, 248, 0.65)",
    glow_primary="rgba(56, 189, 248, 0.45)",
    glow_secondary="rgba(167, 139, 250, 0.4)",
    grid_line="#1e293b",
    chart_sequence=[
        "#38bdf8", "#a78bfa", "#34d399", "#fbbf24",
        "#f87171", "#22d3ee", "#e879f9", "#94a3b8",
    ],
)

PLASMA = Palette(
    name="plasma",
    bg_root="#0d0410",
    bg_surface="#150820",
    bg_elevated="#221033",
    text_primary="#f3e8ff",
    text_secondary="#c4b5fd",
    text_muted="#8b7ba8",
    accent_primary="#e879f9",
    accent_secondary="#f472b6",
    success="#4ade80",
    warning="#fb923c",
    danger="#f43f5e",
    info="#60a5fa",
    border="rgba(232, 121, 249, 0.22)",
    border_active="rgba(232, 121, 249, 0.65)",
    glow_primary="rgba(232, 121, 249, 0.45)",
    glow_secondary="rgba(244, 114, 182, 0.4)",
    grid_line="#2d1b3d",
    chart_sequence=[
        "#e879f9", "#f472b6", "#c084fc", "#fbbf24",
        "#4ade80", "#60a5fa", "#fb923c", "#a1a1aa",
    ],
)

MONO = Palette(
    name="mono",
    bg_root="#0a0a0a",
    bg_surface="#121212",
    bg_elevated="#1c1c1c",
    text_primary="#e5e5e5",
    text_secondary="#a3a3a3",
    text_muted="#666666",
    accent_primary="#e5e5e5",
    accent_secondary="#a3a3a3",
    success="#4ade80",
    warning="#facc15",
    danger="#f87171",
    info="#93c5fd",
    border="rgba(255, 255, 255, 0.14)",
    border_active="rgba(255, 255, 255, 0.55)",
    glow_primary="rgba(255, 255, 255, 0.3)",
    glow_secondary="rgba(200, 200, 200, 0.3)",
    grid_line="#262626",
    chart_sequence=[
        "#e5e5e5", "#a3a3a3", "#737373", "#fbbf24",
        "#4ade80", "#93c5fd", "#f87171", "#c084fc",
    ],
)

SOLAR = Palette(
    name="solar",
    bg_root="#0a0808",
    bg_surface="#151010",
    bg_elevated="#221818",
    text_primary="#faf5f0",
    text_secondary="#c9b5a6",
    text_muted="#8a7a6a",
    accent_primary="#fbbf24",
    accent_secondary="#fb923c",
    success="#84cc16",
    warning="#f59e0b",
    danger="#ef4444",
    info="#7dd3fc",
    border="rgba(251, 191, 36, 0.22)",
    border_active="rgba(251, 191, 36, 0.65)",
    glow_primary="rgba(251, 191, 36, 0.4)",
    glow_secondary="rgba(251, 146, 60, 0.35)",
    grid_line="#2b1e14",
    chart_sequence=[
        "#fbbf24", "#fb923c", "#84cc16", "#38bdf8",
        "#f87171", "#fde047", "#fb7185", "#a3a3a3",
    ],
)


PALETTES: Dict[str, Palette] = {
    "cyber_dark": CYBER_DARK,
    "plasma": PLASMA,
    "mono": MONO,
    "solar": SOLAR,
}


def get_palette(name: str = "cyber_dark") -> Palette:
    """Retrieve a palette by name, defaulting to cyber_dark."""
    return PALETTES.get(name.lower(), CYBER_DARK)


def list_palettes() -> List[str]:
    return list(PALETTES.keys())