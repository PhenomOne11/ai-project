"""
Responsive grid layout helpers built on top of Streamlit columns.
"""

from typing import List, Optional, Callable


def grid_columns(st_module, columns: int, gap: str = "medium") -> list:
    """
    Return a list of Streamlit column handles.

    Args:
        columns: Number of columns (1-6)
        gap: 'small' | 'medium' | 'large'
    """
    gap_map = {'small': 0.3, 'medium': 0.6, 'large': 1.0}
    g = gap_map.get(gap, 0.6)
    return st_module.columns(columns, gap=gap if gap in gap_map else "medium")


def render_row(st_module, items: List[Callable], use_container_width: bool = True):
    """
    Render a horizontal row of render functions, each in its own column.

    Args:
        items: List of zero-argument callables (each writes to its column)
    """
    if not items:
        return
    cols = st_module.columns(len(items))
    for col, fn in zip(cols, items):
        with col:
            fn()


def split_layout(st_module, ratio: tuple, gap: str = "medium") -> list:
    """
    Split the main content area by a ratio tuple (like [2, 1]).

    Args:
        ratio: List/tuple of relative widths, e.g. [2, 1, 1]
    """
    return st_module.columns(list(ratio), gap=gap)


def centered_block(st_module, width_fraction: float = 0.5):
    """
    Return (left_spacer, center, right_spacer) columns for a centered block.
    """
    side = (1 - width_fraction) / 2
    return st_module.columns([side, width_fraction, side])