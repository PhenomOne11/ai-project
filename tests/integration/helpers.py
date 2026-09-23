"""
Integration test utilities.

Provides lightweight stubs so tests don't need to spin up real processes
or the full Streamlit runtime.
"""

import numpy as np
import torch
from typing import Dict, Any, List


class MockST:
    """
    Minimal Streamlit stub that captures render calls instead of
    emitting HTML. Used to smoke-test panel render() functions.
    """
    def __init__(self):
        self.markdown_calls: List[str] = []
        self.plotly_calls: List[Any] = []
        self.metric_calls: List[tuple] = []
        self.button_returns: List[bool] = []
        self.selectbox_returns: List[Any] = []
        self.slider_returns: List[Any] = []
        self.checkbox_returns: List[Any] = []
        self.number_input_returns: List[Any] = []
        self.multiselect_returns: List[Any] = []
        self.toast_calls: List[str] = []
        self.dataframe_calls: List[Any] = []

        # Default return values
        self._button_value = False
        self._selectbox_value = None
        self._slider_value = None
        self._checkbox_value = False
        self._number_input_value = 0.0
        self._multiselect_value = None

    # ---- Layout ----

    def columns(self, spec, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return [_MockColumn(self) for _ in range(n)]

    def container(self):
        return _MockContainer(self)

    def expander(self, *a, **k):
        return _MockContainer(self)

    # ---- Markdown / HTML ----

    def markdown(self, text, **kwargs):
        self.markdown_calls.append(text)
        return None

    def caption(self, *a, **k):
        return None

    def title(self, *a, **k):
        return None

    def header(self, *a, **k):
        return None

    def subheader(self, *a, **k):
        return None

    # ---- Data ----

    def plotly_chart(self, fig, **kwargs):
        self.plotly_calls.append(fig)
        return None

    def metric(self, label, value, delta=None, **kwargs):
        self.metric_calls.append((label, value, delta))
        return None

    def dataframe(self, data, **kwargs):
        self.dataframe_calls.append(data)
        return None

    def table(self, data, **kwargs):
        return None

    # ---- Inputs ----

    def button(self, label, **kwargs):
        self.button_returns.append(True)
        return self._button_value

    def selectbox(self, label, options=None, **kwargs):
        if options is not None:
            self._selectbox_value = options[0] if options else None
        else:
            self._selectbox_value = kwargs.get('index', 0)
        self.selectbox_returns.append(self._selectbox_value)
        return self._selectbox_value

    def multiselect(self, label, options=None, default=None, **kwargs):
        val = default if default is not None else (options or [])
        self._multiselect_value = val
        self.multiselect_returns.append(val)
        return val

    def slider(self, label, min_value=None, max_value=None, value=None, **kwargs):
        self._slider_value = value if value is not None else min_value
        self.slider_returns.append(self._slider_value)
        return self._slider_value

    def number_input(self, label, min_value=None, max_value=None,
                     value=None, **kwargs):
        self._number_input_value = value if value is not None else min_value
        self.number_input_returns.append(self._number_input_value)
        return self._number_input_value

    def checkbox(self, label, value=False, **kwargs):
        self._checkbox_value = value
        self.checkbox_returns.append(value)
        return value

    def radio(self, label, options=None, index=0, **kwargs):
        return options[index] if options else None

    # ---- Feedback ----

    def success(self, *a, **k):
        return None

    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None

    def toast(self, msg, **kwargs):
        self.toast_calls.append(msg)
        return None

    def spinner(self, *a, **k):
        return _MockContainer(self)

    def progress(self, *a, **k):
        return None

    # ---- Control ----

    def rerun(self):
        return None

    def stop(self):
        return None

    def set_page_config(self, **kwargs):
        return None

    def empty(self):
        return None


class _MockColumn:
    def __init__(self, parent: MockST):
        self._parent = parent

    def __enter__(self):
        return self._parent

    def __exit__(self, *args):
        return False

    # Proxy everything to parent
    def __getattr__(self, name):
        return getattr(self._parent, name)


class _MockContainer:
    def __init__(self, parent: MockST):
        self._parent = parent

    def __enter__(self):
        return self._parent

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return getattr(self._parent, name)


# =============================================================================
# SYNTHETIC DATA HELPERS
# =============================================================================

def synthetic_batch(batch_size: int = 4, channels: int = 1,
                    size: int = 32) -> torch.Tensor:
    return torch.randn(batch_size, channels, size, size)


def synthetic_latent(batch_size: int = 4, dim: int = 16) -> torch.Tensor:
    return torch.randn(batch_size, dim)


def synthetic_waveform(length: int = 512) -> np.ndarray:
    t = np.linspace(0, 1, length)
    return np.sin(2 * np.pi * 8 * t) + 0.1 * np.random.randn(length)