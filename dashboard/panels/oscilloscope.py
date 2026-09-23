"""
Waveform Oscilloscope Panel

Real-time signal visualization for multi-channel waveforms:
    - Live waveform traces with rolling trail
    - FFT power spectrum side-panel (using Phase 2 signal processing)
    - Spectrogram waterfall (sliding STFT magnitude)
    - Peak detection markers
    - Zero-crossing markers
    - Signal statistics: RMS, peak, crest factor, SNR, spectral entropy

Accepts:
    - Direct API: `push_waveform(channel, samples)`
    - Phase 2 integration: uses `compute_fft`, `spectral_entropy`, `signal_to_noise_ratio`
    - Phase 10 integration: pull from `AudioStream` / `TimeSeriesStream`
    - Synthetic: `seed_demo(...)` for demonstration
"""

import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import (
    metric_card, section_header, divider, status_badge,
    apply_theme, accent_color, progress_bar, kv_row, alert_box,
)
from dashboard.theme.palette import Palette

from core.math_utils.signal_processing import (
    compute_fft, spectral_entropy, signal_to_noise_ratio,
    find_peaks, zero_crossing_rate, band_energy,
)


# =============================================================================
# STATE
# =============================================================================

class OscilloscopeState:
    """
    Rolling buffer for multi-channel waveform capture.

    Args:
        channel_names: Display names for each channel
        samples_per_view: How many samples visible in the main scope window
        max_channels: Max channel count
        trail_length: Number of historical waveform frames retained for the waterfall
        waterfall_rebin: FFT bin count for the spectrogram waterfall
    """

    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        samples_per_view: int = 512,
        max_channels: int = 4,
        trail_length: int = 64,
        waterfall_rebin: int = 64,
        sample_rate: float = 44100.0,
    ):
        if channel_names is None:
            channel_names = ["ch-0", "ch-1"]

        self.channel_names = list(channel_names)[:max_channels]
        self.samples_per_view = samples_per_view
        self.max_channels = max_channels
        self.trail_length = trail_length
        self.waterfall_rebin = waterfall_rebin
        self.sample_rate = sample_rate

        # channel_name → deque of float samples
        self._channels: Dict[str, deque] = {
            name: deque(maxlen=samples_per_view) for name in self.channel_names
        }
        # channel_name → deque of FFT magnitudes (for waterfall)
        self._waterfalls: Dict[str, deque] = {
            name: deque(maxlen=trail_length) for name in self.channel_names
        }

        self._n_pushed: int = 0

    # ---- Ingestion ----

    def push_waveform(self, channel: str, samples: np.ndarray) -> None:
        """
        Push a fresh window of waveform samples for a channel.
        Only the most recent `samples_per_view` are retained.
        """
        if channel not in self._channels:
            if len(self.channel_names) >= self.max_channels:
                return
            self.channel_names.append(channel)
            self._channels[channel] = deque(maxlen=self.samples_per_view)
            self._waterfalls[channel] = deque(maxlen=self.trail_length)

        arr = np.asarray(samples, dtype=np.float64).flatten()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Retain last samples_per_view values
        take = arr[-self.samples_per_view:]
        self._channels[channel].clear()
        for v in take:
            self._channels[channel].append(float(v))
        self._n_pushed += 1

        # Update waterfall
        self._update_waterfall(channel)

    def _update_waterfall(self, channel: str) -> None:
        """Compute FFT magnitude and store as a waterfall row."""
        samples = list(self._channels[channel])
        if len(samples) < 16:
            return
        _, mags = compute_fft(np.array(samples), sample_rate=self.sample_rate)
        if mags.size == 0:
            return
        # Rebin to waterfall_rebin bins
        rebinned = self._rebin_spectrum(mags, self.waterfall_rebin)
        self._waterfalls[channel].append(rebinned)

    @staticmethod
    def _rebin_spectrum(spec: np.ndarray, target_bins: int) -> np.ndarray:
        """Resample a magnitude spectrum to a fixed number of bins."""
        src = np.asarray(spec, dtype=np.float64)
        n = src.size
        if n == 0:
            return np.zeros(target_bins)
        if n == target_bins:
            return src
        # Simple block-average rebinning
        out = np.zeros(target_bins)
        for i in range(target_bins):
            a = int(np.floor(i * n / target_bins))
            b = int(np.floor((i + 1) * n / target_bins))
            if b <= a:
                b = min(a + 1, n)
            out[i] = float(src[a:b].mean())
        return out

    # ---- Retrieval ----

    def get_waveform(self, channel: str) -> np.ndarray:
        return np.array(self._channels.get(channel, []), dtype=np.float64)

    def get_waterfall(self, channel: str) -> np.ndarray:
        """2D matrix (rows = history frames, cols = bins)."""
        rows = list(self._waterfalls.get(channel, []))
        if not rows:
            return np.zeros((0, self.waterfall_rebin))
        return np.stack(rows, axis=0)

    @property
    def size(self) -> int:
        return self._n_pushed

    @property
    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self._channels.values())

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    # ---- Statistics ----

    def channel_stats(self, channel: str) -> Dict[str, Any]:
        """Full signal analysis for one channel."""
        samples = self.get_waveform(channel)
        if samples.size < 8:
            return self._empty_stats()

        # Basic
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples ** 2)))
        crest = peak / max(rms, 1e-9)
        zcr = float(zero_crossing_rate(samples))

        # Spectral
        se = spectral_entropy(samples)
        try:
            snr = float(signal_to_noise_ratio(samples))
        except Exception:
            snr = 0.0

        # Peaks
        peak_idx, peak_vals = find_peaks(samples, min_distance=5)
        n_peaks = len(peak_idx)

        # Band energies
        try:
            low_energy = band_energy(samples, self.sample_rate,
                                     low_freq=0, high_freq=self.sample_rate * 0.15)
            mid_energy = band_energy(samples, self.sample_rate,
                                     low_freq=self.sample_rate * 0.15,
                                     high_freq=self.sample_rate * 0.5)
            high_energy = band_energy(samples, self.sample_rate,
                                      low_freq=self.sample_rate * 0.5,
                                      high_freq=self.sample_rate * 0.5 + 1)
        except Exception:
            low_energy = mid_energy = high_energy = 0.0

        return {
            'peak': peak,
            'rms': rms,
            'crest_factor': crest,
            'zcr': zcr,
            'snr_db': snr,
            'spectral_entropy': se,
            'num_peaks': n_peaks,
            'low_band': float(low_energy),
            'mid_band': float(mid_energy),
            'high_band': float(high_energy),
            'samples': int(samples.size),
        }

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            'peak': 0.0, 'rms': 0.0, 'crest_factor': 0.0, 'zcr': 0.0,
            'snr_db': 0.0, 'spectral_entropy': 0.0, 'num_peaks': 0,
            'low_band': 0.0, 'mid_band': 0.0, 'high_band': 0.0, 'samples': 0,
        }

    def global_stats(self) -> Dict[str, Any]:
        """Aggregate across channels."""
        if self.is_empty:
            return {'num_channels': self.num_channels, 'active_channels': 0,
                    'total_pushed': 0, 'mean_rms': 0.0, 'max_peak': 0.0}

        stats = [self.channel_stats(ch) for ch in self.channel_names]
        active = sum(1 for s in stats if s['samples'] > 0)
        rms_vals = [s['rms'] for s in stats if s['samples'] > 0]
        peaks = [s['peak'] for s in stats if s['samples'] > 0]

        return {
            'num_channels': self.num_channels,
            'active_channels': active,
            'total_pushed': self._n_pushed,
            'mean_rms': float(np.mean(rms_vals)) if rms_vals else 0.0,
            'max_peak': float(np.max(peaks)) if peaks else 0.0,
        }

    # ---- Demo ----

    def seed_demo(self, n_frames: int = 60, seed: int = 42) -> None:
        """
        Populate the scope with three signal archetypes:
            ch-0: Pure sine (220 Hz → 440 Hz transition)
            ch-1: Sine + noise (SNR ~15 dB)
            ch-2: Multi-tone composite (220 + 440 + 880 Hz)
        """
        rng = np.random.default_rng(seed)
        self._channels.clear()
        self._waterfalls.clear()
        self.channel_names = ["ch-0", "ch-1", "ch-2"]
        for name in self.channel_names:
            self._channels[name] = deque(maxlen=self.samples_per_view)
            self._waterfalls[name] = deque(maxlen=self.trail_length)

        n = self.samples_per_view
        t = np.linspace(0, 1, n)

        for frame in range(n_frames):
            # Phase sweeps slightly each frame for visual variety
            phase = frame * 0.05

            # ch-0: pure sine with slight frequency drift
            f0 = 220.0 + 30.0 * np.sin(frame * 0.1)
            s0 = np.sin(2 * np.pi * f0 / self.sample_rate * np.arange(n) + phase)

            # ch-1: sine + noise
            s1 = 0.8 * np.sin(2 * np.pi * 440 / self.sample_rate * np.arange(n))
            s1 = s1 + 0.15 * rng.standard_normal(n)

            # ch-2: three-tone composite
            s2 = (
                np.sin(2 * np.pi * 220 / self.sample_rate * np.arange(n))
                + 0.5 * np.sin(2 * np.pi * 440 / self.sample_rate * np.arange(n))
                + 0.3 * np.sin(2 * np.pi * 880 / self.sample_rate * np.arange(n))
            ) / 1.8

            self.push_waveform("ch-0", s0)
            self.push_waveform("ch-1", s1)
            self.push_waveform("ch-2", s2)

    def reset(self) -> None:
        for k in self._channels:
            self._channels[k].clear()
        for k in self._waterfalls:
            self._waterfalls[k].clear()
        self._n_pushed = 0


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_waveform_traces(
    waveforms: Dict[str, np.ndarray],
    palette: Palette = None,
    height: int = 320,
    show_peaks: bool = True,
    show_zero_crossings: bool = True,
) -> go.Figure:
    """
    Main scope view: overlay of all channel waveforms with optional
    peak markers and zero-crossing lines.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not waveforms:
        apply_theme(fig, p, height=height)
        return fig

    for i, (name, samples) in enumerate(waveforms.items()):
        if samples.size == 0:
            continue
        color = accent_color(p, i)
        x = np.arange(samples.size)

        fig.add_trace(go.Scatter(
            x=x, y=samples, mode="lines",
            name=name,
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{name}</b><br>n=%{{x}}<br>v=%{{y:.4f}}<extra></extra>",
        ))

        # Peak markers
        if show_peaks and samples.size >= 16:
            peak_idx, peak_vals = find_peaks(samples, min_distance=8)
            if len(peak_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=peak_idx, y=peak_vals, mode="markers",
                    name=f"{name} peaks",
                    marker=dict(color=color, size=5, symbol="diamond",
                                line=dict(width=0)),
                    showlegend=False,
                    hovertemplate=f"peak %{{x}}<br>v=%{{y:.4f}}<extra></extra>",
                ))

        # Zero-crossing markers (light, only for single channel)
        if show_zero_crossings and len(waveforms) == 1:
            signs = np.sign(samples)
            zc_idx = np.where(np.diff(signs) != 0)[0]
            if len(zc_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=zc_idx, y=np.zeros_like(zc_idx), mode="markers",
                    name="zero-X",
                    marker=dict(color=p.text_muted, size=4, symbol="x"),
                    showlegend=False, hoverinfo="skip",
                ))

    # Zero line
    fig.add_hline(y=0.0, line=dict(color=p.text_muted, width=1, dash="dot"))

    fig.update_layout(
        title="Live Waveform Scope",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
    )
    apply_theme(fig, p, height=height)
    return fig


def build_fft_spectrum(
    waveforms: Dict[str, np.ndarray],
    palette: Palette = None,
    sample_rate: float = 44100.0,
    height: int = 320,
    log_scale: bool = True,
) -> go.Figure:
    """
    FFT power spectrum for each channel.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if not waveforms:
        apply_theme(fig, p, height=height)
        return fig

    for i, (name, samples) in enumerate(waveforms.items()):
        if samples.size < 16:
            continue
        color = accent_color(p, i)
        freqs, mags = compute_fft(samples, sample_rate=sample_rate)
        if freqs.size == 0:
            continue
        y = np.log1p(mags) if log_scale else mags
        y_label = "log-magnitude" if log_scale else "magnitude"
        fig.add_trace(go.Scatter(
            x=freqs, y=y, mode="lines",
            name=name,
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{name}</b><br>%{{x:.1f}} Hz<br>{y_label}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="FFT Power Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="log-magnitude" if log_scale else "magnitude",
    )
    apply_theme(fig, p, height=height)
    return fig


def build_waterfall(
    waterfall: np.ndarray,
    palette: Palette = None,
    height: int = 320,
    channel_name: str = "spectrum",
) -> go.Figure:
    """
    Spectrogram waterfall: rows = historical FFT frames, cols = frequency bins.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    if waterfall.size == 0:
        apply_theme(fig, p, height=height)
        return fig

    # Use log scale for visual contrast
    z = np.log1p(waterfall)

    colorscale = [
        [0.0, p.bg_surface],
        [0.25, _hex_to_rgba(p.accent_primary, 0.35)],
        [0.6, p.accent_primary],
        [0.85, p.accent_secondary],
        [1.0, p.warning],
    ]

    fig.add_trace(go.Heatmap(
        z=z,
        colorscale=colorscale,
        showscale=False,
        xgap=0, ygap=0,
        hovertemplate="bin %{x}<br>frame %{y}<br>log|X|=%{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Spectrogram Waterfall · {channel_name}",
        xaxis_title="Frequency Bin",
        yaxis_title="Frame (newest top)",
    )
    apply_theme(fig, p, height=height)
    fig.update_yaxes(autorange="reversed")
    return fig


def build_band_energy_bar(
    stats: Dict[str, Any],
    palette: Palette = None,
    height: int = 240,
) -> go.Figure:
    """
    Horizontal bar chart of low/mid/high band energy fractions.
    """
    from dashboard.theme import get_palette
    p = palette or get_palette("cyber_dark")

    fig = go.Figure()
    bands = ['low', 'mid', 'high']
    values = [stats.get('low_band', 0.0),
              stats.get('mid_band', 0.0),
              stats.get('high_band', 0.0)]
    colors = [p.success, p.warning, p.danger]

    fig.add_trace(go.Bar(
        y=[b.upper() for b in bands],
        x=values,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=p.text_secondary, family="JetBrains Mono, monospace", size=10),
        hovertemplate="%{y}<br>energy: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Band Energy Distribution",
        xaxis_title="Fraction",
        xaxis=dict(range=[0, max(max(values) * 1.15, 0.1)]),
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
    state: OscilloscopeState,
    key_prefix: str = "oscilloscope",
    show_demo_button: bool = True,
) -> None:
    """Render the oscilloscope panel inside a Streamlit context."""
    st.markdown(section_header("📻", "Waveform Oscilloscope"), unsafe_allow_html=True)

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        active_ch = st.selectbox(
            "Active channel",
            options=state.channel_names,
            index=0,
            key=f"{key_prefix}_active_ch",
        )

    with ctrl2:
        view = st.selectbox(
            "Focus view",
            options=["overlay", "active-only"],
            index=0,
            key=f"{key_prefix}_view",
        )

    with ctrl3:
        height = st.slider(
            "Chart height",
            min_value=240, max_value=600, value=320, step=20,
            key=f"{key_prefix}_height",
        )

    with ctrl4:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key=f"{key_prefix}_reset", use_container_width=True):
            state.reset()
            st.toast("Oscilloscope buffers cleared.", icon="🧹")

    if show_demo_button and state.is_empty:
        if st.button("🌱 Seed demo waveforms (3 channels)",
                     key=f"{key_prefix}_seed"):
            state.seed_demo(60)
            st.rerun()

    if state.is_empty:
        st.info(
            "No waveforms captured yet. Boot the engine, or click "
            "**Seed demo waveforms** above."
        )
        return

    # ---- Data ----
    gstats = state.global_stats()
    active_stats = state.channel_stats(active_ch)

    # ---- Top Metric Row ----
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(metric_card(
            "Active Channel",
            active_ch,
            f"{active_stats['samples']} samples",
        ), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_card(
            "RMS",
            f"{active_stats['rms']:.4f}",
            f"peak {active_stats['peak']:.4f}",
        ), unsafe_allow_html=True)

    with m3:
        crest = active_stats['crest_factor']
        crest_accent = (palette.warning if crest > 4.0 else palette.success)
        st.markdown(metric_card(
            "Crest Factor",
            f"{crest:.3f}",
            "peak / RMS",
            accent=crest_accent,
        ), unsafe_allow_html=True)

    with m4:
        se = active_stats['spectral_entropy']
        se_accent = (palette.warning if se > 0.75 else palette.success)
        st.markdown(metric_card(
            "Spectral Entropy",
            f"{se:.4f}",
            "0 = tonal · 1 = noise",
            accent=se_accent,
        ), unsafe_allow_html=True)

    with m5:
        st.markdown(metric_card(
            "SNR",
            f"{active_stats['snr_db']:.2f} dB",
            f"{active_stats['num_peaks']} peaks",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Data for figures ----
    if view == "active-only":
        waveforms = {active_ch: state.get_waveform(active_ch)}
    else:
        waveforms = {name: state.get_waveform(name) for name in state.channel_names}

    # ---- Main waveform ----
    fig_wave = build_waveform_traces(
        waveforms=waveforms,
        palette=palette,
        height=height,
        show_peaks=True,
        show_zero_crossings=(view == "active-only"),
    )
    st.plotly_chart(fig_wave, use_container_width=True, key=f"{key_prefix}_wave")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- FFT | Waterfall ----
    left, right = st.columns(2)

    with left:
        fig_fft = build_fft_spectrum(
            waveforms=waveforms,
            palette=palette,
            sample_rate=state.sample_rate,
            height=320,
            log_scale=True,
        )
        st.plotly_chart(fig_fft, use_container_width=True, key=f"{key_prefix}_fft")

    with right:
        waterfall = state.get_waterfall(active_ch)
        fig_wf = build_waterfall(
            waterfall=waterfall,
            palette=palette,
            height=320,
            channel_name=active_ch,
        )
        st.plotly_chart(fig_wf, use_container_width=True, key=f"{key_prefix}_wf")

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Band energy | Channel stats ----
    stat_left, stat_right = st.columns([1, 1])

    with stat_left:
        fig_band = build_band_energy_bar(active_stats, palette, height=240)
        st.plotly_chart(fig_band, use_container_width=True, key=f"{key_prefix}_band")

    with stat_right:
        st.markdown(section_header("📊", "Signal Statistics"), unsafe_allow_html=True)

        st.markdown(kv_row("RMS", f"{active_stats['rms']:.5f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Peak", f"{active_stats['peak']:.5f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Crest factor", f"{active_stats['crest_factor']:.4f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Zero-crossing rate", f"{active_stats['zcr']:.5f}"), unsafe_allow_html=True)
        st.markdown(kv_row("Spectral entropy", f"{active_stats['spectral_entropy']:.4f}"), unsafe_allow_html=True)
        st.markdown(kv_row("SNR", f"{active_stats['snr_db']:.3f} dB"), unsafe_allow_html=True)

        # Health flag
        st.markdown("<br>", unsafe_allow_html=True)
        if active_stats['spectral_entropy'] > 0.85:
            st.markdown(status_badge("BROADBAND (noisy)", "paused", "●"),
                        unsafe_allow_html=True)
        elif active_stats['crest_factor'] > 5.0:
            st.markdown(status_badge("HIGH CREST (spiky)", "paused", "●"),
                        unsafe_allow_html=True)
        else:
            st.markdown(status_badge("CLEAN SIGNAL", "active", "●"),
                        unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # ---- Channel strip: per-channel quick stats ----
    st.markdown(section_header("🎛️", "Channel Strip"), unsafe_allow_html=True)

    cols = st.columns(min(state.num_channels, 4))
    for i, name in enumerate(state.channel_names):
        col = cols[i % len(cols)]
        s = state.channel_stats(name)
        color = accent_color(palette, i)
        with col:
            st.markdown(f"""
            <div class="hud-card hud-fade-in" style="padding:12px 14px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:var(--mono); font-weight:700;
                                 color:{color};">{name}</span>
                    <span style="font-size:0.7rem; color:var(--text-muted);
                                 font-family:var(--mono);">
                        {s['samples']} smp
                    </span>
                </div>
                <div style="margin-top:8px; font-family:var(--mono);
                            font-size:0.75rem; color:var(--text-secondary);">
                    RMS {s['rms']:.4f} · pk {s['peak']:.4f}
                </div>
                <div style="margin-top:4px; font-family:var(--mono);
                            font-size:0.7rem; color:var(--text-muted);">
                    H {s['spectral_entropy']:.3f} · SNR {s['snr_db']:.1f}dB
                </div>
            </div>
            """, unsafe_allow_html=True)