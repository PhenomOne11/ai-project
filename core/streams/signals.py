"""
Signal Ingestion Streams
Specialized streams for audio waveforms and time-series windowing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Iterator, Optional, Union, List
from pathlib import Path

from core.streams.base import DataStream, infinite_batches, sanitize_batch


class AudioStream(DataStream):
    """
    Audio waveform stream. Converts raw audio into either:

        (a) 1D waveform windows (mode='waveform')
        (b) 2D mel-like spectrograms (mode='spectrogram')

    Compatible with both signal models (waveform) and image models
    (spectrogram).
    """

    def __init__(
        self,
        waveform: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
        window_length: int = 4096,
        batch_size: int = 16,
        mode: str = "spectrogram",
        spectrogram_size: int = 64,
        normalize: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__(batch_size=batch_size, name=f"audio_{mode}")
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.mode = mode
        self.spectrogram_size = spectrogram_size
        self.normalize = normalize

        if waveform is None:
            waveform = self._synthesize()

        # Ensure 1D waveform
        if waveform.dim() > 1:
            waveform = waveform.flatten()

        self.waveform = waveform.float()
        self._rng = np.random.default_rng(seed)

    def _synthesize(self) -> torch.Tensor:
        """Generate a synthetic multi-tone waveform if no file provided."""
        sr = self.sample_rate
        duration = 2.0
        t = torch.linspace(0, duration, int(sr * duration // 4))

        # Multi-tone composite
        tone = (torch.sin(2 * 3.14159 * 220 * t)
                + 0.5 * torch.sin(2 * 3.14159 * 440 * t)
                + 0.3 * torch.sin(2 * 3.14159 * 880 * t))
        # Envelope
        env = torch.exp(-torch.linspace(0, 3, len(t)))
        return (tone * env).float()

    def _extract_window(self, start: int) -> torch.Tensor:
        """Extract one window from the waveform."""
        L = self.window_length
        end = start + L
        if end <= len(self.waveform):
            return self.waveform[start:end]
        # Wrap-around for cycling
        head = self.waveform[start:]
        tail = self.waveform[: end - len(self.waveform)]
        return torch.cat([head, tail])

    def _window_to_spectrogram(self, window: torch.Tensor) -> torch.Tensor:
        """Convert 1D window → 2D log-magnitude spectrogram."""
        # STFT
        n_fft = min(256, window.numel())
        hop = n_fft // 4
        stft = torch.stft(
            window, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
            return_complex=True
        )
        spec = stft.abs()  # (freq, time)
        # Log scale and resize
        spec = torch.log1p(spec)
        spec = spec.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
        spec = F.interpolate(
            spec,
            size=(self.spectrogram_size, self.spectrogram_size),
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)
        return spec

    def __iter__(self) -> Iterator[torch.Tensor]:
        wave_len = len(self.waveform)
        while True:
            if self.mode == "waveform":
                # (B, 1, window_length)
                batch = torch.stack([
                    self._extract_window(
                        self._rng.integers(0, wave_len)
                    ).unsqueeze(0)
                    for _ in range(self.batch_size)
                ], dim=0)
            else:
                # (B, 1, spec_size, spec_size)
                batch = torch.stack([
                    self._window_to_spectrogram(
                        self._extract_window(
                            self._rng.integers(0, wave_len)
                        )
                    ).unsqueeze(0)
                    for _ in range(self.batch_size)
                ], dim=0)

            if self.normalize:
                bmin = batch.amin()
                bmax = batch.amax()
                if float(bmax - bmin) > 1e-8:
                    batch = (batch - bmin) / (bmax - bmin)

            batch = sanitize_batch(batch)
            self._iteration_count += 1
            yield batch


class TimeSeriesStream(DataStream):
    """
    Fixed-length windowing stream over a 1D or multi-channel time-series.

    Yields windows of shape (batch_size, channels, window_length).
    Supports both single-channel (1D input) and multi-channel (2D) series.
    """

    def __init__(
        self,
        series: Union[torch.Tensor, np.ndarray, None] = None,
        window_length: int = 64,
        channels: int = 1,
        stride: int = 1,
        batch_size: int = 16,
        normalize: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__(batch_size=batch_size, name="timeseries")
        self.window_length = window_length
        self.channels = channels
        self.stride = stride
        self.normalize = normalize

        if series is None:
            series = self._synthesize()
        if isinstance(series, np.ndarray):
            series = torch.from_numpy(series)
        series = series.float()

        # Coerce to shape (channels, time)
        if series.dim() == 1:
            series = series.unsqueeze(0)  # (1, T)
        elif series.dim() == 2:
            # Assume (N, C) or (C, T). Heuristic: shorter dim is channels.
            if series.shape[0] < series.shape[1]:
                series = series  # (C, T)
            else:
                series = series.t()  # (T, C) → (C, T)

        self.series = series
        self._rng = np.random.default_rng(seed)

    def _synthesize(self) -> torch.Tensor:
        t = torch.linspace(0, 20, 4096)
        chans = []
        for c in range(self.channels):
            freq = 0.3 + 0.2 * c
            chans.append(
                torch.sin(2 * 3.14159 * freq * t)
                + 0.2 * torch.sin(2 * 3.14159 * (freq * 3.7) * t)
                + 0.05 * torch.randn_like(t)
            )
        return torch.stack(chans, dim=0)  # (C, T)

    def __iter__(self) -> Iterator[torch.Tensor]:
        _, T = self.series.shape
        L = self.window_length
        if L > T:
            # Pad if needed
            pad = torch.zeros(self.channels, L - T)
            series = torch.cat([self.series, pad], dim=1)
        else:
            series = self.series

        while True:
            starts = self._rng.integers(0, max(T - L, 1), size=self.batch_size)
            batch = torch.stack(
                [series[:, s:s+L] for s in starts],
                dim=0
            )  # (B, C, L)

            if self.normalize:
                mean = batch.mean(dim=-1, keepdim=True)
                std = batch.std(dim=-1, keepdim=True).clamp_min(1e-8)
                batch = (batch - mean) / std

            batch = sanitize_batch(batch)
            self._iteration_count += 1
            yield batch