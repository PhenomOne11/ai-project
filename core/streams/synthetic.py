"""
Synthetic Data Streams
Deterministic and stochastic synthetic generators for testing, demos,
and graceful fallbacks when no external data is available.
"""

import torch
import numpy as np
from typing import Iterator, Optional
from core.streams.base import DataStream, normalize_batch


class SyntheticStream(DataStream):
    """
    Random Gaussian noise stream. Simple, fast, baseline generator.

    Output shape: (batch_size, *shape)
    """

    def __init__(
        self,
        batch_size: int = 16,
        shape: tuple = (1, 64, 64),
        mean: float = 0.0,
        std: float = 1.0,
        normalize: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__(batch_size=batch_size, name="synthetic_gaussian")
        self.shape = tuple(shape)
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            batch = torch.randn(self.batch_size, *self.shape, generator=self._rng)
            batch = batch * self.std + self.mean
            if self.normalize:
                batch = normalize_batch(batch, mode="minmax")
            self._iteration_count += 1
            yield batch

    def describe(self) -> dict:
        d = super().describe()
        d.update({'shape': self.shape, 'mean': self.mean, 'std': self.std})
        return d


class SinusoidalStream(DataStream):
    """
    Synthetic sinusoidal signals — ideal for time-series / signal-processing
    testing. Generates batched (channels, length) sinusoids with random
    frequencies and phases.
    """

    def __init__(
        self,
        batch_size: int = 16,
        channels: int = 1,
        length: int = 64,
        freq_range: tuple = (0.05, 0.4),
        noise_std: float = 0.02,
        seed: Optional[int] = None
    ):
        super().__init__(batch_size=batch_size, name="synthetic_sinusoidal")
        self.channels = channels
        self.length = length
        self.freq_range = freq_range
        self.noise_std = noise_std
        if seed is not None:
            torch.manual_seed(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        t = torch.linspace(0, 1, self.length)
        while True:
            freqs = torch.FloatTensor(self.batch_size, self.channels).uniform_(
                *self.freq_range
            )
            phases = torch.FloatTensor(self.batch_size, self.channels).uniform_(0, 6.28)
            # (B, C, L)
            signal = torch.sin(
                2 * 3.14159 * freqs.unsqueeze(-1) * t.view(1, 1, -1)
                + phases.unsqueeze(-1)
            )
            if self.noise_std > 0:
                signal = signal + self.noise_std * torch.randn_like(signal)
            self._iteration_count += 1
            yield signal


class MixedSignalStream(DataStream):
    """
    Synthetic mix of multiple signal types — sinusoids + spikes + noise.
    Provides richer variety than pure Gaussian for testing adaptation,
    drift detection, and anomaly scoring.
    """

    def __init__(
        self,
        batch_size: int = 16,
        channels: int = 1,
        length: int = 64,
        spike_probability: float = 0.03,
        seed: Optional[int] = None
    ):
        super().__init__(batch_size=batch_size, name="synthetic_mixed")
        self.channels = channels
        self.length = length
        self.spike_probability = spike_probability
        if seed is not None:
            torch.manual_seed(seed)

    def _generate(self) -> torch.Tensor:
        B, C, L = self.batch_size, self.channels, self.length
        t = torch.linspace(0, 1, L)

        # Base signal: low-frequency sine
        base_freq = torch.FloatTensor(B, C).uniform_(0.05, 0.2)
        base = torch.sin(2 * 3.14159 * base_freq.unsqueeze(-1) * t.view(1, 1, -1))

        # Add high-freq component to 50% of samples
        mask = (torch.rand(B, C) < 0.5).float().unsqueeze(-1)
        hf_freq = torch.FloatTensor(B, C).uniform_(0.3, 0.45)
        hf = torch.sin(2 * 3.14159 * hf_freq.unsqueeze(-1) * t.view(1, 1, -1))
        signal = base * (1 - mask * 0.3) + hf * mask * 0.3

        # Random spikes
        spikes = (torch.rand(B, C, L) < self.spike_probability).float()
        signal = signal + spikes * torch.randn(B, C, L) * 3.0

        # Add mild noise
        signal = signal + 0.05 * torch.randn_like(signal)
        return signal

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            batch = self._generate()
            self._iteration_count += 1
            yield batch


class SpikeStream(DataStream):
    """
    Sparse spike train stream. Useful for testing isolation forests
    and extreme tail sensitivity in anomaly detectors.
    """

    def __init__(
        self,
        batch_size: int = 16,
        shape: tuple = (1, 64, 64),
        spike_density: float = 0.02,
        spike_amplitude: float = 5.0,
        seed: Optional[int] = None
    ):
        super().__init__(batch_size=batch_size, name="synthetic_spikes")
        self.shape = tuple(shape)
        self.spike_density = spike_density
        self.spike_amplitude = spike_amplitude
        if seed is not None:
            torch.manual_seed(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            spikes = (
                torch.rand(self.batch_size, *self.shape) < self.spike_density
            ).float()
            values = torch.randn(self.batch_size, *self.shape) * self.spike_amplitude
            batch = spikes * values
            self._iteration_count += 1
            yield batch