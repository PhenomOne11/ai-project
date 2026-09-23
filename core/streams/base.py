"""
Base classes and iterators for NeuralStream-Core data pipelines.

Defines the DataStream protocol, infinite cycling iterators, and
shared utilities for shape normalization and normalization scaling.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple, Callable


class DataStream(ABC):
    """
    Abstract base class for all data ingestion streams.

    A DataStream yields batches of tensors of consistent shape,
    either indefinitely (synthetic, cycling) or until exhaustion.

    Subclasses must implement `__iter__`. Optionally override
    `__len__` if finite, and `describe()` for diagnostics.
    """

    def __init__(self, batch_size: int = 16, name: str = "unnamed"):
        self.batch_size = batch_size
        self.name = name
        self._iteration_count = 0

    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        ...

    def __next__(self) -> torch.Tensor:
        """Not implemented by default — use `iter(stream)` then `next()`."""
        raise NotImplementedError(
            "Use `iter(stream)` to obtain an iterator."
        )

    def describe(self) -> dict:
        """Basic stream description for the dashboard."""
        return {
            'name': self.name,
            'batch_size': self.batch_size,
            'class': self.__class__.__name__,
            'iterations_served': self._iteration_count,
        }

    def reset(self) -> None:
        """Reset internal state. Subclasses may override."""
        self._iteration_count = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, batch_size={self.batch_size})"


# ---- Iterator helpers ----

class CyclingIterator:
    """
    Wraps an iterable so it cycles forever, re-yielding from the start.

    Usage:
        loader = CyclingIterator(lambda: iter([1, 2, 3]))
        stream = (next(loader) for _ in range(10))  # yields 1,2,3,1,2,3,1,2,3,1
    """

    def __init__(self, factory: Callable[[], Iterator]):
        self._factory = factory
        self._current_iters: Iterator = iter([])

    def __iter__(self) -> "CyclingIterator":
        return self

    def __next__(self):
        try:
            return next(self._current_iters)
        except StopIteration:
            self._current_iters = self._factory()
            return next(self._current_iters)


def infinite_batches(
    tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    rng: Optional[np.random.Generator] = None
) -> Iterator[torch.Tensor]:
    """
    Yield batches from a tensor infinitely, cycling when exhausted.

    Args:
        tensor: Data of shape (N, ...)
        batch_size: Samples per batch
        shuffle: If True, shuffle indices before each epoch
        drop_last: If True, drop incomplete final batch
        rng: Optional random generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng()
    N = tensor.size(0)
    if N == 0:
        raise ValueError("Cannot iterate over empty tensor")

    while True:
        idx = rng.permutation(N) if shuffle else np.arange(N)
        for start in range(0, N, batch_size):
            end = start + batch_size
            if end > N:
                if drop_last:
                    break
                end = N
            batch_idx = idx[start:end]
            yield tensor[batch_idx]


# ---- Normalization / Sanitization ----

def normalize_batch(
    batch: torch.Tensor,
    mode: str = "minmax",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize a batch in-place-safe manner.

    Modes:
        'minmax': scale to [0, 1] using per-batch min/max
        'zscore': zero mean, unit variance per-batch
        'unit': scale to unit L2 norm (per sample)
        'none':   return unchanged
    """
    if mode == "none":
        return batch

    if mode == "minmax":
        bmin = batch.amin()
        bmax = batch.amax()
        if (bmax - bmin) < eps:
            return batch - bmin
        return (batch - bmin) / (bmax - bmin + eps)

    if mode == "zscore":
        mean = batch.mean()
        std = batch.std()
        return (batch - mean) / (std + eps)

    if mode == "unit":
        # Per-sample L2 across all dims except batch
        flat = batch.view(batch.size(0), -1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(eps)
        return (batch.view(batch.size(0), -1) / norms).view_as(batch)

    raise ValueError(f"Unknown normalization mode: {mode}")


def sanitize_batch(
    batch: torch.Tensor,
    fill_nan: float = 0.0,
    clip_magnitude: Optional[float] = 1e6
) -> torch.Tensor:
    """Replace NaN/Inf and optionally clamp magnitude."""
    out = torch.nan_to_num(batch, nan=fill_nan, posinf=fill_nan, neginf=fill_nan)
    if clip_magnitude is not None:
        out = out.clamp(-clip_magnitude, clip_magnitude)
    return out