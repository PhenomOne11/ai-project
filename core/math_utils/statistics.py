"""
Online Statistical Accumulators
Numerically stable streaming statistics without storing raw history.

Implements:
- Welford's algorithm for online mean/variance
- Exponential Moving Average (EMA) with bias correction
- Rolling percentile estimator (P² algorithm inspired)
- Multi-channel statistical tracker
- Histogram accumulator
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class WelfordAccumulator:
    """
    Welford's online algorithm for numerically stable 
    streaming mean and variance computation.
    
    Complexity: O(1) per update, O(1) memory.
    
    Reference: Welford (1962), "Note on a Method for 
    Calculating Corrected Sums of Squares and Products"
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0          # Sum of squared differences from mean
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, value: float) -> None:
        """Incorporate a single new observation."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def update_batch(self, values: np.ndarray) -> None:
        """Incorporate a batch of observations efficiently."""
        for v in values.flat:
            self.update(float(v))
    
    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count
    
    @property
    def sample_variance(self) -> float:
        """Unbiased sample variance (Bessel's correction)."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)
    
    @property
    def std(self) -> float:
        """Population standard deviation."""
        return np.sqrt(self.variance)
    
    @property
    def sample_std(self) -> float:
        """Sample standard deviation."""
        return np.sqrt(self.sample_variance)
    
    @property
    def range(self) -> float:
        """Observed value range."""
        if self.count == 0:
            return 0.0
        return self.max_val - self.min_val
    
    def to_dict(self) -> Dict[str, float]:
        """Export current state as dictionary."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'sample_std': self.sample_std,
            'variance': self.variance,
            'min': self.min_val,
            'max': self.max_val,
            'range': self.range
        }
    
    def reset(self) -> None:
        """Reset all accumulators to initial state."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')


@dataclass
class WelfordVectorAccumulator:
    """
    Vectorized Welford accumulator for multi-dimensional streams.
    Tracks per-dimension mean and variance simultaneously.
    """
    dim: int = 0
    count: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    _initialized: bool = False
    
    def _init_arrays(self, dim: int) -> None:
        """Lazy initialization on first data arrival."""
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_vals = np.full(dim, float('inf'), dtype=np.float64)
        self.max_vals = np.full(dim, float('-inf'), dtype=np.float64)
        self._initialized = True
    
    def update(self, vector: np.ndarray) -> None:
        """Incorporate a single d-dimensional observation."""
        v = vector.flatten().astype(np.float64)
        if not self._initialized:
            self._init_arrays(len(v))
        
        self.count += 1
        delta = v - self.mean
        self.mean += delta / self.count
        delta2 = v - self.mean
        self.m2 += delta * delta2
        self.min_vals = np.minimum(self.min_vals, v)
        self.max_vals = np.maximum(self.max_vals, v)
    
    def update_batch(self, matrix: np.ndarray) -> None:
        """Incorporate N observations of shape (N, D)."""
        if matrix.ndim == 1:
            self.update(matrix)
            return
        for row in matrix:
            self.update(row)
    
    @property
    def variance(self) -> np.ndarray:
        if self.count < 2 or not self._initialized:
            return np.zeros(self.dim, dtype=np.float64)
        return self.m2 / self.count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)
    
    def reset(self) -> None:
        self.count = 0
        self._initialized = False
        self.mean = None
        self.m2 = None
        self.min_vals = None
        self.max_vals = None


class ExponentialMovingAverage:
    """
    Exponential Moving Average with optional bias correction.
    
    EMA formula:  s_t = α * x_t + (1 - α) * s_{t-1}
    Bias-corrected: ŝ_t = s_t / (1 - (1-α)^t)
    """
    
    def __init__(self, alpha: float = 0.1, bias_correction: bool = True):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.bias_correction = bias_correction
        self.value: float = 0.0
        self.count: int = 0
        self._raw: float = 0.0
    
    def update(self, observation: float) -> float:
        """Update EMA with new observation and return corrected value."""
        self.count += 1
        self._raw = self.alpha * observation + (1.0 - self.alpha) * self._raw
        
        if self.bias_correction and self.count > 0:
            correction = 1.0 - (1.0 - self.alpha) ** self.count
            self.value = self._raw / max(correction, 1e-12)
        else:
            self.value = self._raw
        
        return self.value
    
    def reset(self) -> None:
        self.value = 0.0
        self.count = 0
        self._raw = 0.0


class ExponentialMovingVector:
    """Vectorized EMA for multi-dimensional signals."""
    
    def __init__(self, alpha: float = 0.1, bias_correction: bool = True):
        self.alpha = alpha
        self.bias_correction = bias_correction
        self.count: int = 0
        self._raw: Optional[np.ndarray] = None
        self.value: Optional[np.ndarray] = None
    
    def update(self, vector: np.ndarray) -> np.ndarray:
        v = vector.flatten().astype(np.float64)
        self.count += 1
        
        if self._raw is None:
            self._raw = np.zeros_like(v)
        
        self._raw = self.alpha * v + (1.0 - self.alpha) * self._raw
        
        if self.bias_correction:
            correction = 1.0 - (1.0 - self.alpha) ** self.count
            self.value = self._raw / max(correction, 1e-12)
        else:
            self.value = self._raw.copy()
        
        return self.value
    
    def reset(self) -> None:
        self.count = 0
        self._raw = None
        self.value = None


class RollingWindow:
    """
    Fixed-size rolling window with efficient percentile computation.
    Maintains a circular buffer for O(1) insertion.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.buffer: List[float] = []
        self.position: int = 0
        self.full: bool = False
    
    def update(self, value: float) -> None:
        """Add value to rolling window."""
        if not self.full:
            self.buffer.append(value)
            if len(self.buffer) >= self.window_size:
                self.full = True
                self.position = 0
        else:
            self.buffer[self.position] = value
            self.position = (self.position + 1) % self.window_size
    
    def update_batch(self, values) -> None:
        """Add multiple values."""
        for v in values:
            self.update(float(v))
    
    @property
    def values(self) -> np.ndarray:
        return np.array(self.buffer, dtype=np.float64)
    
    @property
    def count(self) -> int:
        return len(self.buffer)
    
    def percentile(self, q: float) -> float:
        """Compute q-th percentile of current window."""
        if not self.buffer:
            return 0.0
        return float(np.percentile(self.buffer, q))
    
    def mean(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.mean(self.buffer))
    
    def std(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        return float(np.std(self.buffer))
    
    def median(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.median(self.buffer))
    
    def min(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.min(self.buffer))
    
    def max(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.max(self.buffer))
    
    def iqr(self) -> float:
        """Interquartile range."""
        if len(self.buffer) < 4:
            return 0.0
        return self.percentile(75) - self.percentile(25)
    
    def summary(self) -> Dict[str, float]:
        return {
            'count': self.count,
            'mean': self.mean(),
            'std': self.std(),
            'min': self.min(),
            'p25': self.percentile(25),
            'p50': self.median(),
            'p75': self.percentile(75),
            'p95': self.percentile(95),
            'p99': self.percentile(99),
            'max': self.max(),
            'iqr': self.iqr()
        }
    
    def reset(self) -> None:
        self.buffer.clear()
        self.position = 0
        self.full = False


class HistogramAccumulator:
    """
    Online histogram accumulator with fixed bin edges.
    Useful for distribution visualization without storing raw values.
    """
    
    def __init__(self, num_bins: int = 50, range_min: float = -3.0, range_max: float = 3.0):
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.bin_edges = np.linspace(range_min, range_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.counts = np.zeros(num_bins, dtype=np.int64)
        self.underflow: int = 0
        self.overflow: int = 0
        self.total: int = 0
    
    def update(self, values: np.ndarray) -> None:
        """Accumulate values into histogram bins."""
        flat = values.flatten()
        self.total += len(flat)
        
        self.underflow += int(np.sum(flat < self.range_min))
        self.overflow += int(np.sum(flat > self.range_max))
        
        hist, _ = np.histogram(flat, bins=self.bin_edges)
        self.counts += hist
    
    @property
    def density(self) -> np.ndarray:
        """Normalized density (sums to 1)."""
        total = self.counts.sum()
        if total == 0:
            return np.zeros(self.num_bins)
        return self.counts.astype(np.float64) / total
    
    def reset(self) -> None:
        self.counts = np.zeros(self.num_bins, dtype=np.int64)
        self.underflow = 0
        self.overflow = 0
        self.total = 0


class MultiChannelTracker:
    """
    Tracks multiple named statistical channels simultaneously.
    Each channel gets its own Welford accumulator, EMA, and rolling window.
    """
    
    def __init__(self, channel_names: List[str], window_size: int = 100, ema_alpha: float = 0.1):
        self.channel_names = channel_names
        self.accumulators: Dict[str, WelfordAccumulator] = {}
        self.emas: Dict[str, ExponentialMovingAverage] = {}
        self.windows: Dict[str, RollingWindow] = {}
        
        for name in channel_names:
            self.accumulators[name] = WelfordAccumulator()
            self.emas[name] = ExponentialMovingAverage(alpha=ema_alpha)
            self.windows[name] = RollingWindow(window_size=window_size)
    
    def update(self, channel: str, value: float) -> None:
        """Update a specific channel with a new observation."""
        if channel not in self.accumulators:
            raise KeyError(f"Unknown channel: {channel}. Available: {self.channel_names}")
        
        self.accumulators[channel].update(value)
        self.emas[channel].update(value)
        self.windows[channel].update(value)
    
    def get_summary(self, channel: str) -> Dict[str, float]:
        """Get comprehensive statistics for a channel."""
        if channel not in self.accumulators:
            raise KeyError(f"Unknown channel: {channel}")
        
        acc = self.accumulators[channel]
        return {
            'total_count': acc.count,
            'lifetime_mean': acc.mean,
            'lifetime_std': acc.std,
            'lifetime_min': acc.min_val,
            'lifetime_max': acc.max_val,
            'ema': self.emas[channel].value,
            **{f'window_{k}': v for k, v in self.windows[channel].summary().items()}
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all channels."""
        return {name: self.get_summary(name) for name in self.channel_names}
    
    def reset(self) -> None:
        for name in self.channel_names:
            self.accumulators[name].reset()
            self.emas[name].reset()
            self.windows[name].reset()