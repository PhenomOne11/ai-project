"""
CPU Thermal & Load Model
Lightweight software model that estimates thermal state and CPU load
without requiring privileged hardware sensors.

Uses process CPU time, wall time, and observed latency variance
to infer whether the host is thermal-throttled or overloaded.
"""

import time
import psutil
import numpy as np
from typing import Optional, Dict, List
from core.math_utils.statistics import ExponentialMovingAverage, RollingWindow


class ThermalModel:
    """
    Software thermal & load estimator.

    Tracks:
        - CPU utilization (via psutil)
        - Process CPU time accumulation rate
        - Latency ratio vs baseline (proxy for thermal throttling)
        - Load factor [0..1]

    The model classifies the host into one of three thermal states:
        COOL     - Latency is normal, CPU not saturated
        WARM     - Some throttling or moderate load
        HOT      - Significant throttling, aggressive downscale recommended
    """

    STATES = ('COOL', 'WARM', 'HOT')

    def __init__(
        self,
        baseline_latency_ms: float = 3.0,
        hot_ratio: float = 2.0,
        warm_ratio: float = 1.3,
        window_size: int = 30
    ):
        self.baseline_latency_ms = baseline_latency_ms
        self.hot_ratio = hot_ratio
        self.warm_ratio = warm_ratio

        # Rolling latency tracking
        self.window = RollingWindow(window_size=window_size)
        self.latency_ema = ExponentialMovingAverage(alpha=0.15)
        self.cpu_ema = ExponentialMovingAverage(alpha=0.2)

        # Bookkeeping
        self._process = psutil.Process()
        self._last_cpu_times: Optional[float] = None
        self._last_wall_time: Optional[float] = None

        # History
        self.history: List[Dict] = []
        self.max_history = 500

    def record(self, latency_ms: float) -> Dict:
        """
        Record an observed latency and update all internal estimates.

        Returns a snapshot dict with estimated state.
        """
        self.window.update(latency_ms)
        self.latency_ema.update(latency_ms)

        # CPU usage
        try:
            cpu_percent = self._process.cpu_percent(interval=None)
            self.cpu_ema.update(cpu_percent)
        except Exception:
            pass

        snapshot = self.snapshot()
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return snapshot

    def snapshot(self) -> Dict:
        """Current thermal / load state."""
        latency_avg = self.latency_ema.value if self.latency_ema.count > 0 else self.baseline_latency_ms
        latency_ratio = latency_avg / max(self.baseline_latency_ms, 1e-6)
        cpu_used = self.cpu_ema.value

        if latency_ratio >= self.hot_ratio or cpu_used >= 90.0:
            state = 'HOT'
        elif latency_ratio >= self.warm_ratio or cpu_used >= 70.0:
            state = 'WARM'
        else:
            state = 'COOL'

        # Load factor: 0 = idle, 1 = saturated
        load_factor = min(1.0, max(
            (latency_ratio - 1.0) / 2.0,
            cpu_used / 100.0
        ))

        return {
            'state': state,
            'latency_avg_ms': latency_avg,
            'latency_ratio': latency_ratio,
            'cpu_percent': cpu_used,
            'load_factor': load_factor,
            'baseline_ms': self.baseline_latency_ms,
        }

    @property
    def current_state(self) -> str:
        return self.snapshot()['state']

    @property
    def load_factor(self) -> float:
        return self.snapshot()['load_factor']

    @property
    def latency_ratio(self) -> float:
        return self.snapshot()['latency_ratio']

    def update_baseline(self, new_baseline_ms: float) -> None:
        """Adjust the baseline latency reference."""
        self.baseline_latency_ms = float(new_baseline_ms)

    def reset(self) -> None:
        self.window.reset()
        self.latency_ema.reset()
        self.cpu_ema.reset()
        self.history.clear()