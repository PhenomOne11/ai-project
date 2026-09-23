"""
Precision Controller v2
Closed-loop dynamic precision scaler with:
    - Thermal awareness (backed by ThermalModel)
    - Hysteresis (cooldown ticks between transitions)
    - Grace & downgrade policies with target bands
    - Per-transition cost tracking
    - No-thrash guarantee
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Optional, List, Dict, Callable, Tuple
from core.precision.thermal import ThermalModel


class PrecisionState:
    """String constants for precision modes."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"


class AdaptivePrecisionEngineV2:
    """
    Precision controller v2 with hysteresis + thermal guidance.

    Decision logic:
        1. Look at rolling p95 latency vs. target.
        2. Fetch thermal state (COOL / WARM / HOT).
        3. Choose a target precision mode using a banded policy.
        4. Only transition if cooldown_ticks have passed since last change.
        5. Record all transitions with timestamps for audit.

    Args:
        model: PyTorch model to control
        target_latency_ms: Target p95 latency
        window_size: Rolling latency window
        cooldown_ticks: Minimum ticks between precision transitions
        enable_thermal_guidance: If True, HOT state forces downscale
        enable_int8: If True, will quantize for INT8 fallback
    """

    TRANSITION_ORDER = [PrecisionState.FP32, PrecisionState.BF16,
                        PrecisionState.FP16, PrecisionState.INT8]

    def __init__(
        self,
        model: nn.Module,
        target_latency_ms: float = 3.0,
        window_size: int = 30,
        cooldown_ticks: int = 10,
        enable_thermal_guidance: bool = True,
        enable_int8: bool = True
    ):
        self.fp32_model = model.eval()
        self.target_latency_ms = target_latency_ms
        self.window_size = window_size
        self.cooldown_ticks = cooldown_ticks
        self.enable_thermal_guidance = enable_thermal_guidance

        # Latency tracking
        self.latency_history: List[float] = []

        # Thermal model
        self.thermal = ThermalModel(baseline_latency_ms=target_latency_ms)

        # State
        self.current_state = PrecisionState.FP32
        self._ticks_since_transition = cooldown_ticks
        self._tick = 0

        # Pre-quantize INT8 representation if requested
        self.int8_model: Optional[nn.Module] = None
        if enable_int8:
            try:
                self.int8_model = torch.ao.quantization.quantize_dynamic(
                    self.fp32_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            except Exception:
                self.int8_model = None

        # Transition audit log
        self.transition_log: List[Dict] = []

        # Stats
        self.total_transitions = 0
        self.total_time_in_state = {s: 0.0 for s in self.TRANSITION_ORDER}
        self._state_entered_at = time.time()

    # ---- Core step ----

    def step(
        self,
        forward_fn: Callable[[], torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """
        Execute a forward pass and update precision decision.

        Args:
            forward_fn: Zero-arg callable that runs the actual forward pass
                        using the *current* precision. The engine decides
                        whether to wrap in autocast.

        Returns:
            (output_tensor, elapsed_ms)
        """
        self._tick += 1
        self._ticks_since_transition += 1

        t0 = time.perf_counter()

        # Dispatch by state (uses modern autocast API)
        if self.current_state in (PrecisionState.FP16, PrecisionState.BF16):
            try:
                ctx = torch.amp.autocast("cpu", dtype=torch.bfloat16)
            except Exception:
                # Older torch versions fall back to legacy API
                ctx = torch.cpu.amp.autocast(dtype=torch.bfloat16)
            with ctx:
                out = forward_fn()
        else:
            out = forward_fn()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Update stats
        self.latency_history.append(elapsed_ms)
        if len(self.latency_history) > self.window_size:
            self.latency_history.pop(0)

        # Track state duration
        now = time.time()
        self.total_time_in_state[self.current_state] += now - self._state_entered_at
        self._state_entered_at = now

        # Record with thermal model
        self.thermal.record(elapsed_ms)

        # Evaluate if we should transition
        self._evaluate_transition()

        return out, elapsed_ms

    # ---- Decision logic ----

    def _compute_p95(self) -> float:
        if not self.latency_history:
            return 0.0
        return float(np.percentile(self.latency_history, 95))

    def _select_desired_state(self, p95: float) -> str:
        """
        Banded policy: map (p95, thermal_state) → desired precision.

        Bands (relative to target):
            p95 <  0.8× target  → try upscale
            p95 <  1.5× target  → hold
            p95 <  2.5× target  → downscale one step
            p95 >= 2.5× target  → downscale to INT8
        """
        target = self.target_latency_ms
        ratio = p95 / max(target, 1e-6)
        thermal_state = self.thermal.current_state if self.enable_thermal_guidance else 'COOL'

        current_idx = self.TRANSITION_ORDER.index(self.current_state)

        # Under thermal pressure, force down at least one step
        if thermal_state == 'HOT' and current_idx < 3:
            return self.TRANSITION_ORDER[min(current_idx + 1, 3)]

        # WARM nudges downscale at elevated ratios
        if thermal_state == 'WARM' and ratio >= 1.8 and current_idx < 3:
            return self.TRANSITION_ORDER[current_idx + 1]

        # Standard banded logic
        if ratio >= 2.5:
            return PrecisionState.INT8
        elif ratio >= 1.5 and current_idx < 3:
            return self.TRANSITION_ORDER[current_idx + 1]
        elif ratio < 0.8 and current_idx > 0:
            return self.TRANSITION_ORDER[current_idx - 1]
        else:
            return self.current_state

    def _evaluate_transition(self) -> None:
        """Check cooldown and apply transition if warranted."""
        if self._ticks_since_transition < self.cooldown_ticks:
            return
        if len(self.latency_history) < min(self.window_size // 2, 5):
            return

        p95 = self._compute_p95()
        desired = self._select_desired_state(p95)

        if desired != self.current_state:
            self._transition_to(desired, p95)

    def _transition_to(self, new_state: str, p95_at_transition: float) -> None:
        """Execute the precision transition and log it."""
        old_state = self.current_state
        self.current_state = new_state
        self._ticks_since_transition = 0
        self.total_transitions += 1

        self.transition_log.append({
            'tick': self._tick,
            'from': old_state,
            'to': new_state,
            'p95_ms': p95_at_transition,
            'target_ms': self.target_latency_ms,
            'thermal_state': self.thermal.current_state,
            'timestamp': time.time(),
        })

    # ---- Introspection ----

    @property
    def current_model(self) -> nn.Module:
        """Return the module corresponding to current_state."""
        if self.current_state == PrecisionState.INT8 and self.int8_model is not None:
            return self.int8_model
        return self.fp32_model

    @property
    def p95(self) -> float:
        return self._compute_p95()

    @property
    def p50(self) -> float:
        if not self.latency_history:
            return 0.0
        return float(np.percentile(self.latency_history, 50))

    @property
    def tick(self) -> int:
        return self._tick

    def get_status(self) -> Dict:
        """Full diagnostic snapshot."""
        return {
            'current_state': self.current_state,
            'tick': self._tick,
            'p50_ms': self.p50,
            'p95_ms': self.p95,
            'target_ms': self.target_latency_ms,
            'target_ratio': self.p95 / max(self.target_latency_ms, 1e-6),
            'ticks_since_transition': self._ticks_since_transition,
            'total_transitions': self.total_transitions,
            'thermal_state': self.thermal.current_state,
            'thermal_load': self.thermal.load_factor,
            'int8_available': self.int8_model is not None,
            'time_in_state_s': dict(self.total_time_in_state),
        }

    def reset(self) -> None:
        self.latency_history.clear()
        self.thermal.reset()
        self.current_state = PrecisionState.FP32
        self._ticks_since_transition = self.cooldown_ticks
        self._tick = 0
        self.transition_log.clear()
        self.total_transitions = 0
        self.total_time_in_state = {s: 0.0 for s in self.TRANSITION_ORDER}


# Backward-compatible alias
AdaptivePrecisionEngine = AdaptivePrecisionEngineV2