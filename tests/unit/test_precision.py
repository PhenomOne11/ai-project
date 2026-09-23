"""
Unit Tests for core.precision
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time


class TestThermalModel:
    def test_initial_state_cool(self):
        from core.precision.thermal import ThermalModel
        tm = ThermalModel(baseline_latency_ms=3.0)
        assert tm.current_state == 'COOL'
        assert tm.load_factor < 0.5

    def test_hot_after_high_latency(self):
        from core.precision.thermal import ThermalModel
        tm = ThermalModel(baseline_latency_ms=3.0, hot_ratio=2.0)
        for _ in range(30):
            tm.record(10.0)   # 3.33× baseline
        assert tm.current_state == 'HOT'

    def test_warm_after_moderate_latency(self):
        from core.precision.thermal import ThermalModel
        tm = ThermalModel(
            baseline_latency_ms=3.0, hot_ratio=2.5, warm_ratio=1.3
        )
        for _ in range(30):
            tm.record(5.0)    # 1.67× baseline → WARM band
        assert tm.current_state in ('WARM', 'HOT')

    def test_snapshot_keys(self):
        from core.precision.thermal import ThermalModel
        tm = ThermalModel()
        snap = tm.snapshot()
        assert 'state' in snap
        assert 'latency_ratio' in snap
        assert 'load_factor' in snap
        assert 'cpu_percent' in snap

    def test_baseline_update(self):
        from core.precision.thermal import ThermalModel
        tm = ThermalModel(baseline_latency_ms=3.0)
        tm.update_baseline(5.0)
        assert tm.baseline_latency_ms == 5.0

    def test_reset(self):
        from core.precision.thermal import ThermalModel
        tm = ThermalModel()
        for _ in range(20):
            tm.record(10.0)
        tm.reset()
        assert tm.latency_ema.count == 0
        assert len(tm.history) == 0


class TestAdaptivePrecisionEngineV2:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )

    def test_initial_fp32(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2, PrecisionState
        engine = AdaptivePrecisionEngineV2(simple_model)
        assert engine.current_state == PrecisionState.FP32

    def test_step_returns_tuple(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(simple_model, enable_int8=False)
        x = torch.randn(4, 16)
        out, ms = engine.step(lambda: simple_model(x))
        assert out.shape == (4, 16)
        assert ms > 0

    def test_latency_accumulation(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(simple_model, window_size=10, enable_int8=False)
        x = torch.randn(4, 16)
        for _ in range(25):
            engine.step(lambda: simple_model(x))
        assert len(engine.latency_history) == 10  # windowed

    def test_cooldown_prevents_thrash(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(
            simple_model, cooldown_ticks=100, window_size=10, enable_int8=False
        )
        x = torch.randn(4, 16)
        for _ in range(50):
            engine.step(lambda: simple_model(x))
        # No transitions yet because cooldown is 100
        assert engine.total_transitions == 0

    def test_p95_computation(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(simple_model, enable_int8=False)
        engine.latency_history = [1.0, 2.0, 3.0, 4.0, 5.0] * 6
        p95 = engine.p95
        assert 4.0 < p95 < 5.5

    def test_get_status(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(simple_model, enable_int8=False)
        x = torch.randn(4, 16)
        for _ in range(20):
            engine.step(lambda: simple_model(x))
        status = engine.get_status()
        assert 'current_state' in status
        assert 'p95_ms' in status
        assert 'thermal_state' in status
        assert 'total_transitions' in status

    def test_transition_log(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(
            simple_model, cooldown_ticks=2, window_size=4, enable_int8=False
        )
        x = torch.randn(4, 16)

        # Force high latency by adding sleep
        def slow_forward():
            time.sleep(0.02)  # 20ms >> target of 3ms
            return simple_model(x)

        for _ in range(20):
            engine.step(slow_forward)

        # Should have transitioned at least once
        assert engine.total_transitions >= 1
        assert len(engine.transition_log) >= 1

        entry = engine.transition_log[0]
        assert 'from' in entry
        assert 'to' in entry
        assert 'p95_ms' in entry

    def test_reset(self, simple_model):
        from core.precision.controller import AdaptivePrecisionEngineV2
        engine = AdaptivePrecisionEngineV2(simple_model, enable_int8=False)
        x = torch.randn(4, 16)
        for _ in range(10):
            engine.step(lambda: simple_model(x))
        engine.reset()
        assert len(engine.latency_history) == 0
        assert engine.total_transitions == 0
        assert engine.tick == 0

    def test_backward_compat_alias(self):
        from core.precision.controller import AdaptivePrecisionEngine, AdaptivePrecisionEngineV2
        assert AdaptivePrecisionEngine is AdaptivePrecisionEngineV2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])