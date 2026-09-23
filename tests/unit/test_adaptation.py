"""
Unit Tests for core.adaptation
"""

import pytest
import torch
import torch.nn as nn
import time


class TestFisherInformation:
    @pytest.fixture
    def model(self):
        return nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )

    def test_compute_fisher(self, model):
        from core.adaptation.fisher import FisherInformation
        fisher = FisherInformation(model)
        data = torch.randn(20, 16)
        f = fisher.compute(data, nn.MSELoss(), num_samples=10)
        assert len(f) > 0
        assert fisher.has_fisher()

    def test_penalty_zero_at_anchor(self, model):
        from core.adaptation.fisher import FisherInformation
        fisher = FisherInformation(model)
        fisher.compute(torch.randn(20, 16), nn.MSELoss(), num_samples=10)
        # At anchor (no parameter change), penalty = 0
        penalty = fisher.penalty(lambda_=1.0)
        assert penalty.item() < 1e-6

    def test_penalty_positive_after_perturb(self, model):
        from core.adaptation.fisher import FisherInformation
        fisher = FisherInformation(model)
        fisher.compute(torch.randn(20, 16), nn.MSELoss(), num_samples=10)
        # Perturb parameters
        for p in model.parameters():
            p.data += 0.1
        penalty = fisher.penalty(lambda_=1.0)
        assert penalty.item() > 0

    def test_no_fisher_zero_penalty(self, model):
        from core.adaptation.fisher import FisherInformation
        fisher = FisherInformation(model)
        penalty = fisher.penalty()
        assert penalty.item() == 0.0

    def test_summary(self, model):
        from core.adaptation.fisher import FisherInformation
        fisher = FisherInformation(model)
        fisher.compute(torch.randn(20, 16), nn.MSELoss(), num_samples=10)
        summary = fisher.get_summary()
        assert 'total_fisher' in summary
        assert summary['total_fisher'] > 0
        assert summary['num_params'] > 0

    def test_reset(self, model):
        from core.adaptation.fisher import FisherInformation
        fisher = FisherInformation(model)
        fisher.compute(torch.randn(20, 16), nn.MSELoss(), num_samples=10)
        fisher.reset()
        assert not fisher.has_fisher()


class TestPriorityReplayBuffer:
    def test_add_and_sample(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=10, mode='max-error')
        for i in range(5):
            buf.add(torch.randn(1, 4), priority=float(i))
        assert len(buf) == 5
        batch = buf.sample(3)
        # 3 items of shape (1, 4) concatenated along dim=0 → (3, 4)
        assert batch.shape == (3, 4)

    def test_keeps_high_priority(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=5, mode='max-error')
        # Add 10 items with priorities 0-9; only top 5 should survive
        for i in range(10):
            buf.add(torch.full((1, 4), float(i)), priority=float(i))
        assert len(buf) == 5
        # Sample all 5; should be items with priorities 5-9
        batch = buf.sample(5)
        assert batch.shape[0] == 5
        # Each item has shape (1, 4), so column 0 of the (5, 4) result
        values = sorted([float(v) for v in batch[:, 0]])
        assert values == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_overflow_evicts(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=3, mode='max-error')
        buf.add(torch.randn(1, 4), priority=0.5)
        buf.add(torch.randn(1, 4), priority=0.6)
        buf.add(torch.randn(1, 4), priority=0.7)
        assert buf.total_evicted == 0

        buf.add(torch.randn(1, 4), priority=0.1)  # Lower than min → skipped
        assert buf.total_evicted == 0

        buf.add(torch.randn(1, 4), priority=0.9)  # Higher → evicts 0.5
        assert buf.total_evicted == 1

    def test_add_batch(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=20)
        batch = torch.randn(5, 1, 4)
        priorities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        buf.add_batch(batch, priorities)
        assert len(buf) == 5

    def test_drain_clears(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(torch.randn(1, 4), priority=float(i))
        drained = buf.drain()
        assert drained.shape[0] == 5
        assert len(buf) == 0

    def test_is_full(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=3)
        assert not buf.is_full
        for _ in range(3):
            buf.add(torch.randn(1, 4), priority=1.0)
        assert buf.is_full

    def test_get_stats(self):
        from core.adaptation.replay import PriorityReplayBuffer
        buf = PriorityReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(torch.randn(1, 4), priority=float(i))
        stats = buf.get_stats()
        assert stats['size'] == 5
        assert stats['capacity'] == 10
        assert stats['max_priority'] == 4.0


class TestOnlineAdaptationEngineV2:
    @pytest.fixture
    def model(self):
        return nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 1, 3, padding=1), nn.Sigmoid()
        )

    def test_init(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(model, enable_ewc=False)
        assert engine.adaptation_count == 0
        assert engine.current_threshold == 0.18

    def test_evaluate_batch_detects_ood(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(model, error_threshold=0.1, enable_ewc=False)

        inp = torch.randn(4, 1, 8, 8)
        # Simulate bad reconstruction → high error for all
        rec = torch.zeros(4, 1, 8, 8)

        _, num_ood = engine.evaluate_batch(inp, rec)
        assert num_ood == 4
        assert len(engine.replay) == 4

    def test_evaluate_batch_no_ood(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(model, error_threshold=100.0, enable_ewc=False)
        inp = torch.randn(4, 1, 8, 8)
        rec = inp.clone()
        _, num_ood = engine.evaluate_batch(inp, rec)
        assert num_ood == 0

    def test_adaptive_threshold(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(
            model, error_threshold=0.1, adaptive_threshold=True, enable_ewc=False
        )
        # Feed some errors
        inp = torch.randn(4, 1, 8, 8)
        rec = inp + 0.1
        for _ in range(10):
            engine.evaluate_batch(inp, rec)
        # Threshold should have adapted upward
        assert engine.current_threshold >= 0.1

    def test_ewc_baseline(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(model, enable_ewc=True)
        assert not engine.fisher.has_fisher()

        data = torch.rand(20, 1, 8, 8)
        engine.establish_ewc_baseline(data, num_samples=10)
        assert engine.fisher.has_fisher()

    def test_get_status(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(model, enable_ewc=False)
        status = engine.get_status()
        assert 'adaptation_count' in status
        assert 'buffer_size' in status
        assert 'current_threshold' in status
        assert 'ewc_enabled' in status

    def test_backward_compat_alias(self):
        from core.adaptation.engine import (
            OnlineAdaptationEngineV2, ActiveOODAdaptationEngine
        )
        assert OnlineAdaptationEngineV2 is ActiveOODAdaptationEngine

    def test_reset(self, model):
        from core.adaptation.engine import OnlineAdaptationEngineV2
        engine = OnlineAdaptationEngineV2(model, enable_ewc=False)
        inp = torch.randn(4, 1, 8, 8)
        rec = torch.zeros(4, 1, 8, 8)
        engine.evaluate_batch(inp, rec)
        assert len(engine.replay) > 0

        engine.reset()
        assert len(engine.replay) == 0
        assert engine.adaptation_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])