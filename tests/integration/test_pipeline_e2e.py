"""
End-to-end pipeline integration tests.

Validates that config → model → drift → adaptation → anomaly can
run as a coherent system, with data flowing between components.
"""

import numpy as np
import torch
import pytest


class TestEnginePipelineE2E:
    """Full model + calibration + inference pipeline."""

    def test_lsuv_calibration_then_forward(self):
        """Model is created, LSUV-calibrated, then serves forward passes."""
        from core.models import EdgeStreamingAE
        from core.math_utils.initializers import lsuv_init

        model = EdgeStreamingAE(
            in_channels=1, input_size=32, latent_dim=16, base_channels=4,
        )
        calib = torch.randn(8, 1, 32, 32)
        model = lsuv_init(model, calib, max_iterations=3, verbose=False)

        assert model.is_calibrated
        # Forward after calibration must produce valid output
        x = torch.randn(4, 1, 32, 32)
        with torch.no_grad():
            z = model.encode(x)
            x_hat = model.decode(z)
        assert z.shape == (4, 16)
        assert x_hat.shape == x.shape
        assert torch.isfinite(x_hat).all()

    def test_drift_radar_after_calibration(self):
        """Drift radar captures reference then detects a shift."""
        from core.models import EdgeStreamingAE
        from core.drift import DriftRadar

        model = EdgeStreamingAE(
            in_channels=1, input_size=32, latent_dim=16, base_channels=4,
        ).eval()

        radar = DriftRadar(
            layer_names=["encoder_conv.0", "encoder_fc.0"],
            methods=['wasserstein'],
            evaluation_interval=1,
        )

        calib = torch.randn(8, 1, 32, 32)
        refs = radar.capture_reference(model, calib)
        assert radar.is_calibrated
        assert len(refs) >= 1

        # Normal sample → should be nominal
        normal = torch.randn(8, 1, 32, 32)
        reports_normal = radar.evaluate_from_tensor(model, normal)
        assert len(reports_normal) >= 1

        # Shifted sample → should produce higher drift scores
        shifted = normal + 2.0
        reports_shifted = radar.evaluate_from_tensor(model, shifted)
        # At least one layer's composite score should be higher
        max_normal = max(r.composite_score for r in reports_normal.values())
        max_shifted = max(r.composite_score for r in reports_shifted.values())
        assert max_shifted >= max_normal

    def test_adaptation_engine_lifecycle(self):
        """Adaptation engine ingests OOD samples and adapts."""
        from core.models import EdgeStreamingAE
        from core.adaptation import OnlineAdaptationEngineV2

        model = EdgeStreamingAE(
            in_channels=1, input_size=32, latent_dim=16, base_channels=4,
        )

        engine = OnlineAdaptationEngineV2(
            model,
            error_threshold=0.10,
            adaptive_threshold=False,   # Pin the threshold for deterministic OOD
            buffer_capacity=32,
            micro_batch_size=100,        # Larger than we push → no auto-drain
            enable_ewc=False,
        )

        # Feed bad reconstructions to trigger OOD path
        inp = torch.randn(4, 1, 32, 32)
        bad_rec = torch.zeros_like(inp)

        for _ in range(10):
            _, num_ood = engine.evaluate_batch(inp, bad_rec)
            assert num_ood == 4   # All samples should be flagged OOD

        # Cumulative counter is immune to background draining
        assert engine.replay.total_added >= 40
        # Buffer should still hold samples since we set micro_batch_size high
        assert len(engine.replay) > 0

    def test_anomaly_scorer_after_fit(self):
        """Anomaly scorer fits on latents and produces scores in [0,1]."""
        from core.anomaly import AnomalyScorer

        scorer = AnomalyScorer(n_estimators=20)
        latents = np.random.randn(100, 16)
        scorer.fit(latents)

        assert scorer.is_fitted

        latent = torch.randn(4, 16)
        inp = torch.randn(4, 1, 32, 32)
        rec = inp + 0.05 * torch.randn_like(inp)

        result = scorer.score(latent, rec, inp)
        assert 'score' in result
        assert result['score'].shape == (4,)
        assert torch.all(result['score'] >= 0.0)
        assert torch.all(result['score'] <= 1.0)

    def test_precision_engine_full_cycle(self):
        """Precision engine executes forwards and tracks state."""
        from core.models import EdgeStreamingAE
        from core.precision import AdaptivePrecisionEngineV2

        model = EdgeStreamingAE(
            in_channels=1, input_size=32, latent_dim=16, base_channels=4,
        ).eval()
        engine = AdaptivePrecisionEngineV2(
            model, target_latency_ms=3.0, window_size=5, cooldown_ticks=2,
            enable_int8=False,
        )
        x = torch.randn(2, 1, 32, 32)
        for _ in range(10):
            engine.step(lambda: model(x))

        status = engine.get_status()
        assert 'current_state' in status
        assert status['tick'] == 10
        assert status['p95_ms'] >= 0.0

    def test_full_pipeline_data_flow(self):
        """Every analysis subsystem sees the same batch correctly."""
        from core.models import EdgeStreamingAE
        from core.drift import DriftRadar
        from core.anomaly import AnomalyScorer
        from core.precision import AdaptivePrecisionEngineV2

        model = EdgeStreamingAE(
            in_channels=1, input_size=32, latent_dim=16, base_channels=4,
        ).eval()
        x = torch.randn(8, 1, 32, 32)

        # Drift
        radar = DriftRadar(
            methods=['wasserstein'], evaluation_interval=1,
        )
        radar.capture_reference(model, x)
        drift_reports = radar.evaluate_from_tensor(model, x)
        assert len(drift_reports) > 0

        # Precision
        engine = AdaptivePrecisionEngineV2(
            model, target_latency_ms=3.0, window_size=5, enable_int8=False,
        )
        _, elapsed = engine.step(lambda: model(x))
        assert elapsed > 0

        # Anomaly — needs fitted state
        scorer = AnomalyScorer(n_estimators=15)
        scorer.fit(np.random.randn(60, 16))
        with torch.no_grad():
            z = model.encode(x)
            x_hat = model.decode(z)
        anom_result = scorer.score(z, x_hat, x)
        assert anom_result['score'].shape == (8,)


class TestIPCPipeline:
    """Shared-memory bus passing data end-to-end."""

    def test_bus_producer_consumer(self):
        from core.ipc import SharedMemoryBus

        producer = SharedMemoryBus()
        producer.produce("latents", shape=(16,), dtype=np.float32, capacity=20)
        try:
            for i in range(10):
                vec = np.random.randn(16).astype(np.float32)
                producer.publish("latents", vec)

            # Consume
            consumer = SharedMemoryBus()
            consumer.consume("latents", shape=(16,), dtype=np.float32, capacity=20)
            try:
                latest = consumer.read_latest("latents", n=5)
                assert latest.shape == (5, 16)
            finally:
                consumer.close_all()
        finally:
            producer.unlink_all()
            producer.close_all()


class TestMetricsAndEventsPipeline:
    """Metrics registry + event bus work together."""

    def test_metrics_and_events_integration(self):
        from core.metrics import MetricsRegistry
        from core.events import EventBus, EventKind, EventPriority

        reg = MetricsRegistry(name="integration_test")
        bus = EventBus()

        captured = []

        def on_drift(event):
            captured.append(event)
            # Log drift into metrics
            score = event.payload.get('score', 0.0)
            reg.histogram("drift_scores").observe(score)
            reg.counter("drift_events").inc()

        bus.subscribe("drift.*", on_drift)

        for i in range(5):
            bus.emit(
                EventKind.DRIFT_WARNING.value,
                {"score": 0.1 + i * 0.05},
                priority=EventPriority.WARNING,
            )

        assert len(captured) == 5
        assert reg.histogram("drift_scores").count == 5
        assert reg.counter("drift_events").value == 5.0


class TestWorkerPipeline:
    """Worker pool executes tasks that invoke real subsystems."""

    def test_worker_pool_computes_drift(self):
        from core.workers import WorkerPool
        from core.drift import DriftDetector
        import numpy as np

        detector = DriftDetector(methods=['wasserstein'])
        ref = np.random.randn(200)
        live = np.random.randn(200) + 0.5

        pool = WorkerPool(num_workers=2)
        try:
            tid = pool.submit_fn(
                detector.compute, ref, live, "test_layer",
            )
            result = pool.wait_for(tid, timeout_s=5.0)
            assert result is not None
            assert result.ok
            assert result.value.layer_name == "test_layer"
        finally:
            pool.stop()