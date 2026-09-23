"""
Unit Tests for core.drift
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestDriftDetector:
    def test_default_methods(self):
        from core.drift.detector import DriftDetector
        det = DriftDetector()
        assert 'wasserstein' in det.methods
        assert 'mmd' in det.methods
        assert 'kl_divergence' in det.methods

    def test_custom_methods(self):
        from core.drift.detector import DriftDetector
        det = DriftDetector(methods=['wasserstein', 'energy', 'hellinger'])
        assert len(det.methods) == 3

    def test_invalid_method(self):
        from core.drift.detector import DriftDetector
        with pytest.raises(ValueError):
            DriftDetector(methods=['nonexistent_method'])

    def test_identical_distributions(self):
        from core.drift.detector import DriftDetector
        from core.constants import DriftSeverity
        det = DriftDetector(methods=['wasserstein'])
        data = np.random.randn(500)
        report = det.compute(data, data, layer_name="test")
        assert report.composite_score < 0.05
        assert report.severity == DriftSeverity.NOMINAL

    def test_shifted_distributions(self):
        from core.drift.detector import DriftDetector
        det = DriftDetector(methods=['wasserstein'], alert_threshold=0.5)
        ref = np.random.randn(500)
        live = np.random.randn(500) + 3.0
        report = det.compute(ref, live, layer_name="test")
        assert report.composite_score > 1.0

    def test_severity_classification(self):
        from core.drift.detector import DriftDetector
        from core.constants import DriftSeverity
        det = DriftDetector(
            methods=['wasserstein'],
            alert_threshold=0.1,
            critical_threshold=0.5,
            emergency_threshold=1.0
        )
        ref = np.random.randn(500)

        # Small shift → NOMINAL
        report = det.compute(ref, ref + 0.01, "l1")
        assert report.severity == DriftSeverity.NOMINAL

        # Medium shift → WARNING or higher
        report = det.compute(ref, ref + 0.5, "l2")
        assert report.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY)

    def test_multi_method_report(self):
        from core.drift.detector import DriftDetector
        det = DriftDetector(methods=['wasserstein', 'mmd', 'hellinger'])
        ref = np.random.randn(300)
        live = np.random.randn(300) + 1.0
        report = det.compute(ref, live, layer_name="conv1")
        assert len(report.results) == 3
        assert 'wasserstein' in report.method_names
        assert report.layer_name == "conv1"

    def test_batch_compute(self):
        from core.drift.detector import DriftDetector
        det = DriftDetector(methods=['wasserstein'])
        refs = {'l1': np.random.randn(200), 'l2': np.random.randn(200)}
        lives = {'l1': np.random.randn(200) + 1, 'l2': np.random.randn(200)}
        reports = det.compute_batch(refs, lives)
        assert 'l1' in reports
        assert 'l2' in reports

    def test_report_to_dict(self):
        from core.drift.detector import DriftDetector
        det = DriftDetector(methods=['wasserstein'])
        report = det.compute(np.random.randn(100), np.random.randn(100), "test")
        d = report.to_dict()
        assert 'layer' in d
        assert 'composite_score' in d
        assert 'severity' in d


class TestDriftRadar:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

    def test_capture_reference(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'])
        batch = torch.randn(8, 1, 16, 16)
        refs = radar.capture_reference(simple_model, batch)
        assert radar.is_calibrated
        assert len(refs) > 0

    def test_auto_discover_layers(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'])
        batch = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, batch)
        assert len(radar.layer_names) >= 2  # At least 2 conv layers

    def test_evaluate_from_tensor(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'], evaluation_interval=1)
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)

        live = torch.randn(8, 1, 16, 16)
        reports = radar.evaluate_from_tensor(simple_model, live)
        assert len(reports) > 0

    def test_drift_history_accumulation(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'], evaluation_interval=1)
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)

        for _ in range(5):
            live = torch.randn(8, 1, 16, 16)
            radar.evaluate_from_tensor(simple_model, live)

        for layer in radar.layer_names:
            if layer in radar.drift_history:
                assert len(radar.drift_history[layer]) >= 3

    def test_heatmap_data(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'], evaluation_interval=1)
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)

        for _ in range(3):
            radar.evaluate_from_tensor(simple_model, torch.randn(8, 1, 16, 16))

        heatmap = radar.get_heatmap_data()
        assert heatmap.ndim == 2
        assert heatmap.shape[0] >= 1

    def test_trend_detection(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'], evaluation_interval=1)
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)

        for _ in range(10):
            radar.evaluate_from_tensor(simple_model, torch.randn(8, 1, 16, 16))

        for layer in radar.layer_names:
            trend = radar.get_trend(layer, window=5)
            assert trend in ('rising', 'falling', 'stable')

    def test_summary(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'])
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)

        summary = radar.get_summary()
        assert summary['is_calibrated'] is True
        assert summary['num_layers'] >= 2
        assert 'layer_status' in summary

    def test_reset(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'])
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)
        assert radar.is_calibrated

        radar.reset()
        assert not radar.is_calibrated
        assert len(radar.reference_distributions) == 0

    def test_hooks_attach_remove(self, simple_model):
        from core.drift.radar import DriftRadar
        radar = DriftRadar(methods=['wasserstein'])
        calib = torch.randn(8, 1, 16, 16)
        radar.capture_reference(simple_model, calib)

        radar.attach_hooks(simple_model)
        assert len(radar._hook_handles) > 0

        radar.remove_hooks()
        assert len(radar._hook_handles) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])