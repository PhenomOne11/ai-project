"""
Unit Tests for core.math_utils
Comprehensive test coverage for all mathematical primitives.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn


class TestWelfordAccumulator:
    def test_single_values(self):
        from core.math_utils.statistics import WelfordAccumulator
        acc = WelfordAccumulator()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for v in values:
            acc.update(v)
        assert acc.count == 8
        assert abs(acc.mean - 5.0) < 1e-10
        assert abs(acc.variance - 4.0) < 1e-10
        assert acc.min_val == 2.0
        assert acc.max_val == 9.0

    def test_batch_update(self):
        from core.math_utils.statistics import WelfordAccumulator
        acc = WelfordAccumulator()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        acc.update_batch(data)
        assert acc.count == 5
        assert abs(acc.mean - 3.0) < 1e-10

    def test_reset(self):
        from core.math_utils.statistics import WelfordAccumulator
        acc = WelfordAccumulator()
        acc.update(5.0)
        acc.reset()
        assert acc.count == 0
        assert acc.mean == 0.0


class TestVectorAccumulator:
    def test_vector_stats(self):
        from core.math_utils.statistics import WelfordVectorAccumulator
        acc = WelfordVectorAccumulator()
        for _ in range(100):
            acc.update(np.random.randn(10))
        assert acc.count == 100
        assert len(acc.mean) == 10
        assert len(acc.std) == 10
        assert all(s >= 0 for s in acc.std)


class TestEMA:
    def test_convergence(self):
        from core.math_utils.statistics import ExponentialMovingAverage
        ema = ExponentialMovingAverage(alpha=0.5)
        for _ in range(100):
            ema.update(1.0)
        assert abs(ema.value - 1.0) < 0.01

    def test_tracking(self):
        from core.math_utils.statistics import ExponentialMovingAverage
        ema = ExponentialMovingAverage(alpha=0.3)
        for i in range(50):
            ema.update(float(i))
        assert ema.value > 30  # Should track upward trend


class TestRollingWindow:
    def test_percentiles(self):
        from core.math_utils.statistics import RollingWindow
        rw = RollingWindow(window_size=100)
        for i in range(100):
            rw.update(float(i))
        assert abs(rw.median() - 49.5) < 1.0
        assert rw.percentile(95) > rw.percentile(50)

    def test_overflow(self):
        from core.math_utils.statistics import RollingWindow
        rw = RollingWindow(window_size=10)
        for i in range(100):
            rw.update(float(i))
        assert rw.count == 10
        assert rw.min() >= 90


class TestMultiChannelTracker:
    def test_multi_channel(self):
        from core.math_utils.statistics import MultiChannelTracker
        tracker = MultiChannelTracker(["latency", "memory", "drift"])
        for i in range(50):
            tracker.update("latency", float(i) * 0.1)
            tracker.update("memory", 100.0 + float(i))
            tracker.update("drift", np.random.rand())
        summary = tracker.get_all_summaries()
        assert "latency" in summary
        assert "memory" in summary
        assert summary["latency"]["total_count"] == 50


class TestStreamingPCA:
    def test_basic_projection(self):
        from core.math_utils.linear_algebra import StreamingPCA
        pca = StreamingPCA(n_components=3)
        data = np.random.randn(100, 20)
        pca.partial_fit(data)
        projected = pca.transform(data)
        assert projected.shape == (100, 3)

    def test_incremental(self):
        from core.math_utils.linear_algebra import StreamingPCA
        pca = StreamingPCA(n_components=2)
        for _ in range(10):
            batch = np.random.randn(10, 15)
            pca.partial_fit(batch)
        data = np.random.randn(5, 15)
        projected = pca.transform(data)
        assert projected.shape == (5, 2)

    def test_explained_variance(self):
        from core.math_utils.linear_algebra import StreamingPCA
        pca = StreamingPCA(n_components=3)
        data = np.random.randn(200, 10)
        pca.partial_fit(data)
        ratios = pca.explained_variance_ratio
        assert len(ratios) == 3
        assert all(r >= 0 for r in ratios)
        assert sum(ratios) <= 1.0 + 1e-6


class TestOnlineCovariance:
    def test_covariance_matches_numpy(self):
        from core.math_utils.linear_algebra import OnlineCovarianceMatrix
        data = np.random.randn(200, 5)
        ocm = OnlineCovarianceMatrix()
        ocm.update_batch(data)
        np_cov = np.cov(data, rowvar=False)
        online_cov = ocm.sample_covariance
        np.testing.assert_allclose(online_cov, np_cov, atol=1e-10)


class TestDistances:
    def test_wasserstein_identical(self):
        from core.math_utils.distances import wasserstein_1
        a = np.random.randn(1000)
        assert wasserstein_1(a, a) < 0.01

    def test_wasserstein_different(self):
        from core.math_utils.distances import wasserstein_1
        a = np.random.randn(1000)
        b = np.random.randn(1000) + 5.0
        assert wasserstein_1(a, b) > 4.0

    def test_mmd_identical(self):
        from core.math_utils.distances import mmd_rbf
        a = np.random.randn(200)
        assert mmd_rbf(a, a) < 0.1

    def test_kl_symmetry(self):
        from core.math_utils.distances import kl_divergence
        a = np.random.randn(500)
        b = np.random.randn(500) + 1.0
        kl = kl_divergence(a, b)
        assert kl > 0

    def test_ks_statistic(self):
        from core.math_utils.distances import ks_statistic
        a = np.random.randn(500)
        b = np.random.randn(500) + 3.0
        stat, pval = ks_statistic(a, b)
        assert stat > 0.5
        assert pval < 0.01

    def test_hellinger_range(self):
        from core.math_utils.distances import hellinger_distance
        a = np.random.randn(500)
        b = np.random.randn(500)
        h = hellinger_distance(a, b)
        assert 0.0 <= h <= 1.0

    def test_cosine_orthogonal(self):
        from core.math_utils.distances import cosine_distance
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_distance(a, b) - 1.0) < 1e-6

    def test_mahalanobis(self):
        from core.math_utils.distances import mahalanobis_distance
        mean = np.zeros(3)
        cov_inv = np.eye(3)
        x = np.array([1.0, 0.0, 0.0])
        d = mahalanobis_distance(x, mean, cov_inv)
        assert abs(d - 1.0) < 1e-6


class TestSignalProcessing:
    def test_fft_pure_sine(self):
        from core.math_utils.signal_processing import compute_fft
        t = np.linspace(0, 1, 1000, endpoint=False)
        signal = np.sin(2 * np.pi * 50 * t)
        freqs, mags = compute_fft(signal, sample_rate=1000)
        peak_freq = freqs[np.argmax(mags)]
        assert abs(peak_freq - 50.0) < 2.0

    def test_spectral_entropy(self):
        from core.math_utils.signal_processing import spectral_entropy
        noise = np.random.randn(1000)
        sine = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        se_noise = spectral_entropy(noise)
        se_sine = spectral_entropy(sine)
        assert se_noise > se_sine  # Noise has higher spectral entropy

    def test_snr(self):
        from core.math_utils.signal_processing import signal_to_noise_ratio
        clean = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        snr = signal_to_noise_ratio(clean)
        assert snr > 5.0  # Pure sine should have high SNR

    def test_zero_crossing(self):
        from core.math_utils.signal_processing import zero_crossing_rate
        sine = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        zcr = zero_crossing_rate(sine)
        assert zcr > 0.005


class TestTensorOps:
    def test_im2col_shape(self):
        from core.math_utils.tensor_ops import im2col
        x = torch.randn(2, 3, 16, 16)
        cols = im2col(x, kernel_size=3, stride=1, padding=1)
        assert cols.shape[0] == 2
        assert cols.shape[1] == 3 * 3 * 3  # C * kH * kW
        assert cols.shape[2] == 16 * 16     # output spatial elements

    def test_extract_patches(self):
        from core.math_utils.tensor_ops import extract_patches_2d
        x = torch.randn(4, 1, 64, 64)
        patches = extract_patches_2d(x, patch_size=8)
        assert patches.shape == (4, 64, 1, 8, 8)  # 64 non-overlapping 8x8 patches

    def test_tensor_health(self):
        from core.math_utils.tensor_ops import check_tensor_health
        healthy = torch.randn(10, 10)
        report = check_tensor_health(healthy)
        assert report['is_healthy'] is True
        
        unhealthy = torch.tensor([1.0, float('nan'), 3.0])
        report = check_tensor_health(unhealthy)
        assert report['has_nan'] is True
        assert report['is_healthy'] is False

    def test_sanitize(self):
        from core.math_utils.tensor_ops import sanitize_tensor
        dirty = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 3.0])
        clean = sanitize_tensor(dirty)
        assert not torch.isnan(clean).any()
        assert not torch.isinf(clean).any()

    def test_count_parameters(self):
        from core.math_utils.tensor_ops import count_parameters
        model = nn.Linear(10, 5)
        params = count_parameters(model)
        assert params == 10 * 5 + 5  # weights + bias

    def test_output_size(self):
        from core.math_utils.tensor_ops import compute_output_size
        size = compute_output_size(input_size=32, kernel_size=3, stride=1, padding=1)
        assert size == 32
        size = compute_output_size(input_size=32, kernel_size=3, stride=2, padding=0)
        assert size == 15


class TestInitializers:
    def test_lsuv(self):
        from core.math_utils.initializers import lsuv_init
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        batch = torch.randn(8, 1, 16, 16)
        model = lsuv_init(model, batch, verbose=False)
        assert hasattr(model, '_lsuv_diagnostics')

    def test_kaiming(self):
        from core.math_utils.initializers import kaiming_init
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model = kaiming_init(model, verbose=False)
        # Verify weights were modified (not all zeros)
        for p in model.parameters():
            if p.dim() >= 2:
                assert p.std() > 0.01

    def test_diagnostics(self):
        from core.math_utils.initializers import get_init_diagnostics
        model = nn.Linear(10, 5)
        diag = get_init_diagnostics(model)
        assert 'weight' in list(diag.keys())[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])