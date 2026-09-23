"""
Unit Tests for core.anomaly
"""

import pytest
import numpy as np
import torch


class TestIsolationScorer:
    def test_not_fitted_returns_zeros(self):
        from core.anomaly.isolation import IsolationScorer
        scorer = IsolationScorer()
        assert not scorer.is_fitted
        scores = scorer.score(np.random.randn(5, 8))
        assert np.all(scores == 0.0)

    def test_fit_and_score(self):
        from core.anomaly.isolation import IsolationScorer
        scorer = IsolationScorer(n_estimators=50)
        ref = np.random.randn(200, 8)
        scorer.fit(ref)
        assert scorer.is_fitted

        # Normal sample → low score
        normal = np.random.randn(5, 8)
        normal_scores = scorer.score(normal)

        # Extreme outlier → high score
        outlier = np.full((1, 8), 100.0)
        outlier_score = scorer.score(outlier)

        assert outlier_score[0] > normal_scores.mean()

    def test_score_tensor(self):
        from core.anomaly.isolation import IsolationScorer
        scorer = IsolationScorer(n_estimators=30).fit(np.random.randn(100, 8))
        t = torch.randn(4, 8)
        scores = scorer.score_tensor(t)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (4,)
        assert torch.all(scores >= 0.0)
        assert torch.all(scores <= 1.0)

    def test_reset(self):
        from core.anomaly.isolation import IsolationScorer
        scorer = IsolationScorer().fit(np.random.randn(50, 4))
        assert scorer.is_fitted
        scorer.reset()
        assert not scorer.is_fitted


class TestMahalanobisScorer:
    def test_not_fitted_returns_zeros(self):
        from core.anomaly.mahalanobis import MahalanobisScorer
        scorer = MahalanobisScorer()
        assert not scorer.is_fitted
        scores = scorer.score(np.random.randn(5, 8))
        assert np.all(scores == 0.0)

    def test_fit_and_score(self):
        from core.anomaly.mahalanobis import MahalanobisScorer
        scorer = MahalanobisScorer()
        ref = np.random.randn(300, 8)
        scorer.fit(ref)
        assert scorer.is_fitted

        # Normal sample
        normal_scores = scorer.score(np.random.randn(10, 8))

        # Outlier
        outlier = np.full((1, 8), 10.0)
        outlier_score = scorer.score(outlier)

        assert outlier_score[0] > normal_scores.mean()

    def test_all_scores_in_range(self):
        from core.anomaly.mahalanobis import MahalanobisScorer
        scorer = MahalanobisScorer().fit(np.random.randn(200, 4))
        scores = scorer.score(np.random.randn(50, 4))
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_single_sample(self):
        from core.anomaly.mahalanobis import MahalanobisScorer
        scorer = MahalanobisScorer().fit(np.random.randn(100, 8))
        scores = scorer.score(np.random.randn(8))  # 1D
        assert scores.shape == (1,)

    def test_reset(self):
        from core.anomaly.mahalanobis import MahalanobisScorer
        scorer = MahalanobisScorer().fit(np.random.randn(50, 4))
        scorer.reset()
        assert not scorer.is_fitted


class TestAnomalyScorer:
    def test_weight_normalization(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(
            reconstruction_weight=2.0,
            isolation_weight=2.0,
            mahalanobis_weight=2.0
        )
        weights = scorer.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert abs(weights['reconstruction'] - 1/3) < 1e-6

    def test_fit_and_score(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(n_estimators=30)
        latents = np.random.randn(200, 16)
        scorer.fit(latents)
        assert scorer.is_fitted

        latent = torch.randn(4, 16)
        inp = torch.randn(4, 1, 8, 8)
        rec = inp + 0.1 * torch.randn_like(inp)

        result = scorer.score(latent, rec, inp)
        assert 'score' in result
        assert result['score'].shape == (4,)
        assert torch.all(result['score'] >= 0.0)
        assert torch.all(result['score'] <= 1.0)

    def test_high_error_produces_high_score(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(
            reconstruction_weight=1.0, isolation_weight=0.0,
            mahalanobis_weight=0.0, recon_scale=1.0, n_estimators=20
        )
        scorer.fit(np.random.randn(100, 8))

        latent = torch.randn(4, 8)
        inp = torch.randn(4, 1, 8, 8)

        # Good reconstruction → low score
        good_rec = inp.clone()
        score_good = scorer.score(latent, good_rec, inp)['score']

        # Bad reconstruction → high score
        bad_rec = torch.zeros_like(inp)
        score_bad = scorer.score(latent, bad_rec, inp)['score']

        assert score_bad.mean() > score_good.mean()

    def test_ema_smoothing(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(n_estimators=20)
        scorer.fit(np.random.randn(100, 8))

        latent = torch.randn(4, 8)
        inp = torch.randn(4, 1, 8, 8)
        rec = inp.clone()

        scorer.score(latent, rec, inp)
        ema1 = scorer.ema_score
        assert ema1 > 0.0 or ema1 == 0.0

        scorer.score(latent, rec, inp)
        ema2 = scorer.ema_score
        assert ema2 >= 0.0

    def test_vae_tuple_output(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(n_estimators=20)
        scorer.fit(np.random.randn(100, 8))

        latent = torch.randn(4, 8)
        inp = torch.randn(4, 1, 8, 8)
        # VAE-style tuple
        rec = (inp.clone(), torch.randn(4, 8), torch.randn(4, 8))

        result = scorer.score(latent, rec, inp)
        assert result['score'].shape == (4,)

    def test_get_status(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(n_estimators=20)
        scorer.fit(np.random.randn(100, 8))
        status = scorer.get_status()
        assert status['is_fitted'] is True
        assert 'weights' in status
        assert 'recon_scale' in status

    def test_reset(self):
        from core.anomaly.scorer import AnomalyScorer
        scorer = AnomalyScorer(n_estimators=20)
        scorer.fit(np.random.randn(100, 8))
        scorer.reset()
        assert not scorer.is_fitted
        assert scorer.ema_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])