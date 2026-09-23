"""
Mahalanobis Distance Anomaly Scorer
Detects latent embeddings that deviate from the reference distribution's
covariance structure. Complements Isolation Forest by capturing
multivariate correlation anomalies.
"""

import numpy as np
import torch
from typing import Optional
from core.math_utils.linear_algebra import OnlineCovarianceMatrix
from core.math_utils.distances import mahalanobis_batch


class MahalanobisScorer:
    """
    Latent-space Mahalanobis distance scorer.

    Fits an online covariance matrix to reference latents, then scores
    new samples by their squared Mahalanobis distance from the mean.

    Args:
        regularization: Ridge term added to covariance diagonal for stability
        normalize: If True, maps raw distance to [0, 1] via running percentiles
    """

    def __init__(
        self,
        regularization: float = 1e-3,
        normalize: bool = True
    ):
        self.regularization = regularization
        self.normalize = normalize

        self._cov = OnlineCovarianceMatrix()
        self._cov_inv: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._is_fitted: bool = False

        # Normalizer state (set during fit)
        self._p95_distance: float = 1.0
        self._p99_distance: float = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, latent_samples: np.ndarray) -> "MahalanobisScorer":
        """
        Fit on reference latents.

        Args:
            latent_samples: Array of shape (N, latent_dim)
        """
        if latent_samples.ndim == 1:
            latent_samples = latent_samples.reshape(1, -1)

        latent_samples = np.nan_to_num(latent_samples, nan=0.0, posinf=0.0, neginf=0.0)

        # Reset accumulator and feed reference data
        self._cov.reset()
        self._cov.update_batch(latent_samples)

        self._mean = self._cov.mean.copy()
        cov = self._cov.sample_covariance

        # Add ridge for numerical stability
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])
        # Use pseudo-inverse for robustness
        self._cov_inv = np.linalg.pinv(cov_reg)

        # Compute percentile thresholds from reference data
        ref_distances = mahalanobis_batch(latent_samples, self._mean, self._cov_inv)
        ref_distances = np.nan_to_num(ref_distances, nan=0.0)
        self._p95_distance = float(np.percentile(ref_distances, 95))
        self._p99_distance = float(np.percentile(ref_distances, 99))

        self._is_fitted = True
        return self

    def score(self, latent: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for new samples.

        Args:
            latent: Array of shape (N, latent_dim) or (latent_dim,)

        Returns:
            Array of shape (N,) with scores in [0, 1]
        """
        if not self._is_fitted:
            single = latent.ndim == 1
            size = 1 if single else latent.shape[0]
            return np.zeros(size)

        single = latent.ndim == 1
        x = latent.reshape(1, -1) if single else latent
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        dists = mahalanobis_batch(x, self._mean, self._cov_inv)
        dists = np.nan_to_num(dists, nan=0.0)

        if self.normalize:
            # Map [0, p95] → [0, 0.5] and [p95, p99] → [0.5, 1.0]
            scores = np.where(
                dists <= self._p95_distance,
                0.5 * dists / max(self._p95_distance, 1e-6),
                0.5 + 0.5 * (dists - self._p95_distance) /
                max(self._p99_distance - self._p95_distance, 1e-6)
            )
            scores = np.clip(scores, 0.0, 1.0)
        else:
            scores = dists

        return scores[:1] if single else scores

    def score_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        """Torch-friendly wrapper."""
        arr = latent.detach().cpu().numpy()
        result = self.score(arr)
        return torch.from_numpy(np.asarray(result)).float()

    def reset(self) -> None:
        self._cov.reset()
        self._cov_inv = None
        self._mean = None
        self._is_fitted = False
        self._p95_distance = 1.0
        self._p99_distance = 1.0