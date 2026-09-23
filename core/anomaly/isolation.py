"""
Isolation Forest Anomaly Scorer
Unsupervised anomaly detection on latent embeddings.

Isolation Forest builds random decision trees that isolate samples;
anomalies require fewer splits to isolate → shorter path lengths →
higher anomaly scores.

Reference: Liu et al. (2008), "Isolation Forest" (ICDM)
"""

import numpy as np
import torch
from typing import Optional
from sklearn.ensemble import IsolationForest


class IsolationScorer:
    """
    Isolation Forest wrapper for latent-space anomaly detection.

    Args:
        n_estimators: Number of trees in the forest
        contamination: Expected fraction of anomalies (sets threshold)
        random_state: Seed for reproducibility
        max_samples: Samples per tree ('auto' = min(256, n_samples))
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
        max_samples: str = "auto"
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.max_samples = max_samples

        self._model: Optional[IsolationForest] = None
        self._is_fitted: bool = False
        self._score_offset: float = 0.0
        self._score_scale: float = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, latent_samples: np.ndarray) -> "IsolationScorer":
        """
        Fit the Isolation Forest on reference latent embeddings.

        Args:
            latent_samples: Array of shape (N, latent_dim)
        """
        if latent_samples.ndim == 1:
            latent_samples = latent_samples.reshape(-1, 1)

        latent_samples = np.nan_to_num(latent_samples, nan=0.0, posinf=0.0, neginf=0.0)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples=self.max_samples,
            n_jobs=1
        )
        self._model.fit(latent_samples)
        self._is_fitted = True

        # Calibrate normalizer: reference samples should score ~0
        raw = self._model.score_samples(latent_samples)
        # raw is negative: normal ~ -0.5, anomalies more negative
        # We want higher score = more anomalous, so negate
        anomaly_scores = -raw
        self._score_offset = float(np.median(anomaly_scores))
        # Scale by IQR of reference to make scores roughly [0, 1]
        q75, q25 = np.percentile(anomaly_scores, [75, 25])
        iqr = q75 - q25
        self._score_scale = 1.0 / max(iqr * 2.5, 1e-6)

        return self

    def score(self, latent: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for new samples.

        Args:
            latent: Array of shape (N, latent_dim) or (latent_dim,)

        Returns:
            Array of shape (N,) with scores in [0, 1] where 1 = most anomalous
        """
        if not self._is_fitted:
            return np.zeros(1 if latent.ndim == 1 else latent.shape[0])

        single = latent.ndim == 1
        x = latent.reshape(1, -1) if single else latent
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        raw = -self._model.score_samples(x)
        normalized = (raw - self._score_offset) * self._score_scale

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized[:1] if single else normalized

    def score_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        """Torch-friendly wrapper."""
        arr = latent.detach().cpu().numpy()
        result = self.score(arr)
        return torch.from_numpy(np.asarray(result)).float()

    def reset(self) -> None:
        self._model = None
        self._is_fitted = False
        self._score_offset = 0.0
        self._score_scale = 1.0