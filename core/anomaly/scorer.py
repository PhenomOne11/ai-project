"""
Unified Anomaly Scorer
Combines reconstruction error, Isolation Forest, and Mahalanobis distance
into a single confidence-weighted anomaly score.

The three component scorers capture complementary signals:
    - Reconstruction error  → global input novelty
    - Isolation Forest      → local latent density anomaly
    - Mahalanobis distance  → covariance structure deviation

Final score = w_r * recon + w_i * iso + w_m * maha
Weights are normalized to sum to 1.0.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from core.anomaly.isolation import IsolationScorer
from core.anomaly.mahalanobis import MahalanobisScorer


class AnomalyScorer:
    """
    Multi-signal anomaly scorer.

    Args:
        reconstruction_weight: Weight for reconstruction error component
        isolation_weight: Weight for Isolation Forest component
        mahalanobis_weight: Weight for Mahalanobis distance component
        recon_scale: Divisor to map raw MSE into [0, 1] (calibrated)
        ema_alpha: EMA smoothing factor for output score
        fit_contamination: Passed to Isolation Forest
        n_estimators: Trees in Isolation Forest
    """

    def __init__(
        self,
        reconstruction_weight: float = 0.5,
        isolation_weight: float = 0.3,
        mahalanobis_weight: float = 0.2,
        recon_scale: float = 1.0,
        ema_alpha: float = 0.1,
        fit_contamination: float = 0.05,
        n_estimators: int = 100
    ):
        # Normalize weights to sum to 1
        total = reconstruction_weight + isolation_weight + mahalanobis_weight
        assert total > 0, "At least one weight must be positive"

        self.w_recon = reconstruction_weight / total
        self.w_iso = isolation_weight / total
        self.w_maha = mahalanobis_weight / total

        self.recon_scale = max(recon_scale, 1e-6)
        self.ema_alpha = ema_alpha

        # Components
        self.isolation = IsolationScorer(
            n_estimators=n_estimators,
            contamination=fit_contamination
        )
        self.mahalanobis = MahalanobisScorer()

        # EMA state
        self._ema_value: Optional[float] = None

        # Statistics
        self.score_history = []
        self.max_history = 1000

    # ---- Fitting ----

    def fit(
        self,
        latent_samples: np.ndarray,
        reconstruction_errors: Optional[np.ndarray] = None
    ) -> "AnomalyScorer":
        """
        Fit all component scorers on reference data.

        Args:
            latent_samples: Reference latents (N, latent_dim)
            reconstruction_errors: Optional per-sample reconstruction MSE
                                   for calibrating the recon scale
        """
        if latent_samples.ndim == 1:
            latent_samples = latent_samples.reshape(-1, 1)

        # Fit components
        self.isolation.fit(latent_samples)
        self.mahalanobis.fit(latent_samples)

        # Calibrate reconstruction scale from reference distribution
        if reconstruction_errors is not None and len(reconstruction_errors) > 0:
            p95 = float(np.percentile(reconstruction_errors, 95))
            self.recon_scale = max(p95 * 2.0, 1e-6)
        else:
            # Fall back to latent variance
            self.recon_scale = max(float(np.var(latent_samples)) * 2.0, 1e-6)

        return self

    # ---- Scoring ----

    def score(
        self,
        latent: torch.Tensor,
        reconstruction: torch.Tensor,
        input_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the fused anomaly score for a batch.

        Args:
            latent: Latent embeddings (B, latent_dim)
            reconstruction: Reconstructed outputs matching input_batch
            input_batch: Original inputs

        Returns:
            Dictionary with:
                - 'score': Fused anomaly score (B,)
                - 'recon_component': Reconstruction component (B,)
                - 'isolation_component': Isolation Forest component (B,)
                - 'mahalanobis_component': Mahalanobis component (B,)
                - 'ema_score': Smoothed scalar used for dashboard
        """
        B = latent.size(0)

        # --- Component 1: Reconstruction error ---
        # Handle VAE tuple output
        if isinstance(reconstruction, tuple):
            rec = reconstruction[0]
        else:
            rec = reconstruction

        recon_errors = torch.mean((input_batch - rec) ** 2, dim=[1, 2, 3])
        recon_component = torch.clamp(recon_errors / self.recon_scale, 0.0, 1.0)

        # --- Component 2: Isolation Forest ---
        iso_component = self.isolation.score_tensor(latent).to(latent.device)

        # --- Component 3: Mahalanobis ---
        maha_component = self.mahalanobis.score_tensor(latent).to(latent.device)

        # --- Fused score ---
        fused = (
            self.w_recon * recon_component
            + self.w_iso * iso_component
            + self.w_maha * maha_component
        )

        # Update EMA on batch mean
        batch_mean = float(fused.mean().item())
        if self._ema_value is None:
            self._ema_value = batch_mean
        else:
            self._ema_value = (
                self.ema_alpha * batch_mean
                + (1 - self.ema_alpha) * self._ema_value
            )

        # History
        self.score_history.append({
            'mean': batch_mean,
            'max': float(fused.max().item()),
            'ema': self._ema_value,
        })
        if len(self.score_history) > self.max_history:
            self.score_history = self.score_history[-self.max_history:]

        return {
            'score': fused,
            'recon_component': recon_component,
            'isolation_component': iso_component,
            'mahalanobis_component': maha_component,
            'ema_score': self._ema_value,
        }

    # ---- Introspection ----

    @property
    def is_fitted(self) -> bool:
        return self.isolation.is_fitted and self.mahalanobis.is_fitted

    @property
    def ema_score(self) -> float:
        return self._ema_value if self._ema_value is not None else 0.0

    def get_weights(self) -> Dict[str, float]:
        return {
            'reconstruction': self.w_recon,
            'isolation': self.w_iso,
            'mahalanobis': self.w_maha,
        }

    def get_status(self) -> Dict:
        return {
            'is_fitted': self.is_fitted,
            'weights': self.get_weights(),
            'recon_scale': self.recon_scale,
            'ema_score': self.ema_score,
            'history_length': len(self.score_history),
            'isolation_fitted': self.isolation.is_fitted,
            'mahalanobis_fitted': self.mahalanobis.is_fitted,
        }

    def reset(self) -> None:
        self.isolation.reset()
        self.mahalanobis.reset()
        self._ema_value = None
        self.score_history.clear()