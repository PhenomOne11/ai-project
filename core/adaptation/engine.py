"""
Online Adaptation Engine v2
Background micro-tuning of an autoencoder on OOD samples with:
    - Elastic Weight Consolidation (EWC) to prevent forgetting
    - Priority replay buffer that keeps highest-error samples
    - Adaptive error threshold (EMA-based)
    - Full adaptation history with before/after loss
"""

import torch
import torch.nn as nn
import threading
import time
import numpy as np
from typing import Optional, Dict, Tuple, List
from core.adaptation.fisher import FisherInformation
from core.adaptation.replay import PriorityReplayBuffer


class OnlineAdaptationEngineV2:
    """
    Adaptive OOD detection + background micro-fine-tuning.

    Pipeline:
        1. Detect OOD samples via per-sample reconstruction error > threshold.
        2. Push OOD samples into priority replay buffer.
        3. When buffer fills, spawn a background thread that:
            - Drains top-N samples
            - Computes reconstruction loss + EWC penalty
            - Takes one AdamW step with grad clipping
            - Logs before/after loss for audit

    Args:
        model: Autoencoder to adapt
        error_threshold: OOD detection threshold (fixed or adaptive)
        adaptive_threshold: If True, threshold tracks EMA of error
        buffer_capacity: Replay buffer size
        micro_batch_size: Samples per adaptation step
        learning_rate: AdamW learning rate (kept tiny)
        max_grad_norm: Gradient clipping bound
        ewc_lambda: EWC penalty strength
        enable_ewc: Toggle EWC
    """

    def __init__(
        self,
        model: nn.Module,
        error_threshold: float = 0.18,
        adaptive_threshold: bool = True,
        buffer_capacity: int = 64,
        micro_batch_size: int = 16,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
        ewc_lambda: float = 0.5,
        enable_ewc: bool = True
    ):
        self.model = model
        self.error_threshold = error_threshold
        self.adaptive_threshold = adaptive_threshold
        self.micro_batch_size = micro_batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.ewc_lambda = ewc_lambda
        self.enable_ewc = enable_ewc

        # Replay buffer
        self.replay = PriorityReplayBuffer(capacity=buffer_capacity, mode='max-error')

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-6
        )

        # EWC
        self.fisher: Optional[FisherInformation] = None
        if enable_ewc:
            self.fisher = FisherInformation(model)

        # Threshold adaptation (EMA of observed errors)
        self._error_ema = None
        self._error_ema_alpha = 0.05

        # State
        self.is_adapting: bool = False
        self.adaptation_count: int = 0
        self._lock = threading.Lock()

        # Audit log
        self.adaptation_history: List[Dict] = []
        self.max_history = 200

    # ---- OOD detection ----

    @property
    def current_threshold(self) -> float:
        """Effective OOD threshold (may be adaptive)."""
        if self.adaptive_threshold and self._error_ema is not None:
            # 1.5× EMA tends to be a good track
            adaptive = self._error_ema * 1.5
            # Blend with static floor to avoid runaway
            return max(adaptive, self.error_threshold)
        return self.error_threshold

    def evaluate_batch(
        self,
        input_batch: torch.Tensor,
        reconstructed_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute per-sample reconstruction errors, push OOD to buffer.

        Returns:
            (per_sample_mse, num_ood)
        """
        with torch.no_grad():
            # Handle VAE tuple
            if isinstance(reconstructed_batch, tuple):
                reconstructed_batch = reconstructed_batch[0]

            # Compute threshold FIRST (uses pre-update EMA)
            threshold = self.current_threshold

            per_sample_mse = torch.mean(
                (input_batch - reconstructed_batch) ** 2, dim=[1, 2, 3]
            )
            mean_error = float(per_sample_mse.mean().item())

            # Detect OOD against pre-update threshold
            ood_mask = per_sample_mse > threshold
            num_ood = int(ood_mask.sum().item())

            # NOW update EMA (for the NEXT call)
            if self._error_ema is None:
                self._error_ema = mean_error
            else:
                self._error_ema = (
                    self._error_ema_alpha * mean_error
                    + (1 - self._error_ema_alpha) * self._error_ema
                )

            if num_ood > 0:
                ood_samples = input_batch[ood_mask].detach().cpu()
                ood_priorities = per_sample_mse[ood_mask].detach().cpu()
                self.replay.add_batch(ood_samples, ood_priorities)

        # Trigger background adaptation if enough samples queued
        if len(self.replay) >= self.micro_batch_size and not self.is_adapting:
            self._spawn_adaptation()

        return per_sample_mse, num_ood

    # ---- Background adaptation ----

    def _spawn_adaptation(self) -> None:
        """Start a background adaptation thread (non-blocking)."""
        with self._lock:
            if self.is_adapting:
                return
            self.is_adapting = True

        thread = threading.Thread(target=self._run_adaptation, daemon=True)
        thread.start()

    def _run_adaptation(self) -> None:
        """Perform a single micro-adaptation step."""
        try:
            # Drain buffer
            batch = self.replay.sample(self.micro_batch_size)
            if batch is None:
                return

            # Loss before
            loss_before = self._compute_loss(batch)

            # Optimization step
            self.model.train()
            self.optimizer.zero_grad()

            out = self.model(batch)
            if isinstance(out, tuple):
                out = out[0]

            recon_loss = nn.functional.mse_loss(out, batch)

            # Add EWC penalty
            if self.enable_ewc and self.fisher is not None and self.fisher.has_fisher():
                ewc_penalty = self.fisher.penalty(lambda_=self.ewc_lambda)
                total_loss = recon_loss + ewc_penalty
            else:
                total_loss = recon_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.model.eval()

            # Loss after
            loss_after = self._compute_loss(batch)

            # Record
            self.adaptation_count += 1
            entry = {
                'count': self.adaptation_count,
                'loss_before': loss_before,
                'loss_after': loss_after,
                'delta': loss_before - loss_after,
                'batch_size': batch.size(0),
                'ewc_enabled': self.enable_ewc and self.fisher.has_fisher() if self.fisher else False,
                'threshold': self.current_threshold,
                'timestamp': time.time(),
            }
            self.adaptation_history.append(entry)
            if len(self.adaptation_history) > self.max_history:
                self.adaptation_history = self.adaptation_history[-self.max_history:]

        except Exception as e:
            # Log error but don't crash the thread
            self.adaptation_history.append({
                'count': self.adaptation_count,
                'error': str(e),
                'timestamp': time.time(),
            })
        finally:
            with self._lock:
                self.is_adapting = False

    def _compute_loss(self, batch: torch.Tensor) -> float:
        """Evaluate reconstruction MSE on a batch (no grad)."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
            if isinstance(out, tuple):
                out = out[0]
            return float(nn.functional.mse_loss(out, batch).item())

    # ---- EWC baseline ----

    def establish_ewc_baseline(
        self,
        data: torch.Tensor,
        num_samples: int = 100
    ) -> None:
        """
        Compute Fisher Information + anchor parameters.
        Call once after initial calibration to enable EWC protection.
        """
        if self.fisher is None:
            self.fisher = FisherInformation(self.model)
        loss_fn = nn.MSELoss()
        self.fisher.compute(data, loss_fn, num_samples=num_samples)

    # ---- Introspection ----

    def get_status(self) -> dict:
        status = {
            'is_adapting': self.is_adapting,
            'adaptation_count': self.adaptation_count,
            'buffer_size': len(self.replay),
            'buffer_capacity': self.replay.capacity,
            'error_ema': self._error_ema,
            'current_threshold': self.current_threshold,
            'ewc_enabled': self.enable_ewc,
            'ewc_ready': bool(self.fisher and self.fisher.has_fisher()),
            'recent_adaptations': self.adaptation_history[-5:],
            'buffer_stats': self.replay.get_stats(),
        }
        if self.fisher and self.fisher.has_fisher():
            status['fisher_summary'] = self.fisher.get_summary()
        return status

    def reset(self) -> None:
        self.replay.clear()
        self.adaptation_history.clear()
        self.adaptation_count = 0
        self._error_ema = None
        self.is_adapting = False
        if self.fisher:
            self.fisher.reset()


# Backward-compatible alias
ActiveOODAdaptationEngine = OnlineAdaptationEngineV2