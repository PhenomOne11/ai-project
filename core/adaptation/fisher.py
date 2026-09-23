"""
Fisher Information Matrix Approximation for Elastic Weight Consolidation.

The diagonal Fisher encodes how sensitive the loss is to each parameter.
EWC uses it to penalize changes to important weights during fine-tuning,
preventing catastrophic forgetting.

Reference: Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting
in neural networks" (PNAS)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import OrderedDict


class FisherInformation:
    """
    Diagonal Fisher Information Matrix estimator.

    Computes per-parameter squared-gradient averages, which approximate
    how "important" each weight is for the current task. Used by EWC
    to anchor parameters during adaptation.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher: Dict[str, torch.Tensor] = {}
        self._anchors: Dict[str, torch.Tensor] = {}

    def compute(
        self,
        data: torch.Tensor,
        loss_fn: nn.Module,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher from a batch of representative data.

        Args:
            data: Representative input batch
            loss_fn: Reconstruction loss (e.g. MSELoss)
            num_samples: Max samples to iterate over

        Returns:
            Dictionary mapping parameter name → Fisher diagonal
        """
        self.model.eval()
        self.model.zero_grad()

        fisher = OrderedDict()

        # Initialize accumulators
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        n = min(num_samples, data.size(0))
        for i in range(n):
            self.model.zero_grad()

            # Single-sample forward
            sample = data[i:i+1]
            try:
                out = self.model(sample)
                # Handle VAE tuple output
                if isinstance(out, tuple):
                    out = out[0]
                loss = loss_fn(out, sample)
                loss.backward()
            except Exception:
                continue

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

        # Average over samples
        for name in fisher:
            fisher[name] /= max(n, 1)

        self.fisher = dict(fisher)

        # Snapshot current parameters as anchors
        self._anchors = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        return self.fisher

    def penalty(self, lambda_: float = 1.0) -> torch.Tensor:
        """
        EWC penalty term: Σ λ/2 * F_i * (θ_i - θ*_i)²

        Should be added to the training loss to prevent drift from anchors.
        """
        if not self.fisher:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)

        for name, param in self.model.named_parameters():
            if name not in self.fisher or name not in self._anchors:
                continue
            anchor = self._anchors[name].to(device)
            f = self.fisher[name].to(device)
            total = total + (f * (param - anchor) ** 2).sum()

        return (lambda_ / 2.0) * total

    def has_fisher(self) -> bool:
        return len(self.fisher) > 0

    def reset(self) -> None:
        self.fisher.clear()
        self._anchors.clear()

    def get_summary(self) -> Dict[str, float]:
        """Top-level summary of Fisher magnitudes."""
        if not self.fisher:
            return {'total_fisher': 0.0, 'mean_fisher': 0.0, 'num_params': 0}

        total = sum(float(f.sum()) for f in self.fisher.values())
        count = sum(f.numel() for f in self.fisher.values())
        return {
            'total_fisher': total,
            'mean_fisher': total / max(count, 1),
            'num_params': count,
        }