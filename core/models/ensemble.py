"""
Model Ensemble System
Combines multiple autoencoder architectures for robust inference
through weighted averaging, voting, or uncertainty-based fusion.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from core.models.base import BaseModel
from core.math_utils.statistics import ExponentialMovingAverage


class EnsembleMember:
    """Wrapper for a single model within the ensemble."""

    def __init__(
        self,
        name: str,
        model: BaseModel,
        weight: float = 1.0,
        enabled: bool = True
    ):
        self.name = name
        self.model = model
        self.weight = weight
        self.enabled = True

        self.latency_ema = ExponentialMovingAverage(alpha=0.1)
        self.error_ema = ExponentialMovingAverage(alpha=0.1)
        self.call_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: float = 0.0

    @property
    def is_healthy(self) -> bool:
        if self.call_count == 0:
            return True
        return (self.error_count / self.call_count) < 0.1

    @property
    def effective_weight(self) -> float:
        if not self.enabled or not self.is_healthy:
            return 0.0
        return self.weight

    def record_call(self, latency_ms: float, error: Optional[float] = None, failed: bool = False):
        self.call_count += 1
        self.last_latency_ms = latency_ms
        self.latency_ema.update(latency_ms)
        if failed:
            self.error_count += 1
        elif error is not None:
            self.error_ema.update(error)


class ModelEnsemble(nn.Module):
    """
    Multi-Model Ensemble for robust autoencoding.
    """

    def __init__(
        self,
        fusion_strategy: str = "weighted",
        uncertainty_threshold: float = 0.1
    ):
        super().__init__()
        self.members: Dict[str, EnsembleMember] = {}
        self.fusion_strategy = fusion_strategy
        self.uncertainty_threshold = uncertainty_threshold
        self._member_modules = nn.ModuleDict()

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        self.members[name] = EnsembleMember(name, model, weight)
        self._member_modules[name] = model

    def remove_model(self, name: str) -> None:
        if name in self.members:
            del self.members[name]
            del self._member_modules[name]

    def set_weight(self, name: str, weight: float) -> None:
        if name in self.members:
            self.members[name].weight = weight

    def enable_model(self, name: str, enabled: bool = True) -> None:
        if name in self.members:
            self.members[name].enabled = enabled

    @property
    def active_members(self) -> List[EnsembleMember]:
        return [m for m in self.members.values() if m.enabled and m.is_healthy]

    @property
    def latent_dim(self) -> int:
        active = self.active_members
        if not active:
            return 0
        return active[0].model.latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        active = self.active_members
        if not active:
            raise RuntimeError("No active ensemble members")
        return active[0].model.encode(x)

    def encode_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = {}
        for member in self.active_members:
            try:
                t0 = time.perf_counter()
                z = member.model.encode(x)
                lat = (time.perf_counter() - t0) * 1000
                member.record_call(lat)
                results[member.name] = z
            except Exception:
                member.record_call(0, failed=True)
        return results

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        active = self.active_members
        if not active:
            raise RuntimeError("No active ensemble members")
        return active[0].model.decode(z)

    def _extract_reconstruction(self, raw_output) -> torch.Tensor:
        """
        Extract the reconstruction tensor from model output.
        Handles VAE which returns (x_hat, mu, logvar) tuple.
        """
        if isinstance(raw_output, tuple):
            return raw_output[0]
        return raw_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstructions, _ = self.forward_with_uncertainty(x)
        return reconstructions

    def forward_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        active = self.active_members
        if not active:
            raise RuntimeError("No active ensemble members")

        reconstructions = []
        weights = []

        for member in active:
            try:
                t0 = time.perf_counter()

                with torch.no_grad():
                    raw_output = member.model(x)
                    x_hat = self._extract_reconstruction(raw_output)

                lat = (time.perf_counter() - t0) * 1000
                error = float(torch.nn.functional.mse_loss(x_hat, x))
                member.record_call(lat, error=error)

                reconstructions.append(x_hat)

                if self.fusion_strategy == "uncertainty":
                    w = 1.0 / max(member.error_ema.value, 1e-6)
                else:
                    w = member.effective_weight

                weights.append(w)

            except Exception:
                member.record_call(0, failed=True)
                continue

        if len(reconstructions) == 0:
            return x, torch.zeros_like(x)

        recon_stack = torch.stack(reconstructions, dim=0)
        weight_tensor = torch.tensor(weights, device=x.device, dtype=x.dtype)
        weight_tensor = weight_tensor / weight_tensor.sum()

        if self.fusion_strategy == "mean":
            fused = recon_stack.mean(dim=0)
        elif self.fusion_strategy == "weighted":
            fused = (recon_stack * weight_tensor.view(-1, 1, 1, 1, 1)).sum(dim=0)
        elif self.fusion_strategy == "median":
            fused = recon_stack.median(dim=0).values
        elif self.fusion_strategy == "best":
            best_idx = weight_tensor.argmax()
            fused = recon_stack[best_idx]
        elif self.fusion_strategy == "uncertainty":
            fused = (recon_stack * weight_tensor.view(-1, 1, 1, 1, 1)).sum(dim=0)
        else:
            fused = recon_stack.mean(dim=0)

        if len(reconstructions) > 1:
            uncertainty = recon_stack.var(dim=0)
        else:
            uncertainty = torch.zeros_like(fused)

        return fused, uncertainty

    def get_ensemble_report(self) -> Dict[str, Any]:
        """Generate comprehensive ensemble status report."""
        report = {
            'total_members': len(self.members),
            'active_members': len(self.active_members),
            'fusion_strategy': self.fusion_strategy,
            'members': {}
        }

        for name, member in self.members.items():
            report['members'][name] = {
                'enabled': member.enabled,
                'healthy': member.is_healthy,
                'weight': member.weight,
                'effective_weight': member.effective_weight,
                'call_count': member.call_count,
                'error_count': member.error_count,
                'avg_latency_ms': member.latency_ema.value,
                'avg_error': member.error_ema.value,
                'last_latency_ms': member.last_latency_ms,
                'architecture': member.model.__class__.__name__,
                'latent_dim': member.model.latent_dim,
            }

        return report

    def print_report(self) -> None:
        """Pretty-print ensemble status."""
        report = self.get_ensemble_report()

        print("=" * 80)
        print("  MODEL ENSEMBLE STATUS")
        print("=" * 80)
        print(f"  Strategy: {report['fusion_strategy']} | "
              f"Active: {report['active_members']}/{report['total_members']}")
        print("-" * 80)
        print(f"  {'Model':<18} {'Arch':<16} {'Weight':>7} {'Lat(ms)':>8} "
              f"{'Error':>8} {'Calls':>7} {'Status':>8}")
        print("-" * 80)

        for name, info in report['members'].items():
            status = "🟢" if info['healthy'] and info['enabled'] else "🔴"
            print(
                f"  {name:<18} {info['architecture']:<16} "
                f"{info['effective_weight']:>7.3f} "
                f"{info['avg_latency_ms']:>8.2f} "
                f"{info['avg_error']:>8.5f} "
                f"{info['call_count']:>7} "
                f"{status:>8}"
            )

        print("=" * 80)


def create_default_ensemble(
    in_channels: int = 1,
    input_size: int = 64,
    latent_dim: int = 32,
    fusion: str = "weighted"
) -> ModelEnsemble:
    """Factory function to create a pre-configured ensemble."""
    from core.models.edge_ae import EdgeStreamingAE
    from core.models.vae import ConvVAE
    from core.models.unet_ae import UNetAutoencoder

    ensemble = ModelEnsemble(fusion_strategy=fusion)

    edge_model = EdgeStreamingAE(
        in_channels=in_channels, input_size=input_size, latent_dim=latent_dim
    )
    ensemble.add_model("edge_ae", edge_model, weight=2.0)

    vae_model = ConvVAE(
        in_channels=in_channels, input_size=input_size, latent_dim=latent_dim
    )
    ensemble.add_model("vae", vae_model, weight=1.0)

    unet_model = UNetAutoencoder(
        in_channels=in_channels, input_size=input_size, latent_dim=latent_dim,
        depth=3, base_channels=8
    )
    ensemble.add_model("unet_ae", unet_model, weight=1.5)

    return ensemble