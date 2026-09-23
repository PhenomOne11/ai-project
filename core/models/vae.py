"""
Variational Autoencoder (VAE) with KL Annealing
Generative model that learns a smooth, continuous latent space
suitable for interpolation, sampling, and density estimation.

Architecture:
    Encoder: Conv blocks → μ, log(σ²) heads
    Decoder: Linear → ConvTranspose blocks
    Loss: Reconstruction (MSE/BCE) + β-KL Divergence

Features:
    - Reparameterization trick for differentiable sampling
    - KL annealing schedule (linear warmup)
    - Free-bits KL regularization
    - Latent space interpolation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from core.models.base import BaseModel


class ConvVAE(BaseModel):
    """
    Convolutional Variational Autoencoder.
    
    Unlike the deterministic EdgeStreamingAE, the VAE learns a probabilistic
    latent space by parameterizing a Gaussian distribution q(z|x) = N(μ, σ²I)
    and regularizing it against a standard normal prior via KL divergence.
    
    Args:
        in_channels: Number of input channels
        input_size: Spatial dimension of square input
        latent_dim: Dimensionality of the latent space
        base_channels: Starting channel count
        beta: KL divergence weight (β-VAE formulation)
        kl_anneal_steps: Number of steps for linear KL warmup
        free_bits: Minimum KL per dimension (prevents posterior collapse)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 64,
        latent_dim: int = 32,
        base_channels: int = 16,
        beta: float = 1.0,
        kl_anneal_steps: int = 1000,
        free_bits: float = 0.0
    ):
        super().__init__(latent_dim=latent_dim)
        
        self.in_channels = in_channels
        self.input_size = input_size
        self.beta = beta
        self.kl_anneal_steps = kl_anneal_steps
        self.free_bits = free_bits
        self._global_step = 0
        
        spatial = input_size // 4
        self.spatial = spatial
        flat_dim = base_channels * 2 * spatial * spatial
        self.flat_dim = flat_dim
        
        # ---- ENCODER ----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        
        # ---- DECODER ----
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent mean (deterministic path for inference)."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h)
    
    def encode_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to distribution parameters.
        
        Returns:
            (mu, logvar) each of shape (batch, latent_dim)
        """
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I).
        Enables backpropagation through the stochastic sampling step.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(-1, self.flat_dim // (self.spatial * self.spatial), self.spatial, self.spatial)
        return self.decoder_conv(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        
        Returns:
            (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction + β * KL.
        
        Returns dict with individual loss components for logging.
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat, x, reduction=reduction)
        
        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Free-bits regularization
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        
        kl_loss = kl_per_dim.sum(dim=1)
        if reduction == "mean":
            kl_loss = kl_loss.mean()
        else:
            kl_loss = kl_loss.sum()
        
        # KL annealing
        anneal_factor = min(1.0, self._global_step / max(self.kl_anneal_steps, 1))
        effective_beta = self.beta * anneal_factor
        
        total_loss = recon_loss + effective_beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'effective_beta': torch.tensor(effective_beta),
            'anneal_factor': torch.tensor(anneal_factor),
        }
    
    def step(self) -> None:
        """Increment global step counter (call once per training batch)."""
        self._global_step += 1
    
    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate new samples by decoding random latent vectors from the prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Target device
        
        Returns:
            Generated images of shape (num_samples, C, H, W)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decode(z)
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two inputs in latent space.
        
        Args:
            x1: First input (1, C, H, W)
            x2: Second input (1, C, H, W)
            num_steps: Number of interpolation steps
        
        Returns:
            Interpolated reconstructions (num_steps, C, H, W)
        """
        self.eval()
        with torch.no_grad():
            z1 = self.encode(x1)
            z2 = self.encode(x2)
            
            alphas = torch.linspace(0, 1, num_steps, device=z1.device).view(-1, 1)
            z_interp = (1 - alphas) * z1 + alphas * z2
            
            return self.decode(z_interp)
    
    def get_latent_statistics(self, x: torch.Tensor) -> dict:
        """Compute latent distribution statistics."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode_distribution(x)
            return {
                'mu_mean': mu.mean(dim=0).cpu().numpy(),
                'mu_std': mu.std(dim=0).cpu().numpy(),
                'logvar_mean': logvar.mean(dim=0).cpu().numpy(),
                'logvar_std': logvar.std(dim=0).cpu().numpy(),
                'kl_per_dim': (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0).cpu().numpy(),
            }