"""
EdgeStreamingAE — Lightweight Convolutional Autoencoder
Optimized for real-time edge inference with minimal latency.

Architecture:
    Encoder: Conv2d → BN → LeakyReLU → Conv2d → BN → LeakyReLU → Flatten → Linear
    Decoder: Linear → Reshape → ConvTranspose2d → BN → LeakyReLU → ConvTranspose2d → Sigmoid

Features:
    - Configurable depth and channel widths
    - Optional batch normalization
    - Dropout for regularization
    - Residual skip connections (optional)
    - LSUV-compatible layer naming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from core.models.base import BaseModel


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block for convolutional layers.
    Uses identity shortcut when dimensions match, 1x1 conv otherwise.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class EdgeStreamingAE(BaseModel):
    """
    Edge-optimized streaming autoencoder for real-time anomaly detection.
    
    Designed for sub-3ms inference on CPU with 64×64 single-channel inputs.
    The 32-dimensional latent bottleneck captures essential spatial features
    while enabling efficient drift monitoring and OOD detection.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        input_size: Spatial dimension of square input (e.g., 64)
        latent_dim: Dimensionality of the latent bottleneck
        base_channels: Starting channel count (doubles each layer)
        use_batch_norm: Enable batch normalization
        use_residual: Enable residual skip connections
        dropout_rate: Dropout probability in the latent layer
        activation: Activation function name ('leaky_relu', 'relu', 'gelu')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 64,
        latent_dim: int = 32,
        base_channels: int = 16,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        dropout_rate: float = 0.1,
        activation: str = "leaky_relu"
    ):
        super().__init__(latent_dim=latent_dim)
        
        self.in_channels = in_channels
        self.input_size = input_size
        self.base_channels = base_channels
        self.use_residual = use_residual
        
        # Select activation
        act_map = {
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        self.act_fn_name = activation
        act = act_map.get(activation, nn.LeakyReLU(0.1, inplace=True))
        
        # Compute spatial dimensions through encoder
        # Conv1: input_size → input_size//2 (stride 2)
        # Conv2: input_size//2 → input_size//4 (stride 2)
        self.spatial_after_conv = input_size // 4
        self.flat_dim = base_channels * 2 * self.spatial_after_conv * self.spatial_after_conv
        
        # ---- ENCODER ----
        self.encoder_conv = nn.Sequential()
        
        # Block 1: in_channels → base_channels
        self.encoder_conv.add_module('0', nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=not use_batch_norm
        ))
        if use_batch_norm:
            self.encoder_conv.add_module('1_bn', nn.BatchNorm2d(base_channels))
        self.encoder_conv.add_module('1_act', act.__class__(0.1, inplace=True) if hasattr(act, 'negative_slope') else act.__class__())
        
        # Block 2: base_channels → base_channels*2
        self.encoder_conv.add_module('2', nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=not use_batch_norm
        ))
        if use_batch_norm:
            self.encoder_conv.add_module('3_bn', nn.BatchNorm2d(base_channels * 2))
        self.encoder_conv.add_module('3_act', act.__class__(0.1, inplace=True) if hasattr(act, 'negative_slope') else act.__class__())
        
        # Optional residual refinement
        if use_residual:
            self.residual_block = ResidualBlock(base_channels * 2, base_channels * 2)
        
        # Flatten and project to latent
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # ---- DECODER ----
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, self.flat_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.decoder_conv = nn.Sequential()
        
        # Block 1: base_channels*2 → base_channels
        self.decoder_conv.add_module('0', nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=not use_batch_norm
        ))
        if use_batch_norm:
            self.decoder_conv.add_module('1_bn', nn.BatchNorm2d(base_channels))
        self.decoder_conv.add_module('1_act', nn.LeakyReLU(0.1, inplace=True))
        
        # Block 2: base_channels → in_channels
        self.decoder_conv.add_module('2', nn.ConvTranspose2d(
            base_channels, in_channels, kernel_size=4, stride=2, padding=1
        ))
        self.decoder_conv.add_module('3_act', nn.Sigmoid())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent vector."""
        h = self.encoder_conv(x)
        if self.use_residual:
            h = self.residual_block(h)
        z = self.encoder_fc(h)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed image."""
        h = self.decoder_fc(z)
        h = h.view(-1, self.base_channels * 2, self.spatial_after_conv, self.spatial_after_conv)
        x_hat = self.decoder_conv(h)
        return x_hat
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode forward pass."""
        z = self.encode(x)
        return self.decode(z)
    
    def get_latent_statistics(self, x: torch.Tensor) -> dict:
        """Compute statistics of the latent representation for a batch."""
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
            return {
                'mean': z.mean(dim=0).cpu().numpy(),
                'std': z.std(dim=0).cpu().numpy(),
                'min': z.min().item(),
                'max': z.max().item(),
                'norm_mean': z.norm(dim=1).mean().item(),
                'norm_std': z.norm(dim=1).std().item(),
            }