"""
U-Net Autoencoder with Skip Connections (FIXED)
Encoder-decoder architecture with multi-scale skip connections
that preserve fine-grained spatial details lost in deep bottlenecks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from core.models.base import BaseModel
from core.models.attention import SqueezeExcitation, CBAM, SelfAttention2D


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv → BN → LeakyReLU) × 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        attention: Optional[str] = None
    ):
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv = nn.Sequential(*layers)

        self.attention = None
        if attention == "se":
            self.attention = SqueezeExcitation(out_channels)
        elif attention == "cbam":
            self.attention = CBAM(out_channels)
        elif attention == "self":
            self.attention = SelfAttention2D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class EncoderBlock(nn.Module):
    """Single encoder level: DoubleConv + MaxPool downsampling."""

    def __init__(self, in_channels: int, out_channels: int, attention: Optional[str] = None):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, attention=attention)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_output, skip_connection)."""
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """
    Single decoder level: Upsample + Concat Skip + DoubleConv.

    FIX: The DoubleConv input channels must account for the concatenation
    of the upsampled tensor (in_channels // 2) with the skip tensor
    (skip_channels), giving a total of (in_channels // 2 + skip_channels).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention: Optional[str] = None
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)

        # KEY FIX: concat_channels = upsampled + skip
        concat_channels = in_channels // 2 + skip_channels
        self.conv = DoubleConv(concat_channels, out_channels, attention=attention)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch from odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetAutoencoder(BaseModel):
    """
    U-Net Autoencoder with multi-scale skip connections.

    The latent representation is the bottleneck feature map projected
    to a flat vector via global average pooling for drift detection
    compatibility.
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 64,
        latent_dim: int = 32,
        base_channels: int = 16,
        depth: int = 3,
        attention: Optional[str] = None,
        use_batch_norm: bool = True
    ):
        super().__init__(latent_dim=latent_dim)

        assert 2 <= depth <= 5, f"Depth must be 2-5, got {depth}"
        assert input_size % (2 ** depth) == 0, \
            f"input_size ({input_size}) must be divisible by 2^{depth}={2**depth}"

        self.in_channels = in_channels
        self.input_size = input_size
        self.depth = depth
        self.base_channels = base_channels

        # Channel progression: [base, base*2, base*4, ...]
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # ---- ENCODER ----
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(
                EncoderBlock(in_ch, channels[i + 1], attention=attention)
            )
            in_ch = channels[i + 1]

        # ---- BOTTLENECK ----
        self.bottleneck = DoubleConv(channels[depth], channels[depth], attention=attention)
        self.bottleneck_spatial = input_size // (2 ** depth)
        self.bottleneck_channels = channels[depth]

        # Flat latent projection
        self.latent_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[depth], latent_dim)
        )
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, channels[depth]),
            nn.Unflatten(1, (channels[depth], 1, 1)),
            nn.Upsample(
                size=(self.bottleneck_spatial, self.bottleneck_spatial),
                mode='bilinear', align_corners=False
            )
        )

                # ---- DECODER ----
        self.decoders = nn.ModuleList()
        self._skip_channels = []
        self._skip_spatials = []

        for i in range(depth - 1, -1, -1):
            dec_in_ch = channels[i + 1]
            skip_ch = channels[i + 1]
            dec_out_ch = channels[i]
            # FIX: Skip is captured BEFORE MaxPool, so spatial = input_size // 2^i
            skip_spatial = input_size // (2 ** i)

            self._skip_channels.append(skip_ch)
            self._skip_spatials.append(skip_spatial)

            self.decoders.append(
                DecoderBlock(dec_in_ch, skip_ch, dec_out_ch, attention=attention)
            )

        # ---- OUTPUT ----
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[0], in_channels, 1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to flat latent vector via bottleneck + global pooling."""
        h = x
        for encoder in self.encoders:
            h, _ = encoder(h)
        h = self.bottleneck(h)
        z = self.latent_proj(h)
        return z

    def encode_multiscale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode and return all multi-scale feature maps."""
        features = {}
        h = x
        for i, encoder in enumerate(self.encoders):
            h, skip = encoder(h)
            features[f'encoder_level_{i}'] = skip
        features['bottleneck'] = self.bottleneck(h)
        return features

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode flat latent vector to reconstruction.
        Uses zero-filled skip placeholders for standalone decoding.
        """
        h = self.latent_expand(z)

        for j, decoder in enumerate(self.decoders):
            B = h.size(0)
            skip_ch = self._skip_channels[j]
            skip_sp = self._skip_spatials[j]
            skip = torch.zeros(B, skip_ch, skip_sp, skip_sp, device=h.device)
            h = decoder(h, skip)

        return self.output_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full U-Net forward pass with skip connections."""
        skips = []
        h = x

        for encoder in self.encoders:
            h, skip = encoder(h)
            skips.append(skip)

        h = self.bottleneck(h)

        for i, decoder in enumerate(self.decoders):
            skip = skips[self.depth - 1 - i]
            h = decoder(h, skip)

        return self.output_conv(h)

    def encode_decode_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full encode-decode returning both flat latent and reconstruction."""
        skips = []
        h = x

        for encoder in self.encoders:
            h, skip = encoder(h)
            skips.append(skip)

        h = self.bottleneck(h)
        z = self.latent_proj(h)

        for i, decoder in enumerate(self.decoders):
            skip = skips[self.depth - 1 - i]
            h = decoder(h, skip)

        x_hat = self.output_conv(h)
        return z, x_hat

    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-level anomaly map via reconstruction error."""
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            anomaly_map = torch.mean((x - x_hat) ** 2, dim=1, keepdim=True)
        return anomaly_map

    def get_reconstruction_error_per_region(
        self,
        x: torch.Tensor,
        grid_size: int = 4
    ) -> torch.Tensor:
        """Compute reconstruction error averaged over spatial grid regions."""
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            error = torch.mean((x - x_hat) ** 2, dim=1)

            B, H, W = error.shape
            h_step = H // grid_size
            w_step = W // grid_size

            regional = torch.zeros(B, grid_size, grid_size, device=x.device)
            for i in range(grid_size):
                for j in range(grid_size):
                    region = error[:, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                    regional[:, i, j] = region.mean(dim=[1, 2])

            return regional