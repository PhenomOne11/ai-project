"""
Pluggable Attention Modules for NeuralStream-Core Models
Drop-in attention blocks that can be inserted into any convolutional
architecture to improve feature selectivity and reconstruction quality.

Implements:
- Squeeze-and-Excitation (SE) Channel Attention
- Convolutional Block Attention Module (CBAM)
- Spatial Self-Attention (Non-Local style)
- Multi-Head Spatial Attention
- Gated Attention Unit (GAU)

All modules are designed as nn.Module wrappers that can be inserted
between existing layers with zero architectural changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention.
    
    Learns channel-wise importance weights by:
    1. Squeeze: Global average pooling → (B, C, 1, 1)
    2. Excitation: Two FC layers with bottleneck → channel weights
    3. Scale: Multiply input features by learned weights
    
    Reference: Hu et al. (2018), "Squeeze-and-Excitation Networks"
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for bottleneck (default 16)
        activation: Activation function for the bottleneck
    """
    
    def __init__(self, channels: int, reduction: int = 16, activation: str = "relu"):
        super().__init__()
        
        mid_channels = max(channels // reduction, 4)
        
        act_fn = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            act_fn,
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Channel-recalibrated features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Squeeze
        s = self.squeeze(x).view(B, C)  # (B, C)
        
        # Excitation
        w = self.excitation(s).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Scale
        return x * w


class ChannelAttention(nn.Module):
    """
    Channel attention component of CBAM.
    Uses both average and max pooling for richer channel descriptors.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        
        avg_out = self.shared_mlp(self.avg_pool(x).view(B, C))
        max_out = self.shared_mlp(self.max_pool(x).view(B, C))
        
        weights = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    """
    Spatial attention component of CBAM.
    Generates a 2D spatial attention map by pooling across channels.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv(combined))     # (B, 1, H, W)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies channel attention then spatial attention.
    Can be inserted after any convolutional block.
    
    Reference: Woo et al. (2018), "CBAM: Convolutional Block Attention Module"
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio
        spatial_kernel: Kernel size for spatial attention
    """
    
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attention-refined features (B, C, H, W)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SelfAttention2D(nn.Module):
    """
    Spatial Self-Attention for 2D feature maps (Non-Local style).
    
    Computes pairwise attention between all spatial positions:
    Attention(i, j) = softmax(φ(x_i)^T θ(x_j)) * g(x_j)
    
    Uses 1×1 convolutions for query, key, and value projections.
    Includes a learnable residual scaling factor (initialized to 0).
    
    Reference: Wang et al. (2018), "Non-local Neural Networks"
    
    Args:
        in_channels: Number of input channels
        inter_channels: Intermediate channel count (default: in_channels // 8)
        mode: Attention mode ('embedded_gaussian', 'dot_product', 'gaussian')
    """
    
    def __init__(
        self,
        in_channels: int,
        inter_channels: Optional[int] = None,
        mode: str = "embedded_gaussian"
    ):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = max(in_channels // 8, 1)
        
        self.mode = mode
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.query = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.key = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.value = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.output_proj = nn.Conv2d(inter_channels, in_channels, 1, bias=False)
        
        # Initialize output projection to zero for residual stability
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        
        self.norm = nn.GroupNorm(
            num_groups=min(8, in_channels),
            num_channels=in_channels
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Self-attended features (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W  # Number of spatial positions
        
        # Project to query, key, value
        q = self.query(x).view(B, self.inter_channels, N)   # (B, C', N)
        k = self.key(x).view(B, self.inter_channels, N)     # (B, C', N)
        v = self.value(x).view(B, self.inter_channels, N)   # (B, C', N)
        
        # Compute attention: (B, N, N)
        if self.mode == "embedded_gaussian" or self.mode == "gaussian":
            attn = torch.bmm(q.transpose(1, 2), k)  # (B, N, N)
            attn = attn / math.sqrt(self.inter_channels)
            attn = F.softmax(attn, dim=-1)
        elif self.mode == "dot_product":
            attn = torch.bmm(q.transpose(1, 2), k)  # (B, N, N)
            attn = attn / N  # Normalize by spatial size
        else:
            attn = torch.bmm(q.transpose(1, 2), k)
            attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values: (B, C', N)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, self.inter_channels, H, W)
        
        # Project back and residual
        out = self.output_proj(out)
        out = self.norm(out)
        
        return x + out


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-Head Spatial Attention for 2D feature maps.
    Splits channels into multiple heads, each computing independent
    spatial attention, then concatenates and projects.
    
    Args:
        in_channels: Number of input channels
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: in_channels // num_heads)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = head_dim or in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)
        
        nn.init.zeros_(self.proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        # Compute Q, K, V
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, N)
        
        # Attention per head: (B, heads, N, N)
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention: (B, heads, head_dim, N)
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        
        # Reshape and project
        out = out.contiguous().view(B, C, H, W)
        out = self.proj(out)
        out = self.norm(out)
        
        return x + out


class GatedAttentionUnit(nn.Module):
    """
    Gated Attention Unit (GAU) — simplified efficient attention.
    
    Combines gating mechanism with quadratic attention for
    linear-complexity spatial reasoning.
    
    Reference: Hua et al. (2022), "Transformer Quality in Linear Time"
    
    Args:
        channels: Number of input/output channels
        expansion_factor: Hidden dimension expansion (default 2)
    """
    
    def __init__(self, channels: int, expansion_factor: int = 2):
        super().__init__()
        
        hidden = channels * expansion_factor
        
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.proj_uv = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.proj_qk = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_out = nn.Conv2d(hidden, channels, 1, bias=False)
        
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        residual = x
        x_norm = self.norm(x)
        
        # Gate and value projections
        uv = self.proj_uv(x_norm)  # (B, 2*hidden, H, W)
        u, v = uv.chunk(2, dim=1)  # Each: (B, hidden, H, W)
        u = F.silu(u)
        
        # Query-Key projection
        qk = self.proj_qk(x_norm)  # (B, C, H, W)
        qk = qk.view(B, C, N)
        
        # Quadratic attention: QK^T / N
        attn = torch.bmm(qk.transpose(1, 2), qk) / N  # (B, N, N)
        attn = F.relu(attn) ** 2  # Squared ReLU for efficiency
        
        # Apply attention to value
        v_flat = v.view(B, -1, N)  # (B, hidden, N)
        out = torch.bmm(v_flat, attn)  # (B, hidden, N)
        out = out.view(B, -1, H, W)
        
        # Gate
        out = u * out
        
        # Project and residual
        out = self.proj_out(out)
        out = out * self.gamma + self.beta
        
        return residual + out