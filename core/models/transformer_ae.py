"""
Transformer Autoencoder
Attention-based autoencoder that captures long-range spatial dependencies
missed by purely convolutional architectures.

Architecture:
    Encoder: Patch Embedding → Positional Encoding → Transformer Encoder Layers
    Decoder: Learned Query → Transformer Decoder Layers → Patch Unembedding

Features:
    - Sinusoidal or learned positional encoding
    - Multi-head self-attention with configurable heads
    - Patch-based tokenization (ViT-style)
    - Cross-attention decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from core.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer sequences.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Convert image patches to token embeddings (ViT-style).
    
    Splits input image into non-overlapping patches and projects
    each to a d_model-dimensional vector.
    """
    
    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, d_model)
        """
        x = self.proj(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        return x


class PatchUnembedding(nn.Module):
    """
    Reverse patch embedding: reconstruct spatial feature map from tokens.
    """
    
    def __init__(self, d_model: int, patch_size: int, out_channels: int, spatial_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.spatial_size = spatial_size
        self.num_patches_h = spatial_size // patch_size
        self.num_patches_w = spatial_size // patch_size
        
        self.proj = nn.ConvTranspose2d(
            d_model, out_channels,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, d_model)
        Returns:
            (B, out_channels, H, W)
        """
        B, N, D = x.shape
        x = x.transpose(1, 2).view(B, D, self.num_patches_h, self.num_patches_w)
        x = self.proj(x)
        return x


class TransformerAutoencoder(BaseModel):
    """
    Transformer-based Autoencoder for spatial data.
    
    Tokenizes the input into patches, processes through transformer
    encoder layers, compresses to a latent bottleneck via pooling,
    then reconstructs through transformer decoder layers.
    
    Args:
        in_channels: Input channel count
        input_size: Spatial dimension (must be divisible by patch_size)
        latent_dim: Latent bottleneck dimensionality
        patch_size: Size of each image patch token
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Transformer encoder depth
        num_decoder_layers: Transformer decoder depth
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 64,
        latent_dim: int = 32,
        patch_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__(latent_dim=latent_dim)
        
        assert input_size % patch_size == 0, \
            f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
        
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (input_size // patch_size) ** 2
        
        # ---- TOKENIZATION ----
        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 1, dropout=dropout)
        
        # Learnable CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # ---- TRANSFORMER ENCODER ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # ---- LATENT BOTTLENECK ----
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
        # ---- TRANSFORMER DECODER ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Learnable query tokens for decoder
        self.decoder_queries = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )
        
        # ---- RECONSTRUCTION ----
        self.patch_unembed = PatchUnembedding(d_model, patch_size, in_channels, input_size)
        self.output_act = nn.Sigmoid()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent vector via CLS token pooling."""
        B = x.size(0)
        
        # Tokenize
        tokens = self.patch_embed(x)  # (B, num_patches, d_model)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, num_patches+1, d_model)
        
        # Add positional encoding
        tokens = self.pos_encoder(tokens)
        
        # Transformer encoding
        encoded = self.transformer_encoder(tokens)
        
        # Pool via CLS token
        cls_output = encoded[:, 0, :]  # (B, d_model)
        
        # Project to latent
        z = self.latent_proj(cls_output)  # (B, latent_dim)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image via cross-attention."""
        B = z.size(0)
        
        # Expand latent to memory tokens
        memory = self.latent_expand(z).unsqueeze(1)  # (B, 1, d_model)
        
        # Decoder queries
        queries = self.decoder_queries.expand(B, -1, -1)  # (B, num_patches, d_model)
        queries = self.pos_encoder(queries)
        
        # Cross-attention decoding
        decoded = self.transformer_decoder(queries, memory)  # (B, num_patches, d_model)
        
        # Reconstruct spatial output
        x_hat = self.patch_unembed(decoded)  # (B, C, H, W)
        x_hat = self.output_act(x_hat)
        
        return x_hat
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode forward pass."""
        z = self.encode(x)
        return self.decode(z)
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract self-attention weights from the last encoder layer.
        Useful for visualizing which patches the model attends to.
        
        Note: Requires PyTorch >= 2.0 with need_weights=True support.
        """
        try:
            B = x.size(0)
            tokens = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            tokens = self.pos_encoder(tokens)
            
            # Pass through all but last layer
            for i in range(len(self.transformer_encoder.layers) - 1):
                tokens = self.transformer_encoder.layers[i](tokens)
            
            # Last layer with attention weights
            last_layer = self.transformer_encoder.layers[-1]
            output, attn_weights = last_layer.self_attn(
                tokens, tokens, tokens, need_weights=True, average_attn_weights=True
            )
            return attn_weights
        except Exception:
            return None