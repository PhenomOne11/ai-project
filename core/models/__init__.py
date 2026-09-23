"""
NeuralStream-Core Model Zoo
All available model architectures, attention modules, and ensemble system.
"""

# Base
from core.models.base import BaseModel

# Architectures
from core.models.edge_ae import EdgeStreamingAE, ResidualBlock
from core.models.vae import ConvVAE
from core.models.transformer_ae import (
    TransformerAutoencoder,
    PositionalEncoding,
    PatchEmbedding,
    PatchUnembedding,
)
from core.models.unet_ae import UNetAutoencoder, DoubleConv, EncoderBlock, DecoderBlock

# Attention Modules
from core.models.attention import (
    SqueezeExcitation,
    ChannelAttention,
    SpatialAttention,
    CBAM,
    SelfAttention2D,
    MultiHeadSpatialAttention,
    GatedAttentionUnit,
)

# Ensemble
from core.models.ensemble import (
    ModelEnsemble,
    EnsembleMember,
    create_default_ensemble,
)

# Registry
from core.models.registry import ModelRegistry

__all__ = [
    # Base
    'BaseModel',
    # Architectures
    'EdgeStreamingAE', 'ResidualBlock',
    'ConvVAE',
    'TransformerAutoencoder', 'PositionalEncoding', 'PatchEmbedding', 'PatchUnembedding',
    'UNetAutoencoder', 'DoubleConv', 'EncoderBlock', 'DecoderBlock',
    # Attention
    'SqueezeExcitation', 'ChannelAttention', 'SpatialAttention',
    'CBAM', 'SelfAttention2D', 'MultiHeadSpatialAttention', 'GatedAttentionUnit',
    # Ensemble
    'ModelEnsemble', 'EnsembleMember', 'create_default_ensemble',
    # Registry
    'ModelRegistry',
]