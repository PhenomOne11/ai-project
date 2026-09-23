"""
NeuralStream-Core Data Ingestion Package
"""

from core.streams.base import (
    DataStream, CyclingIterator, infinite_batches,
    normalize_batch, sanitize_batch,
)
from core.streams.synthetic import (
    SyntheticStream, SinusoidalStream, MixedSignalStream, SpikeStream,
)
from core.streams.loaders import (
    load_npy_to_tensor, load_pt_to_tensor,
    load_images_to_tensor, load_csv_to_tensor,
)
from core.streams.signals import AudioStream, TimeSeriesStream
from core.streams.augment import AugmentationPipeline


def build_stream(source_type: str, **kwargs) -> DataStream:
    """
    Factory for creating streams by string name.

    Supported types:
        'synthetic'    → SyntheticStream
        'sinusoidal'   → SinusoidalStream
        'mixed'        → MixedSignalStream
        'spike'        → SpikeStream
        'audio'        → AudioStream
        'timeseries'   → TimeSeriesStream
    """
    registry = {
        'synthetic': SyntheticStream,
        'sinusoidal': SinusoidalStream,
        'mixed': MixedSignalStream,
        'spike': SpikeStream,
        'audio': AudioStream,
        'timeseries': TimeSeriesStream,
    }
    key = source_type.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown stream type '{source_type}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[key](**kwargs)


__all__ = [
    # Base
    'DataStream', 'CyclingIterator', 'infinite_batches',
    'normalize_batch', 'sanitize_batch',
    # Synthetic
    'SyntheticStream', 'SinusoidalStream', 'MixedSignalStream', 'SpikeStream',
    # Loaders
    'load_npy_to_tensor', 'load_pt_to_tensor',
    'load_images_to_tensor', 'load_csv_to_tensor',
    # Signals
    'AudioStream', 'TimeSeriesStream',
    # Augment
    'AugmentationPipeline',
    # Factory
    'build_stream',
]