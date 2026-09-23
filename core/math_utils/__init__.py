"""
NeuralStream-Core Math Utilities Package
Public API for all mathematical primitives.
"""

# Statistics
from core.math_utils.statistics import (
    WelfordAccumulator,
    WelfordVectorAccumulator,
    ExponentialMovingAverage,
    ExponentialMovingVector,
    RollingWindow,
    HistogramAccumulator,
    MultiChannelTracker,
)

# Linear Algebra
from core.math_utils.linear_algebra import (
    StreamingPCA,
    OnlineCovarianceMatrix,
    gram_schmidt,
    batch_svd_project,
)

# Distance Functions
from core.math_utils.distances import (
    wasserstein_1,
    wasserstein_2,
    mmd_rbf,
    kl_divergence,
    jensen_shannon,
    ks_statistic,
    energy_distance,
    mahalanobis_distance,
    mahalanobis_batch,
    cosine_distance,
    hellinger_distance,
)

# Signal Processing
from core.math_utils.signal_processing import (
    compute_fft,
    power_spectral_density,
    spectral_entropy,
    signal_to_noise_ratio,
    find_peaks,
    band_energy,
    zero_crossing_rate,
    apply_window,
)

# Tensor Operations
from core.math_utils.tensor_ops import (
    im2col,
    col2im,
    extract_patches_2d,
    safe_normalize,
    safe_log,
    safe_divide,
    check_tensor_health,
    sanitize_tensor,
    flatten_except_batch,
    compute_output_size,
    count_parameters,
    model_memory_mb,
)

# Initializers
from core.math_utils.initializers import (
    lsuv_init,
    kaiming_init,
    xavier_init,
    orthogonal_init,
    get_init_diagnostics,
)

__all__ = [
    # Statistics
    'WelfordAccumulator', 'WelfordVectorAccumulator',
    'ExponentialMovingAverage', 'ExponentialMovingVector',
    'RollingWindow', 'HistogramAccumulator', 'MultiChannelTracker',
    # Linear Algebra
    'StreamingPCA', 'OnlineCovarianceMatrix', 'gram_schmidt', 'batch_svd_project',
    # Distances
    'wasserstein_1', 'wasserstein_2', 'mmd_rbf', 'kl_divergence',
    'jensen_shannon', 'ks_statistic', 'energy_distance',
    'mahalanobis_distance', 'mahalanobis_batch', 'cosine_distance', 'hellinger_distance',
    # Signal Processing
    'compute_fft', 'power_spectral_density', 'spectral_entropy',
    'signal_to_noise_ratio', 'find_peaks', 'band_energy', 'zero_crossing_rate', 'apply_window',
    # Tensor Ops
    'im2col', 'col2im', 'extract_patches_2d', 'safe_normalize', 'safe_log', 'safe_divide',
    'check_tensor_health', 'sanitize_tensor', 'flatten_except_batch',
    'compute_output_size', 'count_parameters', 'model_memory_mb',
    # Initializers
    'lsuv_init', 'kaiming_init', 'xavier_init', 'orthogonal_init', 'get_init_diagnostics',
]