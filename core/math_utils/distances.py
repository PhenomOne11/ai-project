"""
Statistical Distance & Divergence Functions
Complete library of distribution comparison metrics for drift detection.

Implements:
- Wasserstein-1 (Earth Mover's Distance)
- Wasserstein-2
- Maximum Mean Discrepancy (MMD) with RBF kernel
- KL Divergence (symmetrized)
- Jensen-Shannon Divergence
- Kolmogorov-Smirnov Statistic
- Energy Distance
- Mahalanobis Distance
- Cosine Distance
- Hellinger Distance
"""

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein
from scipy.stats import ks_2samp, entropy
from typing import Optional, Tuple


def wasserstein_1(p: np.ndarray, q: np.ndarray) -> float:
    """
    Wasserstein-1 (Earth Mover's) distance between two 1D distributions.
    
    W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx
    
    Uses scipy's efficient implementation based on sorted CDF differences.
    """
    p_flat = p.flatten().astype(np.float64)
    q_flat = q.flatten().astype(np.float64)
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    # Remove non-finite values
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    return float(scipy_wasserstein(p_flat, q_flat))


def wasserstein_2(p: np.ndarray, q: np.ndarray) -> float:
    """
    Wasserstein-2 distance between two 1D distributions.
    
    W_2(P, Q) = sqrt(integral (F_P^{-1}(t) - F_Q^{-1}(t))^2 dt)
    
    Computed via sorted quantile differences.
    """
    p_flat = np.sort(p.flatten().astype(np.float64))
    q_flat = np.sort(q.flatten().astype(np.float64))
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    # Interpolate to common quantile grid
    n = min(len(p_flat), len(q_flat), 1000)
    quantiles = np.linspace(0, 1, n)
    
    p_quantiles = np.quantile(p_flat, quantiles)
    q_quantiles = np.quantile(q_flat, quantiles)
    
    return float(np.sqrt(np.mean((p_quantiles - q_quantiles) ** 2)))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: Optional[float] = None) -> float:
    """
    Maximum Mean Discrepancy with Radial Basis Function kernel.
    
    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    where k(a,b) = exp(-||a-b||² / (2σ²))
    
    Args:
        X: Samples from distribution P, shape (n, d) or (n,)
        Y: Samples from distribution Q, shape (m, d) or (m,)
        bandwidth: RBF kernel bandwidth σ. If None, uses median heuristic.
    
    Returns:
        MMD² estimate (non-negative float)
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X.astype(np.float64)
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y.astype(np.float64)
    
    X = X[np.all(np.isfinite(X), axis=1)]
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    # Subsample for efficiency
    max_samples = 500
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
    if len(Y) > max_samples:
        idx = np.random.choice(len(Y), max_samples, replace=False)
        Y = Y[idx]
    
    # Bandwidth via median heuristic
    if bandwidth is None:
        XY = np.vstack([X, Y])
        dists = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=2)
        bandwidth = max(float(np.sqrt(np.median(dists[dists > 0]))), 1e-6)
    
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    
    # Compute kernel matrices
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY_dist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY_dist)
    
    n = len(X)
    m = len(Y)
    
    # Unbiased estimator
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    if n > 1:
        term1 = K_XX.sum() / (n * (n - 1))
    else:
        term1 = 0.0
    
    if m > 1:
        term2 = K_YY.sum() / (m * (m - 1))
    else:
        term2 = 0.0
    
    term3 = 2.0 * K_XY.sum() / (n * m)
    
    mmd_squared = max(term1 + term2 - term3, 0.0)
    return float(np.sqrt(mmd_squared))


def kl_divergence(p: np.ndarray, q: np.ndarray, num_bins: int = 50) -> float:
    """
    Symmetrized KL Divergence via histogram estimation.
    
    KL_sym(P||Q) = 0.5 * [KL(P||Q) + KL(Q||P)]
    
    Uses Laplace smoothing to avoid division by zero.
    """
    p_flat = p.flatten().astype(np.float64)
    q_flat = q.flatten().astype(np.float64)
    
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    # Common range
    all_vals = np.concatenate([p_flat, q_flat])
    range_min, range_max = all_vals.min(), all_vals.max()
    
    if range_max - range_min < 1e-12:
        return 0.0
    
    bins = np.linspace(range_min, range_max, num_bins + 1)
    
    p_hist, _ = np.histogram(p_flat, bins=bins, density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, density=True)
    
    # Laplace smoothing
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    
    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    
    kl_pq = float(entropy(p_hist, q_hist))
    kl_qp = float(entropy(q_hist, p_hist))
    
    return 0.5 * (kl_pq + kl_qp)


def jensen_shannon(p: np.ndarray, q: np.ndarray, num_bins: int = 50) -> float:
    """
    Jensen-Shannon Divergence (symmetric, bounded [0, ln(2)]).
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
    """
    p_flat = p.flatten().astype(np.float64)
    q_flat = q.flatten().astype(np.float64)
    
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    all_vals = np.concatenate([p_flat, q_flat])
    bins = np.linspace(all_vals.min(), all_vals.max(), num_bins + 1)
    
    p_hist, _ = np.histogram(p_flat, bins=bins, density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, density=True)
    
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    
    m = 0.5 * (p_hist + q_hist)
    
    jsd = 0.5 * entropy(p_hist, m) + 0.5 * entropy(q_hist, m)
    return float(jsd)


def ks_statistic(p: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov two-sample test statistic.
    
    Returns:
        (ks_statistic, p_value)
    """
    p_flat = p.flatten().astype(np.float64)
    q_flat = q.flatten().astype(np.float64)
    
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) < 2 or len(q_flat) < 2:
        return 0.0, 1.0
    
    stat, pval = ks_2samp(p_flat, q_flat)
    return float(stat), float(pval)


def energy_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Energy distance between two distributions.
    
    D_E(P, Q) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|
    
    Computed via pairwise L2 norms.
    """
    p_flat = p.flatten().astype(np.float64)
    q_flat = q.flatten().astype(np.float64)
    
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    # Subsample for efficiency
    max_n = 500
    if len(p_flat) > max_n:
        p_flat = np.random.choice(p_flat, max_n, replace=False)
    if len(q_flat) > max_n:
        q_flat = np.random.choice(q_flat, max_n, replace=False)
    
    # E|X - Y|
    cross = np.mean(np.abs(p_flat[:, None] - q_flat[None, :]))
    
    # E|X - X'|
    if len(p_flat) > 1:
        pp = np.mean(np.abs(p_flat[:, None] - p_flat[None, :]))
    else:
        pp = 0.0
    
    # E|Y - Y'|
    if len(q_flat) > 1:
        qq = np.mean(np.abs(q_flat[:, None] - q_flat[None, :]))
    else:
        qq = 0.0
    
    return float(max(2.0 * cross - pp - qq, 0.0))


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray
) -> float:
    """
    Mahalanobis distance of point x from distribution (mean, covariance).
    
    D_M = sqrt((x - μ)^T Σ^{-1} (x - μ))
    """
    x = x.flatten().astype(np.float64)
    mean = mean.flatten().astype(np.float64)
    
    delta = x - mean
    left = delta @ cov_inv
    dist_sq = left @ delta
    
    return float(np.sqrt(max(dist_sq, 0.0)))


def mahalanobis_batch(
    X: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray
) -> np.ndarray:
    """
    Mahalanobis distance for a batch of points.
    
    Args:
        X: Shape (N, D)
        mean: Shape (D,)
        cov_inv: Shape (D, D)
    
    Returns:
        Array of shape (N,) with distances
    """
    X = X.astype(np.float64)
    mean = mean.flatten().astype(np.float64)
    
    delta = X - mean[None, :]
    left = delta @ cov_inv  # (N, D)
    dist_sq = np.sum(left * delta, axis=1)  # (N,)
    
    return np.sqrt(np.maximum(dist_sq, 0.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance: 1 - cosine_similarity.
    Range: [0, 2].
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    
    similarity = np.dot(a, b) / (norm_a * norm_b)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return float(1.0 - similarity)


def hellinger_distance(p: np.ndarray, q: np.ndarray, num_bins: int = 50) -> float:
    """
    Hellinger distance between two distributions.
    
    H(P, Q) = (1/√2) * sqrt(Σ(√p_i - √q_i)²)
    
    Range: [0, 1].
    """
    p_flat = p.flatten().astype(np.float64)
    q_flat = q.flatten().astype(np.float64)
    
    p_flat = p_flat[np.isfinite(p_flat)]
    q_flat = q_flat[np.isfinite(q_flat)]
    
    if len(p_flat) == 0 or len(q_flat) == 0:
        return 0.0
    
    all_vals = np.concatenate([p_flat, q_flat])
    bins = np.linspace(all_vals.min(), all_vals.max(), num_bins + 1)
    
    p_hist, _ = np.histogram(p_flat, bins=bins, density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, density=True)
    
    p_hist = p_hist / max(p_hist.sum(), 1e-12)
    q_hist = q_hist / max(q_hist.sum(), 1e-12)
    
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_hist) - np.sqrt(q_hist)) ** 2)))