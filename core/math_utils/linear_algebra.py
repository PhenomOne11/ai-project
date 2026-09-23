"""
Streaming Linear Algebra Utilities
Incremental PCA, covariance tracking, and projection operations
designed for continuous real-time data streams.

Implements:
- Incremental SVD-based PCA (no full dataset required)
- Online covariance matrix accumulator
- Efficient low-rank projections
- Gram-Schmidt orthogonalization
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StreamingPCA:
    """
    Incremental PCA via rank-1 SVD updates.
    Projects high-dimensional streaming vectors into a low-rank subspace
    without ever storing the full dataset.
    
    Algorithm: Maintains running mean and top-k singular vectors.
    On each new sample, performs a rank-1 update to the SVD.
    """
    n_components: int = 3
    count: int = 0
    mean: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None     # (n_components, dim)
    singular_values: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None
    _initialized: bool = False
    _buffer: list = field(default_factory=list)
    _warmup_size: int = 0
    
    def __post_init__(self):
        self._warmup_size = max(self.n_components * 3, 20)
        self._buffer = []
    
    def partial_fit(self, X: np.ndarray) -> None:
        """
        Incrementally update PCA with new observations.
        
        Args:
            X: Array of shape (n_samples, n_features) or (n_features,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X = X.astype(np.float64)
        
        if not self._initialized:
            self._buffer.extend(X)
            if len(self._buffer) >= self._warmup_size:
                self._bootstrap(np.array(self._buffer))
                self._buffer = []
            return
        
        for row in X:
            self._rank1_update(row)
    
    def _bootstrap(self, X: np.ndarray) -> None:
        """Initial full SVD on warmup buffer."""
        self.count = X.shape[0]
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        
        k = min(self.n_components, len(S))
        self.components = Vt[:k].copy()
        self.singular_values = S[:k].copy()
        self.explained_variance = (S[:k] ** 2) / (self.count - 1)
        self._initialized = True
    
    def _rank1_update(self, x: np.ndarray) -> None:
        """
        Rank-1 SVD update using the projection-residual method.
        Updates mean, components, and singular values incrementally.
        """
        self.count += 1
        
        # Update running mean
        old_mean = self.mean.copy()
        self.mean = old_mean + (x - old_mean) / self.count
        
        # Center the new observation
        xc = x - self.mean
        
        # Project onto existing components
        projections = self.components @ xc  # (k,)
        
        # Compute residual
        reconstruction = self.components.T @ projections  # (dim,)
        residual = xc - reconstruction
        residual_norm = np.linalg.norm(residual)
        
        if residual_norm < 1e-12:
            return
        
        residual_unit = residual / residual_norm
        
        # Build augmented matrix for SVD update
        k = len(self.singular_values)
        B = np.zeros((k + 1, k + 1))
        B[:k, :k] = np.diag(self.singular_values)
        B[:k, k] = projections
        B[k, k] = residual_norm
        
        # SVD of small augmented matrix
        U_small, S_new, Vt_small = np.linalg.svd(B, full_matrices=False)
        
        # Keep only top n_components
        keep = min(self.n_components, len(S_new))
        
        # Update components
        augmented_basis = np.vstack([self.components, residual_unit.reshape(1, -1)])
        self.components = (Vt_small[:keep] @ augmented_basis) if keep > 0 else self.components
        self.singular_values = S_new[:keep]
        self.explained_variance = (self.singular_values ** 2) / max(self.count - 1, 1)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data into the learned principal subspace.
        
        Args:
            X: Array of shape (n_samples, n_features) or (n_features,)
        
        Returns:
            Projected array of shape (n_samples, n_components)
        """
        if not self._initialized:
            raise RuntimeError("PCA not initialized. Call partial_fit first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        Xc = X.astype(np.float64) - self.mean
        return Xc @ self.components.T
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct from principal components back to original space."""
        if not self._initialized:
            raise RuntimeError("PCA not initialized.")
        
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        
        return Z @ self.components + self.mean
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each component."""
        if self.explained_variance is None:
            return np.array([])
        total = self.explained_variance.sum()
        if total < 1e-12:
            return np.zeros_like(self.explained_variance)
        return self.explained_variance / total
    
    def reset(self) -> None:
        self.count = 0
        self.mean = None
        self.components = None
        self.singular_values = None
        self.explained_variance = None
        self._initialized = False
        self._buffer = []


class OnlineCovarianceMatrix:
    """
    Online covariance matrix estimation using Welford's parallel algorithm.
    Computes the full D×D covariance matrix in a streaming fashion.
    
    Memory: O(D²), Update: O(D²) per sample.
    """
    
    def __init__(self):
        self.count: int = 0
        self.mean: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None  # Co-moment matrix
        self._initialized: bool = False
    
    def _init(self, dim: int) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.C = np.zeros((dim, dim), dtype=np.float64)
        self._initialized = True
    
    def update(self, x: np.ndarray) -> None:
        """Update with a single observation vector."""
        x = x.flatten().astype(np.float64)
        
        if not self._initialized:
            self._init(len(x))
        
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.C += np.outer(delta, delta2)
    
    def update_batch(self, X: np.ndarray) -> None:
        """Update with multiple observations (N, D)."""
        if X.ndim == 1:
            self.update(X)
            return
        for row in X:
            self.update(row)
    
    @property
    def covariance(self) -> np.ndarray:
        """Population covariance matrix."""
        if self.count < 2 or not self._initialized:
            dim = len(self.mean) if self.mean is not None else 0
            return np.zeros((dim, dim))
        return self.C / self.count
    
    @property
    def sample_covariance(self) -> np.ndarray:
        """Unbiased sample covariance matrix."""
        if self.count < 2 or not self._initialized:
            dim = len(self.mean) if self.mean is not None else 0
            return np.zeros((dim, dim))
        return self.C / (self.count - 1)
    
    @property
    def correlation(self) -> np.ndarray:
        """Correlation matrix derived from covariance."""
        cov = self.sample_covariance
        diag = np.sqrt(np.diag(cov))
        diag[diag < 1e-12] = 1.0
        return cov / np.outer(diag, diag)
    
    def eigendecomposition(self, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of the covariance matrix.
        
        Returns:
            (eigenvalues, eigenvectors) sorted by descending eigenvalue
        """
        cov = self.sample_covariance
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        if top_k is not None:
            eigenvalues = eigenvalues[:top_k]
            eigenvectors = eigenvectors[:, :top_k]
        
        return eigenvalues, eigenvectors
    
    def reset(self) -> None:
        self.count = 0
        self.mean = None
        self.C = None
        self._initialized = False


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Classical Gram-Schmidt orthogonalization.
    
    Args:
        vectors: Matrix of shape (n_vectors, dim) - rows are vectors
    
    Returns:
        Orthonormal matrix of same shape
    """
    vectors = vectors.astype(np.float64).copy()
    n = vectors.shape[0]
    
    for i in range(n):
        for j in range(i):
            proj = np.dot(vectors[i], vectors[j]) / max(np.dot(vectors[j], vectors[j]), 1e-12)
            vectors[i] -= proj * vectors[j]
        
        norm = np.linalg.norm(vectors[i])
        if norm > 1e-12:
            vectors[i] /= norm
        else:
            vectors[i] = 0.0
    
    return vectors


def batch_svd_project(data: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-shot SVD projection for batch data.
    
    Args:
        data: Array of shape (N, D)
        n_components: Number of principal components
    
    Returns:
        (projected_data, explained_variance_ratio, components)
    """
    data = data.astype(np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    k = min(n_components, len(S))
    projected = centered @ Vt[:k].T
    
    total_var = (S ** 2).sum()
    if total_var < 1e-12:
        explained_ratio = np.zeros(k)
    else:
        explained_ratio = (S[:k] ** 2) / total_var
    
    return projected, explained_ratio, Vt[:k]