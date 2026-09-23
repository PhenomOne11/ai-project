"""
Tensor Operations Library
Low-level tensor manipulation utilities including im2col unfolding,
patch extraction, safe arithmetic, and shape transformations.

All operations work on both NumPy arrays and PyTorch tensors
with automatic type detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Optional

TensorLike = Union[torch.Tensor, np.ndarray]


def im2col(
    input_tensor: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    """
    Extract image patches and reshape into column matrix (im2col).
    
    Converts a 4D tensor (B, C, H, W) into a 2D matrix where each column
    represents one flattened patch. This is the foundation of efficient
    convolution via matrix multiplication.
    
    Args:
        input_tensor: Shape (B, C, H, W)
        kernel_size: Square kernel side length
        stride: Convolution stride
        padding: Zero padding
    
    Returns:
        Column matrix of shape (B, C*kH*kW, L) where L is the number of patches
    """
    if padding > 0:
        input_tensor = F.pad(input_tensor, [padding] * 4)
    
    B, C, H, W = input_tensor.shape
    
    # Use unfold to extract patches
    patches = input_tensor.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    # Shape: (B, C, out_h, out_w, kH, kW)
    
    out_h, out_w = patches.shape[2], patches.shape[3]
    
    # Reshape to (B, C*kH*kW, out_h*out_w)
    columns = patches.contiguous().view(B, C * kernel_size * kernel_size, out_h * out_w)
    
    return columns


def col2im(
    columns: torch.Tensor,
    output_shape: Tuple[int, int, int, int],
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    """
    Reverse im2col: reconstruct image from column matrix.
    Uses fold operation for accumulation with overlap averaging.
    
    Args:
        columns: Shape (B, C*kH*kW, L)
        output_shape: Target (B, C, H, W)
        kernel_size: Square kernel side length
        stride: Convolution stride
        padding: Zero padding
    
    Returns:
        Reconstructed tensor of shape (B, C, H, W)
    """
    B, C, H, W = output_shape
    
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    
    output = F.fold(
        columns,
        output_size=(H_padded, W_padded),
        kernel_size=kernel_size,
        stride=stride
    )
    
    if padding > 0:
        output = output[:, :, padding:-padding, padding:-padding]
    
    return output


def extract_patches_2d(
    images: torch.Tensor,
    patch_size: int,
    stride: Optional[int] = None
) -> torch.Tensor:
    """
    Extract non-overlapping or overlapping 2D patches from images.
    
    Args:
        images: Shape (B, C, H, W)
        patch_size: Square patch side length
        stride: Step between patches. Default = patch_size (non-overlapping)
    
    Returns:
        Patches of shape (B, num_patches, C, patch_size, patch_size)
    """
    if stride is None:
        stride = patch_size
    
    B, C, H, W = images.shape
    
    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # Shape: (B, C, n_h, n_w, patch_size, patch_size)
    
    n_h, n_w = patches.shape[2], patches.shape[3]
    
    patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)
    # Shape: (B, n_h, n_w, C, patch_size, patch_size)
    
    patches = patches.reshape(B, n_h * n_w, C, patch_size, patch_size)
    
    return patches


def safe_normalize(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    L2-normalize tensor along a dimension with numerical safety.
    """
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def safe_log(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically safe logarithm."""
    return torch.log(tensor.clamp(min=eps))


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Numerically safe division."""
    return numerator / (denominator + eps)


def check_tensor_health(tensor: torch.Tensor) -> dict:
    """
    Comprehensive tensor health check.
    
    Returns dictionary with:
    - has_nan: bool
    - has_inf: bool
    - has_neg_inf: bool
    - max_magnitude: float
    - fraction_zero: float
    - is_healthy: bool
    """
    flat = tensor.detach().float()
    
    has_nan = bool(torch.isnan(flat).any())
    has_inf = bool(torch.isinf(flat).any())
    has_neg_inf = bool((flat == float('-inf')).any())
    
    finite_vals = flat[torch.isfinite(flat)]
    
    if len(finite_vals) > 0:
        max_magnitude = float(finite_vals.abs().max())
        fraction_zero = float((finite_vals == 0).sum()) / max(len(finite_vals), 1)
    else:
        max_magnitude = float('inf')
        fraction_zero = 0.0
    
    is_healthy = not has_nan and not has_inf and max_magnitude < 1e6
    
    return {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'has_neg_inf': has_neg_inf,
        'max_magnitude': max_magnitude,
        'fraction_zero': fraction_zero,
        'num_elements': int(tensor.numel()),
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'is_healthy': is_healthy
    }


def sanitize_tensor(
    tensor: torch.Tensor,
    replace_nan: float = 0.0,
    replace_inf: float = 0.0,
    clamp_magnitude: Optional[float] = 1e6
) -> torch.Tensor:
    """
    Replace NaN/Inf values and optionally clamp magnitude.
    Returns a clean copy.
    """
    result = tensor.clone()
    
    result = torch.where(torch.isnan(result), torch.full_like(result, replace_nan), result)
    result = torch.where(torch.isinf(result), torch.full_like(result, replace_inf), result)
    
    if clamp_magnitude is not None:
        result = result.clamp(-clamp_magnitude, clamp_magnitude)
    
    return result


def flatten_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten all dimensions except the batch dimension."""
    return tensor.view(tensor.size(0), -1)


def compute_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    """Compute convolution output spatial dimension."""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_memory_mb(model: torch.nn.Module) -> float:
    """Estimate model parameter memory in megabytes."""
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 * 1024)