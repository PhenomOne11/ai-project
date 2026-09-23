"""
Channel-safe augmentation pipeline for images and signal tensors.

All augmentations are implemented with torch ops (no external deps)
and preserve the (B, C, H, W) shape contract.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, List


class AugmentationPipeline:
    """
    Composable augmentation pipeline.

    Each transform is a callable: (batch: Tensor) → Tensor.
    Transforms are applied sequentially with the given probabilities.

    Example:
        aug = AugmentationPipeline([
            (AugmentationPipeline.gaussian_noise(0.02), 0.5),
            (AugmentationPipeline.horizontal_flip(), 0.5),
            (AugmentationPipeline.random_crop(56), 0.3),
        ])
        batch = aug(batch)
    """

    def __init__(self, transforms: Optional[List] = None, seed: Optional[int] = None):
        self.transforms = transforms or []
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def add(self, transform: Callable, probability: float = 1.0) -> "AugmentationPipeline":
        """Add a transform with a given probability."""
        self.transforms.append((transform, probability))
        return self

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        out = batch
        for transform, p in self.transforms:
            if p >= 1.0 or float(torch.rand(1, generator=self.generator)) < p:
                out = transform(out)
        return out

    # ---- Static transform factories ----

    @staticmethod
    def gaussian_noise(std: float = 0.02) -> Callable:
        def _apply(x):
            return x + std * torch.randn_like(x)
        return _apply

    @staticmethod
    def uniform_noise(amplitude: float = 0.05) -> Callable:
        def _apply(x):
            return x + amplitude * (torch.rand_like(x) * 2 - 1)
        return _apply

    @staticmethod
    def horizontal_flip() -> Callable:
        def _apply(x):
            return torch.flip(x, dims=[-1])
        return _apply

    @staticmethod
    def vertical_flip() -> Callable:
        def _apply(x):
            return torch.flip(x, dims=[-2])
        return _apply

    @staticmethod
    def random_crop(size: int) -> Callable:
        def _apply(x):
            if x.dim() != 4:
                return x
            H, W = x.shape[-2], x.shape[-1]
            if size >= H or size >= W:
                return x
            i = torch.randint(0, H - size + 1, (1,)).item()
            j = torch.randint(0, W - size + 1, (1,)).item()
            crop = x[..., i:i+size, j:j+size]
            # Resize back for shape-preserving augmentation
            return F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
        return _apply

    @staticmethod
    def random_erasing(p: float = 0.1, fill: float = 0.0) -> Callable:
        def _apply(x):
            if x.dim() != 4:
                return x
            B, C, H, W = x.shape
            for b in range(B):
                if float(torch.rand(1)) < p:
                    eh = max(1, int(H * 0.15))
                    ew = max(1, int(W * 0.15))
                    top = torch.randint(0, H - eh, (1,)).item()
                    left = torch.randint(0, W - ew, (1,)).item()
                    x[b, :, top:top+eh, left:left+ew] = fill
            return x
        return _apply

    @staticmethod
    def gaussian_blur(kernel_size: int = 3) -> Callable:
        def _apply(x):
            if x.dim() != 4:
                return x
            k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
            coords = torch.arange(k, dtype=x.dtype) - (k - 1) / 2
            g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            g = g / g.sum()
            kernel_h = g.view(1, 1, -1, 1)
            kernel_w = g.view(1, 1, 1, -1)
            C = x.size(1)
            pad = k // 2
            xp = F.pad(x, (pad, pad, 0, 0), mode='reflect')
            x = F.conv2d(xp, kernel_w.repeat(C, 1, 1, 1), groups=C)
            xp = F.pad(x, (0, 0, pad, pad), mode='reflect')
            x = F.conv2d(xp, kernel_h.repeat(C, 1, 1, 1), groups=C)
            return x
        return _apply

    @staticmethod
    def brightness_scale(factor_range: tuple = (0.8, 1.2)) -> Callable:
        def _apply(x):
            lo, hi = factor_range
            factor = lo + (hi - lo) * float(torch.rand(1))
            return x * factor
        return _apply

    @staticmethod
    def amplitude_jitter(max_db: float = 3.0) -> Callable:
        """For 3D signals (B, C, L): multiplicative amplitude scaling."""
        def _apply(x):
            if x.dim() != 3:
                return x
            scale = 10 ** (torch.randn(x.size(1), 1) * (max_db / 20.0))
            return x * scale.unsqueeze(0)
        return _apply

    @staticmethod
    def time_shift(max_shift: int = 8) -> Callable:
        """For 3D signals (B, C, L): circular time shift per channel."""
        def _apply(x):
            if x.dim() != 3:
                return x
            B, C, L = x.shape
            out = x.clone()
            for b in range(B):
                for c in range(C):
                    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                    out[b, c] = torch.roll(x[b, c], shifts=shift, dims=-1)
            return out
        return _apply

    # ---- Presets ----

    @staticmethod
    def image_preset(seed: Optional[int] = None) -> "AugmentationPipeline":
        """Standard image augmentation preset."""
        return AugmentationPipeline([
            (AugmentationPipeline.gaussian_noise(0.02), 0.5),
            (AugmentationPipeline.horizontal_flip(), 0.5),
            (AugmentationPipeline.brightness_scale((0.9, 1.1)), 0.3),
            (AugmentationPipeline.random_erasing(0.05), 0.2),
        ], seed=seed)

    @staticmethod
    def signal_preset(seed: Optional[int] = None) -> "AugmentationPipeline":
        """Standard signal augmentation preset (for waveform / time-series)."""
        return AugmentationPipeline([
            (AugmentationPipeline.gaussian_noise(0.02), 0.5),
            (AugmentationPipeline.amplitude_jitter(2.0), 0.3),
            (AugmentationPipeline.time_shift(4), 0.3),
        ], seed=seed)