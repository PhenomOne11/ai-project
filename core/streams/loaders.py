"""
Unified file loaders for external data sources.

Supports:
    - .npy / .npz (NumPy archives)
    - .pt / .pth (PyTorch tensors)
    - images (single, list, or PIL)
    - .csv (tabular time-series)
    - .ndjson / .jsonl (streaming records)

All loaders return torch.Tensor of shape (N, C, H, W) or (N, C, L).
"""

import torch
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, List, Tuple, Any, BinaryIO
from PIL import Image


# ---- Tensor / NumPy loaders ----

def load_npy_to_tensor(
    source: Union[str, Path, BinaryIO, bytes],
    target_size: int = 64,
    channels: int = 1,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load a .npy file into a 4D tensor of shape (N, C, H, W).

    Automatically reshapes 1D/2D/3D inputs into the standard format.
    """
    if isinstance(source, (str, Path)):
        arr = np.load(source, allow_pickle=False)
    elif isinstance(source, (bytes, bytearray)):
        arr = np.load(BytesIO(source), allow_pickle=False)
    else:
        arr = np.load(source, allow_pickle=False)

    return _array_to_4d_tensor(arr, target_size, channels, normalize)


def load_pt_to_tensor(
    source: Union[str, Path, BinaryIO, bytes],
    target_size: int = 64,
    channels: int = 1,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load a .pt/.pth file containing a tensor or dict of tensors.
    """
    if isinstance(source, (str, Path)):
        obj = torch.load(source, map_location="cpu", weights_only=False)
    elif isinstance(source, (bytes, bytearray)):
        obj = torch.load(BytesIO(source), map_location="cpu", weights_only=False)
    else:
        obj = torch.load(source, map_location="cpu", weights_only=False)

    # Unwrap common container patterns
    if isinstance(obj, dict):
        for key in ('data', 'tensor', 'samples', 'x', 'input'):
            if key in obj and isinstance(obj[key], torch.Tensor):
                obj = obj[key]
                break

    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"Unsupported .pt container: {type(obj).__name__}")

    arr = obj.detach().cpu().numpy()
    return _array_to_4d_tensor(arr, target_size, channels, normalize)


def _array_to_4d_tensor(
    arr: np.ndarray,
    target_size: int,
    channels: int,
    normalize: bool
) -> torch.Tensor:
    """Convert a numpy array of any shape to (N, C, target_size, target_size)."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if arr.ndim == 1:
        # Scalar sequence → (N, 1, 1, 1) — reshape to a pseudo-image
        arr = arr.reshape(-1, 1, 1, 1)
        tensor = torch.from_numpy(arr).expand(-1, channels, target_size, target_size).contiguous()
    elif arr.ndim == 2:
        # (N, F) → treat as N samples of feature vectors
        N, F = arr.shape
        side = int(np.ceil(np.sqrt(F)))
        padded = np.zeros((N, side * side), dtype=np.float32)
        padded[:, :F] = arr
        arr = padded.reshape(N, 1, side, side)
        tensor = torch.from_numpy(arr)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
    elif arr.ndim == 3:
        # (N, H, W) → add channel dim
        arr = arr[:, None, :, :]
        tensor = torch.from_numpy(arr)
        if tensor.shape[2] != target_size or tensor.shape[3] != target_size:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
            )
    elif arr.ndim == 4:
        tensor = torch.from_numpy(arr)
        if tensor.shape[2] != target_size or tensor.shape[3] != target_size:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
            )
    else:
        raise ValueError(f"Cannot handle {arr.ndim}-D array")

    # Ensure channel count matches
    if tensor.shape[1] == 1 and channels > 1:
        tensor = tensor.repeat(1, channels, 1, 1)
    elif tensor.shape[1] != channels:
        # Average over channel dim and repeat
        tensor = tensor.mean(dim=1, keepdim=True).repeat(1, channels, 1, 1)

    if normalize:
        tmin = tensor.amin()
        tmax = tensor.amax()
        if float(tmax - tmin) > 1e-8:
            tensor = (tensor - tmin) / (tmax - tmin)

    return tensor.contiguous()


# ---- Image loaders ----

def load_images_to_tensor(
    sources: Union[list, str, Path, None],
    target_size: int = 64,
    channels: int = 1,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load one or more images (paths, PIL, or uploaded file objects)
    into a normalized 4D tensor.
    """
    if sources is None:
        return torch.zeros(1, channels, target_size, target_size)

    if not isinstance(sources, list):
        sources = [sources]

    tensors = []
    for src in sources:
        try:
            if hasattr(src, 'read'):
                # Streamlit UploadedFile or BytesIO
                img = Image.open(src).convert('RGB' if channels == 3 else 'L')
            elif isinstance(src, (str, Path)):
                img = Image.open(src).convert('RGB' if channels == 3 else 'L')
            elif isinstance(src, Image.Image):
                img = src.convert('RGB' if channels == 3 else 'L')
            else:
                continue

            img = img.resize((target_size, target_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)

            if arr.ndim == 2:
                arr = arr[None, :, :]
            else:
                arr = arr.transpose(2, 0, 1)

            tensors.append(torch.from_numpy(arr))
        except Exception:
            continue

    if not tensors:
        return torch.zeros(1, channels, target_size, target_size)

    stacked = torch.stack(tensors, dim=0)

    # Channel reconciliation
    if stacked.shape[1] != channels:
        if stacked.shape[1] == 1 and channels == 3:
            stacked = stacked.repeat(1, 3, 1, 1)
        elif stacked.shape[1] == 3 and channels == 1:
            stacked = stacked.mean(dim=1, keepdim=True)

    if normalize:
        s = stacked.amax().clamp_min(1e-8)
        stacked = stacked / s

    return stacked.contiguous()


# ---- CSV / Time-series loaders ----

def load_csv_to_tensor(
    source: Union[str, Path, BinaryIO, bytes],
    value_columns: Optional[List[str]] = None,
    time_column: Optional[str] = None,
    target_size: Optional[int] = None,
    normalize: bool = True,
    delimiter: str = ","
) -> torch.Tensor:
    """
    Load a CSV file into a 2D tensor of shape (N, C).

    If target_size is provided and the array is 1D, it will be reshaped to
    (N_blocks, target_size, target_size) for compatibility with image models.
    """
    import csv

    if isinstance(source, (str, Path)):
        with open(source, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
    elif isinstance(source, (bytes, bytearray)):
        from io import StringIO
        text = source.decode('utf-8', errors='ignore')
        reader = csv.reader(StringIO(text), delimiter=delimiter)
        rows = list(reader)
    else:
        # Assume file-like
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        from io import StringIO
        reader = csv.reader(StringIO(text), delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("Empty CSV")

    # Parse header
    header = rows[0]
    data_rows = rows[1:] if not _is_number(header[0]) else rows

    # Locate value columns
    if value_columns is not None:
        indices = [header.index(c) for c in value_columns if c in header]
    else:
        indices = [i for i, _ in enumerate(header) if not (
            time_column is not None and header[i] == time_column
        )]

    if not indices:
        indices = list(range(len(header)))

    values = []
    for row in data_rows:
        try:
            values.append([float(row[i]) for i in indices])
        except (ValueError, IndexError):
            continue

    if not values:
        raise ValueError("No numeric rows parsed from CSV")

    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        a_min = arr.min()
        a_max = arr.max()
        if a_max - a_min > 1e-8:
            arr = (arr - a_min) / (a_max - a_min)

    tensor = torch.from_numpy(arr)

    if target_size is not None and tensor.dim() == 2:
        # Chunk into windows
        N, C = tensor.shape
        window = target_size * target_size
        if C == 1:
            flat = tensor.flatten()
            total = (flat.numel() // window) * window
            if total > 0:
                tensor = flat[:total].reshape(-1, 1, target_size, target_size)
            else:
                tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(target_size, target_size)
                )
        else:
            # Multi-channel: pad/crop columns to target_size, then window
            if C < target_size:
                pad = torch.zeros(N, target_size - C)
                tensor = torch.cat([tensor, pad], dim=1)
            elif C > target_size:
                tensor = tensor[:, :target_size]
            if N < target_size:
                pad = torch.zeros(target_size - N, tensor.size(1))
                tensor = torch.cat([tensor, pad], dim=0)
            else:
                tensor = tensor[:target_size]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_size, target_size)
            )

    return tensor.contiguous()


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False