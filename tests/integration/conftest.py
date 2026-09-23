"""
Shared pytest fixtures for integration tests.
"""

import sys
import torch
import numpy as np
import pytest
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    return _PROJECT_ROOT


@pytest.fixture
def cfg_default():
    from config.loader import load_config
    return load_config(profile="default")


@pytest.fixture
def cfg_ci():
    from config.loader import load_config
    return load_config(profile="ci")


@pytest.fixture
def small_batch():
    """A small input tensor for quick integration runs."""
    return torch.randn(4, 1, 32, 32)


@pytest.fixture
def small_input_size():
    return 32


@pytest.fixture(scope="session")
def tiny_model():
    """Reusable tiny EdgeStreamingAE (built once per session)."""
    from core.models import EdgeStreamingAE
    return EdgeStreamingAE(
        in_channels=1, input_size=32, latent_dim=16,
        base_channels=4,
    ).eval()