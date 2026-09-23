"""
NeuralStream-Core Precision Package
"""

from core.precision.thermal import ThermalModel
from core.precision.controller import (
    PrecisionState,
    AdaptivePrecisionEngineV2,
    AdaptivePrecisionEngine,  # alias
)

__all__ = [
    'ThermalModel',
    'PrecisionState',
    'AdaptivePrecisionEngineV2',
    'AdaptivePrecisionEngine',
]