"""
NeuralStream-Core System Constants
Centralized definitions for all magic numbers, thresholds, and system limits.
"""

import torch
from enum import Enum, auto
from typing import Final


# ============================================================
# System Identity
# ============================================================
PROJECT_NAME: Final[str] = "NeuralStream-Core"
VERSION: Final[str] = "2.0.0"
BUILD_TAG: Final[str] = "enterprise-50x"


# ============================================================
# Numerical Constants
# ============================================================
EPS: Final[float] = 1e-8
INF: Final[float] = float('inf')
NEG_INF: Final[float] = float('-inf')
PI: Final[float] = 3.141592653589793


# ============================================================
# Default Tensor Properties
# ============================================================
DEFAULT_DTYPE: Final[torch.dtype] = torch.float32
DEFAULT_DEVICE: Final[str] = "cpu"
DEFAULT_BATCH_SIZE: Final[int] = 16
DEFAULT_INPUT_SIZE: Final[int] = 64
DEFAULT_CHANNELS: Final[int] = 1
DEFAULT_LATENT_DIM: Final[int] = 32


# ============================================================
# Memory Limits
# ============================================================
MAX_RSS_MB: Final[float] = 2048.0
MAX_TENSOR_ELEMENTS: Final[int] = 100_000_000
MAX_BUFFER_SIZE: Final[int] = 10_000
RING_BUFFER_DEFAULT_CAPACITY: Final[int] = 100


# ============================================================
# Latency Targets (milliseconds)
# ============================================================
LATENCY_TARGET_FAST_MS: Final[float] = 1.0
LATENCY_TARGET_NORMAL_MS: Final[float] = 3.0
LATENCY_TARGET_RELAXED_MS: Final[float] = 10.0
CIRCUIT_BREAKER_RECOVERY_TARGET_MS: Final[float] = 2.0


# ============================================================
# Drift Detection Thresholds
# ============================================================
DRIFT_ALERT_THRESHOLD: Final[float] = 0.15
DRIFT_CRITICAL_THRESHOLD: Final[float] = 0.35
DRIFT_SAMPLE_SIZE: Final[int] = 500


# ============================================================
# Anomaly Detection
# ============================================================
OOD_RECONSTRUCTION_THRESHOLD: Final[float] = 0.18
ANOMALY_SCORE_WARNING: Final[float] = 0.5
ANOMALY_SCORE_CRITICAL: Final[float] = 0.8


# ============================================================
# Adaptation Parameters
# ============================================================
ADAPTATION_LR: Final[float] = 1e-5
ADAPTATION_MAX_GRAD_NORM: Final[float] = 1.0
EWC_LAMBDA: Final[float] = 0.5


# ============================================================
# Dashboard
# ============================================================
MAX_SPARKLINE_POINTS: Final[int] = 100
MAX_TRAJECTORY_POINTS: Final[int] = 500
MAX_HEATMAP_HISTORY: Final[int] = 200
DASHBOARD_REFRESH_MS: Final[int] = 30


# ============================================================
# System State Enums
# ============================================================
class EngineState(Enum):
    UNINITIALIZED = auto()
    BOOTING = auto()
    CALIBRATING = auto()
    IDLE = auto()
    STREAMING = auto()
    PAUSED = auto()
    ADAPTING = auto()
    FAULT = auto()
    SHUTDOWN = auto()


class PrecisionState(Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


class DriftSeverity(Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class WorkerState(Enum):
    IDLE = auto()
    RUNNING = auto()
    BLOCKED = auto()
    ERROR = auto()
    TERMINATED = auto()