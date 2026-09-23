"""
NeuralStream-Core Configuration Schema
Pydantic-validated configuration structures for all system components.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class PrecisionMode(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


class StreamFormat(str, Enum):
    SYNTHETIC = "synthetic"
    NPY = "npy"
    PT = "pt"
    IMAGE = "image"
    CSV = "csv"
    AUDIO = "audio"
    VIDEO = "video"


class ModelArchitecture(str, Enum):
    EDGE_AE = "edge_ae"
    VAE = "vae"
    TRANSFORMER_AE = "transformer_ae"
    UNET_AE = "unet_ae"
    CONV_VAE = "conv_vae"


class DriftMethod(str, Enum):
    # Methods supported by core.drift.detector.DriftDetector.METHOD_REGISTRY
    WASSERSTEIN = "wasserstein"        # Wasserstein-1 (Earth Mover's)
    WASSERSTEIN_2 = "wasserstein_2"    # Wasserstein-2
    MMD = "mmd"                        # Maximum Mean Discrepancy (RBF)
    KL_DIVERGENCE = "kl_divergence"    # Symmetric KL divergence
    JENSEN_SHANNON = "jensen_shannon"  # Jensen-Shannon divergence
    ENERGY = "energy"                  # Energy distance
    HELLINGER = "hellinger"            # Hellinger distance


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    architecture: ModelArchitecture = ModelArchitecture.EDGE_AE
    in_channels: int = Field(default=1, ge=1, le=16)
    input_size: int = Field(default=64, ge=8, le=1024)
    latent_dim: int = Field(default=32, ge=2, le=512)
    hidden_dims: List[int] = Field(default=[32, 64, 128])
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    use_batch_norm: bool = True
    activation: str = "leaky_relu"


class StreamConfig(BaseModel):
    """Configuration for data ingestion streams."""
    format: StreamFormat = StreamFormat.SYNTHETIC
    batch_size: int = Field(default=16, ge=1, le=512)
    shape: List[int] = Field(default=[1, 64, 64])
    buffer_size: int = Field(default=1000, ge=10)
    shuffle: bool = True
    normalize: bool = True
    augment: bool = False


class PrecisionConfig(BaseModel):
    """Configuration for dynamic precision scaling."""
    initial_mode: PrecisionMode = PrecisionMode.FP32
    target_latency_ms: float = Field(default=3.0, ge=0.1, le=100.0)
    window_size: int = Field(default=30, ge=5, le=200)
    upscale_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    downscale_threshold_fp16: float = Field(default=1.5, ge=1.0, le=5.0)
    downscale_threshold_int8: float = Field(default=2.5, ge=1.5, le=10.0)
    cooldown_ticks: int = Field(default=10, ge=1, le=100)
    enable_thermal_model: bool = True


class DriftConfig(BaseModel):
    """Configuration for drift detection."""
    methods: List[DriftMethod] = Field(default=[DriftMethod.WASSERSTEIN])
    sample_size: int = Field(default=500, ge=50, le=10000)
    alert_threshold: float = Field(default=0.15, ge=0.01, le=1.0)
    critical_threshold: float = Field(default=0.35, ge=0.05, le=2.0)
    evaluation_interval: int = Field(default=5, ge=1, le=100)
    layer_names: List[str] = Field(default=["encoder_conv.0", "encoder_conv.2", "encoder_fc.0"])
    enable_heatmap: bool = True
    history_length: int = Field(default=200, ge=10, le=5000)


class AdaptationConfig(BaseModel):
    """Configuration for online OOD adaptation."""
    error_threshold: float = Field(default=0.18, ge=0.01, le=1.0)
    buffer_capacity: int = Field(default=64, ge=8, le=1024)
    micro_batch_size: int = Field(default=16, ge=4, le=128)
    learning_rate: float = Field(default=1e-5, ge=1e-8, le=1e-2)
    max_grad_norm: float = Field(default=1.0, ge=0.1, le=10.0)
    ewc_lambda: float = Field(default=0.5, ge=0.0, le=10.0)
    enable_ewc: bool = True
    adaptation_cooldown_ms: float = Field(default=100.0, ge=10.0, le=5000.0)


class AnomalyConfig(BaseModel):
    """Configuration for anomaly scoring engine."""
    reconstruction_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    isolation_forest_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    mahalanobis_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    contamination: float = Field(default=0.05, ge=0.001, le=0.5)
    n_estimators: int = Field(default=100, ge=10, le=1000)
    score_ema_alpha: float = Field(default=0.1, ge=0.01, le=1.0)


class IPCConfig(BaseModel):
    """Configuration for inter-process communication."""
    ring_buffer_capacity: int = Field(default=100, ge=10, le=10000)
    channel_count: int = Field(default=4, ge=1, le=16)
    channel_names: List[str] = Field(default=["latents", "drift", "telemetry", "anomaly"])
    enable_compression: bool = False
    heartbeat_interval_ms: float = Field(default=500.0, ge=100.0, le=5000.0)


class TelemetryConfig(BaseModel):
    """Configuration for metrics collection."""
    collection_interval_ms: float = Field(default=100.0, ge=10.0, le=5000.0)
    history_length: int = Field(default=1000, ge=25, le=50000)
    enable_cpu_tracking: bool = True
    enable_memory_tracking: bool = True
    enable_disk_tracking: bool = False
    enable_network_tracking: bool = False
    export_format: str = "json"


class DashboardConfig(BaseModel):
    """Configuration for the visual HUD."""
    theme: str = "cyber_dark"
    refresh_rate_ms: int = Field(default=30, ge=16, le=1000)
    max_sparkline_points: int = Field(default=100, ge=20, le=1000)
    max_trajectory_points: int = Field(default=500, ge=50, le=5000)
    enable_3d_plots: bool = True
    enable_heatmaps: bool = True
    enable_oscilloscope: bool = True
    animation_duration_ms: int = Field(default=300, ge=0, le=2000)
    panel_columns: int = Field(default=2, ge=1, le=4)


class CircuitBreakerConfig(BaseModel):
    """Configuration for fault isolation."""
    enable: bool = True
    nan_detection: bool = True
    inf_detection: bool = True
    magnitude_threshold: float = Field(default=1e6, ge=1e3, le=1e12)
    max_consecutive_faults: int = Field(default=5, ge=1, le=100)
    recovery_timeout_ms: float = Field(default=2.0, ge=0.1, le=100.0)
    quarantine_buffer_size: int = Field(default=50, ge=5, le=500)


class WorkerConfig(BaseModel):
    """Configuration for distributed worker pool."""
    num_inference_workers: int = Field(default=2, ge=1, le=16)
    num_adaptation_workers: int = Field(default=1, ge=0, le=4)
    num_drift_workers: int = Field(default=1, ge=0, le=4)
    task_queue_size: int = Field(default=100, ge=10, le=10000)
    worker_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    enable_auto_scaling: bool = False


class SystemConfig(BaseModel):
    """
    Master system configuration.
    Aggregates all subsystem configurations into a single validated structure.
    """
    project_name: str = "NeuralStream-Core"
    version: str = "2.0.0"
    debug: bool = False
    seed: int = Field(default=42, ge=0)
    log_level: str = "INFO"
    log_dir: str = "logs"

    model: ModelConfig = ModelConfig()
    stream: StreamConfig = StreamConfig()
    precision: PrecisionConfig = PrecisionConfig()
    drift: DriftConfig = DriftConfig()
    adaptation: AdaptationConfig = AdaptationConfig()
    anomaly: AnomalyConfig = AnomalyConfig()
    ipc: IPCConfig = IPCConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    dashboard: DashboardConfig = DashboardConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    workers: WorkerConfig = WorkerConfig()