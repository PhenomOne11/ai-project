"""
NeuralStream-Core Exception Hierarchy
Structured exception classes for all system fault conditions.
"""


class NeuralStreamError(Exception):
    """Base exception for all NeuralStream-Core errors."""
    pass


# --- Engine Lifecycle Exceptions ---

class EngineBootError(NeuralStreamError):
    """Raised when the engine fails to initialize."""
    pass


class EngineShutdownError(NeuralStreamError):
    """Raised during abnormal engine shutdown."""
    pass


class CalibrationError(NeuralStreamError):
    """Raised when LSUV or other calibration procedures fail."""
    pass


# --- Data Pipeline Exceptions ---

class StreamExhaustedError(NeuralStreamError):
    """Raised when a finite data stream is fully consumed."""
    pass


class InvalidDataFormatError(NeuralStreamError):
    """Raised when input data fails validation checks."""
    pass


class DataCorruptionError(NeuralStreamError):
    """Raised when NaN, Inf, or invalid values are detected in tensors."""
    pass


# --- Batch Processing Exceptions ---

class CancelBatchException(NeuralStreamError):
    """
    Non-fatal exception to cancel processing of a single corrupted batch.
    The circuit breaker catches this and continues the pipeline.
    """
    pass


class BatchTimeoutError(NeuralStreamError):
    """Raised when a batch exceeds maximum allowed processing time."""
    pass


# --- Model Exceptions ---

class ModelLoadError(NeuralStreamError):
    """Raised when a model fails to load or initialize."""
    pass


class PrecisionScalingError(NeuralStreamError):
    """Raised when precision mode transition fails."""
    pass


class QuantizationError(NeuralStreamError):
    """Raised when dynamic quantization encounters an error."""
    pass


# --- Drift Detection Exceptions ---

class DriftDetectionError(NeuralStreamError):
    """Raised when drift computation encounters invalid distributions."""
    pass


class ReferenceNotCapturedError(NeuralStreamError):
    """Raised when drift is computed before reference distributions are recorded."""
    pass


# --- Adaptation Exceptions ---

class AdaptationError(NeuralStreamError):
    """Raised when online micro-adaptation fails."""
    pass


class OODQueueOverflowError(NeuralStreamError):
    """Raised when the OOD replay buffer exceeds capacity limits."""
    pass


# --- IPC Exceptions ---

class IPCError(NeuralStreamError):
    """Base exception for inter-process communication failures."""
    pass


class SharedMemoryError(IPCError):
    """Raised when shared memory allocation or access fails."""
    pass


class RingBufferOverflowError(IPCError):
    """Raised when the ring buffer write head laps the read head."""
    pass


# --- Worker Exceptions ---

class WorkerError(NeuralStreamError):
    """Raised when a worker process encounters a fatal error."""
    pass


class WorkerTimeoutError(WorkerError):
    """Raised when a worker exceeds its execution timeout."""
    pass


# --- Configuration Exceptions ---

class ConfigError(NeuralStreamError):
    """Raised when configuration validation fails."""
    pass


class ProfileNotFoundError(ConfigError):
    """Raised when a configuration profile file is not found."""
    pass


# --- Circuit Breaker States ---

class CircuitBreakerTripped(NeuralStreamError):
    """Raised when the circuit breaker trips due to excessive consecutive faults."""
    pass


class CircuitBreakerRecovery(NeuralStreamError):
    """Raised during circuit breaker recovery phase."""
    pass