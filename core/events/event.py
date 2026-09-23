"""
Event and Priority definitions for the NeuralStream event bus.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventPriority(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EventKind(str, Enum):
    """Standard event categories emitted by NeuralStream subsystems."""
    # Lifecycle
    ENGINE_BOOT = "engine.boot"
    ENGINE_START = "engine.start"
    ENGINE_PAUSE = "engine.pause"
    ENGINE_STOP = "engine.stop"

    # Data
    BATCH_PROCESSED = "batch.processed"
    DATA_CORRUPTED = "data.corrupted"

    # Drift
    DRIFT_NOMINAL = "drift.nominal"
    DRIFT_WARNING = "drift.warning"
    DRIFT_CRITICAL = "drift.critical"
    DRIFT_EMERGENCY = "drift.emergency"

    # Precision
    PRECISION_TRANSITION = "precision.transition"

    # Adaptation
    OOD_DETECTED = "adaptation.ood_detected"
    ADAPTATION_STEP = "adaptation.step"
    ADAPTATION_FAILED = "adaptation.failed"

    # Anomaly
    ANOMALY_SCORE_HIGH = "anomaly.score_high"

    # Circuit breaker
    CIRCUIT_BREAKER_TRIP = "circuit_breaker.trip"
    CIRCUIT_BREAKER_RECOVER = "circuit_breaker.recover"

    # Workers
    WORKER_TASK_FAILED = "worker.task_failed"
    WORKER_TIMEOUT = "worker.timeout"

    # Metrics
    METRICS_SNAPSHOT = "metrics.snapshot"

    # Generic
    CUSTOM = "custom"


@dataclass
class Event:
    """
    Immutable event record.

    Fields:
        kind:       Event category (string, typically from EventKind)
        priority:   Severity level
        payload:    Arbitrary data dict
        source:     Optional emitter identifier
        event_id:   Unique hex ID
        timestamp:  Unix time of creation
    """
    kind: str
    priority: EventPriority = EventPriority.INFO
    payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'event_id': self.event_id,
            'kind': self.kind,
            'priority': self.priority.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'payload': self.payload,
        }

    def __repr__(self) -> str:
        return (f"Event(id={self.event_id}, kind={self.kind!r}, "
                f"priority={self.priority.value}, source={self.source!r})")


def make_event(
    kind: str,
    payload: Optional[Dict[str, Any]] = None,
    priority: EventPriority = EventPriority.INFO,
    source: Optional[str] = None
) -> Event:
    """Convenience constructor."""
    return Event(
        kind=kind,
        priority=priority,
        payload=payload or {},
        source=source,
    )