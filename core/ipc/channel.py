"""
Named IPC Channel
A single named data stream built on top of RingBuffer, with metadata,
heartbeat monitoring, and stats tracking.
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from core.ipc.ring_buffer import RingBuffer


@dataclass
class ChannelStats:
    """Runtime statistics for an IPC channel."""
    write_count: int = 0
    read_count: int = 0
    created_at: float = 0.0
    last_write_time: float = 0.0
    last_read_time: float = 0.0
    total_bytes_written: int = 0

    @property
    def age_seconds(self) -> float:
        if self.created_at == 0:
            return 0.0
        return time.time() - self.created_at

    @property
    def seconds_since_last_write(self) -> float:
        if self.last_write_time == 0:
            return float('inf')
        return time.time() - self.last_write_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'write_count': self.write_count,
            'read_count': self.read_count,
            'age_seconds': self.age_seconds,
            'seconds_since_last_write': self.seconds_since_last_write,
            'total_bytes_written': self.total_bytes_written,
        }


class Channel:
    """
    Named IPC channel wrapping a RingBuffer with metadata and heartbeat.
    """

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        capacity: int = 100,
        create: bool = False,
        heartbeat_timeout_s: float = 2.0
    ):
        self.name = name
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.stats = ChannelStats(created_at=time.time())

        # Shared memory segment name (namespace by channel name)
        shm_name = f"neuralstream_{name}"

        self.ring = RingBuffer(
            name=shm_name,
            shape=shape,
            dtype=dtype,
            capacity=capacity,
            create=create
        )

    # ---- Producer API ----

    def write(self, data: np.ndarray) -> None:
        """Write one item to the channel."""
        self.ring.write(data)
        self.stats.write_count += 1
        self.stats.last_write_time = time.time()
        self.stats.total_bytes_written += data.nbytes

    # ---- Consumer API ----

    def read_latest(self, n: int = 1) -> np.ndarray:
        """Read the N most recent items."""
        out = self.ring.read_latest(n)
        self.stats.read_count += 1
        self.stats.last_read_time = time.time()
        return out

    def read_all(self) -> np.ndarray:
        """Read all available items."""
        out = self.ring.read_all()
        self.stats.read_count += 1
        self.stats.last_read_time = time.time()
        return out

    # ---- Health ----

    @property
    def is_alive(self) -> bool:
        """True if the producer is actively writing."""
        return self.stats.seconds_since_last_write < self.heartbeat_timeout_s

    @property
    def size(self) -> int:
        """Number of items currently stored."""
        return len(self.ring)

    @property
    def write_head(self) -> int:
        return self.ring.write_head

    # ---- Lifecycle ----

    def close(self) -> None:
        self.ring.close()

    def unlink(self) -> None:
        self.ring.unlink()

    def __len__(self) -> int:
        return len(self.ring)

    def __repr__(self) -> str:
        return (
            f"Channel(name={self.name!r}, shape={self.ring.shape}, "
            f"items={len(self)}, writes={self.stats.write_count})"
        )


class ChannelSpec:
    """Specification for creating a channel (used by the bus)."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        capacity: int = 100,
        heartbeat_timeout_s: float = 2.0
    ):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.capacity = int(capacity)
        self.heartbeat_timeout_s = heartbeat_timeout_s

    def create(self) -> Channel:
        return Channel(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            capacity=self.capacity,
            create=True,
            heartbeat_timeout_s=self.heartbeat_timeout_s
        )

    def attach(self) -> Channel:
        return Channel(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            capacity=self.capacity,
            create=False,
            heartbeat_timeout_s=self.heartbeat_timeout_s
        )