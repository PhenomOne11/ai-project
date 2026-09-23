"""
Lock-Free Shared Memory Ring Buffer
Zero-copy inter-process communication for high-throughput streaming.
Uses an atomic write-head counter to allow a producer process to write
while a consumer process reads without any mutex overhead.

Layout of the shared memory segment:
    [8 bytes: atomic int64 write head][capacity * item_bytes: data slots]
"""

import numpy as np
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple
import time


class RingBuffer:
    """
    Fixed-capacity circular buffer in shared memory.

    Thread-safe for single-producer / single-consumer scenarios.
    Producer only writes to `index_array[0]` and the current slot.
    Consumer only reads `index_array[0]` and previously written slots.

    Args:
        name: Unique identifier for the shared memory segment
        shape: Shape of a single item
        dtype: NumPy dtype for items
        capacity: Number of items the buffer can hold
        create: True for producer (creates segment), False for consumer
    """

    HEADER_BYTES = 8  # int64 write head

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        capacity: int = 100,
        create: bool = False
    ):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.capacity = int(capacity)

        self.item_bytes = int(np.prod(self.shape) * self.dtype.itemsize)
        self.total_bytes = self.item_bytes * self.capacity + self.HEADER_BYTES

        if create:
            self.shm = SharedMemory(name=name, create=True, size=self.total_bytes)
            # Zero the write head
            np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf[:self.HEADER_BYTES])[0] = 0
        else:
            self.shm = SharedMemory(name=name, create=False)

        self.index_array = np.ndarray(
            (1,), dtype=np.int64, buffer=self.shm.buf[:self.HEADER_BYTES]
        )
        self.buffer_array = np.ndarray(
            (self.capacity, *self.shape),
            dtype=self.dtype,
            buffer=self.shm.buf[self.HEADER_BYTES:]
        )

    # ---- Producer side ----

    def write(self, data: np.ndarray) -> int:
        """
        Write one item to the next slot.
        Returns the write index position (0-based within capacity).
        """
        current_head = int(self.index_array[0])
        idx = current_head % self.capacity
        self.buffer_array[idx] = data
        # Atomic head advance (fence semantics implicit in CPython bytecode)
        self.index_array[0] = current_head + 1
        return idx

    # ---- Consumer side ----

    @property
    def write_head(self) -> int:
        """Total number of writes since creation."""
        return int(self.index_array[0])

    def read_latest(self, n: int = 1) -> np.ndarray:
        """
        Read the N most recent items, newest first.
        Returns array of shape (min(n, available), *item_shape).
        """
        head = int(self.index_array[0])
        if head == 0:
            return np.empty((0, *self.shape), dtype=self.dtype)

        n = min(n, head, self.capacity)
        # Fancy indexing into shared buffer
        indices = [(head - 1 - i) % self.capacity for i in range(n)]
        return self.buffer_array[indices].copy()

    def read_range(self, start: int, end: int) -> np.ndarray:
        """
        Read a range of items [start, end) in write-order (oldest first).
        Values are clipped to the available history window.
        """
        head = int(self.index_array[0])
        available = min(head, self.capacity)
        oldest = head - available

        start = max(start, oldest)
        end = min(end, head)

        if start >= end:
            return np.empty((0, *self.shape), dtype=self.dtype)

        count = end - start
        indices = [(start + i) % self.capacity for i in range(count)]
        return self.buffer_array[indices].copy()

    def read_all(self) -> np.ndarray:
        """Read all available items in write-order."""
        head = int(self.index_array[0])
        available = min(head, self.capacity)
        return self.read_range(head - available, head)

    def __len__(self) -> int:
        """Number of items currently available for reading."""
        head = int(self.index_array[0])
        return min(head, self.capacity)

    # ---- Lifecycle ----

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        """Deallocate the shared memory segment (call from creating process only)."""
        try:
            self.shm.unlink()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---- Static helpers ----

def create_ring_buffer(
    name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    capacity: int = 100
) -> RingBuffer:
    """Convenience function to create a new ring buffer."""
    return RingBuffer(name=name, shape=shape, dtype=dtype, capacity=capacity, create=True)


def attach_ring_buffer(name: str) -> RingBuffer:
    """
    Attach to an existing ring buffer.
    Note: requires knowing shape/dtype/capacity.
    In practice, use the SharedMemoryBus which stores metadata.
    """
    # Try common defaults first
    shm = SharedMemory(name=name, create=False)
    available_bytes = shm.size
    shm.close()
    raise RuntimeError(
        f"Cannot infer shape/dtype from raw segment (size={available_bytes} bytes). "
        f"Use SharedMemoryBus.collect(name, shape, dtype, capacity) instead."
    )