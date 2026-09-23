"""
Shared Memory Bus
Manages a collection of named channels, providing a single point of
access for multi-channel inter-process communication.

Typical usage pattern:
    Producer process:
        bus = SharedMemoryBus()
        bus.produce("latents", shape=(32,), capacity=200)
        bus.produce("drift", shape=(8,), capacity=200)
        bus.produce("telemetry", shape=(4,), capacity=500)

        bus.publish("latents", latent_vector)
        bus.publish("drift", drift_scores)
        bus.publish("telemetry", [cpu, mem, lat, rss])

    Consumer process (UI):
        bus = SharedMemoryBus()
        bus.consume("latents", shape=(32,), capacity=200)
        bus.consume("drift", shape=(8,), capacity=200)
        bus.consume("telemetry", shape=(4,), capacity=500)

        latest_latents = bus.read_latest("latents", n=10)
"""

import numpy as np
import atexit
from typing import Dict, Optional, Tuple, Any, List
from core.ipc.channel import Channel, ChannelSpec


class SharedMemoryBus:
    """
    Manages multiple IPC channels in a single namespace.

    Roles:
        - Producer: calls .produce(name, shape, ...) then .publish(name, data)
        - Consumer: calls .consume(name, shape, ...) then .read_latest(name)
        - Dual: can do both (use register_channel for explicit control)
    """

    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._owned_channels: List[str] = []  # Channels we created (responsible for unlink)

        # Ensure cleanup on interpreter shutdown
        atexit.register(self.close_all)

    # ---- Setup ----

    def produce(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        capacity: int = 100,
        heartbeat_timeout_s: float = 2.0
    ) -> Channel:
        """Create a new channel (producer role)."""
        if name in self._channels:
            return self._channels[name]

        spec = ChannelSpec(name, shape, dtype, capacity, heartbeat_timeout_s)
        ch = spec.create()
        self._channels[name] = ch
        self._owned_channels.append(name)
        return ch

    def consume(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        capacity: int = 100,
        heartbeat_timeout_s: float = 2.0
    ) -> Channel:
        """Attach to an existing channel (consumer role)."""
        if name in self._channels:
            return self._channels[name]

        spec = ChannelSpec(name, shape, dtype, capacity, heartbeat_timeout_s)
        ch = spec.attach()
        self._channels[name] = ch
        return ch

    def register_channel(self, channel: Channel, owned: bool = False) -> None:
        """Register a pre-built channel."""
        self._channels[channel.name] = channel
        if owned:
            self._owned_channels.append(channel.name)

    # ---- Data Flow ----

    def publish(self, name: str, data: np.ndarray) -> None:
        """Write to a channel by name."""
        if name not in self._channels:
            raise KeyError(f"Channel '{name}' not registered. Call .produce() or .consume() first.")
        self._channels[name].write(data)

    def read_latest(self, name: str, n: int = 1) -> np.ndarray:
        """Read the N most recent items from a channel."""
        if name not in self._channels:
            raise KeyError(f"Channel '{name}' not registered.")
        return self._channels[name].read_latest(n)

    def read_all(self, name: str) -> np.ndarray:
        """Read all items from a channel."""
        if name not in self._channels:
            raise KeyError(f"Channel '{name}' not registered.")
        return self._channels[name].read_all()

    def get_channel(self, name: str) -> Channel:
        if name not in self._channels:
            raise KeyError(f"Channel '{name}' not registered.")
        return self._channels[name]

    # ---- Introspection ----

    @property
    def channel_names(self) -> List[str]:
        return list(self._channels.keys())

    def list_channels(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata for all registered channels."""
        out = {}
        for name, ch in self._channels.items():
            out[name] = {
                'shape': ch.ring.shape,
                'dtype': str(ch.ring.dtype),
                'capacity': ch.ring.capacity,
                'items_available': len(ch),
                'write_head': ch.write_head,
                'is_alive': ch.is_alive,
                'stats': ch.stats.to_dict()
            }
        return out

    def print_status(self) -> None:
        """Pretty-print status of all channels."""
        print("=" * 90)
        print("  SHARED MEMORY BUS STATUS")
        print("=" * 90)
        print(f"  {'Channel':<16} {'Shape':<14} {'Cap':>5} {'Items':>6} "
              f"{'Writes':>8} {'Reads':>8} {'Alive':>6}")
        print("-" * 90)

        for name, ch in self._channels.items():
            shape_str = str(ch.ring.shape)
            if len(shape_str) > 12:
                shape_str = shape_str[:12] + "…"
            print(
                f"  {name:<16} {shape_str:<14} "
                f"{ch.ring.capacity:>5} "
                f"{len(ch):>6} "
                f"{ch.stats.write_count:>8} "
                f"{ch.stats.read_count:>8} "
                f"{'✓' if ch.is_alive else '✗':>6}"
            )

        print("=" * 90)

    # ---- Lifecycle ----

    def close_channel(self, name: str) -> None:
        if name in self._channels:
            self._channels[name].close()
            del self._channels[name]

    def close_all(self) -> None:
        for ch in self._channels.values():
            try:
                ch.close()
            except Exception:
                pass
        self._channels.clear()

    def unlink_all(self) -> None:
        """Remove shared memory segments (producer must call this)."""
        for name in list(self._owned_channels):
            try:
                self._channels[name].unlink()
            except Exception:
                pass
        self._owned_channels.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlink_all()
        self.close_all()


# ---- Convenience factory functions ----

def create_producer_bus(
    channel_specs: Optional[Dict[str, Tuple[Tuple[int, ...], int]]] = None
) -> SharedMemoryBus:
    """
    Create a pre-configured producer bus.

    Args:
        channel_specs: {name: (shape, capacity)}
            If None, uses defaults for the standard NeuralStream channels.
    """
    if channel_specs is None:
        channel_specs = {
            "latents": ((32,), 200),
            "drift": ((8,), 200),
            "telemetry": ((4,), 500),
            "anomaly": ((1,), 200),
        }

    bus = SharedMemoryBus()
    for name, (shape, capacity) in channel_specs.items():
        bus.produce(name, shape=shape, capacity=capacity)
    return bus


def create_consumer_bus(
    channel_specs: Optional[Dict[str, Tuple[Tuple[int, ...], int]]] = None
) -> SharedMemoryBus:
    """Create a pre-configured consumer bus with matching specs."""
    if channel_specs is None:
        channel_specs = {
            "latents": ((32,), 200),
            "drift": ((8,), 200),
            "telemetry": ((4,), 500),
            "anomaly": ((1,), 200),
        }

    bus = SharedMemoryBus()
    for name, (shape, capacity) in channel_specs.items():
        bus.consume(name, shape=shape, capacity=capacity)
    return bus