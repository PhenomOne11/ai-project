"""
NeuralStream-Core IPC Package
Lock-free shared memory inter-process communication.
"""

from core.ipc.ring_buffer import RingBuffer, create_ring_buffer
from core.ipc.channel import Channel, ChannelSpec, ChannelStats
from core.ipc.bus import (
    SharedMemoryBus,
    create_producer_bus,
    create_consumer_bus,
)

__all__ = [
    'RingBuffer',
    'create_ring_buffer',
    'Channel',
    'ChannelSpec',
    'ChannelStats',
    'SharedMemoryBus',
    'create_producer_bus',
    'create_consumer_bus',
]