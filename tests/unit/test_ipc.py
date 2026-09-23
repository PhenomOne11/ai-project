"""
Unit Tests for core.ipc
"""

import pytest
import numpy as np
import time


class TestRingBuffer:
    def test_create_and_write_read(self):
        from core.ipc.ring_buffer import create_ring_buffer
        rb = create_ring_buffer("test_rb_1", shape=(4,), dtype=np.float32, capacity=10)
        try:
            data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            rb.write(data)
            assert rb.write_head == 1
            latest = rb.read_latest(1)
            assert latest.shape == (1, 4)
            np.testing.assert_allclose(latest[0], data)
        finally:
            rb.unlink()
            rb.close()

    def test_capacity_wrap(self):
        from core.ipc.ring_buffer import create_ring_buffer
        rb = create_ring_buffer("test_rb_2", shape=(2,), dtype=np.float32, capacity=5)
        try:
            for i in range(12):
                rb.write(np.array([float(i), float(i * 2)], dtype=np.float32))

            # Only last 5 should be readable
            assert len(rb) == 5
            latest = rb.read_latest(5)
            # Newest first: 11, 10, 9, 8, 7
            assert latest[0, 0] == 11.0
            assert latest[4, 0] == 7.0
        finally:
            rb.unlink()
            rb.close()

    def test_read_latest_n(self):
        from core.ipc.ring_buffer import create_ring_buffer
        rb = create_ring_buffer("test_rb_3", shape=(1,), dtype=np.float32, capacity=20)
        try:
            for i in range(15):
                rb.write(np.array([float(i)], dtype=np.float32))

            latest_5 = rb.read_latest(5)
            assert latest_5.shape == (5, 1)
            assert latest_5[0, 0] == 14.0
            assert latest_5[4, 0] == 10.0
        finally:
            rb.unlink()
            rb.close()

    def test_read_all(self):
        from core.ipc.ring_buffer import create_ring_buffer
        rb = create_ring_buffer("test_rb_4", shape=(3,), dtype=np.float32, capacity=10)
        try:
            for i in range(4):
                rb.write(np.array([float(i), float(i), float(i)], dtype=np.float32))

            all_items = rb.read_all()
            assert all_items.shape == (4, 3)
            # Write order preserved
            assert all_items[0, 0] == 0.0
            assert all_items[3, 0] == 3.0
        finally:
            rb.unlink()
            rb.close()

    def test_empty_read(self):
        from core.ipc.ring_buffer import create_ring_buffer
        rb = create_ring_buffer("test_rb_5", shape=(2,), dtype=np.float32, capacity=10)
        try:
            empty = rb.read_latest(5)
            assert empty.shape == (0, 2)
        finally:
            rb.unlink()
            rb.close()


class TestChannel:
    def test_channel_write_read(self):
        from core.ipc.channel import Channel
        ch = Channel("chan_test_1", shape=(4,), dtype=np.float32, capacity=10, create=True)
        try:
            for i in range(5):
                ch.write(np.array([float(i)] * 4, dtype=np.float32))

            assert ch.stats.write_count == 5
            assert len(ch) == 5

            latest = ch.read_latest(2)
            assert latest.shape == (2, 4)
            assert latest[0, 0] == 4.0
        finally:
            ch.unlink()
            ch.close()

    def test_channel_heartbeat(self):
        from core.ipc.channel import Channel
        ch = Channel(
            "chan_test_2", shape=(1,), dtype=np.float32, capacity=5,
            create=True, heartbeat_timeout_s=0.1
        )
        try:
            # No writes yet → not alive
            time.sleep(0.15)
            assert not ch.is_alive

            # Write → alive
            ch.write(np.array([1.0], dtype=np.float32))
            assert ch.is_alive

            # Wait for timeout
            time.sleep(0.15)
            assert not ch.is_alive
        finally:
            ch.unlink()
            ch.close()


class TestSharedMemoryBus:
    def test_producer_consumer_flow(self):
        from core.ipc.bus import SharedMemoryBus
        producer = SharedMemoryBus()
        producer.produce("test_flow", shape=(4,), dtype=np.float32, capacity=10)

        try:
            # Write some data
            for i in range(5):
                producer.publish("test_flow", np.array([float(i)] * 4, dtype=np.float32))

            # Consumer attaches (in-process for test)
            consumer = SharedMemoryBus()
            consumer.consume("test_flow", shape=(4,), dtype=np.float32, capacity=10)
            try:
                latest = consumer.read_latest("test_flow", n=3)
                assert latest.shape == (3, 4)
                assert latest[0, 0] == 4.0
            finally:
                consumer.close_all()
        finally:
            producer.unlink_all()
            producer.close_all()

    def test_multiple_channels(self):
        from core.ipc.bus import SharedMemoryBus
        bus = SharedMemoryBus()
        try:
            bus.produce("latents", shape=(32,), dtype=np.float32, capacity=50)
            bus.produce("drift", shape=(8,), dtype=np.float32, capacity=50)

            for _ in range(10):
                bus.publish("latents", np.random.randn(32).astype(np.float32))
                bus.publish("drift", np.random.randn(8).astype(np.float32))

            assert len(bus.channel_names) == 2

            lats = bus.read_latest("latents", n=5)
            drifts = bus.read_latest("drift", n=5)

            assert lats.shape == (5, 32)
            assert drifts.shape == (5, 8)
        finally:
            bus.unlink_all()
            bus.close_all()

    def test_publish_unknown_channel(self):
        from core.ipc.bus import SharedMemoryBus
        bus = SharedMemoryBus()
        try:
            with pytest.raises(KeyError):
                bus.publish("nonexistent", np.array([1.0]))
        finally:
            bus.close_all()

    def test_list_channels(self):
        from core.ipc.bus import create_producer_bus
        bus = create_producer_bus({
            "a": ((4,), 10),
            "b": ((8,), 20),
        })
        try:
            listing = bus.list_channels()
            assert "a" in listing
            assert "b" in listing
            assert listing["a"]["shape"] == (4,)
            assert listing["b"]["capacity"] == 20
        finally:
            bus.unlink_all()
            bus.close_all()

    def test_convenience_buses(self):
        from core.ipc.bus import create_producer_bus, create_consumer_bus
        specs = {"test_conv": ((2,), 10)}

        producer = create_producer_bus(specs)
        try:
            producer.publish("test_conv", np.array([1.0, 2.0], dtype=np.float32))

            consumer = create_consumer_bus(specs)
            try:
                latest = consumer.read_latest("test_conv", n=1)
                assert latest.shape == (1, 2)
                np.testing.assert_allclose(latest[0], [1.0, 2.0])
            finally:
                consumer.close_all()
        finally:
            producer.unlink_all()
            producer.close_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])