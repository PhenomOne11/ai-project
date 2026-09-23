"""
Phase 6 Verification Script
"""

import sys
import time
import numpy as np


def verify_ring_buffer():
    print("=" * 60)
    print("  PHASE 6: Lock-Free Ring Buffer")
    print("=" * 60)

    from core.ipc.ring_buffer import create_ring_buffer

    rb = create_ring_buffer("verify_rb", shape=(8,), dtype=np.float32, capacity=100)
    try:
        t0 = time.perf_counter()
        for i in range(1000):
            rb.write(np.array([float(i)] * 8, dtype=np.float32))
        write_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(1000):
            rb.read_latest(10)
        read_time = (time.perf_counter() - t0) * 1000

        print(f"  ✅ Wrote 1000 items in {write_time:.2f} ms ({write_time/10:.4f} ms/write)")
        print(f"  ✅ Read 1000× latest=10 in {read_time:.2f} ms ({read_time/10:.4f} ms/read)")
        print(f"  ✅ Write head: {rb.write_head}")
        print(f"  ✅ Items available: {len(rb)}")

        latest = rb.read_latest(3)
        assert latest.shape == (3, 8)
        assert latest[0, 0] == 999.0
        print(f"  ✅ Latest values: {latest[0, 0]}, {latest[1, 0]}, {latest[2, 0]}")
    finally:
        rb.unlink()
        rb.close()

    return True


def verify_channel():
    print("\n" + "=" * 60)
    print("  PHASE 6: Named IPC Channel")
    print("=" * 60)

    from core.ipc.channel import Channel

    ch = Channel(
        "verify_chan", shape=(32,), dtype=np.float32,
        capacity=200, create=True, heartbeat_timeout_s=1.0
    )
    try:
        for _ in range(50):
            ch.write(np.random.randn(32).astype(np.float32))

        print(f"  ✅ Writes: {ch.stats.write_count}")
        print(f"  ✅ Items: {len(ch)}")
        print(f"  ✅ Write head: {ch.write_head}")
        print(f"  ✅ Alive: {ch.is_alive}")
        print(f"  ✅ Total bytes written: {ch.stats.total_bytes_written:,}")

        assert ch.is_alive
        assert len(ch) == 50
    finally:
        ch.unlink()
        ch.close()

    return True


def verify_bus():
    print("\n" + "=" * 60)
    print("  PHASE 6: Shared Memory Bus")
    print("=" * 60)

    from core.ipc.bus import SharedMemoryBus

    bus = SharedMemoryBus()
    try:
        # Setup 4 channels
        bus.produce("latents", shape=(32,), dtype=np.float32, capacity=200)
        bus.produce("drift", shape=(8,), dtype=np.float32, capacity=200)
        bus.produce("telemetry", shape=(4,), dtype=np.float32, capacity=500)
        bus.produce("anomaly", shape=(1,), dtype=np.float32, capacity=200)

        print(f"  ✅ Registered {len(bus.channel_names)} channels")

        # Simulate 100 write ticks
        for tick in range(100):
            bus.publish("latents", np.random.randn(32).astype(np.float32))
            bus.publish("drift", np.random.randn(8).astype(np.float32) * 0.1)
            bus.publish("telemetry", np.array([
                45.2 + tick * 0.01,   # CPU %
                512.0 + tick * 0.5,   # Memory MB
                2.5 + np.random.rand(),  # Latency ms
                128.0,                # RSS MB
            ], dtype=np.float32))
            bus.publish("anomaly", np.array([np.random.rand()], dtype=np.float32))

        # Status print
        bus.print_status()

        # Verify contents
        lats = bus.read_latest("latents", n=10)
        drifts = bus.read_latest("drift", n=10)
        tele = bus.read_latest("telemetry", n=3)
        anom = bus.read_latest("anomaly", n=1)

        assert lats.shape == (10, 32)
        assert drifts.shape == (10, 8)
        assert tele.shape == (3, 4)
        assert anom.shape == (1, 1)

        print(f"\n  ✅ Latents: {lats.shape}")
        print(f"  ✅ Drift:   {drifts.shape}")
        print(f"  ✅ Telemetry: {tele.shape}")
        print(f"  ✅ Anomaly: {anom.shape}")
    finally:
        bus.unlink_all()
        bus.close_all()

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 6: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_ipc.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 6 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Ring Buffer", verify_ring_buffer()))
    results.append(("Named Channel", verify_channel()))
    results.append(("Shared Memory Bus", verify_bus()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 6 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 6 COMPLETE — Ready for Phase 7!")
        print("  Next: Precision Controller v2 (Gradient-Aware + Thermal Modeling)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)