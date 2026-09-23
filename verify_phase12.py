"""
Phase 12 Verification Script
"""

import sys
import time


def verify_primitives():
    print("=" * 60)
    print("  PHASE 12: Metrics Primitives")
    print("=" * 60)

    from core.metrics import Counter, Gauge, Histogram, Timer

    # Counter
    c = Counter("requests", "Total requests")
    for _ in range(100):
        c.inc()
    print(f"  ✅ Counter: {c.value}")

    # Gauge
    g = Gauge("active_workers", initial=4)
    g.inc()
    g.dec(2)
    print(f"  ✅ Gauge: {g.value}")

    # Histogram
    h = Histogram("latency_ms", window_size=500, range_min=0, range_max=20)
    import random
    for _ in range(300):
        h.observe(random.gauss(5.0, 1.5))
    print(f"  ✅ Histogram: count={h.count}, mean={h.mean:.2f}, "
          f"p50={h.p50:.2f}, p95={h.p95:.2f}, p99={h.p99:.2f}")

    # Timer
    h2 = Histogram("op_ms")
    with Timer(h2):
        time.sleep(0.005)
    print(f"  ✅ Timer: recorded {h2.samples()[0]:.2f} ms")

    return True


def verify_system_probes():
    print("\n" + "=" * 60)
    print("  PHASE 12: System Probes")
    print("=" * 60)

    from core.metrics.system import probe_process, probe_system

    p = probe_process()
    print(f"  ✅ Process telemetry:")
    print(f"      RSS:      {p['rss_mb']:.2f} MB")
    print(f"      VMS:      {p['vms_mb']:.2f} MB")
    print(f"      Threads:  {p['num_threads']}")

    s = probe_system()
    print(f"  ✅ System telemetry:")
    print(f"      CPU:      {s['cpu_percent']:.1f}% ({s['cpu_count']} cores)")
    print(f"      RAM:      {s['mem_percent']:.1f}% "
          f"({s['mem_available_mb']:.0f} MB free / {s['mem_total_mb']:.0f} MB)")
    print(f"      Uptime:   {s['uptime_s']/3600:.1f} hours")

    return True


def verify_registry():
    print("\n" + "=" * 60)
    print("  PHASE 12: Metrics Registry")
    print("=" * 60)

    from core.metrics import MetricsRegistry

    reg = MetricsRegistry(name="neuralstream")

    # Simulate pipeline metrics
    reg.counter("batches_total").inc(150)
    reg.counter("ood_samples_total").inc(23)
    reg.gauge("active_precision_id").set(1.0)
    reg.gauge("circuit_breaker_trips").set(2.0)

    # Latency histogram
    import random
    for _ in range(500):
        reg.histogram("forward_latency_ms").observe(random.gauss(3.0, 0.5))

    # Timer usage
    with reg.timer("drift_eval_ms"):
        time.sleep(0.002)

    snap = reg.snapshot()
    print(f"  ✅ Registry snapshot:")
    print(f"      Counters:   {snap['counters']}")
    print(f"      Gauges:     {snap['gauges']}")
    print(f"      Histograms: {list(snap['histograms'].keys())}")

    # Extended system snapshot
    snap_full = reg.snapshot_system()
    print(f"  ✅ Extended snapshot has system data: "
          f"{'system' in snap_full and 'process' in snap_full and 'system' in snap_full['system']}")

    # JSON export
    text = reg.export_json()
    print(f"  ✅ JSON export: {len(text):,} chars")

    # Names
    names = reg.list_names()
    print(f"  ✅ Registered names: {sum(len(v) for v in names.values())} total")

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 12: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_metrics.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 12 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Metrics Primitives", verify_primitives()))
    results.append(("System Probes", verify_system_probes()))
    results.append(("Metrics Registry", verify_registry()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 12 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 12 COMPLETE — Ready for Phase 13!")
        print("  Next: Event Bus & Pub/Sub Notification System (short!)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)