"""
Phase 17 Verification Script
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np


def verify_state():
    print("=" * 60)
    print("  PHASE 17: TelemetryState")
    print("=" * 60)

    from dashboard.panels.telemetry import TelemetryState

    s = TelemetryState(history_length=200)
    print(f"  ✅ Initial: size={s.size}, empty={s.is_empty}")

    # Push live auto-probes
    for i in range(5):
        sample = s.append_auto(latency_ms=2.0 + i * 0.1)
        print(f"      sample {i}: CPU={sample.cpu_percent:.1f}%, "
              f"RSS={sample.rss_mb:.1f}MB, latency={sample.latency_ms:.2f}ms")

    stats = s.stats()
    print(f"  ✅ Live stats:")
    print(f"      samples:      {stats['n_samples']}")
    print(f"      CPU mean/p95: {stats['cpu']['mean']:.2f} / {stats['cpu']['p95']:.2f}")
    print(f"      RSS mean/p95: {stats['rss']['mean']:.2f} / {stats['rss']['p95']:.2f}")
    print(f"      Latency p95:  {stats['latency']['p95']:.3f} ms")
    print(f"      health:       {stats['health']}")
    print(f"      overall:      {stats['overall']}")

    return True


def verify_demo():
    print("\n" + "=" * 60)
    print("  PHASE 17: Demo Seeding")
    print("=" * 60)

    from dashboard.panels.telemetry import TelemetryState

    s = TelemetryState(
        history_length=300,
        cpu_warning=70.0, cpu_critical=90.0,
        latency_warning_ms=3.0, latency_critical_ms=6.0,
    )
    s.seed_demo(200)

    stats = s.stats()
    print(f"  ✅ Seeded {s.size} samples")
    for key in ['cpu', 'rss', 'vms', 'latency']:
        sub = stats[key]
        print(f"      {key:<8}: μ={sub['mean']:7.2f}  p95={sub['p95']:7.2f}  "
              f"max={sub['max']:7.2f}")
    print(f"  ✅ Health: {stats['health']}")
    print(f"  ✅ Overall: {stats['overall']}")

    # Demo should trigger at least a latency warning (spike in seed)
    assert stats['latency']['max'] > 3.0
    print(f"  ✅ Demo latency spike present: max={stats['latency']['max']:.2f}ms")

    return True


def verify_figures():
    print("\n" + "=" * 60)
    print("  PHASE 17: Figure Builders")
    print("=" * 60)

    from dashboard.panels.telemetry import (
        build_multi_metric_chart, build_gauge, build_latency_histogram,
        TelemetryState,
    )
    from dashboard.theme import get_palette

    p = get_palette("cyber_dark")
    s = TelemetryState()
    s.seed_demo(200)
    cols = s.to_arrays()

    fig1 = build_multi_metric_chart(cols, p, height=500)
    print(f"  ✅ Multi-metric chart: {len(fig1.data)} traces, height={fig1.layout.height}")

    fig2 = build_gauge("CPU", 55.0, 100.0, 70.0, 90.0, p)
    print(f"  ✅ Gauge: {len(fig2.data)} indicator")

    counts, edges = s.latency_histogram(30)
    fig3 = build_latency_histogram(counts, edges, p, height=320)
    print(f"  ✅ Histogram: {len(fig3.data)} traces ({counts.sum()} samples)")

    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        pal = get_palette(pname)
        build_multi_metric_chart(cols, pal)
        build_gauge("x", 40.0, 100.0, 70.0, 90.0, pal)
        build_latency_histogram(counts, edges, pal)
    print(f"  ✅ All 4 palettes render")

    return True


def verify_health_classification():
    print("\n" + "=" * 60)
    print("  PHASE 17: Health Classification")
    print("=" * 60)

    from dashboard.panels.telemetry import TelemetryState

    test_cases = [
        ("nominal",  dict(cpu_percent=30.0, rss_mb=300.0, latency_ms=1.0)),
        ("warning",  dict(cpu_percent=75.0, rss_mb=300.0, latency_ms=1.0)),
        ("critical", dict(cpu_percent=95.0, rss_mb=300.0, latency_ms=1.0)),
    ]

    for expected, kwargs in test_cases:
        s = TelemetryState(history_length=50)
        for _ in range(20):
            s.append(**kwargs)
        stats = s.stats()
        actual = stats['overall']
        print(f"  ✅ {expected:<10} → {actual}")
        assert actual == expected, f"Expected {expected}, got {actual}"

    # Latency-driven criticality
    s = TelemetryState(history_length=50, latency_warning_ms=3.0, latency_critical_ms=6.0)
    for _ in range(20):
        s.append(cpu_percent=10.0, latency_ms=8.0)
    assert s.stats()['overall'] == 'critical'
    print(f"  ✅ Latency critical: 8.0ms → critical")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 17: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/panels/telemetry.py",
        "dashboard/panels/__init__.py",
        "dashboard/preview_panels.py",
        "tests/unit/test_telemetry_panel.py",
    ]
    all_ok = True
    for f in files:
        exists = Path(f).is_file()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {f}")
    return all_ok


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 17: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_telemetry_panel.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 17 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("TelemetryState", verify_state()))
    results.append(("Demo Seeding", verify_demo()))
    results.append(("Figure Builders", verify_figures()))
    results.append(("Health Classification", verify_health_classification()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 17 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 17 COMPLETE — Ready for Phase 18!")
        print("\n  ┌──────────────────────────────────────────────┐")
        print("  │  Preview the panel live:                     │")
        print("  │                                              │")
        print("  │     streamlit run dashboard/preview_panels.py │")
        print("  │                                              │")
        print("  │  Switch to 'Hardware Telemetry' in sidebar.  │")
        print("  │                                              │")
        print("  │  Next: Phase 18 — Anomaly & OOD Console      │")
        print("  └──────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)