"""
Phase 5 Verification Script
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np


def verify_detector():
    print("=" * 60)
    print("  PHASE 5: Multi-Method Drift Detector")
    print("=" * 60)

    from core.drift.detector import DriftDetector
    from core.constants import DriftSeverity

    all_methods = ['wasserstein', 'wasserstein_2', 'mmd', 'kl_divergence',
                   'jensen_shannon', 'energy', 'hellinger']

    det = DriftDetector(methods=all_methods, alert_threshold=0.25, critical_threshold=0.5)

    ref = np.random.randn(500)
    live_nominal = np.random.randn(500) + 0.05
    live_shifted = np.random.randn(500) + 2.0

    report_nom = det.compute(ref, live_nominal, "nominal_layer")
    report_shift = det.compute(ref, live_shifted, "shifted_layer")

    print(f"  ✅ Nominal drift:")
    print(f"      Composite: {report_nom.composite_score:.4f} | Severity: {report_nom.severity.value}")
    for m, r in report_nom.results.items():
        print(f"      {m:<16}: {r.score:.4f}")

    print(f"\n  ✅ Shifted drift:")
    print(f"      Composite: {report_shift.composite_score:.4f} | Severity: {report_shift.severity.value}")
    for m, r in report_shift.results.items():
        print(f"      {m:<16}: {r.score:.4f}")

    assert report_nom.severity == DriftSeverity.NOMINAL
    assert report_shift.severity in (DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY)
    print(f"\n  ✅ Severity classification verified")

    return True


def verify_radar():
    print("\n" + "=" * 60)
    print("  PHASE 5: Production Drift Radar")
    print("=" * 60)

    from core.drift.radar import DriftRadar

    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )

    radar = DriftRadar(
        methods=['wasserstein', 'mmd'],
        evaluation_interval=1,
        alert_threshold=0.1,
        critical_threshold=0.3
    )

    # Calibrate
    calib = torch.randn(16, 1, 32, 32)
    refs = radar.capture_reference(model, calib)
    print(f"  ✅ Calibration: {len(refs)} layers captured")
    for name, dist in refs.items():
        print(f"      {name}: {len(dist)} samples, range=[{dist.min():.2f}, {dist.max():.2f}]")

    # Simulate drift over time
    print(f"\n  Simulating 10 evaluation ticks...")
    for tick in range(10):
        # Gradually increase drift
        shift = tick * 0.3
        live = torch.randn(16, 1, 32, 32) + shift
        reports = radar.evaluate_from_tensor(model, live)

        if tick % 3 == 0:
            for layer, report in reports.items():
                print(f"      Tick {tick:2d} | {layer:<12} | "
                      f"score={report.composite_score:.4f} | "
                      f"severity={report.severity.value}")

    # Heatmap
    heatmap = radar.get_heatmap_data()
    print(f"\n  ✅ Heatmap data: {heatmap.shape}")

    # Trends
    print(f"  ✅ Layer trends:")
    for layer in radar.get_layer_names():
        trend = radar.get_trend(layer, window=5)
        print(f"      {layer}: {trend}")

    # Summary
    summary = radar.get_summary()
    print(f"\n  ✅ Summary: {summary['num_layers']} layers, "
          f"{summary['alert_count']} alerts, "
          f"{summary['critical_count']} critical")

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 5: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_drift.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 5 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Drift Detector", verify_detector()))
    results.append(("Drift Radar", verify_radar()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 5 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 5 COMPLETE — Ready for Phase 6!")
        print("  Next: Advanced IPC Multi-Channel Shared Memory Bus\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)