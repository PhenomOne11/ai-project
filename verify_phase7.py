"""
Phase 7 Verification Script
"""

import sys
import time
import torch
import torch.nn as nn


def verify_thermal():
    print("=" * 60)
    print("  PHASE 7: Thermal Model")
    print("=" * 60)

    from core.precision.thermal import ThermalModel

    tm = ThermalModel(baseline_latency_ms=3.0)

    print(f"  Initial state: {tm.current_state} "
          f"(load={tm.load_factor:.3f}, ratio={tm.latency_ratio:.3f})")

    # Simulate increasing thermal load
    for i in range(30):
        tm.record(3.0 + i * 0.5)

    print(f"  After heating: state={tm.current_state}, "
          f"load={tm.load_factor:.3f}, ratio={tm.latency_ratio:.3f}")

    assert tm.current_state in ('WARM', 'HOT')

    # Cool down
    for _ in range(60):
        tm.record(2.0)

    print(f"  After cooling: state={tm.current_state}, "
          f"load={tm.load_factor:.3f}, ratio={tm.latency_ratio:.3f}")

    return True


def verify_controller():
    print("\n" + "=" * 60)
    print("  PHASE 7: Precision Controller v2")
    print("=" * 60)

    from core.precision.controller import (
        AdaptivePrecisionEngineV2, PrecisionState
    )

    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32)
    )

    engine = AdaptivePrecisionEngineV2(
        model,
        target_latency_ms=3.0,
        window_size=15,
        cooldown_ticks=5,
        enable_int8=False
    )

    x = torch.randn(8, 64)

    print(f"  Initial state: {engine.current_state}")
    print(f"\n  Simulating 40 fast ticks...")

    for i in range(40):
        engine.step(lambda: model(x))

    status = engine.get_status()
    print(f"  After fast ticks:")
    print(f"    State:      {status['current_state']}")
    print(f"    p50:        {status['p50_ms']:.3f} ms")
    print(f"    p95:        {status['p95_ms']:.3f} ms")
    print(f"    Thermal:    {status['thermal_state']}")
    print(f"    Transitions:{status['total_transitions']}")

    # Now simulate heavy load (add sleep to force high latency)
    print(f"\n  Simulating 40 slow ticks (20ms per tick)...")

    def slow_forward():
        time.sleep(0.02)
        return model(x)

    for i in range(40):
        engine.step(slow_forward)

    status = engine.get_status()
    print(f"\n  After slow ticks:")
    print(f"    State:      {status['current_state']}")
    print(f"    p50:        {status['p50_ms']:.3f} ms")
    print(f"    p95:        {status['p95_ms']:.3f} ms")
    print(f"    Thermal:    {status['thermal_state']}")
    print(f"    Transitions:{status['total_transitions']}")

    assert status['total_transitions'] >= 1
    print(f"\n  ✅ Transition log:")
    for entry in engine.transition_log[:5]:
        print(f"    Tick {entry['tick']:3d}: {entry['from']} → {entry['to']} "
              f"(p95={entry['p95_ms']:.2f}ms, thermal={entry['thermal_state']})")

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 7: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_precision.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 7 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Thermal Model", verify_thermal()))
    results.append(("Precision Controller v2", verify_controller()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 7 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 7 COMPLETE — Ready for Phase 8!")
        print("  Next: Online Adaptation v2 (EWC + Replay Buffer)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)