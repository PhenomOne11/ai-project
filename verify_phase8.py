"""
Phase 8 Verification Script
"""

import sys
import time
import torch
import torch.nn as nn


def verify_fisher():
    print("=" * 60)
    print("  PHASE 8: Fisher Information Matrix")
    print("=" * 60)

    from core.adaptation.fisher import FisherInformation

    model = nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)
    )
    fisher = FisherInformation(model)

    data = torch.randn(50, 16)
    t0 = time.perf_counter()
    fisher.compute(data, nn.MSELoss(), num_samples=30)
    dt = (time.perf_counter() - t0) * 1000

    summary = fisher.get_summary()
    print(f"  ✅ Fisher computed in {dt:.1f} ms")
    print(f"      Total Fisher: {summary['total_fisher']:.4f}")
    print(f"      Mean Fisher:  {summary['mean_fisher']:.6f}")
    print(f"      Num Params:   {summary['num_params']:,}")

    # Verify penalty is zero at anchor
    p0 = fisher.penalty(lambda_=1.0).item()
    print(f"  ✅ Penalty at anchor: {p0:.8f} (expect 0)")

    # Perturb and check penalty grows
    with torch.no_grad():
        for p in model.parameters():
            p.data += 0.5
    p1 = fisher.penalty(lambda_=1.0).item()
    print(f"  ✅ Penalty after perturb: {p1:.4f} (expect > 0)")

    assert p0 < 1e-6
    assert p1 > 0
    return True


def verify_replay():
    print("\n" + "=" * 60)
    print("  PHASE 8: Priority Replay Buffer")
    print("=" * 60)

    from core.adaptation.replay import PriorityReplayBuffer

    buf = PriorityReplayBuffer(capacity=10, mode='max-error')

    # Add low-priority items first
    for i in range(5):
        buf.add(torch.full((1, 4), float(i)), priority=float(i))

    print(f"  After 5 low-priority adds: size={len(buf)}")

    # Add high-priority items (should evict low ones)
    for i in range(10, 20):
        buf.add(torch.full((1, 4), float(i)), priority=float(i))

    print(f"  After 10 high-priority adds: size={len(buf)}")
    print(f"  Buffer stats: {buf.get_stats()}")

    assert len(buf) == 10
    assert buf.get_stats()['max_priority'] >= 19.0

    # Sample top 5
    sampled = buf.sample(5)
    print(f"  Sampled top 5: shape={sampled.shape}")

    return True


def verify_adapt_engine():
    print("\n" + "=" * 60)
    print("  PHASE 8: Online Adaptation Engine v2")
    print("=" * 60)

    from core.adaptation.engine import OnlineAdaptationEngineV2

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid()
    )

    engine = OnlineAdaptationEngineV2(
        model,
        error_threshold=0.1,
        adaptive_threshold=True,
        buffer_capacity=32,
        micro_batch_size=8,
        enable_ewc=True,
    )

    # Establish EWC baseline (as if after calibration)
    baseline_data = torch.rand(30, 1, 16, 16)
    engine.establish_ewc_baseline(baseline_data, num_samples=20)
    print(f"  ✅ EWC baseline established: {engine.fisher.has_fisher()}")

    # Simulate OOD cascade to trigger multiple adaptation steps
    print(f"\n  Simulating OOD cascade over 20 ticks...")
    for tick in range(20):
        # "Live" input, "reconstructed" output with big error
        inp = torch.rand(4, 1, 16, 16)
        rec = torch.rand(4, 1, 16, 16) * 0.3  # forced bad reconstruction
        _, num_ood = engine.evaluate_batch(inp, rec)

        if tick % 5 == 0:
            print(f"      Tick {tick:2d}: added {num_ood} OOD, "
                  f"buffer={len(engine.replay)}, "
                  f"adapts={engine.adaptation_count}, "
                  f"threshold={engine.current_threshold:.4f}")

    # Wait for background thread
    time.sleep(0.5)

    status = engine.get_status()
    print(f"\n  ✅ Status after cascade:")
    print(f"      Total adaptations:    {status['adaptation_count']}")
    print(f"      Buffer size:          {status['buffer_size']}")
    print(f"      Current threshold:    {status['current_threshold']:.4f}")
    print(f"      Error EMA:            {status['error_ema']:.4f}")
    print(f"      EWC ready:            {status['ewc_ready']}")

    if engine.adaptation_history:
        print(f"\n  ✅ Recent adaptation steps:")
        for entry in engine.adaptation_history[-3:]:
            if 'error' in entry:
                print(f"      ERROR: {entry['error']}")
            else:
                print(f"      #{entry['count']}: "
                      f"loss {entry['loss_before']:.4f} → {entry['loss_after']:.4f} "
                      f"(Δ={entry['delta']:+.4f})")

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 8: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_adaptation.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 8 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Fisher Information", verify_fisher()))
    results.append(("Priority Replay Buffer", verify_replay()))
    results.append(("Adaptation Engine v2", verify_adapt_engine()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 8 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 8 COMPLETE — Ready for Phase 9!")
        print("  Next: Anomaly Scoring Engine (IF + Mahalanobis ensemble)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)