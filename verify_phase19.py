"""
Phase 19 Verification Script
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
    print("  PHASE 19: ModelComparisonState")
    print("=" * 60)

    from dashboard.panels.model_comparison import ModelComparisonState

    s = ModelComparisonState()
    print(f"  ✅ Initial: size={s.size}, empty={s.is_empty}")

    # Manually add two models
    s.add_model("fast", architecture="EdgeAE",
                parameters=150_000, memory_mb=0.55, throughput_fps=1000)
    s.add_model("slow", architecture="BigAE",
                parameters=500_000, memory_mb=2.10, throughput_fps=250)

    # Record samples
    rng = np.random.default_rng(7)
    for _ in range(50):
        s.record_latency("fast", 2.0 + rng.normal(0, 0.2))
        s.record_latency("slow", 5.5 + rng.normal(0, 0.5))
        s.record_recon("fast", 0.004 + rng.normal(0, 0.0005))
        s.record_recon("slow", 0.002 + rng.normal(0, 0.0003))

    print(f"  ✅ Models: {s.model_names()}")

    for name in s.model_names():
        m = s.get_model(name)
        ls, rs = m.latency_stats(), m.recon_stats()
        print(f"      {name:<6}: params={m.parameters:,} "
              f"lat={ls['p95']:.3f}ms recon={rs['mean']:.5f}")

    print(f"  ✅ Latency leaderboard: {s.rank_by('latency_mean')}")
    print(f"  ✅ Recon leaderboard: {s.rank_by('recon_mean')}")
    print(f"  ✅ Summary: {s.summary()}")

    return True


def verify_demo():
    print("\n" + "=" * 60)
    print("  PHASE 19: Demo Seeding")
    print("=" * 60)

    from dashboard.panels.model_comparison import ModelComparisonState

    s = ModelComparisonState()
    s.seed_demo()

    print(f"  ✅ Seeded {s.size} models")
    for name in s.model_names():
        m = s.get_model(name)
        ls, rs = m.latency_stats(), m.recon_stats()
        print(f"      {name:<16}: "
              f"params={m.parameters:>7,}  "
              f"mem={m.memory_mb:>5.3f}MB  "
              f"p95={ls['p95']:>5.2f}ms  "
              f"recon={rs['mean']:.5f}")

    summary = s.summary()
    print(f"  ✅ Ranking:")
    print(f"      Fastest (p95):           {summary['best_latency']}")
    print(f"      Sharpest reconstruction: {summary['best_recon']}")
    print(f"      Best trade-off:          {summary['best_efficiency']}")

    # Sanity checks on which models should win
    assert summary['best_latency'] == "edge_ae"
    assert summary['best_recon'] == "unet_ae"
    print(f"  ✅ Winner assertions passed")

    return True


def verify_figures():
    print("\n" + "=" * 60)
    print("  PHASE 19: Figure Builders")
    print("=" * 60)

    from dashboard.panels.model_comparison import (
        ModelComparisonState,
        build_params_bar, build_latency_box,
        build_latency_percentile_grouped, build_recon_bar,
        build_radar, build_throughput_fps,
    )
    from dashboard.theme import get_palette

    s = ModelComparisonState()
    s.seed_demo()
    rows = s.to_dataframe_like()
    samples = [(n, list(s.get_model(n).latency_samples)) for n in s.model_names()]
    p = get_palette("cyber_dark")

    fig1 = build_params_bar(rows, p, height=280)
    print(f"  ✅ Params bar: {len(fig1.data)} trace(s)")

    fig2 = build_latency_box(samples, p, height=320)
    print(f"  ✅ Latency box: {len(fig2.data)} trace(s)")

    fig3 = build_latency_percentile_grouped(rows, p, height=320)
    print(f"  ✅ Latency percentiles: {len(fig3.data)} traces (p50/p95/p99)")

    fig4 = build_recon_bar(rows, p, height=280)
    print(f"  ✅ Recon bar: {len(fig4.data)} trace(s)")

    fig5 = build_radar(rows, p, height=440)
    print(f"  ✅ Radar: {len(fig5.data)} traces (1 per model)")

    fig6 = build_throughput_fps(rows, p, height=280)
    print(f"  ✅ Throughput: {len(fig6.data)} trace(s)")

    # All palettes
    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        pal = get_palette(pname)
        build_params_bar(rows, pal)
        build_latency_box(samples, pal)
        build_latency_percentile_grouped(rows, pal)
        build_recon_bar(rows, pal)
        build_radar(rows, pal)
        build_throughput_fps(rows, pal)
    print(f"  ✅ All 4 palettes render successfully")

    return True


def verify_phase3_integration():
    print("\n" + "=" * 60)
    print("  PHASE 19: Phase 3 (ModelRegistry) Integration")
    print("=" * 60)

    from dashboard.panels.model_comparison import ModelComparisonState

    class FakeRegistry:
        def compare_architectures(self, **kwargs):
            return {
                'edge_ae': {
                    'parameters': 149_024, 'memory_mb': 0.57,
                    'latency_mean_ms': 2.4, 'latency_p95_ms': 3.1,
                },
                'vae': {
                    'parameters': 168_448, 'memory_mb': 0.64,
                    'latency_mean_ms': 3.1, 'latency_p95_ms': 4.2,
                },
                'unet_ae': {
                    'parameters': 209_617, 'memory_mb': 0.80,
                    'latency_mean_ms': 5.2, 'latency_p95_ms': 6.8,
                },
            }

    s = ModelComparisonState()
    count = s.import_from_registry(FakeRegistry())
    print(f"  ✅ Imported {count} models from FakeRegistry")
    for name in s.model_names():
        m = s.get_model(name)
        ls = m.latency_stats()
        print(f"      {name:<10}: params={m.parameters:>7,}  "
              f"p95≈{ls['p95']:.3f}ms  samples={ls['n']}")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 19: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/panels/model_comparison.py",
        "dashboard/panels/__init__.py",
        "dashboard/preview_panels.py",
        "tests/unit/test_model_comparison.py",
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
    print("  PHASE 19: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_model_comparison.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 19 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("ModelComparisonState", verify_state()))
    results.append(("Demo Seeding", verify_demo()))
    results.append(("Figure Builders", verify_figures()))
    results.append(("Phase 3 Integration", verify_phase3_integration()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 19 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 19 COMPLETE — Ready for Phase 20!")
        print("\n  ┌──────────────────────────────────────────────┐")
        print("  │  Preview the panel live:                     │")
        print("  │                                              │")
        print("  │     streamlit run dashboard/preview_panels.py │")
        print("  │                                              │")
        print("  │  Choose 'Model Comparison' in the sidebar.   │")
        print("  │                                              │")
        print("  │  Next: Phase 20 — Waveform Oscilloscope      │")
        print("  └──────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)