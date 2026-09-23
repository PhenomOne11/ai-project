"""
Phase 16 Verification Script
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
    print("  PHASE 16: DriftHeatmapState")
    print("=" * 60)

    from dashboard.panels.drift_heatmap import DriftHeatmapState

    s = DriftHeatmapState(layer_names=["enc.0", "enc.2", "fc.0"], history_length=100)
    print(f"  ✅ Initial: layers={s.num_layers}, empty={s.is_empty}")

    # Push 30 ticks with a rising layer
    for i in range(30):
        s.push_snapshot({
            "enc.0": 0.04 + np.random.rand() * 0.02,
            "enc.2": 0.05 + i * 0.01 + np.random.rand() * 0.02,
            "fc.0":  0.08 + np.random.rand() * 0.03,
        })

    matrix, names = s.as_matrix()
    print(f"  ✅ Pushed 30 ticks: matrix={matrix.shape}")

    gs = s.global_stats()
    print(f"  ✅ Global stats:")
    for k, v in gs.items():
        print(f"      {k}: {v}")

    # Per-layer detail
    print(f"\n  ✅ Per-layer analysis:")
    for layer in names:
        st = s.layer_stats(layer)
        print(f"      {layer:<8}: latest={st['latest']:.4f} "
              f"mean={st['mean']:.4f} max={st['max']:.4f} "
              f"alerts={st['alerts']} trend={st['trend']}")

    return True


def verify_demo():
    print("\n" + "=" * 60)
    print("  PHASE 16: Demo Seeding")
    print("=" * 60)

    from dashboard.panels.drift_heatmap import DriftHeatmapState

    s = DriftHeatmapState(
        history_length=200,
        warning_threshold=0.15,
        critical_threshold=0.35,
    )
    s.seed_demo(n_ticks=120)
    print(f"  ✅ Seeded {s.num_layers} layers × 120 ticks")

    gs = s.global_stats()
    print(f"  ✅ Global stats:")
    for k, v in gs.items():
        print(f"      {k}: {v}")

    # Alert timeline
    events = s.alert_events(limit=5)
    print(f"  ✅ Top 5 alerts:")
    for e in events:
        print(f"      tick={e['tick']:3d} | {e['layer']:<20} | "
              f"score={e['score']:.4f} | {e['level']}")

    # Should show a spike from layer 2 near the end
    assert gs['critical_alerts'] > 0

    return True


def verify_figures():
    print("\n" + "=" * 60)
    print("  PHASE 16: Figure Builders")
    print("=" * 60)

    from dashboard.panels.drift_heatmap import (
        build_heatmap_figure, build_layer_trajectories, build_current_scores_bar,
    )
    from dashboard.theme import get_palette

    p = get_palette("cyber_dark")
    matrix = np.random.rand(4, 80) * 0.6
    names = ["enc.0", "enc.2", "fc.0", "dec.0"]

    f1 = build_heatmap_figure(matrix, names, p, height=380)
    print(f"  ✅ Heatmap: {len(f1.data)} trace(s), height={f1.layout.height}")

    f2 = build_layer_trajectories(matrix, names, p, height=320)
    print(f"  ✅ Trajectories: {len(f2.data)} trace(s) for {len(names)} layers")

    f3 = build_current_scores_bar(names, [0.05, 0.22, 0.45, 0.10], p, height=260)
    print(f"  ✅ Current-score bar: {len(f3.data)} trace(s)")

    # All 4 palettes
    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        pal = get_palette(pname)
        build_heatmap_figure(matrix, names, pal)
        build_layer_trajectories(matrix, names, pal)
        build_current_scores_bar(names, [0.1, 0.2, 0.3, 0.4], pal)
    print(f"  ✅ All 4 palettes render successfully")

    return True


def verify_radar_import():
    print("\n" + "=" * 60)
    print("  PHASE 16: DriftRadar Integration")
    print("=" * 60)

    from dashboard.panels.drift_heatmap import DriftHeatmapState

    # Simulate what a real DriftRadar produces
    class Fake:
        drift_history = {
            "encoder_conv.0": [
                {"composite_score": 0.04, "severity": "nominal"},
                {"composite_score": 0.05, "severity": "nominal"},
                {"composite_score": 0.05, "severity": "nominal"},
                {"composite_score": 0.06, "severity": "nominal"},
                {"composite_score": 0.07, "severity": "nominal"},
                {"composite_score": 0.08, "severity": "nominal"},
            ],
            "encoder_conv.2": [
                {"composite_score": 0.10, "severity": "nominal"},
                {"composite_score": 0.14, "severity": "nominal"},
                {"composite_score": 0.20, "severity": "warning"},
                {"composite_score": 0.28, "severity": "warning"},
                {"composite_score": 0.35, "severity": "critical"},
                {"composite_score": 0.42, "severity": "critical"},
            ],
            "encoder_fc.0": [
                {"composite_score": 0.03, "severity": "nominal"},
                {"composite_score": 0.04, "severity": "nominal"},
                {"composite_score": 0.04, "severity": "nominal"},
                {"composite_score": 0.05, "severity": "nominal"},
                {"composite_score": 0.05, "severity": "nominal"},
                {"composite_score": 0.06, "severity": "nominal"},
            ],
        }

    s = DriftHeatmapState()
    n_imported = s.import_from_radar(Fake())
    print(f"  ✅ Imported {n_imported} layers from DriftRadar-like object")

    matrix, names = s.as_matrix()
    print(f"      Matrix shape: {matrix.shape}")

    # Check that the rising layer is detected
    enc2_trend = s.layer_stats("encoder_conv.2")
    print(f"      encoder_conv.2: trend={enc2_trend['trend']}, "
          f"latest={enc2_trend['latest']:.4f}")
    assert enc2_trend['trend'] == 'rising'

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 16: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/panels/drift_heatmap.py",
        "dashboard/panels/__init__.py",
        "dashboard/preview_panels.py",
        "tests/unit/test_drift_panel.py",
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
    print("  PHASE 16: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_drift_panel.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 16 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("DriftHeatmapState", verify_state()))
    results.append(("Demo Seeding", verify_demo()))
    results.append(("Figure Builders", verify_figures()))
    results.append(("DriftRadar Integration", verify_radar_import()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 16 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 16 COMPLETE — Ready for Phase 17!")
        print("\n  ┌──────────────────────────────────────────────┐")
        print("  │  Preview the panel live:                     │")
        print("  │                                              │")
        print("  │     streamlit run dashboard/preview_panels.py │")
        print("  │                                              │")
        print("  │  Switch to 'Drift Heatmap' in the sidebar.   │")
        print("  │                                              │")
        print("  │  Next: Phase 17 — Hardware Telemetry Dashboard│")
        print("  └──────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)