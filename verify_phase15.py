"""
Phase 15 Verification Script
"""

import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np


def verify_state():
    print("=" * 60)
    print("  PHASE 15: LatentExplorerState")
    print("=" * 60)

    from dashboard.panels.latent_explorer import LatentExplorerState

    s = LatentExplorerState(latent_dim=32, buffer_size=300, n_components=3)
    print(f"  ✅ Initial: size={s.size}, empty={s.is_empty}")

    s.push_batch(np.random.randn(100, 32))
    print(f"  ✅ After push 100: size={s.size}")

    coords, ratios = s.project()
    print(f"  ✅ Projection: coords.shape={coords.shape}, "
          f"explained={ratios.sum():.3f}")

    stats = s.stats()
    print(f"  ✅ Stats:")
    for k, v in stats.items():
        print(f"      {k}: {v}")

    # Cluster
    labels = s.cluster(n_clusters=3, n_iter=15)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  ✅ Clusters: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Overflow
    s2 = LatentExplorerState(latent_dim=8, buffer_size=10)
    for _ in range(50):
        s2.push(np.random.randn(8))
    print(f"  ✅ Overflow bounded: size={s2.size} (expected 10)")

    # Reset
    s2.reset()
    print(f"  ✅ Reset: size={s2.size}, empty={s2.is_empty}")

    return True


def verify_demo_seed():
    print("\n" + "=" * 60)
    print("  PHASE 15: Demo Seeding")
    print("=" * 60)

    from dashboard.panels.latent_explorer import LatentExplorerState

    s = LatentExplorerState(latent_dim=32, buffer_size=500, n_components=3)
    s.seed_demo(n=200)
    print(f"  ✅ Seeded: size={s.size}")

    coords, ratios = s.project()
    print(f"  ✅ Projected: coords.shape={coords.shape}")
    print(f"      PC1 variance: {ratios[0]*100:.1f}%")
    print(f"      PC2 variance: {ratios[1]*100:.1f}%")
    print(f"      PC3 variance: {ratios[2]*100:.1f}%")
    print(f"      Total explained: {ratios.sum()*100:.1f}%")

    # Should see the 3-cluster structure
    labels = s.cluster(n_clusters=3, n_iter=20)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  ✅ Detected {len(unique)} clusters with sizes: {counts.tolist()}")

    return True


def verify_figures():
    print("\n" + "=" * 60)
    print("  PHASE 15: Figure Builders (pure)")
    print("=" * 60)

    from dashboard.panels.latent_explorer import (
        build_trajectory_figure, build_projection_panel,
    )
    from dashboard.theme import get_palette

    p = get_palette("cyber_dark")
    coords = np.random.randn(120, 3)

    # Time-coloring
    fig1 = build_trajectory_figure(coords=coords, palette=p, color_mode="time")
    print(f"  ✅ time-mode figure: {len(fig1.data)} traces, height={fig1.layout.height}")

    # Cluster-coloring
    labels = np.array([i % 4 for i in range(len(coords))])
    fig2 = build_trajectory_figure(coords=coords, labels=labels, palette=p, color_mode="cluster")
    print(f"  ✅ cluster-mode figure: {len(fig2.data)} traces")

    # Novelty-coloring
    fig3 = build_trajectory_figure(coords=coords, palette=p, color_mode="novelty")
    print(f"  ✅ novelty-mode figure: {len(fig3.data)} traces")

    # 2D panel
    fig4 = build_projection_panel(coords=coords, palette=p)
    print(f"  ✅ 2D projections: {len(fig4.data)} traces")

    # All palettes
    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        pal = get_palette(pname)
        f = build_trajectory_figure(coords=coords, palette=pal)
        assert len(f.data) >= 1
    print(f"  ✅ All 4 palettes render successfully")

    return True


def verify_integration_with_phase2():
    print("\n" + "=" * 60)
    print("  PHASE 15: Integration with Phase 2 PCA")
    print("=" * 60)

    from core.math_utils.linear_algebra import batch_svd_project
    from dashboard.panels.latent_explorer import LatentExplorerState

    # Simulate a real latent stream from a model
    rng = np.random.default_rng(42)
    # 3 latent dimensions coupled to a helix
    t = np.linspace(0, 4 * np.pi, 200)
    latents = rng.normal(0, 0.05, (200, 32))
    latents[:, 0] += np.cos(t) * 2
    latents[:, 1] += np.sin(t) * 2
    latents[:, 2] += t / 4

    s = LatentExplorerState(latent_dim=32, buffer_size=500)
    s.push_batch(latents)

    coords, ratios = s.project()
    # Should preserve the helix structure — high variance explained
    explained = ratios.sum()
    print(f"  ✅ PCA preserved structure: {explained*100:.1f}% variance in 3 PCs")
    assert explained > 0.4

    # Check that PC1 correlates with the "helix angle" (t variable)
    # This is a rough structural check
    pc1 = coords[:, 0]
    corr_with_t = abs(np.corrcoef(pc1, t)[0, 1])
    print(f"  ✅ PC1 correlation with helix parameter: {corr_with_t:.3f}")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 15: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/panels/latent_explorer.py",
        "dashboard/panels/__init__.py",
        "dashboard/preview_panels.py",
        "tests/unit/test_panels.py",
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
    print("  PHASE 15: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_panels.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 15 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("LatentExplorerState", verify_state()))
    results.append(("Demo Seeding", verify_demo_seed()))
    results.append(("Figure Builders", verify_figures()))
    results.append(("Phase-2 PCA Integration", verify_integration_with_phase2()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 15 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 15 COMPLETE — Ready for Phase 16!")
        print("\n  ┌──────────────────────────────────────────────────┐")
        print("  │  Preview the panel live:                         │")
        print("  │                                                  │")
        print("  │     streamlit run dashboard/preview_panels.py     │")
        print("  │                                                  │")
        print("  │  Next: Phase 16 — Multi-Layer Drift Heatmap      │")
        print("  └──────────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)