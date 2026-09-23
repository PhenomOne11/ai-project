"""
Phase 9 Verification Script
"""

import sys
import torch
import numpy as np


def verify_components():
    print("=" * 60)
    print("  PHASE 9: Component Scorers")
    print("=" * 60)

    from core.anomaly.isolation import IsolationScorer
    from core.anomaly.mahalanobis import MahalanobisScorer

    ref = np.random.randn(300, 16)

    # Isolation Forest
    iso = IsolationScorer(n_estimators=50).fit(ref)
    normal_iso = iso.score(np.random.randn(20, 16)).mean()
    outlier_iso = iso.score(np.full((1, 16), 8.0))[0]
    print(f"  ✅ Isolation Forest: normal={normal_iso:.4f}, outlier={outlier_iso:.4f}")

    # Mahalanobis
    maha = MahalanobisScorer().fit(ref)
    normal_maha = maha.score(np.random.randn(20, 16)).mean()
    outlier_maha = maha.score(np.full((1, 16), 8.0))[0]
    print(f"  ✅ Mahalanobis:      normal={normal_maha:.4f}, outlier={outlier_maha:.4f}")

    assert outlier_iso > normal_iso
    assert outlier_maha > normal_maha
    return True


def verify_scorer():
    print("\n" + "=" * 60)
    print("  PHASE 9: Unified Anomaly Scorer")
    print("=" * 60)

    from core.anomaly.scorer import AnomalyScorer

    scorer = AnomalyScorer(
        reconstruction_weight=0.5,
        isolation_weight=0.3,
        mahalanobis_weight=0.2,
        n_estimators=50
    )

    # Fit on reference latents
    ref_latents = np.random.randn(300, 32)
    scorer.fit(ref_latents)

    print(f"  ✅ Fitted: {scorer.is_fitted}")
    print(f"      Weights: {scorer.get_weights()}")

    # Simulate normal batch
    latent_norm = torch.randn(16, 32)
    inp_norm = torch.randn(16, 1, 64, 64)
    rec_norm = inp_norm + 0.05 * torch.randn_like(inp_norm)

    result_norm = scorer.score(latent_norm, rec_norm, inp_norm)
    print(f"\n  Normal batch:")
    print(f"    Fused score:      {result_norm['score'].mean():.4f}")
    print(f"    Recon component:  {result_norm['recon_component'].mean():.4f}")
    print(f"    Isolation:        {result_norm['isolation_component'].mean():.4f}")
    print(f"    Mahalanobis:      {result_norm['mahalanobis_component'].mean():.4f}")

    # Simulate anomalous batch
    latent_anom = torch.randn(16, 32) * 5.0   # far from distribution
    inp_anom = torch.randn(16, 1, 64, 64)
    rec_anom = torch.zeros_like(inp_anom)      # garbage reconstruction

    result_anom = scorer.score(latent_anom, rec_anom, inp_anom)
    print(f"\n  Anomalous batch:")
    print(f"    Fused score:      {result_anom['score'].mean():.4f}")
    print(f"    Recon component:  {result_anom['recon_component'].mean():.4f}")
    print(f"    Isolation:        {result_anom['isolation_component'].mean():.4f}")
    print(f"    Mahalanobis:      {result_anom['mahalanobis_component'].mean():.4f}")

    assert result_anom['score'].mean() > result_norm['score'].mean()

    print(f"\n  ✅ Anomaly detection separation verified")
    print(f"      Delta: {result_anom['score'].mean() - result_norm['score'].mean():.4f}")

    # Status
    status = scorer.get_status()
    print(f"\n  ✅ Scorer status:")
    for key, val in status.items():
        print(f"      {key}: {val}")

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 9: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_anomaly.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 9 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Component Scorers", verify_components()))
    results.append(("Unified Anomaly Scorer", verify_scorer()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 9 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 9 COMPLETE — Ready for Phase 10!")
        print("  Next: Data Pipeline v2 (Video, Audio, Time-Series, Multi-Modal)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)