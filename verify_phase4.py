"""
Phase 4 Verification Script
Validates attention modules, U-Net AE, and ensemble system.
"""

import sys
import time
import torch
import numpy as np


def verify_attention():
    print("=" * 60)
    print("  PHASE 4: Attention Modules")
    print("=" * 60)
    
    from core.models.attention import (
        SqueezeExcitation, CBAM, SelfAttention2D,
        MultiHeadSpatialAttention, GatedAttentionUnit
    )
    
    x = torch.randn(2, 32, 16, 16)
    
    modules = {
        'SE': SqueezeExcitation(32),
        'CBAM': CBAM(32),
        'SelfAttn': SelfAttention2D(32),
        'MultiHead': MultiHeadSpatialAttention(32, num_heads=4),
        'GAU': GatedAttentionUnit(32),
    }
    
    for name, mod in modules.items():
        t0 = time.perf_counter()
        out = mod(x)
        lat = (time.perf_counter() - t0) * 1000
        params = sum(p.numel() for p in mod.parameters())
        print(f"  ✅ {name:<12}: {x.shape} → {out.shape} | {lat:.2f}ms | {params:,} params")
    
    return True


def verify_unet():
    print("\n" + "=" * 60)
    print("  PHASE 4: U-Net Autoencoder")
    print("=" * 60)
    
    from core.models.unet_ae import UNetAutoencoder
    
    configs = [
        {"depth": 2, "base_channels": 16, "attention": None},
        {"depth": 3, "base_channels": 8, "attention": None},
        {"depth": 3, "base_channels": 8, "attention": "se"},
        {"depth": 3, "base_channels": 8, "attention": "cbam"},
    ]
    
    x = torch.randn(4, 1, 64, 64)
    
    for cfg in configs:
        model = UNetAutoencoder(
            in_channels=1, input_size=64, latent_dim=32, **cfg
        )
        model.eval()
        
        t0 = time.perf_counter()
        out = model(x)
        lat = (time.perf_counter() - t0) * 1000
        
        z = model.encode(x)
        amap = model.get_anomaly_map(x)
        params = sum(p.numel() for p in model.parameters())
        
        attn_str = cfg['attention'] or 'none'
        print(
            f"  ✅ depth={cfg['depth']}, attn={attn_str:<5}: "
            f"{lat:.2f}ms | {params:,} params | "
            f"recon={out.shape}, latent={z.shape}, amap={amap.shape}"
        )
    
    # Test multiscale features
    model = UNetAutoencoder(in_channels=1, input_size=64, latent_dim=32, depth=3, base_channels=8)
    features = model.encode_multiscale(x)
    print(f"\n  ✅ Multi-scale features:")
    for name, feat in features.items():
        print(f"      {name}: {feat.shape}")
    
    # Test regional error
    regional = model.get_reconstruction_error_per_region(x, grid_size=4)
    print(f"  ✅ Regional error map: {regional.shape}")
    
    return True


def verify_ensemble():
    print("\n" + "=" * 60)
    print("  PHASE 4: Model Ensemble")
    print("=" * 60)
    
    from core.models.ensemble import create_default_ensemble
    
    x = torch.randn(4, 1, 64, 64)
    
    for strategy in ["mean", "weighted", "median", "best", "uncertainty"]:
        ens = create_default_ensemble(fusion=strategy)
        
        t0 = time.perf_counter()
        fused, uncertainty = ens.forward_with_uncertainty(x)
        lat = (time.perf_counter() - t0) * 1000
        
        print(
            f"  ✅ Strategy '{strategy:<12}': "
            f"fused={fused.shape}, uncert={uncertainty.shape}, "
            f"latency={lat:.1f}ms"
        )
    
    # Detailed report
    ens = create_default_ensemble(fusion="weighted")
    _ = ens(x)
    _ = ens(x)
    print(f"\n  Ensemble Report:")
    ens.print_report()
    
    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 4: Running Unit Tests")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_models_p2.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 4 FULL VERIFICATION")
    print("🔬" * 30 + "\n")
    
    results = []
    results.append(("Attention Modules", verify_attention()))
    results.append(("U-Net Autoencoder", verify_unet()))
    results.append(("Model Ensemble", verify_ensemble()))
    results.append(("Unit Tests (pytest)", run_pytest()))
    
    print("\n" + "=" * 60)
    print("  PHASE 4 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Result: {total_pass}/{total} checks passed")
    
    if total_pass == total:
        print("\n  🎉 PHASE 4 COMPLETE — Ready for Phase 5!")
        print("  Next: Multi-Head Drift Detection Engine")
        print("  (Phases 5+ are shorter and faster!)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding to Phase 5.\n")
    
    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)