"""
Phase 3 Verification Script
Validates all model architectures and runs unit tests.
"""

import sys
import time
import torch
import numpy as np


def verify_edge_ae():
    print("=" * 60)
    print("  PHASE 3: EdgeStreamingAE")
    print("=" * 60)
    
    from core.models.edge_ae import EdgeStreamingAE
    
    model = EdgeStreamingAE(in_channels=1, input_size=64, latent_dim=32)
    x = torch.randn(8, 1, 64, 64)
    
    t0 = time.perf_counter()
    out = model(x)
    lat = (time.perf_counter() - t0) * 1000
    
    z = model.encode(x)
    diag = model.diagnostics(x)
    
    print(f"  ✅ Forward: {x.shape} → {out.shape} in {lat:.2f} ms")
    print(f"  ✅ Latent: {z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
    print(f"  ✅ Parameters: {diag['total_parameters']:,}")
    print(f"  ✅ Memory: {diag['memory_mb']:.4f} MB")
    print(f"  ✅ Recon MSE: {diag['reconstruction_error']:.6f}")
    
    # Test residual variant
    model_res = EdgeStreamingAE(in_channels=1, input_size=64, latent_dim=32, use_residual=True)
    out_res = model_res(x)
    print(f"  ✅ Residual variant: {out_res.shape}")
    
    return True


def verify_vae():
    print("\n" + "=" * 60)
    print("  PHASE 3: ConvVAE")
    print("=" * 60)
    
    from core.models.vae import ConvVAE
    
    model = ConvVAE(in_channels=1, input_size=64, latent_dim=32, beta=1.0)
    x = torch.randn(8, 1, 64, 64)
    
    x_hat, mu, logvar = model(x)
    losses = model.compute_loss(x, x_hat, mu, logvar)
    
    print(f"  ✅ Forward: x_hat={x_hat.shape}, mu={mu.shape}, logvar={logvar.shape}")
    print(f"  ✅ Total Loss: {losses['total_loss'].item():.4f}")
    print(f"  ✅ Recon Loss: {losses['recon_loss'].item():.4f}")
    print(f"  ✅ KL Loss:    {losses['kl_loss'].item():.4f}")
    print(f"  ✅ Beta:       {losses['effective_beta'].item():.4f}")
    
    samples = model.sample(4)
    print(f"  ✅ Sampling: {samples.shape}, range=[{samples.min():.3f}, {samples.max():.3f}]")
    
    interp = model.interpolate(x[:1], x[1:2], num_steps=5)
    print(f"  ✅ Interpolation: {interp.shape}")
    
    return True


def verify_transformer_ae():
    print("\n" + "=" * 60)
    print("  PHASE 3: TransformerAutoencoder")
    print("=" * 60)
    
    from core.models.transformer_ae import TransformerAutoencoder
    
    model = TransformerAutoencoder(
        in_channels=1, input_size=64, latent_dim=32,
        patch_size=8, d_model=32, nhead=4,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=64
    )
    x = torch.randn(4, 1, 64, 64)
    
    t0 = time.perf_counter()
    out = model(x)
    lat = (time.perf_counter() - t0) * 1000
    
    z = model.encode(x)
    diag = model.diagnostics(x)
    
    print(f"  ✅ Forward: {x.shape} → {out.shape} in {lat:.2f} ms")
    print(f"  ✅ Latent: {z.shape}")
    print(f"  ✅ Parameters: {diag['total_parameters']:,}")
    print(f"  ✅ Memory: {diag['memory_mb']:.4f} MB")
    print(f"  ✅ Num Patches: {model.num_patches}")
    
    return True


def verify_registry():
    print("\n" + "=" * 60)
    print("  PHASE 3: Model Registry & Comparison")
    print("=" * 60)
    
    from core.models.registry import ModelRegistry
    
    archs = ModelRegistry.list_architectures()
    print(f"  ✅ Registered: {archs}")
    
    for arch in archs:
        if arch == "transformer_ae":
            m = ModelRegistry.create(arch, in_channels=1, input_size=64, latent_dim=32, patch_size=8, d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64)
        else:
            m = ModelRegistry.create(arch, in_channels=1, input_size=64, latent_dim=32)
        x = torch.randn(2, 1, 64, 64)
        out = m(x) if arch != "vae" else m(x)[0]
        print(f"  ✅ {arch}: output={out.shape}")
    
    print("\n  Running architecture comparison benchmark...")
    ModelRegistry.print_comparison(in_channels=1, input_size=64, latent_dim=32)
    
    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 3: Running Unit Tests")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_models.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 3 FULL VERIFICATION")
    print("🔬" * 30 + "\n")
    
    results = []
    results.append(("EdgeStreamingAE", verify_edge_ae()))
    results.append(("ConvVAE", verify_vae()))
    results.append(("TransformerAutoencoder", verify_transformer_ae()))
    results.append(("Model Registry", verify_registry()))
    results.append(("Unit Tests (pytest)", run_pytest()))
    
    print("\n" + "=" * 60)
    print("  PHASE 3 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Result: {total_pass}/{total} checks passed")
    
    if total_pass == total:
        print("\n  🎉 PHASE 3 COMPLETE — Ready for Phase 4!")
        print("  Next: Advanced Model Zoo Part 2 (U-Net AE, Attention Modules)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding to Phase 4.\n")
    
    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)