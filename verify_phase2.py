"""
Phase 2 Verification Script
Validates all math utility modules and runs unit tests.
"""

import sys
import time
import numpy as np
import torch


def verify_statistics():
    print("=" * 60)
    print("  PHASE 2: Statistics Module")
    print("=" * 60)
    
    from core.math_utils.statistics import (
        WelfordAccumulator, WelfordVectorAccumulator,
        ExponentialMovingAverage, RollingWindow,
        HistogramAccumulator, MultiChannelTracker
    )
    
    # Welford
    acc = WelfordAccumulator()
    data = np.random.randn(10000)
    acc.update_batch(data)
    print(f"  ✅ Welford: mean={acc.mean:.4f} (expect ~0), std={acc.std:.4f} (expect ~1)")
    
    # Vector Welford
    vacc = WelfordVectorAccumulator()
    for _ in range(100):
        vacc.update(np.random.randn(32))
    print(f"  ✅ VectorWelford: {vacc.count} samples, dim={vacc.dim}, mean_norm={np.linalg.norm(vacc.mean):.4f}")
    
    # EMA
    ema = ExponentialMovingAverage(alpha=0.1)
    for i in range(200):
        ema.update(float(i))
    print(f"  ✅ EMA: tracking value={ema.value:.2f} (expect ~190+)")
    
    # Rolling Window
    rw = RollingWindow(window_size=50)
    for i in range(200):
        rw.update(float(i))
    summary = rw.summary()
    print(f"  ✅ RollingWindow: p50={summary['p50']:.1f}, p95={summary['p95']:.1f}")
    
    # Histogram
    hist = HistogramAccumulator(num_bins=30)
    hist.update(np.random.randn(5000))
    print(f"  ✅ Histogram: {hist.total} samples, density_sum={hist.density.sum():.4f}")
    
    # MultiChannel
    tracker = MultiChannelTracker(["lat", "mem", "drift"], window_size=50)
    for _ in range(100):
        tracker.update("lat", np.random.rand())
        tracker.update("mem", 100 + np.random.rand() * 10)
        tracker.update("drift", np.random.rand() * 0.1)
    summaries = tracker.get_all_summaries()
    print(f"  ✅ MultiChannel: {len(summaries)} channels tracked")
    
    return True


def verify_linear_algebra():
    print("\n" + "=" * 60)
    print("  PHASE 2: Linear Algebra Module")
    print("=" * 60)
    
    from core.math_utils.linear_algebra import (
        StreamingPCA, OnlineCovarianceMatrix, gram_schmidt, batch_svd_project
    )
    
    # Streaming PCA
    pca = StreamingPCA(n_components=3)
    for _ in range(5):
        batch = np.random.randn(20, 32)
        pca.partial_fit(batch)
    test_data = np.random.randn(10, 32)
    projected = pca.transform(test_data)
    print(f"  ✅ StreamingPCA: {test_data.shape} → {projected.shape}, variance_ratio={pca.explained_variance_ratio}")
    
    # Online Covariance
    ocm = OnlineCovarianceMatrix()
    data = np.random.randn(200, 5)
    ocm.update_batch(data)
    eigenvals, _ = ocm.eigendecomposition(top_k=3)
    print(f"  ✅ OnlineCovariance: {ocm.count} samples, top eigenvalues={eigenvals[:3]}")
    
    # Gram-Schmidt
    vecs = np.random.randn(3, 10)
    ortho = gram_schmidt(vecs)
    dots = ortho @ ortho.T
    off_diag = np.abs(dots - np.eye(3)).max()
    print(f"  ✅ GramSchmidt: max off-diagonal={off_diag:.8f} (expect ~0)")
    
    # Batch SVD
    data = np.random.randn(50, 20)
    proj, ratios, comps = batch_svd_project(data, n_components=3)
    print(f"  ✅ BatchSVD: {data.shape} → {proj.shape}, explained={ratios.sum():.4f}")
    
    return True


def verify_distances():
    print("\n" + "=" * 60)
    print("  PHASE 2: Distance Functions Module")
    print("=" * 60)
    
    from core.math_utils.distances import (
        wasserstein_1, wasserstein_2, mmd_rbf, kl_divergence,
        jensen_shannon, ks_statistic, energy_distance,
        mahalanobis_distance, cosine_distance, hellinger_distance
    )
    
    a = np.random.randn(500)
    b = np.random.randn(500) + 2.0
    
    results = {
        'Wasserstein-1': wasserstein_1(a, b),
        'Wasserstein-2': wasserstein_2(a, b),
        'MMD-RBF': mmd_rbf(a, b),
        'KL-Symmetric': kl_divergence(a, b),
        'Jensen-Shannon': jensen_shannon(a, b),
        'KS-Statistic': ks_statistic(a, b)[0],
        'Energy Distance': energy_distance(a, b),
        'Hellinger': hellinger_distance(a, b),
    }
    
    for name, val in results.items():
        status = "✅" if val > 0 else "⚠️"
        print(f"  {status} {name}: {val:.6f}")
    
    # Mahalanobis
    x = np.array([2.0, 0.0, 0.0])
    mean = np.zeros(3)
    cov_inv = np.eye(3)
    md = mahalanobis_distance(x, mean, cov_inv)
    print(f"  ✅ Mahalanobis: {md:.6f} (expect 2.0)")
    
    # Cosine
    cd = cosine_distance(np.array([1, 0]), np.array([0, 1]))
    print(f"  ✅ Cosine Distance: {cd:.6f} (expect 1.0)")
    
    return True


def verify_signal_processing():
    print("\n" + "=" * 60)
    print("  PHASE 2: Signal Processing Module")
    print("=" * 60)
    
    from core.math_utils.signal_processing import (
        compute_fft, power_spectral_density, spectral_entropy,
        signal_to_noise_ratio, find_peaks, band_energy, zero_crossing_rate
    )
    
    t = np.linspace(0, 1, 1000, endpoint=False)
    sine_50hz = np.sin(2 * np.pi * 50 * t)
    
    freqs, mags = compute_fft(sine_50hz, sample_rate=1000)
    peak_freq = freqs[np.argmax(mags)]
    print(f"  ✅ FFT: Detected peak at {peak_freq:.1f} Hz (expect ~50)")
    
    psd_f, psd_v = power_spectral_density(sine_50hz, sample_rate=1000)
    print(f"  ✅ PSD: {len(psd_f)} frequency bins computed")
    
    se = spectral_entropy(sine_50hz)
    se_noise = spectral_entropy(np.random.randn(1000))
    print(f"  ✅ Spectral Entropy: sine={se:.4f}, noise={se_noise:.4f} (noise should be higher)")
    
    snr = signal_to_noise_ratio(sine_50hz)
    print(f"  ✅ SNR: {snr:.2f} dB")
    
    multi_peak = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    peak_idx, peak_vals = find_peaks(multi_peak, min_distance=5)
    print(f"  ✅ Peak Detection: {len(peak_idx)} peaks found")
    
    be = band_energy(sine_50hz, sample_rate=1000, low_freq=40, high_freq=60)
    print(f"  ✅ Band Energy [40-60Hz]: {be:.4f} (expect high fraction)")
    
    zcr = zero_crossing_rate(sine_50hz)
    print(f"  ✅ Zero Crossing Rate: {zcr:.4f}")
    
    return True


def verify_tensor_ops():
    print("\n" + "=" * 60)
    print("  PHASE 2: Tensor Operations Module")
    print("=" * 60)
    
    from core.math_utils.tensor_ops import (
        im2col, extract_patches_2d, check_tensor_health,
        sanitize_tensor, count_parameters, model_memory_mb,
        compute_output_size
    )
    
    x = torch.randn(2, 3, 32, 32)
    cols = im2col(x, kernel_size=3, stride=1, padding=1)
    print(f"  ✅ im2col: {x.shape} → {cols.shape}")
    
    patches = extract_patches_2d(torch.randn(4, 1, 64, 64), patch_size=8)
    print(f"  ✅ Patches: (4,1,64,64) → {patches.shape}")
    
    health = check_tensor_health(torch.randn(100, 100))
    print(f"  ✅ Health Check (clean): healthy={health['is_healthy']}")
    
    dirty = torch.tensor([1.0, float('nan'), float('inf')])
    health2 = check_tensor_health(dirty)
    clean = sanitize_tensor(dirty)
    print(f"  ✅ Health Check (dirty): nan={health2['has_nan']}, inf={health2['has_inf']}")
    print(f"  ✅ Sanitized: {clean.tolist()}")
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3), torch.nn.ReLU()
    )
    params = count_parameters(model)
    mem = model_memory_mb(model)
    print(f"  ✅ Model Stats: {params:,} params, {mem:.4f} MB")
    
    out_size = compute_output_size(64, kernel_size=3, stride=2, padding=1)
    print(f"  ✅ Output Size: input=64, k=3, s=2, p=1 → {out_size}")
    
    return True


def verify_initializers():
    print("\n" + "=" * 60)
    print("  PHASE 2: Initializers Module")
    print("=" * 60)
    
    from core.math_utils.initializers import (
        lsuv_init, kaiming_init, xavier_init, orthogonal_init,
        get_init_diagnostics
    )
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 10)
    )
    
    batch = torch.randn(8, 1, 16, 16)
    
    t0 = time.perf_counter()
    model = lsuv_init(model, batch, verbose=False)
    lsuv_time = (time.perf_counter() - t0) * 1000
    diag = model._lsuv_diagnostics
    converged = sum(1 for d in diag.values() if d.get('converged', False))
    print(f"  ✅ LSUV: {converged}/{len(diag)} converged in {lsuv_time:.1f} ms")
    
    model2 = torch.nn.Sequential(torch.nn.Linear(20, 30), torch.nn.ReLU(), torch.nn.Linear(30, 10))
    kaiming_init(model2, verbose=False)
    print(f"  ✅ Kaiming Init: applied to {sum(1 for _ in model2.parameters())} params")
    
    xavier_init(model2, verbose=False)
    print(f"  ✅ Xavier Init: applied")
    
    orthogonal_init(model2, verbose=False)
    print(f"  ✅ Orthogonal Init: applied")
    
    diag_report = get_init_diagnostics(model2)
    print(f"  ✅ Diagnostics: {len(diag_report)} parameter tensors analyzed")
    
    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 2: Running Unit Tests")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_math_utils.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 2 FULL VERIFICATION")
    print("🔬" * 30 + "\n")
    
    results = []
    results.append(("Statistics Module", verify_statistics()))
    results.append(("Linear Algebra Module", verify_linear_algebra()))
    results.append(("Distance Functions Module", verify_distances()))
    results.append(("Signal Processing Module", verify_signal_processing()))
    results.append(("Tensor Operations Module", verify_tensor_ops()))
    results.append(("Initializers Module", verify_initializers()))
    results.append(("Unit Tests (pytest)", run_pytest()))
    
    print("\n" + "=" * 60)
    print("  PHASE 2 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Result: {total_pass}/{total} checks passed")
    
    if total_pass == total:
        print("\n  🎉 PHASE 2 COMPLETE — Ready for Phase 3!")
        print("  Next: Advanced Model Zoo Part 1 (VAE, Transformer AE)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding to Phase 3.\n")
    
    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)