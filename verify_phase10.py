"""
Phase 10 Verification Script
"""

import sys
import time
import torch
import numpy as np
import tempfile
from pathlib import Path


def verify_synthetic():
    print("=" * 60)
    print("  PHASE 10: Synthetic Streams")
    print("=" * 60)

    from core.streams import (
        SyntheticStream, SinusoidalStream, MixedSignalStream, SpikeStream
    )

    configs = [
        ('Gaussian', SyntheticStream(batch_size=4, shape=(1, 32, 32))),
        ('Sinusoidal', SinusoidalStream(batch_size=4, channels=2, length=64)),
        ('Mixed', MixedSignalStream(batch_size=4, channels=1, length=64)),
        ('Spike', SpikeStream(batch_size=4, shape=(1, 32, 32), spike_density=0.05)),
    ]

    for name, stream in configs:
        batch = next(iter(stream))
        print(f"  ✅ {name:<12}: {batch.shape} | "
              f"range=[{batch.min():.3f}, {batch.max():.3f}] | "
              f"mean={batch.mean():.3f}")

    return True


def verify_loaders():
    print("\n" + "=" * 60)
    print("  PHASE 10: File Loaders")
    print("=" * 60)

    from core.streams import (
        load_npy_to_tensor, load_pt_to_tensor, load_csv_to_tensor
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # NPY (3D image stack)
        arr = np.random.rand(10, 24, 24).astype(np.float32)
        np.save(tmp_path / "test.npy", arr)
        t1 = load_npy_to_tensor(str(tmp_path / "test.npy"), target_size=24)
        print(f"  ✅ NPY (10, 24, 24) → {t1.shape}")

        # PT
        tensor = torch.rand(6, 1, 16, 16)
        torch.save(tensor, tmp_path / "test.pt")
        t2 = load_pt_to_tensor(str(tmp_path / "test.pt"), target_size=16)
        print(f"  ✅ PT (6, 1, 16, 16) → {t2.shape}")

        # CSV
        (tmp_path / "test.csv").write_text(
            "time,val\n" + "\n".join(f"{i},{np.sin(i*0.1):.4f}" for i in range(200))
        )
        t3 = load_csv_to_tensor(
            str(tmp_path / "test.csv"), value_columns=["val"]
        )
        print(f"  ✅ CSV (200 rows) → {t3.shape}")

    return True


def verify_signals():
    print("\n" + "=" * 60)
    print("  PHASE 10: Signal Streams")
    print("=" * 60)

    from core.streams import AudioStream, TimeSeriesStream

    # Audio (waveform + spectrogram)
    for mode in ['waveform', 'spectrogram']:
        s = AudioStream(batch_size=4, mode=mode, spectrogram_size=32)
        t0 = time.perf_counter()
        batch = next(iter(s))
        lat = (time.perf_counter() - t0) * 1000
        print(f"  ✅ Audio [{mode:<12}]: {batch.shape} in {lat:.1f} ms")

    # Time-series
    s = TimeSeriesStream(window_length=64, channels=2, batch_size=4)
    batch = next(iter(s))
    print(f"  ✅ TimeSeries: {batch.shape}")

    return True


def verify_augment():
    print("\n" + "=" * 60)
    print("  PHASE 10: Augmentation Pipeline")
    print("=" * 60)

    from core.streams import AugmentationPipeline

    x_img = torch.rand(8, 1, 32, 32)
    x_sig = torch.randn(8, 2, 64)

    aug_img = AugmentationPipeline.image_preset(seed=1)
    aug_sig = AugmentationPipeline.signal_preset(seed=1)

    out_img = aug_img(x_img)
    out_sig = aug_sig(x_sig)

    print(f"  ✅ Image augment:  {x_img.shape} → {out_img.shape}")
    print(f"  ✅ Signal augment: {x_sig.shape} → {out_sig.shape}")

    # Consistency check
    for _ in range(10):
        assert aug_img(x_img).shape == x_img.shape
        assert aug_sig(x_sig).shape == x_sig.shape

    print(f"  ✅ Consistency verified over 10 iterations")
    return True


def verify_factory():
    print("\n" + "=" * 60)
    print("  PHASE 10: Stream Factory")
    print("=" * 60)

    from core.streams import build_stream

    types = ['synthetic', 'sinusoidal', 'mixed', 'spike', 'audio', 'timeseries']

    for t in types:
        kwargs = {'batch_size': 2}
        if t in ('sinusoidal', 'mixed', 'timeseries'):
            kwargs.update({'channels': 1, 'length': 16} if t != 'timeseries' else {'channels': 1, 'window_length': 16})
        stream = build_stream(t, **kwargs)
        batch = next(iter(stream))
        print(f"  ✅ '{t}': {batch.shape}")

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 10: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_streams_p2.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 10 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Synthetic Streams", verify_synthetic()))
    results.append(("File Loaders", verify_loaders()))
    results.append(("Signal Streams", verify_signals()))
    results.append(("Augmentation", verify_augment()))
    results.append(("Stream Factory", verify_factory()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 10 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 10 COMPLETE — Ready for Phase 11!")
        print("  The heavy phases are done. Next: Distributed Worker Pool (short!)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)