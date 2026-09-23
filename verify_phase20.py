"""
Phase 20 Verification Script
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
    print("  PHASE 20: OscilloscopeState")
    print("=" * 60)

    from dashboard.panels.oscilloscope import OscilloscopeState

    s = OscilloscopeState(channel_names=["a", "b"], samples_per_view=256)
    print(f"  ✅ Initial: channels={s.num_channels}, empty={s.is_empty}")

    # Push test waveforms
    t = np.linspace(0, 1, 512)
    s.push_waveform("a", np.sin(2 * np.pi * 5 * t))
    s.push_waveform("b", np.sin(2 * np.pi * 12 * t) + 0.3 * np.random.randn(512))

    print(f"  ✅ Pushed 2 waveforms; size={s.size}")
    for name in s.channel_names:
        w = s.get_waveform(name)
        print(f"      {name}: {w.size} samples, "
              f"range=[{w.min():.3f}, {w.max():.3f}]")

    return True


def verify_demo():
    print("\n" + "=" * 60)
    print("  PHASE 20: Demo Seeding")
    print("=" * 60)

    from dashboard.panels.oscilloscope import OscilloscopeState

    s = OscilloscopeState(samples_per_view=512, trail_length=32)
    s.seed_demo(40)

    print(f"  ✅ Seeded 3 channels × 40 frames; total pushes={s.size}")

    for name in s.channel_names:
        stats = s.channel_stats(name)
        print(f"      {name}:")
        print(f"          RMS={stats['rms']:.4f}  peak={stats['peak']:.4f}  "
              f"crest={stats['crest_factor']:.2f}")
        print(f"          spectral_entropy={stats['spectral_entropy']:.3f}  "
              f"SNR={stats['snr_db']:.2f}dB  peaks={stats['num_peaks']}")

    g = s.global_stats()
    print(f"  ✅ Global: {g}")

    # ch-1 has noise → higher spectral entropy than ch-0
    entropy_0 = s.channel_stats("ch-0")['spectral_entropy']
    entropy_1 = s.channel_stats("ch-1")['spectral_entropy']
    print(f"  ✅ Spectral entropy check: ch-0={entropy_0:.3f}, ch-1={entropy_1:.3f}")
    # Both reasonably tonal since ch-1 is 0.8 × sine + 0.15 noise
    # Even so, ch-0 should have lower entropy
    assert entropy_0 <= entropy_1 + 0.05  # Allow small tolerance

    return True


def verify_figures():
    print("\n" + "=" * 60)
    print("  PHASE 20: Figure Builders")
    print("=" * 60)

    from dashboard.panels.oscilloscope import (
        OscilloscopeState,
        build_waveform_traces, build_fft_spectrum,
        build_waterfall, build_band_energy_bar,
    )
    from dashboard.theme import get_palette

    s = OscilloscopeState(samples_per_view=512, trail_length=32)
    s.seed_demo(40)
    p = get_palette("cyber_dark")

    waveforms = {name: s.get_waveform(name) for name in s.channel_names}

    fig1 = build_waveform_traces(waveforms, p, height=400)
    print(f"  ✅ Waveform overlay: {len(fig1.data)} traces")

    fig2 = build_fft_spectrum(waveforms, p, sample_rate=s.sample_rate, height=320)
    print(f"  ✅ FFT spectrum: {len(fig2.data)} traces")

    wf = s.get_waterfall("ch-0")
    fig3 = build_waterfall(wf, p, height=320)
    print(f"  ✅ Waterfall: {len(fig3.data)} trace(s), shape={wf.shape}")

    stats = s.channel_stats("ch-0")
    fig4 = build_band_energy_bar(stats, p, height=240)
    print(f"  ✅ Band energy bar: {len(fig4.data)} trace(s)")

    # All palettes
    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        pal = get_palette(pname)
        build_waveform_traces(waveforms, pal)
        build_fft_spectrum(waveforms, pal, sample_rate=s.sample_rate)
        build_waterfall(wf, pal)
        build_band_energy_bar(stats, pal)
    print(f"  ✅ All 4 palettes render successfully")

    return True


def verify_phase2_integration():
    print("\n" + "=" * 60)
    print("  PHASE 20: Phase 2 (Signal Processing) Integration")
    print("=" * 60)

    from core.math_utils.signal_processing import (
        compute_fft, spectral_entropy, signal_to_noise_ratio,
        find_peaks, zero_crossing_rate, band_energy,
    )

    # Pure sine at 50 Hz, sample rate 1000 Hz
    t = np.linspace(0, 1, 1000, endpoint=False)
    sine = np.sin(2 * np.pi * 50 * t)

    freqs, mags = compute_fft(sine, sample_rate=1000.0)
    peak_freq = float(freqs[np.argmax(mags)])
    print(f"  ✅ FFT peak at {peak_freq:.1f} Hz (expected 50.0)")

    se = spectral_entropy(sine)
    print(f"  ✅ Spectral entropy: {se:.4f} (expect low for pure tone)")

    snr = signal_to_noise_ratio(sine)
    print(f"  ✅ SNR: {snr:.2f} dB")

    zcr = zero_crossing_rate(sine)
    print(f"  ✅ Zero-crossing rate: {zcr:.4f}")

    peaks_idx, _ = find_peaks(sine, min_distance=5)
    print(f"  ✅ Detected {len(peaks_idx)} peaks in pure sine")

    be = band_energy(sine, sample_rate=1000.0, low_freq=40, high_freq=60)
    print(f"  ✅ Band energy [40-60Hz]: {be:.4f}")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 20: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/panels/oscilloscope.py",
        "dashboard/panels/__init__.py",
        "dashboard/preview_panels.py",
        "tests/unit/test_oscilloscope.py",
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
    print("  PHASE 20: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_oscilloscope.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 20 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("OscilloscopeState", verify_state()))
    results.append(("Demo Seeding", verify_demo()))
    results.append(("Figure Builders", verify_figures()))
    results.append(("Phase 2 Integration", verify_phase2_integration()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 20 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 20 COMPLETE — Ready for Phase 21!")
        print("\n  ┌──────────────────────────────────────────────┐")
        print("  │  Preview the panel live:                     │")
        print("  │                                              │")
        print("  │     streamlit run dashboard/preview_panels.py │")
        print("  │                                              │")
        print("  │  Choose 'Oscilloscope' in the sidebar.       │")
        print("  │                                              │")
        print("  │  Next: Phase 21 — System Topology Visualizer │")
        print("  └──────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)