"""
Phase 24 Verification Script
"""

import sys
import io
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def verify_pipeline_e2e():
    print("=" * 60)
    print("  PHASE 24: Full Pipeline E2E")
    print("=" * 60)

    import torch
    import numpy as np
    from core.models import EdgeStreamingAE
    from core.math_utils.initializers import lsuv_init
    from core.drift import DriftRadar
    from core.anomaly import AnomalyScorer
    from core.precision import AdaptivePrecisionEngineV2

    # Build model
    model = EdgeStreamingAE(
        in_channels=1, input_size=32, latent_dim=16, base_channels=4,
    )
    print(f"  ✅ Model built: {sum(p.numel() for p in model.parameters()):,} params")

    # LSUV calibrate
    calib = torch.randn(8, 1, 32, 32)
    model = lsuv_init(model, calib, max_iterations=3, verbose=False)
    print(f"  ✅ LSUV calibrated: {model.is_calibrated}")

    # Drift radar
    radar = DriftRadar(methods=['wasserstein'], evaluation_interval=1)
    radar.capture_reference(model, calib)
    reports = radar.evaluate_from_tensor(model, torch.randn(8, 1, 32, 32))
    print(f"  ✅ Drift radar: {len(reports)} layers monitored")

    # Precision engine
    engine = AdaptivePrecisionEngineV2(
        model, target_latency_ms=3.0, window_size=5, enable_int8=False,
    )
    x = torch.randn(2, 1, 32, 32)
    for _ in range(5):
        engine.step(lambda: model(x))
    print(f"  ✅ Precision engine: state={engine.current_state}, ticks={engine.tick}")

    # Anomaly scorer
    scorer = AnomalyScorer(n_estimators=15)
    scorer.fit(np.random.randn(60, 16))
    with torch.no_grad():
        z = model.encode(x)
        rec = model.decode(z)
    result = scorer.score(z, rec, x)
    print(f"  ✅ Anomaly scorer: mean score={result['score'].mean().item():.4f}")

    # Return types are what downstream expects
    assert result['score'].shape == (2,)
    assert radar.is_calibrated

    return True


def verify_config_matrix():
    print("\n" + "=" * 60)
    print("  PHASE 24: Config Matrix (5 profiles × model build)")
    print("=" * 60)

    import torch
    from config.loader import load_config
    from core.models import ModelRegistry

    profiles = ["default", "high_performance", "edge", "ci", "debug"]
    for name in profiles:
        cfg = load_config(profile=name)
        model = ModelRegistry.create(
            cfg.model.architecture,
            in_channels=cfg.model.in_channels,
            input_size=cfg.model.input_size,
            latent_dim=cfg.model.latent_dim,
        )
        x = torch.randn(2, cfg.model.in_channels,
                        cfg.model.input_size, cfg.model.input_size)
        with torch.no_grad():
            z = model.encode(x)
        print(f"  ✅ {name:<18} → input={cfg.model.input_size} "
              f"latent={cfg.model.latent_dim} batch={cfg.stream.batch_size} "
              f"z={tuple(z.shape)}")

    return True


def verify_dashboard_smoke():
    print("\n" + "=" * 60)
    print("  PHASE 24: Dashboard Panel Smoke (7 panels)")
    print("=" * 60)

    from tests.integration.helpers import MockST
    from dashboard.theme import get_palette
    from dashboard.panels import (
        LatentExplorerState, render_latent_explorer,
        DriftHeatmapState, render_drift_heatmap,
        TelemetryState, render_telemetry,
        AnomalyConsoleState, render_anomaly_console,
        ModelComparisonState, render_model_comparison,
        OscilloscopeState, render_oscilloscope,
        TopologyState, seed_default_topology, render_topology,
    )

    palette = get_palette("cyber_dark")

    # Latent
    st = MockST()
    state = LatentExplorerState(latent_dim=16, buffer_size=50)
    state.seed_demo(30)
    render_latent_explorer(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Latent Explorer:      {len(st.plotly_calls)} plot(s)")

    # Drift
    st = MockST()
    state = DriftHeatmapState()
    state.seed_demo(40)
    render_drift_heatmap(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Drift Heatmap:       {len(st.plotly_calls)} plot(s)")

    # Telemetry
    st = MockST()
    state = TelemetryState()
    state.seed_demo(50)
    render_telemetry(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Hardware Telemetry:  {len(st.plotly_calls)} plot(s)")

    # Anomaly
    st = MockST()
    state = AnomalyConsoleState()
    state.seed_demo(50)
    render_anomaly_console(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Anomaly Console:     {len(st.plotly_calls)} plot(s)")

    # Comparison
    st = MockST()
    state = ModelComparisonState()
    state.seed_demo()
    render_model_comparison(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Model Comparison:    {len(st.plotly_calls)} plot(s)")

    # Oscilloscope
    st = MockST()
    state = OscilloscopeState(samples_per_view=256)
    state.seed_demo(15)
    render_oscilloscope(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Oscilloscope:        {len(st.plotly_calls)} plot(s)")

    # Topology
    st = MockST()
    state = TopologyState()
    seed_default_topology(state)
    render_topology(st, palette, state, key_prefix="v", show_demo_button=False)
    print(f"  ✅ Topology Visualizer: {len(st.plotly_calls)} plot(s)")

    return True


def verify_cli_e2e():
    print("\n" + "=" * 60)
    print("  PHASE 24: CLI End-to-End")
    print("=" * 60)

    project_root = Path.cwd()

    # Info
    r = subprocess.run(
        [sys.executable, "-m", "cli", "info", "--profile", "ci"],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(project_root), timeout=60,
    )
    assert r.returncode == 0
    print(f"  ✅ info --profile ci:     exit={r.returncode}")

    # Validate all
    r = subprocess.run(
        [sys.executable, "-m", "cli", "config", "validate"],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(project_root), timeout=60,
    )
    assert r.returncode == 0
    assert "5 valid" in r.stdout
    print(f"  ✅ config validate:       5/5 valid")

    # Config dump JSON
    r = subprocess.run(
        [sys.executable, "-m", "cli", "config", "dump",
         "--profile", "edge", "--format", "json"],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(project_root), timeout=60,
    )
    assert r.returncode == 0
    import json
    data = json.loads(r.stdout)
    assert data['model']['input_size'] == 32
    print(f"  ✅ config dump --profile edge: input_size=32")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 24: File Checklist")
    print("=" * 60)

    files = [
        "tests/integration/__init__.py",
        "tests/integration/conftest.py",
        "tests/integration/helpers.py",
        "tests/integration/test_pipeline_e2e.py",
        "tests/integration/test_config_driven.py",
        "tests/integration/test_dashboard_smoke.py",
        "tests/integration/test_cli_e2e.py",
    ]
    all_ok = True
    for f in files:
        exists = Path(f).is_file()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {f}")
    return all_ok


def run_integration_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 24: Running Integration Tests")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration/",
         "-v", "--tb=short", "-m", "not slow"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=600,
    )
    print(r.stdout)
    if r.returncode != 0:
        # Print only the short summary for readability
        lines = r.stdout.split('\n')
        for line in lines[-40:]:
            if 'FAIL' in line or 'passed' in line or 'failed' in line:
                print(line)
    return r.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 24 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Full Pipeline E2E", verify_pipeline_e2e()))
    results.append(("Config Matrix", verify_config_matrix()))
    results.append(("Dashboard Panels", verify_dashboard_smoke()))
    results.append(("CLI End-to-End", verify_cli_e2e()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Integration Pytest", run_integration_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 24 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 24 COMPLETE — Ready for Phase 25 (final)!")
        print("\n  ┌───────────────────────────────────────────────────────┐")
        print("  │  Run integration tests manually:                      │")
        print("  │                                                       │")
        print("  │     python -m pytest tests/integration/ -v            │")
        print("  │     python -m pytest tests/ -v                        │")
        print("  │     python -m cli verify all                          │")
        print("  │                                                       │")
        print("  │  Next: Phase 25 — Final Assembly & Documentation      │")
        print("  └───────────────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)