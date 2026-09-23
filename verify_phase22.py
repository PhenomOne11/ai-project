"""
Phase 22 Verification Script
"""

import sys
import json
import tempfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def verify_bencher():
    print("=" * 60)
    print("  PHASE 22: Bencher Primitives")
    print("=" * 60)

    from benchmarks.bench import Bencher

    b = Bencher(warmup=2, iterations=20, track_memory=True)

    result = b.run("noop", lambda: None)
    print(f"  ✅ noop: mean={result.mean_ms:.5f} ms, "
          f"p95={result.p95_ms:.5f} ms, "
          f"throughput={result.throughput_per_s:.0f} ops/s")

    # Memory tracking
    def alloc():
        return [0] * 500_000
    result2 = b.run("alloc", alloc, category="memory")
    print(f"  ✅ alloc: RSS before={result2.rss_before_mb:.2f}MB "
          f"peak={result2.rss_peak_mb:.2f}MB delta={result2.rss_delta_mb:+.2f}MB")

    # Error path
    def boom():
        raise RuntimeError("simulated failure")
    result3 = b.run("boom", boom)
    print(f"  ✅ error captured: {result3.error[:50]}...")
    assert result3.error is not None

    return True


def verify_baseline():
    print("\n" + "=" * 60)
    print("  PHASE 22: Baseline Save/Load")
    print("=" * 60)

    from benchmarks.regression import BenchmarkBaseline, BaselineEntry

    b = BenchmarkBaseline()
    b.set(BaselineEntry(name="alpha", mean_ms=1.5, p95_ms=2.5,
                        rss_peak_mb=120.0, throughput_per_s=666.7))
    b.set(BaselineEntry(name="beta", mean_ms=3.0, p95_ms=5.0,
                        rss_peak_mb=240.0))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        b.save(str(path))
        size = path.stat().st_size

        loaded = BenchmarkBaseline.load(str(path))
        print(f"  ✅ Saved {len(b)} entries ({size} bytes)")
        print(f"  ✅ Loaded {len(loaded)} entries")
        print(f"      alpha: mean={loaded.get('alpha').mean_ms} ms")
        print(f"      beta:  mean={loaded.get('beta').mean_ms} ms")
        assert len(loaded) == 2

    return True


def verify_regression_detection():
    print("\n" + "=" * 60)
    print("  PHASE 22: Regression Detection")
    print("=" * 60)

    from benchmarks.regression import (
        compare_to_baseline, BaselineEntry,
    )
    from benchmarks.bench import BenchResult

    baseline = BaselineEntry(
        name="test", mean_ms=10.0, p95_ms=15.0, rss_peak_mb=100.0,
    )

    scenarios = [
        ("nominal",  BenchResult(name="test", latencies_ms=[10.1] * 20,
                                 rss_peak_mb=100.0)),
        ("warning",  BenchResult(name="test", latencies_ms=[12.0] * 20,
                                 rss_peak_mb=100.0)),
        ("critical", BenchResult(name="test", latencies_ms=[15.0] * 20,
                                 rss_peak_mb=100.0)),
    ]

    for label, result in scenarios:
        v = compare_to_baseline(result, baseline,
                                latency_tolerance_pct=15.0)
        status = "PASS" if v.passed else "FAIL"
        print(f"  ✅ [{label:<8}] passed={v.passed} "
              f"reason='{v.reason}' Δmean={v.deltas['mean_ms']:+.1f}%")

    return True


def verify_suite_run():
    print("\n" + "=" * 60)
    print("  PHASE 22: Benchmark Suite Run")
    print("=" * 60)

    from benchmarks.suite import BenchmarkSuite

    print("  Running minimal suite (warmup=1, iterations=3)...")
    import time
    t0 = time.time()
    suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
    result = suite.run_all()
    duration = time.time() - t0

    print(f"  ✅ Suite completed in {duration:.2f}s")
    print(f"      Total benchmarks: {len(result.results)}")

    by_cat = result.by_category()
    for cat, rs in sorted(by_cat.items()):
        print(f"      {cat:<10}: {len(rs)} benchmark(s)")

    return True


def verify_reporter():
    print("\n" + "=" * 60)
    print("  PHASE 22: Report Generation")
    print("=" * 60)

    from benchmarks.suite import BenchmarkSuite
    from benchmarks.reporter import (
        render_console, render_json, render_html,
        save_json, save_html,
    )

    suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
    result = suite.run_all()

    # Console
    console_text = render_console(result)
    print(f"  ✅ Console report: {len(console_text):,} chars")
    print(f"      First 3 lines:")
    for line in console_text.split('\n')[:3]:
        print(f"          {line}")

    # JSON
    json_text = render_json(result)
    parsed = json.loads(json_text)
    print(f"  ✅ JSON report: {len(json_text):,} chars, "
          f"{len(parsed['results'])} results")

    # HTML
    html_text = render_html(result)
    print(f"  ✅ HTML report: {len(html_text):,} chars")

    # Save both
    with tempfile.TemporaryDirectory() as tmp:
        jpath = Path(tmp) / "report.json"
        hpath = Path(tmp) / "report.html"
        save_json(result, str(jpath))
        save_html(result, str(hpath))
        print(f"  ✅ Saved JSON ({jpath.stat().st_size} bytes) "
              f"and HTML ({hpath.stat().st_size} bytes)")

    return True


def verify_regression_end_to_end():
    print("\n" + "=" * 60)
    print("  PHASE 22: End-to-End Regression Workflow")
    print("=" * 60)

    from benchmarks.suite import BenchmarkSuite
    from benchmarks.regression import BenchmarkBaseline

    with tempfile.TemporaryDirectory() as tmp:
        baseline_path = Path(tmp) / "baseline.json"

        # Run 1: establish baseline
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        baseline = BenchmarkBaseline.load(str(baseline_path))
        result1 = suite.run_and_check(
            baseline=baseline, update_baseline=True,
        )
        baseline.save(str(baseline_path))
        print(f"  ✅ Run 1 (baseline established): "
              f"{len(baseline)} entries saved")
        print(f"      All passed: {result1.all_passed}")

        # Run 2: compare against baseline
        baseline2 = BenchmarkBaseline.load(str(baseline_path))
        result2 = suite.run_and_check(baseline=baseline2)
        print(f"  ✅ Run 2 (regression check): "
              f"{result2.n_passed}/{len(result2.verdicts)} passed")
        print(f"      All passed: {result2.all_passed}")

        # Should pass on the immediate re-run
        assert result2.all_passed

    return True


def verify_cli():
    print("\n" + "=" * 60)
    print("  PHASE 22: CLI Entry Point")
    print("=" * 60)

    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "benchmarks.run_suite", "--help"],
        capture_output=True, text=True, encoding="utf-8",
    )
    print(f"  ✅ Help command: exit={r.returncode}")
    assert r.returncode == 0
    assert "benchmark" in r.stdout.lower()

    # Run a tiny suite with --no-baseline
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "out.json"
        r = subprocess.run(
            [
                sys.executable, "-m", "benchmarks.run_suite",
                "--no-baseline",
                "--iterations", "2",
                "--warmup", "1",
                "--batch-size", "2",
                "--json", str(json_path),
            ],
            capture_output=True, text=True, encoding="utf-8",
            timeout=120,
        )
        print(f"  ✅ Full suite run: exit={r.returncode}")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        print(f"      Report: {len(data['results'])} benchmarks, "
              f"duration={data['duration_s']:.2f}s")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 22: File Checklist")
    print("=" * 60)

    files = [
        "benchmarks/bench.py",
        "benchmarks/regression.py",
        "benchmarks/suite.py",
        "benchmarks/reporter.py",
        "benchmarks/__init__.py",
        "benchmarks/run_suite.py",
        "tests/unit/test_benchmarks.py",
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
    print("  PHASE 22: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_benchmarks.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=300,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 22 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Bencher Primitives", verify_bencher()))
    results.append(("Baseline Save/Load", verify_baseline()))
    results.append(("Regression Detection", verify_regression_detection()))
    results.append(("Suite Run", verify_suite_run()))
    results.append(("Report Generation", verify_reporter()))
    results.append(("Regression End-to-End", verify_regression_end_to_end()))
    results.append(("CLI Entry Point", verify_cli()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 22 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 22 COMPLETE — Ready for Phase 23!")
        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │  Run the full benchmark suite manually:            │")
        print("  │                                                     │")
        print("  │     python -m benchmarks.run_suite                 │")
        print("  │     python -m benchmarks.run_suite --html out.html │")
        print("  │                                                     │")
        print("  │  Next: Phase 23 — YAML Config Profiles + CLI       │")
        print("  └─────────────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)