"""
Unit Tests for the benchmarks package
"""

import json
import time
import tempfile
import torch
import numpy as np
import pytest
from pathlib import Path


class TestBencher:
    def test_basic_run(self):
        from benchmarks.bench import Bencher
        b = Bencher(warmup=1, iterations=5, track_memory=False)
        result = b.run("noop", lambda: None)
        assert result.iterations == 5
        assert len(result.latencies_ms) == 5
        assert result.mean_ms >= 0

    def test_stats(self):
        from benchmarks.bench import Bencher
        b = Bencher(warmup=0, iterations=20, track_memory=False)
        result = b.run("sleep", lambda: time.sleep(0.001))
        assert result.mean_ms >= 0.5
        assert result.p95_ms >= result.p50_ms
        assert result.p99_ms >= result.p95_ms
        assert result.throughput_per_s > 0

    def test_memory_tracking(self):
        from benchmarks.bench import Bencher
        b = Bencher(warmup=0, iterations=3, track_memory=True)
        result = b.run("alloc", lambda: [0] * 100_000)
        assert result.rss_peak_mb > 0
        assert result.rss_before_mb > 0

    def test_error_handling(self):
        from benchmarks.bench import Bencher
        b = Bencher(warmup=0, iterations=3, track_memory=False)

        def boom():
            raise ValueError("intentional")

        result = b.run("boom", boom)
        assert result.error is not None
        assert "intentional" in result.error

    def test_to_dict(self):
        from benchmarks.bench import Bencher
        b = Bencher(warmup=0, iterations=3, track_memory=False)
        result = b.run("noop", lambda: None)
        d = result.to_dict()
        assert d['name'] == 'noop'
        assert 'mean_ms' in d
        assert 'p95_ms' in d

    def test_latency_summary_helper(self):
        from benchmarks.bench import latency_summary
        s = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s['mean'] == 3.0
        assert s['min'] == 1.0
        assert s['max'] == 5.0
        assert s['n'] == 5


class TestBaseline:
    def test_baseline_save_load(self):
        from benchmarks.regression import BenchmarkBaseline, BaselineEntry
        b = BenchmarkBaseline()
        b.set(BaselineEntry(name="x", mean_ms=1.5, p95_ms=2.5,
                            rss_peak_mb=100.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            b.save(str(path))
            loaded = BenchmarkBaseline.load(str(path))
            assert len(loaded) == 1
            e = loaded.get("x")
            assert e is not None
            assert abs(e.mean_ms - 1.5) < 1e-6
            assert abs(e.p95_ms - 2.5) < 1e-6

    def test_load_missing_returns_empty(self):
        from benchmarks.regression import BenchmarkBaseline
        b = BenchmarkBaseline.load("/nonexistent/path/x.json")
        assert len(b) == 0

    def test_update_from_result(self):
        from benchmarks.regression import BenchmarkBaseline
        from benchmarks.bench import BenchResult
        b = BenchmarkBaseline()
        # mean_ms / p95_ms are computed properties, not constructor fields.
        # Set the source data (latencies_ms) and let them be derived.
        result = BenchResult(name="x", latencies_ms=[2.0] * 10,
                             rss_peak_mb=100.0)
        b.update_from_result(result)
        entry = b.get("x")
        assert entry is not None
        assert abs(entry.mean_ms - 2.0) < 1e-6
        assert abs(entry.p95_ms - 2.0) < 1e-6


class TestRegressionCompare:
    def test_no_baseline_passes(self):
        from benchmarks.regression import compare_to_baseline
        from benchmarks.bench import BenchResult
        result = BenchResult(name="x", latencies_ms=[1.0] * 10)
        v = compare_to_baseline(result, None)
        assert v.passed
        assert v.reason == "no baseline yet"

    def test_nominal_passes(self):
        from benchmarks.regression import (
            compare_to_baseline, BaselineEntry,
        )
        from benchmarks.bench import BenchResult
        result = BenchResult(
            name="x", latencies_ms=[1.0] * 20, rss_peak_mb=100.0,
        )
        baseline = BaselineEntry(
            name="x", mean_ms=1.0, p95_ms=1.0, rss_peak_mb=100.0,
        )
        v = compare_to_baseline(result, baseline)
        assert v.passed
        assert v.reason == "nominal"

    def test_warning_regression(self):
        from benchmarks.regression import (
            compare_to_baseline, BaselineEntry,
        )
        from benchmarks.bench import BenchResult

        # Baseline 5ms → current 6ms = +20% (above 15% tolerance),
        # absolute delta 1ms > 0.5ms floor → warning
        result = BenchResult(name="x", latencies_ms=[6.0] * 20)
        baseline = BaselineEntry(name="x", mean_ms=5.0, p95_ms=5.0,
                                 rss_peak_mb=0.0)
        v = compare_to_baseline(result, baseline,
                                latency_tolerance_pct=15.0,
                                absolute_latency_floor_ms=0.5)
        assert v.passed
        assert v.reason == "warning-level regression"
        assert v.levels['mean_ms'] == "warning"

    def test_critical_fails(self):
        from benchmarks.regression import (
            compare_to_baseline, BaselineEntry,
        )
        from benchmarks.bench import BenchResult

        # Baseline 5ms → current 10ms = +100% (well above 2× tolerance of 30%),
        # absolute delta 5ms > 0.5ms floor → critical
        result = BenchResult(name="x", latencies_ms=[10.0] * 20)
        baseline = BaselineEntry(name="x", mean_ms=5.0, p95_ms=5.0,
                                 rss_peak_mb=0.0)
        v = compare_to_baseline(result, baseline,
                                latency_tolerance_pct=15.0,
                                absolute_latency_floor_ms=0.5)
        assert not v.passed
        assert v.reason == "critical regression detected"
        assert v.levels['mean_ms'] == "critical"

    def test_memory_regression(self):
        from benchmarks.regression import (
            compare_to_baseline, BaselineEntry,
        )
        from benchmarks.bench import BenchResult
        # Memory jump: 100 → 200 (100% increase); tolerance 20%
        result = BenchResult(name="x", latencies_ms=[1.0] * 20, rss_peak_mb=200.0)
        baseline = BaselineEntry(name="x", mean_ms=1.0, p95_ms=1.0, rss_peak_mb=100.0)
        v = compare_to_baseline(result, baseline, memory_tolerance_pct=20.0)
        assert not v.passed


class TestSuite:
    def test_suite_runs_all_groups(self):
        from benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite(warmup=1, iterations=5, batch_size=4)
        result = suite.run_all()
        assert len(result.results) >= 5
        assert not result.all_passed is False  # all_passed default True until verdicts run

    def test_suite_has_model_benchmarks(self):
        from benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        names = [r.name for r in result.results]
        assert any('model.edge_ae' in n for n in names)

    def test_suite_by_category(self):
        from benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        cats = result.by_category()
        assert 'model' in cats
        assert 'ipc' in cats
        assert 'drift' in cats

    def test_suite_find(self):
        from benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        r = result.find("model.edge_ae")
        assert r is not None
        assert r.mean_ms > 0

    def test_suite_to_dict(self):
        from benchmarks.suite import BenchmarkSuite
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        d = result.to_dict()
        assert 'results' in d
        assert 'suite_name' in d
        assert len(d['results']) == len(result.results)

    def test_run_and_check_fresh_baseline(self):
        from benchmarks.suite import BenchmarkSuite
        from benchmarks.regression import BenchmarkBaseline
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        baseline = BenchmarkBaseline()
        result = suite.run_and_check(baseline=baseline, update_baseline=True)
        # First run against empty baseline → should pass
        assert result.all_passed
        assert len(baseline) > 0


class TestReporter:
    def test_console_render(self):
        from benchmarks.suite import BenchmarkSuite
        from benchmarks.reporter import render_console
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        text = render_console(result)
        assert "NEURALSTREAM-CORE" in text
        assert "model" in text.lower()

    def test_json_render(self):
        from benchmarks.suite import BenchmarkSuite
        from benchmarks.reporter import render_json, save_json
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        js = render_json(result)
        parsed = json.loads(js)
        assert 'results' in parsed

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            save_json(result, str(path))
            assert path.exists()

    def test_html_render(self):
        from benchmarks.suite import BenchmarkSuite
        from benchmarks.reporter import render_html, save_html
        suite = BenchmarkSuite(warmup=1, iterations=3, batch_size=4)
        result = suite.run_all()
        html_str = render_html(result)
        assert "<html" in html_str
        assert "NeuralStream" in html_str
        assert "model.edge_ae" in html_str

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            save_html(result, str(path))
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "<html" in content


class TestCLI:
    def test_cli_help_runs(self):
        import subprocess
        import sys as _sys
        result = subprocess.run(
            [_sys.executable, "-m", "benchmarks.run_suite", "--help"],
            capture_output=True, text=True, encoding="utf-8"
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])