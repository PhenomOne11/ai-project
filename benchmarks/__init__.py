"""
NeuralStream-Core Benchmark Package
"""

from benchmarks.bench import Bencher, BenchResult, latency_summary
from benchmarks.regression import (
    BaselineEntry, BenchmarkBaseline, RegressionVerdict,
    compare_to_baseline,
)
from benchmarks.suite import BenchmarkSuite, SuiteResult
from benchmarks.reporter import (
    render_console, render_json, save_json,
    render_html, save_html,
)

__all__ = [
    'Bencher', 'BenchResult', 'latency_summary',
    'BaselineEntry', 'BenchmarkBaseline', 'RegressionVerdict',
    'compare_to_baseline',
    'BenchmarkSuite', 'SuiteResult',
    'render_console', 'render_json', 'save_json',
    'render_html', 'save_html',
]