"""
Full NeuralStream-Core benchmark suite.

Runs a curated set of benchmarks covering every major subsystem:
    - Model forward pass (per architecture)
    - IPC ring buffer write/read
    - Drift computation (Wasserstein)
    - Anomaly scoring
    - Precision controller step
    - Metrics registry updates
    - Event bus publish/subscribe
    - Worker pool submission

Produces a `SuiteResult` with all BenchResults + regression verdicts.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from benchmarks.bench import Bencher, BenchResult
from benchmarks.regression import (
    BenchmarkBaseline, BaselineEntry, RegressionVerdict,
    compare_to_baseline,
)


# =============================================================================
# SUITE RESULT
# =============================================================================

@dataclass
class SuiteResult:
    """Aggregate result of running the full suite."""
    results: List[BenchResult] = field(default_factory=list)
    verdicts: List[RegressionVerdict] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    suite_name: str = "neuralstream"

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    @property
    def n_passed(self) -> int:
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for v in self.verdicts if not v.passed)

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0

    def by_category(self) -> Dict[str, List[BenchResult]]:
        out: Dict[str, List[BenchResult]] = {}
        for r in self.results:
            out.setdefault(r.category, []).append(r)
        return out

    def find(self, name: str) -> Optional[BenchResult]:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'suite_name': self.suite_name,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'duration_s': self.duration_s,
            'n_passed': self.n_passed,
            'n_failed': self.n_failed,
            'all_passed': self.all_passed,
            'results': [r.to_dict() for r in self.results],
            'verdicts': [
                {
                    'name': v.name,
                    'passed': v.passed,
                    'reason': v.reason,
                    'deltas': dict(v.deltas),
                    'levels': dict(v.levels),
                }
                for v in self.verdicts
            ],
        }


# =============================================================================
# SUITE
# =============================================================================

class BenchmarkSuite:
    """
    Curated suite of NeuralStream benchmarks.

    Usage:
        suite = BenchmarkSuite()
        result = suite.run_all()
        baseline = BenchmarkBaseline.load("benchmarks/baseline.json")
        for r in result.results:
            v = compare_to_baseline(r, baseline.get(r.name))
    """

    def __init__(
        self,
        warmup: int = 3,
        iterations: int = 30,
        batch_size: int = 16,
        input_size: int = 64,
        channels: int = 1,
        latent_dim: int = 32,
        device: str = "cpu",
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.batch_size = batch_size
        self.input_size = input_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.device = device

    # ---- Individual benchmark builders ----

    def _bench_model_forward(self) -> List[BenchResult]:
        """Latency per model architecture."""
        results: List[BenchResult] = []
        from core.models import (
            EdgeStreamingAE, ConvVAE, UNetAutoencoder,
        )

        x = torch.randn(
            self.batch_size, self.channels, self.input_size, self.input_size,
            device=self.device,
        )

        archs = [
            ("model.edge_ae",
             EdgeStreamingAE(
                 in_channels=self.channels,
                 input_size=self.input_size,
                 latent_dim=self.latent_dim,
             ).to(self.device).eval()),
            ("model.vae",
             ConvVAE(
                 in_channels=self.channels,
                 input_size=self.input_size,
                 latent_dim=self.latent_dim,
             ).to(self.device).eval()),
            ("model.unet_ae",
             UNetAutoencoder(
                 in_channels=self.channels,
                 input_size=self.input_size,
                 latent_dim=self.latent_dim,
                 base_channels=8, depth=3,
             ).to(self.device).eval()),
        ]

        bencher = Bencher(warmup=self.warmup, iterations=self.iterations,
                          track_memory=True, gc_between=True)

        for name, model in archs:
            def _run(m=model):
                with torch.no_grad():
                    out = m(x)
                return out

            result = bencher.run(
                name=name,
                fn=_run,
                category="model",
                extra={
                    'params': sum(p.numel() for p in model.parameters()),
                    'batch_size': self.batch_size,
                },
            )
            results.append(result)

        return results

    def _bench_ipc(self) -> List[BenchResult]:
        """IPC ring buffer write and read throughput."""
        from core.ipc import create_ring_buffer
        results = []

        rb = create_ring_buffer("bench_rb", shape=(32,),
                                dtype=np.float32, capacity=1000)
        try:
            write_data = np.random.randn(32).astype(np.float32)

            write_bench = Bencher(
                warmup=10, iterations=max(self.iterations, 200),
                track_memory=False,
            )
            write_result = write_bench.run(
                name="ipc.ring_buffer.write",
                fn=lambda: rb.write(write_data),
                category="ipc",
            )
            # Writer likely sub-microsecond
            results.append(write_result)

            # Fill some data first
            for _ in range(500):
                rb.write(write_data)

            read_bench = Bencher(
                warmup=10, iterations=max(self.iterations, 200),
                track_memory=False,
            )
            read_result = read_bench.run(
                name="ipc.ring_buffer.read_latest_10",
                fn=lambda: rb.read_latest(10),
                category="ipc",
            )
            results.append(read_result)
        finally:
            rb.unlink()
            rb.close()

        return results

    def _bench_drift(self) -> List[BenchResult]:
        """Drift computation throughput."""
        from core.drift import DriftDetector
        results = []

        ref = np.random.randn(500)
        live = np.random.randn(500) + 0.1

        for method in ['wasserstein', 'mmd', 'kl_divergence', 'hellinger']:
            detector = DriftDetector(methods=[method])
            bencher = Bencher(warmup=2, iterations=20, track_memory=False)
            result = bencher.run(
                name=f"drift.{method}",
                fn=lambda d=detector, r=ref, l=live: d.compute(r, l, "bench"),
                category="drift",
            )
            results.append(result)

        return results

    def _bench_anomaly(self) -> List[BenchResult]:
        """Anomaly scoring throughput."""
        from core.anomaly import AnomalyScorer
        results = []

        scorer = AnomalyScorer(n_estimators=50)
        scorer.fit(np.random.randn(200, 32))

        latent = torch.randn(self.batch_size, 32)
        inp = torch.randn(self.batch_size, 1, self.input_size, self.input_size)
        rec = inp + 0.05 * torch.randn_like(inp)

        bencher = Bencher(warmup=2, iterations=20, track_memory=False)
        result = bencher.run(
            name="anomaly.fused_score",
            fn=lambda: scorer.score(latent, rec, inp),
            category="anomaly",
            extra={'batch_size': self.batch_size},
        )
        results.append(result)

        return results

    def _bench_metrics(self) -> List[BenchResult]:
        """Metrics registry update throughput."""
        from core.metrics import MetricsRegistry
        results = []

        reg = MetricsRegistry(name="bench")
        reg.counter("c")
        reg.gauge("g")
        reg.histogram("h", window_size=1000)

        bencher = Bencher(warmup=10, iterations=500, track_memory=False)
        result = bencher.run(
            name="metrics.record_triple",
            fn=lambda: (
                reg.counter("c").inc(),
                reg.gauge("g").set(0.5),
                reg.histogram("h").observe(2.5),
            ),
            category="metrics",
        )
        results.append(result)

        return results

    def _bench_events(self) -> List[BenchResult]:
        """Event bus publish throughput."""
        from core.events import EventBus
        results = []

        bus = EventBus(history_size=500)
        # Add subscribers to exercise the dispatch path
        for i in range(5):
            bus.subscribe(f"bench.topic.{i % 3}",
                          lambda e: None)

        bencher = Bencher(warmup=10, iterations=500, track_memory=False)
        result = bencher.run(
            name="events.publish",
            fn=lambda: bus.emit("bench.topic.0", {"v": 1}),
            category="events",
        )
        results.append(result)

        return results

    # ---- Orchestrator ----

    def run_all(self) -> SuiteResult:
        """Execute every benchmark group and aggregate."""
        suite = SuiteResult(suite_name="neuralstream",
                            started_at=time.time())

        groups = [
            self._bench_model_forward,
            self._bench_ipc,
            self._bench_drift,
            self._bench_anomaly,
            self._bench_metrics,
            self._bench_events,
        ]

        for group in groups:
            try:
                results = group()
                suite.results.extend(results)
            except Exception as e:
                # Skip a group gracefully if it fails
                suite.results.append(BenchResult(
                    name=f"group.{group.__name__}",
                    category="error",
                    error=f"{type(e).__name__}: {e}",
                ))

        suite.finished_at = time.time()
        return suite

    def run_and_check(
        self,
        baseline: Optional[BenchmarkBaseline] = None,
        latency_tolerance_pct: float = 15.0,
        memory_tolerance_pct: float = 20.0,
        absolute_latency_floor_ms: float = 0.5,
        absolute_memory_floor_mb: float = 10.0,
        update_baseline: bool = False,
    ) -> SuiteResult:
        """
        Run all benchmarks and compare to a baseline (if provided).

        If `update_baseline=True`, the baseline is refreshed with new results.
        """
        suite = self.run_all()

        for r in suite.results:
            if r.error:
                suite.verdicts.append(RegressionVerdict(
                    name=r.name, passed=False,
                    reason=f"benchmark error: {r.error}",
                ))
                continue

            base_entry = baseline.get(r.name) if baseline else None
            verdict = compare_to_baseline(
                r, base_entry,
                latency_tolerance_pct=latency_tolerance_pct,
                memory_tolerance_pct=memory_tolerance_pct,
                absolute_latency_floor_ms=absolute_latency_floor_ms,
                absolute_memory_floor_mb=absolute_memory_floor_mb,
            )
            suite.verdicts.append(verdict)

            if update_baseline and baseline is not None:
                baseline.update_from_result(r)

        return suite