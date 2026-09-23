"""
CLI entry point for the NeuralStream benchmark suite.

Usage:
    python -m benchmarks.run_suite                    # console output
    python -m benchmarks.run_suite --json out.json    # also save JSON
    python -m benchmarks.run_suite --html out.html    # also save HTML
    python -m benchmarks.run_suite --update-baseline  # refresh baseline
    python -m benchmarks.run_suite --baseline path    # custom baseline
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from benchmarks.suite import BenchmarkSuite
from benchmarks.regression import BenchmarkBaseline
from benchmarks.reporter import (
    render_console, save_json, save_html,
)


DEFAULT_BASELINE = Path("benchmarks") / "baseline.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.run_suite",
        description="NeuralStream-Core benchmark suite",
    )
    parser.add_argument("--baseline", type=str,
                        default=str(DEFAULT_BASELINE),
                        help="Path to baseline JSON")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Refresh baseline after run")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline comparison entirely")
    parser.add_argument("--json", type=str, default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--html", type=str, default=None,
                        help="Save HTML report to this path")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for model benchmarks")
    parser.add_argument("--latency-tolerance", type=float, default=15.0,
                        help="%% increase in mean latency tolerated")
    parser.add_argument("--memory-tolerance", type=float, default=20.0,
                        help="%% increase in peak RSS tolerated")
    parser.add_argument("--latency-floor-ms", type=float, default=0.5,
                        help="Absolute latency delta (ms) below which "
                             "regressions are ignored (microbenchmark guard)")
    parser.add_argument("--memory-floor-mb", type=float, default=10.0,
                        help="Absolute memory delta (MB) below which "
                             "regressions are ignored")
    args = parser.parse_args()

    # Load baseline
    baseline = None
    if not args.no_baseline:
        baseline = BenchmarkBaseline.load(args.baseline)
        print(f"[baseline] loaded {len(baseline)} entries "
              f"from {args.baseline}")

    suite = BenchmarkSuite(
        warmup=args.warmup,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )

    print(f"[suite] running benchmarks "
          f"(iterations={args.iterations}, warmup={args.warmup})...")
    result = suite.run_and_check(
        baseline=baseline,
        latency_tolerance_pct=args.latency_tolerance,
        memory_tolerance_pct=args.memory_tolerance,
        absolute_latency_floor_ms=args.latency_floor_ms,
        absolute_memory_floor_mb=args.memory_floor_mb,
        update_baseline=args.update_baseline,
    )

    # Console
    print()
    print(render_console(result))

    # Save reports
    if args.json:
        save_json(result, args.json)
        print(f"[report] JSON saved to {args.json}")

    if args.html:
        save_html(result, args.html)
        print(f"[report] HTML saved to {args.html}")

    # Update baseline (either implied or explicit)
    if args.update_baseline and baseline is not None:
        baseline.save(args.baseline)
        print(f"[baseline] updated {args.baseline}")

    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())