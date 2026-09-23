"""
`neuralstream bench` — Run the benchmark suite.
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_bench(args) -> int:
    from config.loader import load_config
    from benchmarks.suite import BenchmarkSuite
    from benchmarks.regression import BenchmarkBaseline
    from benchmarks.reporter import render_console, save_json, save_html

    profile = getattr(args, 'profile', None)
    cfg = load_config(profile=profile)

    # Resolve iteration / batch / warmup
    iterations = getattr(args, 'iterations', None) or 30
    warmup = getattr(args, 'warmup', None) or 3
    batch_size = cfg.stream.batch_size

    use_baseline = not getattr(args, 'no_baseline', False)
    update_baseline = getattr(args, 'update_baseline', False)
    baseline_path = getattr(args, 'baseline', None) or "benchmarks/baseline.json"

    print(f"[bench] profile: {profile or 'default'}")
    print(f"[bench] iterations={iterations}, warmup={warmup}, batch_size={batch_size}")

    baseline = None
    if use_baseline:
        baseline = BenchmarkBaseline.load(baseline_path)
        print(f"[bench] baseline: {len(baseline)} entries from {baseline_path}")
    else:
        print(f"[bench] baseline: (disabled)")

    suite = BenchmarkSuite(
        warmup=warmup,
        iterations=iterations,
        batch_size=batch_size,
        input_size=cfg.model.input_size,
        channels=cfg.model.in_channels,
        latent_dim=cfg.model.latent_dim,
    )

    print(f"[bench] running suite...")
    result = suite.run_and_check(
        baseline=baseline,
        latency_tolerance_pct=getattr(args, 'latency_tolerance', 15.0),
        memory_tolerance_pct=getattr(args, 'memory_tolerance', 20.0),
        update_baseline=update_baseline,
    )

    print()
    print(render_console(result))

    json_path = getattr(args, 'json', None)
    if json_path:
        save_json(result, json_path)
        print(f"[bench] JSON saved: {json_path}")

    html_path = getattr(args, 'html', None)
    if html_path:
        save_html(result, html_path)
        print(f"[bench] HTML saved: {html_path}")

    if update_baseline and baseline is not None:
        baseline.save(baseline_path)
        print(f"[bench] baseline updated: {baseline_path}")

    return 0 if result.all_passed else 1