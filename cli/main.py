"""
NeuralStream-Core Unified CLI

Entry point for all system operations:
    neuralstream info                        System + config summary
    neuralstream config show|validate|dump   Config inspection
    neuralstream demo                        Launch dashboard
    neuralstream bench                       Run benchmarks
    neuralstream verify [N|all|a-b]          Run phase verifiers
"""

import sys
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    from config.loader import list_profiles

    profiles = list_profiles() or ['default']

    parser = argparse.ArgumentParser(
        prog="neuralstream",
        description="NeuralStream-Core unified command line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="NeuralStream-Core 2.0.0",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- info ----
    p_info = sub.add_parser("info", help="System + environment summary")
    p_info.add_argument("--profile", choices=profiles, default=None,
                        help=f"Config profile ({', '.join(profiles)})")
    p_info.set_defaults(func=_cmd_info)

    # ---- config ----
    p_cfg = sub.add_parser("config", help="Inspect / validate configuration")
    cfg_sub = p_cfg.add_subparsers(dest="config_action", required=True)

    p_cfg_show = cfg_sub.add_parser("show", help="Print config summary")
    p_cfg_show.add_argument("--profile", choices=profiles, default=None)
    p_cfg_show.set_defaults(func=_cmd_config, config_action='show')

    p_cfg_val = cfg_sub.add_parser("validate", help="Validate profile(s)")
    p_cfg_val.add_argument("--profile", choices=profiles, default=None,
                           help="Validate a single profile (default: all)")
    p_cfg_val.set_defaults(func=_cmd_config, config_action='validate')

    p_cfg_dump = cfg_sub.add_parser("dump", help="Dump full config")
    p_cfg_dump.add_argument("--profile", choices=profiles, default=None)
    p_cfg_dump.add_argument("--format", choices=['yaml', 'json'], default='yaml')
    p_cfg_dump.set_defaults(func=_cmd_config, config_action='dump')

    # ---- demo ----
    p_demo = sub.add_parser("demo", help="Launch the dashboard")
    p_demo.add_argument("--profile", choices=profiles, default=None)
    p_demo.add_argument("--port", type=int, default=8501)
    p_demo.add_argument("--app", type=str, default=None,
                        help="Path override for the Streamlit app")
    p_demo.set_defaults(func=_cmd_demo)

    # ---- bench ----
    p_bench = sub.add_parser("bench", help="Run the benchmark suite")
    p_bench.add_argument("--profile", choices=profiles, default=None)
    p_bench.add_argument("--iterations", type=int, default=None)
    p_bench.add_argument("--warmup", type=int, default=None)
    p_bench.add_argument("--baseline", type=str, default=None)
    p_bench.add_argument("--no-baseline", action="store_true")
    p_bench.add_argument("--update-baseline", action="store_true")
    p_bench.add_argument("--latency-tolerance", type=float, default=15.0)
    p_bench.add_argument("--memory-tolerance", type=float, default=20.0)
    p_bench.add_argument("--json", type=str, default=None)
    p_bench.add_argument("--html", type=str, default=None)
    p_bench.set_defaults(func=_cmd_bench)

    # ---- verify ----
    p_ver = sub.add_parser("verify", help="Run phase verifier(s)")
    p_ver.add_argument("phases", nargs="?", default="all",
                       help="Phase number, range (5-10), or 'all'")
    p_ver.set_defaults(func=_cmd_verify)

    return parser


def _cmd_info(args) -> int:
    from cli.commands import run_info
    return run_info(args)


def _cmd_config(args) -> int:
    from cli.commands import run_config
    return run_config(args)


def _cmd_demo(args) -> int:
    from cli.commands import run_demo
    return run_demo(args)


def _cmd_bench(args) -> int:
    from cli.commands import run_bench
    return run_bench(args)


def _cmd_verify(args) -> int:
    from cli.commands import run_verify
    return run_verify(args)


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\n[cli] interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[cli] error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())