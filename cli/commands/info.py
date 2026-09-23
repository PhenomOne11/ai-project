"""
`neuralstream info` — Display system, environment, and config summary.
"""

import sys
import platform
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_info(args) -> int:
    """Print a compact system summary."""
    from config.loader import (
        load_config, summarize_config, list_profiles,
    )

    profile = getattr(args, 'profile', None)
    cfg = load_config(profile=profile)

    # System header
    print("=" * 68)
    print(f"  {cfg.project_name} v{cfg.version}")
    print("=" * 68)

    print(f"  Active profile:  {profile or 'default'}")
    print(f"  Available:       {', '.join(list_profiles()) or '(none)'}")
    print()
    print(f"  Python:          {sys.version.split()[0]}")
    print(f"  Platform:        {platform.system()} {platform.release()}")
    print(f"  Machine:         {platform.machine()}")

    try:
        import torch
        print(f"  PyTorch:         {torch.__version__}")
        print(f"  CUDA available:  {torch.cuda.is_available()}")
    except ImportError:
        print(f"  PyTorch:         (not installed)")

    print()

    # Config summary
    print("-" * 68)
    print("  CONFIG SUMMARY")
    print("-" * 68)

    summary = summarize_config(cfg)
    for section, values in summary.items():
        print(f"  [{section}]")
        for k, v in values.items():
            print(f"      {k:<24} = {v}")
        print()

    return 0