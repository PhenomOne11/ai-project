"""
`neuralstream verify` — Run a phase verification script.

Examples:
    neuralstream verify 5          # run verify_phase5.py
    neuralstream verify all        # run every verifier in sequence
    neuralstream verify 1-10       # run a range
"""

import sys
import re
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _discover_verifiers(project_root: Path) -> dict:
    """Return {phase_number: path} for every verify_phaseN.py script found."""
    verifiers = {}
    for path in project_root.glob("verify_phase*.py"):
        m = re.match(r"verify_phase(\d+)\.py$", path.name)
        if m:
            verifiers[int(m.group(1))] = path
    return verifiers


def _parse_phase_arg(arg: str, available: dict) -> list:
    """Parse a phase argument into a list of phase numbers."""
    arg = str(arg).strip().lower()

    if arg == 'all':
        return sorted(available.keys())

    m = re.match(r"^(\d+)-(\d+)$", arg)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return [n for n in sorted(available.keys()) if lo <= n <= hi]

    if arg.isdigit():
        return [int(arg)]

    return []


def run_verify(args) -> int:
    phases_arg = getattr(args, 'phases', 'all')
    project_root = Path.cwd()

    available = _discover_verifiers(project_root)
    if not available:
        print("No verify_phase*.py scripts found in the current directory.",
              file=sys.stderr)
        return 1

    phases = _parse_phase_arg(phases_arg, available)
    if not phases:
        print(f"Invalid phase argument: {phases_arg}", file=sys.stderr)
        print(f"Available phases: {sorted(available.keys())}", file=sys.stderr)
        return 2

    print("=" * 68)
    print(f"  Running {len(phases)} verifier(s): {phases}")
    print("=" * 68)

    passed, failed = [], []
    for phase in phases:
        path = available.get(phase)
        if path is None:
            print(f"  ? [phase {phase:>2}] no verifier found")
            continue

        print(f"\n  ▶ [phase {phase:>2}] {path.name}")
        print("  " + "─" * 64)

        r = subprocess.run(
            [sys.executable, str(path)],
            cwd=project_root,
        )
        if r.returncode == 0:
            passed.append(phase)
        else:
            failed.append(phase)

    print()
    print("=" * 68)
    print(f"  PASSED: {passed}")
    if failed:
        print(f"  FAILED: {failed}")
    print("=" * 68)

    return 0 if not failed else 1