"""
Phase 25 Verification Script
The final assembly check.
"""

import sys
import subprocess
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def verify_app_imports():
    print("=" * 60)
    print("  PHASE 25: app.py Imports")
    print("=" * 60)

    # Compile-check the file without executing Streamlit calls
    import ast
    path = Path("app.py")
    assert path.is_file(), "app.py not found"

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Collect all `from X import Y` statements
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, []))

    print(f"  ✅ app.py parsed: {len(imports)} import statements")

    # Verify each module resolves
    for module, names in imports:
        if module is None:
            continue
        try:
            __import__(module)
        except Exception as e:
            raise AssertionError(f"Failed to import {module}: {e}")

    print(f"  ✅ All imported modules resolve")

    # Sanity: 7 panel imports present
    src_lower = src
    for panel in ['latent_explorer', 'drift_heatmap', 'telemetry',
                  'anomaly_console', 'model_comparison', 'oscilloscope',
                  'topology']:
        assert panel in src_lower or panel.replace('_', '') in src_lower

    print(f"  ✅ All 7 panels referenced")

    return True


def verify_launch_scripts():
    print("\n" + "=" * 60)
    print("  PHASE 25: Launch Scripts")
    print("=" * 60)

    bats = Path("run.bat")
    shs = Path("run.sh")

    assert bats.is_file(), "run.bat missing"
    assert shs.is_file(), "run.sh missing"

    bat_content = bats.read_text(encoding="utf-8")
    sh_content = shs.read_text(encoding="utf-8")

    # .bat checks
    assert "streamlit run app.py" in bat_content
    assert "NEURALSTREAM_PROFILE" in bat_content
    print(f"  ✅ run.bat: {len(bat_content)} chars, references app.py + profile env")

    # .sh checks
    assert "streamlit run app.py" in sh_content
    assert "NEURALSTREAM_PROFILE" in sh_content
    assert sh_content.startswith("#!/usr/bin/env bash") or sh_content.startswith("#!/bin/bash")
    print(f"  ✅ run.sh:  {len(sh_content)} chars, has shebang + profile env")

    return True


def verify_readme():
    print("\n" + "=" * 60)
    print("  PHASE 25: README")
    print("=" * 60)

    readme = Path("README.md")
    assert readme.is_file(), "README.md missing"

    content = readme.read_text(encoding="utf-8")

    # Required sections
    sections = [
        "## Quick Start",
        "## CLI Reference",
        "## Configuration Profiles",
        "## Architecture",
        "## Development",
        "## License",
    ]
    for section in sections:
        assert section in content, f"Missing section: {section}"
        print(f"  ✅ Section: {section}")

    # Panel coverage
    for panel in ['Latent Explorer', 'Drift Radar', 'Telemetry',
                  'Anomaly Console', 'Model Comparison',
                  'Oscilloscope', 'Topology']:
        assert panel in content, f"Missing panel mention: {panel}"
    print(f"  ✅ All 7 panels documented")

    # Profile coverage
    for profile in ['default', 'high_performance', 'edge', 'ci', 'debug']:
        assert profile in content, f"Missing profile: {profile}"
    print(f"  ✅ All 5 profiles documented")

    print(f"  ✅ README total: {len(content):,} chars, {len(content.splitlines())} lines")

    return True


def verify_license():
    print("\n" + "=" * 60)
    print("  PHASE 25: LICENSE")
    print("=" * 60)

    lic = Path("LICENSE")
    assert lic.is_file(), "LICENSE missing"
    content = lic.read_text(encoding="utf-8")
    assert "MIT License" in content
    assert "Permission is hereby granted" in content
    print(f"  ✅ MIT License present ({len(content)} chars)")

    return True


def verify_gitignore():
    print("\n" + "=" * 60)
    print("  PHASE 25: .gitignore")
    print("=" * 60)

    gi = Path(".gitignore")
    assert gi.is_file(), ".gitignore missing"
    content = gi.read_text(encoding="utf-8")

    for entry in ["__pycache__/", "venv/", ".pytest_cache/", "logs/"]:
        assert entry in content, f"Missing ignore entry: {entry}"
        print(f"  ✅ Ignores: {entry}")

    return True


def verify_app_syntax():
    print("\n" + "=" * 60)
    print("  PHASE 25: app.py Compilation")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "app.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"Compile failed:\n{r.stderr}"
    print(f"  ✅ app.py compiles cleanly")

    # Also compile every launcher for good measure
    for script in ["verify_phase25.py"]:
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", script],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
    print(f"  ✅ Self-compile OK")

    return True


def verify_tree_completeness():
    print("\n" + "=" * 60)
    print("  PHASE 25: Project Tree Completeness")
    print("=" * 60)

    critical_paths = [
        # Top-level
        "app.py", "README.md", "LICENSE", ".gitignore",
        "requirements.txt", "pyproject.toml",
        "run.bat", "run.sh",
        # Config
        "config/schema.py", "config/loader.py",
        "config/profiles/default.yaml",
        "config/profiles/high_performance.yaml",
        "config/profiles/edge.yaml",
        "config/profiles/ci.yaml",
        "config/profiles/debug.yaml",
        # Core packages
        "core/models/__init__.py",
        "core/drift/__init__.py",
        "core/ipc/__init__.py",
        "core/precision/__init__.py",
        "core/adaptation/__init__.py",
        "core/anomaly/__init__.py",
        "core/streams/__init__.py",
        "core/workers/__init__.py",
        "core/metrics/__init__.py",
        "core/events/__init__.py",
        # Dashboard
        "dashboard/theme/__init__.py",
        "dashboard/layouts/__init__.py",
        "dashboard/panels/__init__.py",
        "dashboard/panels/latent_explorer.py",
        "dashboard/panels/drift_heatmap.py",
        "dashboard/panels/telemetry.py",
        "dashboard/panels/anomaly_console.py",
        "dashboard/panels/model_comparison.py",
        "dashboard/panels/oscilloscope.py",
        "dashboard/panels/topology.py",
        # Benchmark
        "benchmarks/suite.py",
        "benchmarks/regression.py",
        "benchmarks/reporter.py",
        "benchmarks/run_suite.py",
        # CLI
        "cli/main.py", "cli/__main__.py",
        "cli/commands/__init__.py",
        # Tests
        "tests/unit", "tests/integration",
    ]

    missing = []
    for path in critical_paths:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        for m in missing:
            print(f"  ❌ {m}")
        raise AssertionError(f"Missing {len(missing)} critical paths")

    print(f"  ✅ All {len(critical_paths)} critical paths present")
    return True


def verify_all_tests_pass():
    print("\n" + "=" * 60)
    print("  PHASE 25: Full Test Suite")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=900,
    )
    # Print just the summary line
    for line in r.stdout.split('\n'):
        if 'passed' in line.lower() or 'failed' in line.lower():
            print(f"      {line}")
    return r.returncode == 0


def verify_cli_smoke():
    print("\n" + "=" * 60)
    print("  PHASE 25: CLI Smoke Test")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "cli", "info"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=60,
    )
    assert r.returncode == 0
    print(f"  ✅ neuralstream info: exit=0")

    r = subprocess.run(
        [sys.executable, "-m", "cli", "config", "validate"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=60,
    )
    assert r.returncode == 0
    assert "5 valid" in r.stdout
    print(f"  ✅ config validate: all 5 profiles valid")

    return True


def print_final_banner(file_counts):
    print()
    print("╔" + "═" * 66 + "╗")
    print("║" + " " * 66 + "║")
    print("║" + "  🎉  NEURALSTREAM-CORE · BUILD COMPLETE".ljust(66) + "║")
    print("║" + " " * 66 + "║")
    print("╠" + "═" * 66 + "╣")
    print("║" + f"  Total Python files:        {file_counts['total_py']:>4}".ljust(66) + "║")
    print("║" + f"  Test files:                {file_counts['test_py']:>4}".ljust(66) + "║")
    print("║" + f"  Verifier scripts:          {file_counts['verifiers']:>4}".ljust(66) + "║")
    print("║" + f"  Config profiles:           {file_counts['profiles']:>4}".ljust(66) + "║")
    print("║" + " " * 66 + "║")
    print("╠" + "═" * 66 + "╣")
    print("║" + "  LAUNCH THE DASHBOARD:".ljust(66) + "║")
    print("║" + " " * 66 + "║")
    print("║" + "      python -m cli demo".ljust(66) + "║")
    print("║" + "      streamlit run app.py".ljust(66) + "║")
    print("║" + "      run.bat               (Windows)".ljust(66) + "║")
    print("║" + "      ./run.sh              (Linux/macOS)".ljust(66) + "║")
    print("║" + " " * 66 + "║")
    print("╚" + "═" * 66 + "╝")
    print()


def _count_files():
    total_py = len(list(Path(".").rglob("*.py")))
    test_py = len(list(Path("tests").rglob("*.py"))) if Path("tests").exists() else 0
    verifiers = len(list(Path(".").glob("verify_phase*.py")))
    profiles = len(list(Path("config/profiles").glob("*.yaml"))) if Path("config/profiles").exists() else 0
    return {
        'total_py': total_py,
        'test_py': test_py,
        'verifiers': verifiers,
        'profiles': profiles,
    }


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 25 FINAL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    for name, fn in [
        ("app.py Imports",       verify_app_imports),
        ("Launch Scripts",       verify_launch_scripts),
        ("README",               verify_readme),
        ("LICENSE",              verify_license),
        (".gitignore",           verify_gitignore),
        ("app.py Compile",       verify_app_syntax),
        ("Project Tree",         verify_tree_completeness),
        ("CLI Smoke",            verify_cli_smoke),
        ("Full Test Suite",      verify_all_tests_pass),
    ]:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"\n  ❌ {name}: {type(e).__name__}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("  PHASE 25 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print_final_banner(_count_files())
    else:
        print("\n  ⚠️  Fix failing checks before shipping.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)