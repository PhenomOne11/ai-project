"""
Phase 23 Verification Script
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def verify_profiles_exist():
    print("=" * 60)
    print("  PHASE 23: Config Profiles")
    print("=" * 60)

    from config.loader import list_profiles, resolve_profile

    profiles = list_profiles()
    print(f"  ✅ Discovered {len(profiles)} profiles: {profiles}")

    expected = ['default', 'edge', 'ci', 'debug', 'high_performance']
    for name in expected:
        assert name in profiles, f"Missing profile: {name}"
        path = resolve_profile(name)
        size = path.stat().st_size
        print(f"      {name:<18} → {path.name} ({size} bytes)")

    return True


def verify_profile_loading():
    print("\n" + "=" * 60)
    print("  PHASE 23: Profile Loading")
    print("=" * 60)

    from config.loader import load_config

    cases = [
        ("default",          dict(input_size=64, latent_dim=32)),
        ("edge",             dict(input_size=32, latent_dim=16)),
        ("ci",               dict(input_size=32, latent_dim=16)),
        ("debug",            dict(input_size=64, latent_dim=32)),
        ("high_performance", dict(input_size=64, latent_dim=64)),
    ]

    for name, expected in cases:
        cfg = load_config(profile=name)
        for k, v in expected.items():
            actual = getattr(cfg.model, k)
            assert actual == v, f"{name}.{k}: expected {v}, got {actual}"
        print(f"  ✅ {name:<18} input_size={cfg.model.input_size} "
              f"latent_dim={cfg.model.latent_dim}")

    # Overrides
    cfg = load_config(profile="ci", overrides={"model": {"latent_dim": 128}})
    assert cfg.model.latent_dim == 128
    assert cfg.seed == 1337
    print(f"  ✅ Overrides preserved: latent_dim=128, seed={cfg.seed}")

    return True


def verify_summary():
    print("\n" + "=" * 60)
    print("  PHASE 23: Config Summary")
    print("=" * 60)

    from config.loader import load_config, summarize_config

    cfg = load_config(profile="edge")
    summary = summarize_config(cfg)

    print(f"  ✅ Summary generated with {len(summary)} sections")
    for section, values in summary.items():
        print(f"      {section:<12} {len(values)} keys")

    assert 'model' in summary
    assert 'dashboard' in summary
    assert summary['model']['input_size'] == 32

    return True


def verify_cli_help():
    print("\n" + "=" * 60)
    print("  PHASE 23: CLI Subcommands")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "cli", "--help"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0

    expected = ['info', 'config', 'demo', 'bench', 'verify']
    for name in expected:
        assert name in r.stdout, f"Missing subcommand: {name}"
        print(f"  ✅ Subcommand '{name}' present")

    return True


def verify_cli_info():
    print("\n" + "=" * 60)
    print("  PHASE 23: `neuralstream info`")
    print("=" * 60)

    for profile in ['default', 'edge', 'ci']:
        r = subprocess.run(
            [sys.executable, "-m", "cli", "info", "--profile", profile],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        assert "NeuralStream-Core" in r.stdout
        assert profile in r.stdout
        print(f"  ✅ info --profile {profile}: exit=0, {len(r.stdout)} chars")

    return True


def verify_cli_config_validate():
    print("\n" + "=" * 60)
    print("  PHASE 23: `neuralstream config validate`")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "cli", "config", "validate"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    print(f"  ✅ All profiles valid")
    print()
    for line in r.stdout.split('\n'):
        if line.strip().startswith('✓') or line.strip().startswith('✗'):
            print(f"      {line}")

    # Single profile
    r2 = subprocess.run(
        [sys.executable, "-m", "cli", "config", "validate",
         "--profile", "edge"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r2.returncode == 0
    print(f"  ✅ Single profile validation OK")

    return True


def verify_cli_config_dump():
    print("\n" + "=" * 60)
    print("  PHASE 23: `neuralstream config dump`")
    print("=" * 60)

    # JSON
    r = subprocess.run(
        [sys.executable, "-m", "cli", "config", "dump",
         "--profile", "ci", "--format", "json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data['project_name'] == "NeuralStream-Core"
    assert data['seed'] == 1337
    print(f"  ✅ JSON dump: {len(r.stdout)} chars, "
          f"{len(data)} top-level keys")

    # YAML
    r2 = subprocess.run(
        [sys.executable, "-m", "cli", "config", "dump",
         "--profile", "ci", "--format", "yaml"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r2.returncode == 0
    assert "project_name" in r2.stdout
    print(f"  ✅ YAML dump: {len(r2.stdout)} chars")

    return True


def verify_cli_verify_dry():
    print("\n" + "=" * 60)
    print("  PHASE 23: `neuralstream verify` (dry discovery)")
    print("=" * 60)

    from cli.commands.verify import _discover_verifiers, _parse_phase_arg
    root = Path.cwd()
    verifiers = _discover_verifiers(root)
    print(f"  ✅ Discovered {len(verifiers)} verifiers")
    print(f"      Keys: {sorted(verifiers.keys())}")

    # Verify parser handles range
    phases = _parse_phase_arg("1-5", verifiers)
    print(f"  ✅ Range parse '1-5' → {phases}")

    # Help text
    r = subprocess.run(
        [sys.executable, "-m", "cli", "verify", "--help"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    assert "phase" in r.stdout.lower()
    print(f"  ✅ verify --help works ({len(r.stdout)} chars)")

    return True


def verify_pyproject():
    print("\n" + "=" * 60)
    print("  PHASE 23: pyproject.toml")
    print("=" * 60)

    p = Path("pyproject.toml")
    assert p.is_file()
    content = p.read_text(encoding="utf-8")

    checks = [
        ('name = "neuralstream-core"', "package name"),
        ('version = "2.0.0"', "version"),
        ('neuralstream = "cli.main:main"', "console entry point"),
        ('requires-python = ">=3.10"', "python version"),
    ]
    for pattern, label in checks:
        assert pattern in content, f"Missing {label}: {pattern}"
        print(f"  ✅ {label}")

    return True


def verify_cli_bench_dry():
    print("\n" + "=" * 60)
    print("  PHASE 23: `neuralstream bench` (dry help)")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "cli", "bench", "--help"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    assert "--iterations" in r.stdout
    assert "--update-baseline" in r.stdout
    assert "--html" in r.stdout
    print(f"  ✅ bench --help: {len(r.stdout)} chars")
    print(f"      --iterations flag present")
    print(f"      --update-baseline flag present")
    print(f"      --html flag present")

    return True


def verify_cli_demo_dry():
    print("\n" + "=" * 60)
    print("  PHASE 23: `neuralstream demo` (dry help)")
    print("=" * 60)

    r = subprocess.run(
        [sys.executable, "-m", "cli", "demo", "--help"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    assert "--port" in r.stdout
    assert "--profile" in r.stdout
    print(f"  ✅ demo --help: {len(r.stdout)} chars")

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 23: File Checklist")
    print("=" * 60)

    files = [
        "config/profiles/edge.yaml",
        "config/profiles/ci.yaml",
        "config/profiles/debug.yaml",
        "config/profiles/README.md",
        "config/loader.py",
        "cli/__init__.py",
        "cli/__main__.py",
        "cli/main.py",
        "cli/commands/__init__.py",
        "cli/commands/info.py",
        "cli/commands/config_cmd.py",
        "cli/commands/demo.py",
        "cli/commands/bench.py",
        "cli/commands/verify.py",
        "pyproject.toml",
        "tests/unit/test_cli.py",
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
    print("  PHASE 23: Running Unit Tests")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_cli.py",
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
    print("  NEURALSTREAM-CORE: PHASE 23 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Config Profiles", verify_profiles_exist()))
    results.append(("Profile Loading", verify_profile_loading()))
    results.append(("Config Summary", verify_summary()))
    results.append(("CLI Subcommands", verify_cli_help()))
    results.append(("CLI info", verify_cli_info()))
    results.append(("CLI config validate", verify_cli_config_validate()))
    results.append(("CLI config dump", verify_cli_config_dump()))
    results.append(("CLI verify helper", verify_cli_verify_dry()))
    results.append(("pyproject.toml", verify_pyproject()))
    results.append(("CLI bench help", verify_cli_bench_dry()))
    results.append(("CLI demo help", verify_cli_demo_dry()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 23 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 23 COMPLETE — Ready for Phase 24!")
        print("\n  ┌──────────────────────────────────────────────────────┐")
        print("  │  Try the CLI now:                                    │")
        print("  │                                                      │")
        print("  │     python -m cli info                               │")
        print("  │     python -m cli config validate                    │")
        print("  │     python -m cli config dump --profile ci --format json │")
        print("  │     python -m cli bench --profile ci --no-baseline   │")
        print("  │     python -m cli verify 22                          │")
        print("  │     python -m cli demo --profile debug               │")
        print("  │                                                      │")
        print("  │  (Optional) install as a real CLI command:           │")
        print("  │     pip install -e .                                 │")
        print("  │     neuralstream info                                │")
        print("  │                                                      │")
        print("  │  Next: Phase 24 — Integration Testing Harness        │")
        print("  └──────────────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)