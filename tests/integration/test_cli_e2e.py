"""
CLI subprocess end-to-end tests.

Invokes the CLI as a real subprocess and verifies behaviour across
all subcommands and profiles.
"""

import sys
import json
import subprocess
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_cli(*args, timeout=60):
    """Helper: invoke the CLI as a subprocess with UTF-8 I/O."""
    import os
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    r = subprocess.run(
        [sys.executable, "-m", "cli", *args],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=timeout,
    )
    return r


class TestCLIInfo:
    def test_info_default(self):
        r = _run_cli("info")
        assert r.returncode == 0
        assert "NeuralStream-Core" in r.stdout

    @pytest.mark.parametrize("profile",
                             ["default", "edge", "ci", "debug",
                              "high_performance"])
    def test_info_all_profiles(self, profile):
        r = _run_cli("info", "--profile", profile)
        assert r.returncode == 0
        assert profile in r.stdout


class TestCLIConfig:
    def test_config_validate_all(self):
        r = _run_cli("config", "validate")
        assert r.returncode == 0
        assert "5 valid" in r.stdout

    def test_config_show_json(self):
        r = _run_cli("config", "show", "--profile", "edge")
        assert r.returncode == 0
        # Find the JSON block
        lines = r.stdout.split('\n')
        idx = next(i for i, l in enumerate(lines) if l.strip().startswith('{'))
        payload = json.loads('\n'.join(lines[idx:]))
        assert payload['model']['input_size'] == 32

    def test_config_dump_json(self):
        r = _run_cli("config", "dump", "--profile", "ci",
                     "--format", "json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data['seed'] == 1337

    def test_config_dump_yaml(self):
        r = _run_cli("config", "dump", "--profile", "ci",
                     "--format", "yaml")
        assert r.returncode == 0
        assert "project_name" in r.stdout
        assert "seed: 1337" in r.stdout


class TestCLIHelp:
    @pytest.mark.parametrize("subcommand",
                             ["info", "config", "demo", "bench", "verify"])
    def test_subcommand_help(self, subcommand):
        r = _run_cli(subcommand, "--help")
        assert r.returncode == 0
        assert len(r.stdout) > 50


class TestCLIVerifyHelper:
    """Verify phase discovery works via CLI helper."""

    def test_verify_help(self):
        r = _run_cli("verify", "--help")
        assert r.returncode == 0
        assert "phase" in r.stdout.lower()

    def test_verify_short_run(self):
        """Run a known-fast verifier for a small phase."""
        r = _run_cli("verify", "13", timeout=180)
        # Phase 13 was a short phase; verifier should finish quickly
        assert r.returncode == 0