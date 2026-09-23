"""
Unit Tests for the CLI package
"""

import sys
import subprocess
from pathlib import Path
import pytest


class TestProfileResolution:
    def test_list_profiles(self):
        from config.loader import list_profiles
        profiles = list_profiles()
        assert 'default' in profiles
        assert 'edge' in profiles
        assert 'ci' in profiles
        assert 'debug' in profiles
        assert 'high_performance' in profiles

    def test_resolve_by_name(self):
        from config.loader import resolve_profile
        path = resolve_profile("edge")
        assert path is not None
        assert path.name == "edge.yaml"
        assert path.is_file()

    def test_resolve_with_extension(self):
        from config.loader import resolve_profile
        path = resolve_profile("edge.yaml")
        assert path is not None
        assert path.name == "edge.yaml"

    def test_resolve_missing_raises(self):
        from config.loader import resolve_profile
        with pytest.raises(FileNotFoundError):
            resolve_profile("does_not_exist")

    def test_resolve_none_returns_none(self):
        from config.loader import resolve_profile
        assert resolve_profile(None) is None


class TestLoadConfigWithProfile:
    def test_load_edge_profile(self):
        from config.loader import load_config
        cfg = load_config(profile="edge")
        assert cfg.model.input_size == 32
        assert cfg.model.latent_dim == 16
        assert cfg.precision.target_latency_ms == 5.0
        assert cfg.dashboard.enable_3d_plots is False

    def test_load_ci_profile(self):
        from config.loader import load_config
        cfg = load_config(profile="ci")
        assert cfg.seed == 1337
        assert cfg.stream.batch_size == 4
        assert cfg.adaptation.enable_ewc is False

    def test_load_debug_profile(self):
        from config.loader import load_config
        cfg = load_config(profile="debug")
        assert cfg.debug is True
        assert cfg.log_level == "DEBUG"
        assert len(cfg.drift.methods) == 4

    def test_load_high_performance(self):
        from config.loader import load_config
        cfg = load_config(profile="high_performance")
        assert cfg.stream.batch_size == 32
        assert cfg.workers.num_inference_workers == 4

    def test_load_default(self):
        from config.loader import load_config
        cfg = load_config()
        assert cfg.project_name == "NeuralStream-Core"
        assert cfg.model.architecture == "edge_ae"

    def test_profile_with_overrides(self):
        from config.loader import load_config
        cfg = load_config(
            profile="ci",
            overrides={"model": {"latent_dim": 64}, "debug": True},
        )
        assert cfg.model.latent_dim == 64
        assert cfg.debug is True
        # Other CI values should still apply
        assert cfg.seed == 1337


class TestSummarizeConfig:
    def test_summary_has_sections(self):
        from config.loader import load_config, summarize_config
        cfg = load_config()
        summary = summarize_config(cfg)
        for key in ['project', 'model', 'stream', 'precision',
                    'drift', 'adaptation', 'dashboard', 'workers']:
            assert key in summary

    def test_summary_contains_key_values(self):
        from config.loader import load_config, summarize_config
        cfg = load_config(profile="edge")
        summary = summarize_config(cfg)
        assert summary['model']['input_size'] == 32
        assert summary['precision']['target_latency_ms'] == 5.0


class TestCLIParser:
    def test_parser_builds(self):
        from cli.main import _build_parser
        parser = _build_parser()
        assert parser is not None

    def test_parse_info(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(['info'])
        assert args.command == 'info'
        assert callable(args.func)

    def test_parse_config_show(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(['config', 'show', '--profile', 'edge'])
        assert args.command == 'config'
        assert args.config_action == 'show'
        assert args.profile == 'edge'

    def test_parse_config_validate_default_profile(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(['config', 'validate'])
        assert args.config_action == 'validate'
        # No explicit profile → validate all

    def test_parse_bench(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            'bench', '--profile', 'ci', '--iterations', '5',
            '--no-baseline', '--json', '/tmp/out.json',
        ])
        assert args.command == 'bench'
        assert args.iterations == 5
        assert args.no_baseline is True

    def test_parse_verify(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(['verify', '5'])
        assert args.command == 'verify'
        assert args.phases == '5'

    def test_parse_verify_default_all(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(['verify'])
        assert args.phases == 'all'

    def test_parse_demo(self):
        from cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(['demo', '--port', '9000'])
        assert args.command == 'demo'
        assert args.port == 9000


class TestPhaseArgParsing:
    def test_single(self):
        from cli.commands.verify import _parse_phase_arg
        assert _parse_phase_arg("5", {}) == [5]

    def test_all(self):
        from cli.commands.verify import _parse_phase_arg
        available = {1: 'v1', 2: 'v2', 3: 'v3'}
        assert _parse_phase_arg("all", available) == [1, 2, 3]

    def test_range(self):
        from cli.commands.verify import _parse_phase_arg
        available = {n: f"v{n}" for n in range(1, 11)}
        assert _parse_phase_arg("3-7", available) == [3, 4, 5, 6, 7]

    def test_invalid(self):
        from cli.commands.verify import _parse_phase_arg
        assert _parse_phase_arg("abc", {}) == []


class TestVerifierDiscovery:
    def test_discover_finds_scripts(self):
        from cli.commands.verify import _discover_verifiers
        root = Path.cwd()
        found = _discover_verifiers(root)
        # We've created verifiers for phases 1-22
        assert len(found) >= 5
        assert 1 in found


class TestCLISubprocess:
    """End-to-end checks by invoking the CLI as a subprocess."""

    def test_cli_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli", "--help"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        assert "neuralstream" in r.stdout.lower()

    def test_cli_info(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli", "info"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        assert "NeuralStream-Core" in r.stdout

    def test_cli_info_with_profile(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli", "info", "--profile", "edge"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        assert "edge" in r.stdout
        assert "32" in r.stdout  # input_size

    def test_cli_config_validate_all(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli", "config", "validate"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        assert "5 valid" in r.stdout

    def test_cli_config_show_json(self):
        import json
        r = subprocess.run(
            [sys.executable, "-m", "cli", "config", "show",
             "--profile", "ci"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        # Extract the JSON line (after the header)
        lines = r.stdout.split('\n')
        # Find first {
        json_start = next(
            (i for i, l in enumerate(lines) if l.strip().startswith('{')),
            None,
        )
        assert json_start is not None
        payload = '\n'.join(lines[json_start:])
        parsed = json.loads(payload)
        assert parsed['model']['latent_dim'] == 16

    def test_cli_config_dump_json(self):
        import json
        r = subprocess.run(
            [sys.executable, "-m", "cli", "config", "dump",
             "--profile", "ci", "--format", "json"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert parsed['project_name'] == "NeuralStream-Core"
        assert parsed['seed'] == 1337


class TestPyprojectToml:
    def test_pyproject_exists(self):
        p = Path("pyproject.toml")
        assert p.is_file()

    def test_pyproject_has_entry_point(self):
        content = Path("pyproject.toml").read_text(encoding="utf-8")
        assert "neuralstream" in content
        assert "cli.main:main" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])