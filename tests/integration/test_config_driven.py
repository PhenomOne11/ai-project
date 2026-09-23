"""
Config-driven integration tests.

Each profile triggers a different set of subsystem values. These tests
verify that a config loaded from YAML correctly propagates to the
runtime components that consume it.
"""

import pytest


PROFILE_CONFIGS = [
    # (profile, input_size, latent_dim, batch_size)
    ("default",          64, 32, 16),
    ("high_performance", 64, 64, 32),
    ("edge",             32, 16, 8),
    ("ci",               32, 16, 4),
    ("debug",            64, 32, 4),
]


class TestConfigPropagation:
    """Every profile routes correctly through the loader."""

    @pytest.mark.parametrize("profile,expected_size,expected_latent,expected_batch",
                             PROFILE_CONFIGS)
    def test_model_built_for_profile(self, profile, expected_size,
                                     expected_latent, expected_batch):
        from config.loader import load_config
        from core.models import ModelRegistry

        cfg = load_config(profile=profile)
        assert cfg.model.input_size == expected_size
        assert cfg.model.latent_dim == expected_latent
        assert cfg.stream.batch_size == expected_batch

        # Build a model matching config
        model = ModelRegistry.create(
            cfg.model.architecture,
            in_channels=cfg.model.in_channels,
            input_size=cfg.model.input_size,
            latent_dim=cfg.model.latent_dim,
        )
        assert model.latent_dim == expected_latent

        # Forward pass with the config-specified shape
        import torch
        x = torch.randn(2, cfg.model.in_channels,
                        cfg.model.input_size, cfg.model.input_size)
        with torch.no_grad():
            z = model.encode(x)
        assert z.shape == (2, expected_latent)

    @pytest.mark.parametrize("profile", [p[0] for p in PROFILE_CONFIGS])
    def test_precision_engine_built_for_profile(self, profile):
        from config.loader import load_config
        from core.precision import AdaptivePrecisionEngineV2
        from core.models import EdgeStreamingAE

        cfg = load_config(profile=profile)
        model = EdgeStreamingAE(
            in_channels=cfg.model.in_channels,
            input_size=cfg.model.input_size,
            latent_dim=cfg.model.latent_dim,
            base_channels=4,
        )
        engine = AdaptivePrecisionEngineV2(
            model,
            target_latency_ms=cfg.precision.target_latency_ms,
            window_size=cfg.precision.window_size,
            cooldown_ticks=cfg.precision.cooldown_ticks,
            enable_int8=False,
        )
        assert engine.target_latency_ms == cfg.precision.target_latency_ms
        assert engine.cooldown_ticks == cfg.precision.cooldown_ticks

    @pytest.mark.parametrize("profile", [p[0] for p in PROFILE_CONFIGS])
    def test_drift_radar_built_for_profile(self, profile):
        from config.loader import load_config
        from core.drift import DriftRadar

        cfg = load_config(profile=profile)
        radar = DriftRadar(
            layer_names=cfg.drift.layer_names,
            methods=cfg.drift.methods,
            sample_size=cfg.drift.sample_size,
            evaluation_interval=cfg.drift.evaluation_interval,
        )
        assert radar.detector.methods == cfg.drift.methods
        assert radar.evaluation_interval == cfg.drift.evaluation_interval

    @pytest.mark.parametrize("profile", [p[0] for p in PROFILE_CONFIGS])
    def test_adaptation_built_for_profile(self, profile):
        from config.loader import load_config
        from core.adaptation import OnlineAdaptationEngineV2
        from core.models import EdgeStreamingAE

        cfg = load_config(profile=profile)
        model = EdgeStreamingAE(
            in_channels=cfg.model.in_channels,
            input_size=cfg.model.input_size,
            latent_dim=cfg.model.latent_dim,
            base_channels=4,
        )
        engine = OnlineAdaptationEngineV2(
            model,
            error_threshold=cfg.adaptation.error_threshold,
            buffer_capacity=cfg.adaptation.buffer_capacity,
            micro_batch_size=cfg.adaptation.micro_batch_size,
            learning_rate=cfg.adaptation.learning_rate,
            enable_ewc=cfg.adaptation.enable_ewc,
        )
        assert engine.error_threshold == cfg.adaptation.error_threshold
        assert engine.replay.capacity == cfg.adaptation.buffer_capacity


class TestOverridesCombination:
    """Config + runtime overrides combine correctly."""

    def test_override_latent_dim(self):
        from config.loader import load_config
        cfg = load_config(profile="ci",
                          overrides={"model": {"latent_dim": 64}})
        assert cfg.model.latent_dim == 64
        # Other CI values preserved
        assert cfg.seed == 1337
        assert cfg.stream.batch_size == 4

    def test_override_debug_flag(self):
        from config.loader import load_config
        cfg = load_config(profile="default", overrides={"debug": True})
        assert cfg.debug is True

    def test_override_nested(self):
        from config.loader import load_config
        cfg = load_config(
            profile="edge",
            overrides={
                "precision": {"target_latency_ms": 1.0},
                "workers": {"num_inference_workers": 4},
            },
        )
        assert cfg.precision.target_latency_ms == 1.0
        assert cfg.workers.num_inference_workers == 4
        # Other edge values preserved
        assert cfg.model.input_size == 32