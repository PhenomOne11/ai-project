"""
Unit Tests for Phase 4: Attention Modules, U-Net AE, Ensemble
"""

import pytest
import torch
import torch.nn as nn


class TestAttentionModules:
    def test_se_shape(self):
        from core.models.attention import SqueezeExcitation
        se = SqueezeExcitation(channels=32)
        x = torch.randn(2, 32, 16, 16)
        out = se(x)
        assert out.shape == x.shape
    
    def test_se_output_range(self):
        from core.models.attention import SqueezeExcitation
        se = SqueezeExcitation(channels=16)
        x = torch.randn(2, 16, 8, 8)
        out = se(x)
        # SE scales features, so output range depends on input
        assert torch.isfinite(out).all()
    
    def test_cbam_shape(self):
        from core.models.attention import CBAM
        cbam = CBAM(channels=32)
        x = torch.randn(2, 32, 16, 16)
        out = cbam(x)
        assert out.shape == x.shape
    
    def test_cbam_gradient_flow(self):
        from core.models.attention import CBAM
        cbam = CBAM(channels=16)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        out = cbam(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_self_attention_shape(self):
        from core.models.attention import SelfAttention2D
        sa = SelfAttention2D(in_channels=32)
        x = torch.randn(2, 32, 8, 8)
        out = sa(x)
        assert out.shape == x.shape
    
    def test_self_attention_residual(self):
        from core.models.attention import SelfAttention2D
        sa = SelfAttention2D(in_channels=16)
        # Output projection initialized to zero, so output ≈ input initially
        x = torch.randn(2, 16, 4, 4)
        out = sa(x)
        assert torch.allclose(out, x, atol=1e-5)
    
    def test_multihead_spatial(self):
        from core.models.attention import MultiHeadSpatialAttention
        mha = MultiHeadSpatialAttention(in_channels=32, num_heads=4)
        x = torch.randn(2, 32, 8, 8)
        out = mha(x)
        assert out.shape == x.shape
    
    def test_gau(self):
        from core.models.attention import GatedAttentionUnit
        gau = GatedAttentionUnit(channels=16)
        x = torch.randn(2, 16, 8, 8)
        out = gau(x)
        assert out.shape == x.shape


class TestUNetAutoencoder:
    @pytest.fixture
    def model(self):
        from core.models.unet_ae import UNetAutoencoder
        return UNetAutoencoder(
            in_channels=1, input_size=64, latent_dim=32,
            base_channels=8, depth=3
        )
    
    def test_forward_shape(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_output_range(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    
    def test_encode_shape(self, model):
        x = torch.randn(2, 1, 64, 64)
        z = model.encode(x)
        assert z.shape == (2, 32)
    
    def test_decode_shape(self, model):
        z = torch.randn(2, 32)
        x_hat = model.decode(z)
        assert x_hat.shape == (2, 1, 64, 64)
    
    def test_encode_decode_full(self, model):
        x = torch.randn(2, 1, 64, 64)
        z, x_hat = model.encode_decode_full(x)
        assert z.shape == (2, 32)
        assert x_hat.shape == (2, 1, 64, 64)
    
    def test_multiscale_features(self, model):
        x = torch.randn(2, 1, 64, 64)
        features = model.encode_multiscale(x)
        assert 'encoder_level_0' in features
        assert 'encoder_level_1' in features
        assert 'encoder_level_2' in features
        assert 'bottleneck' in features
        # Skips are captured BEFORE MaxPool, so spatial = input / 2^level
        assert features['encoder_level_0'].shape[2] == 64   # 64 / 2^0
        assert features['encoder_level_1'].shape[2] == 32   # 64 / 2^1
        assert features['encoder_level_2'].shape[2] == 16
    
    def test_anomaly_map(self, model):
        x = torch.randn(2, 1, 64, 64)
        amap = model.get_anomaly_map(x)
        assert amap.shape == (2, 1, 64, 64)
        assert amap.min() >= 0.0
    
    def test_regional_error(self, model):
        x = torch.randn(2, 1, 64, 64)
        regional = model.get_reconstruction_error_per_region(x, grid_size=4)
        assert regional.shape == (2, 4, 4)
        assert regional.min() >= 0.0
    
    def test_with_se_attention(self):
        from core.models.unet_ae import UNetAutoencoder
        model = UNetAutoencoder(
            in_channels=1, input_size=64, latent_dim=32,
            base_channels=8, depth=3, attention="se"
        )
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_with_cbam_attention(self):
        from core.models.unet_ae import UNetAutoencoder
        model = UNetAutoencoder(
            in_channels=1, input_size=64, latent_dim=32,
            base_channels=8, depth=3, attention="cbam"
        )
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_rgb_input(self):
        from core.models.unet_ae import UNetAutoencoder
        model = UNetAutoencoder(
            in_channels=3, input_size=64, latent_dim=32,
            base_channels=8, depth=3
        )
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 3, 64, 64)
    
    def test_diagnostics(self, model):
        x = torch.randn(2, 1, 64, 64)
        diag = model.diagnostics(x)
        assert diag['architecture'] == 'UNetAutoencoder'
        assert diag['total_parameters'] > 0
        assert 'reconstruction_error' in diag


class TestModelEnsemble:
    @pytest.fixture
    def ensemble(self):
        from core.models.ensemble import create_default_ensemble
        return create_default_ensemble(
            in_channels=1, input_size=64, latent_dim=32, fusion="weighted"
        )
    
    def test_member_count(self, ensemble):
        assert len(ensemble.members) == 3
        assert "edge_ae" in ensemble.members
        assert "vae" in ensemble.members
        assert "unet_ae" in ensemble.members
    
    def test_forward_shape(self, ensemble):
        x = torch.randn(2, 1, 64, 64)
        out = ensemble(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_uncertainty_output(self, ensemble):
        x = torch.randn(2, 1, 64, 64)
        fused, uncertainty = ensemble.forward_with_uncertainty(x)
        assert fused.shape == (2, 1, 64, 64)
        assert uncertainty.shape == (2, 1, 64, 64)
        assert uncertainty.min() >= 0.0
    
    def test_encode_all(self, ensemble):
        x = torch.randn(2, 1, 64, 64)
        latents = ensemble.encode_all(x)
        assert len(latents) == 3
        for name, z in latents.items():
            assert z.shape == (2, 32)
    
    def test_weight_adjustment(self, ensemble):
        ensemble.set_weight("edge_ae", 5.0)
        assert ensemble.members["edge_ae"].weight == 5.0
    
    def test_model_disable(self, ensemble):
        ensemble.enable_model("vae", False)
        active = ensemble.active_members
        active_names = [m.name for m in active]
        assert "vae" not in active_names
        ensemble.enable_model("vae", True)
    
    def test_fusion_strategies(self):
        from core.models.ensemble import create_default_ensemble
        x = torch.randn(2, 1, 64, 64)
        
        for strategy in ["mean", "weighted", "median", "best", "uncertainty"]:
            ens = create_default_ensemble(fusion=strategy)
            out = ens(x)
            assert out.shape == (2, 1, 64, 64), f"Failed for strategy: {strategy}"
    
    def test_ensemble_report(self, ensemble):
        x = torch.randn(2, 1, 64, 64)
        _ = ensemble(x)  # Run once to populate metrics
        
        report = ensemble.get_ensemble_report()
        assert report['total_members'] == 3
        assert report['active_members'] == 3
        assert 'edge_ae' in report['members']
        assert report['members']['edge_ae']['call_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])