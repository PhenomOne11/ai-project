"""
Unit Tests for core.models
Validates all model architectures, forward passes, and diagnostics.
"""

import pytest
import torch
import torch.nn as nn


class TestEdgeStreamingAE:
    @pytest.fixture
    def model(self):
        from core.models.edge_ae import EdgeStreamingAE
        return EdgeStreamingAE(in_channels=1, input_size=64, latent_dim=32)
    
    def test_forward_shape(self, model):
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, 1, 64, 64)
    
    def test_encode_shape(self, model):
        x = torch.randn(4, 1, 64, 64)
        z = model.encode(x)
        assert z.shape == (4, 32)
    
    def test_decode_shape(self, model):
        z = torch.randn(4, 32)
        x_hat = model.decode(z)
        assert x_hat.shape == (4, 1, 64, 64)
    
    def test_output_range(self, model):
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    
    def test_diagnostics(self, model):
        x = torch.randn(2, 1, 64, 64)
        diag = model.diagnostics(x)
        assert diag['total_parameters'] > 0
        assert diag['memory_mb'] > 0
        assert diag['latent_dim'] == 32
        assert 'reconstruction_error' in diag
    
    def test_freeze_unfreeze(self, model):
        model.freeze_encoder()
        encoder_trainable = sum(
            p.numel() for n, p in model.named_parameters()
            if 'encoder' in n and p.requires_grad
        )
        assert encoder_trainable == 0
        
        model.unfreeze_encoder()
        encoder_trainable = sum(
            p.numel() for n, p in model.named_parameters()
            if 'encoder' in n and p.requires_grad
        )
        assert encoder_trainable > 0
    
    def test_activation_hooks(self, model):
        model.register_activation_hooks()
        x = torch.randn(2, 1, 64, 64)
        _ = model(x)
        activations = model.get_activations()
        assert len(activations) > 0
        model.remove_activation_hooks()
    
    def test_residual_variant(self):
        from core.models.edge_ae import EdgeStreamingAE
        model = EdgeStreamingAE(in_channels=1, input_size=64, latent_dim=32, use_residual=True)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_rgb_input(self):
        from core.models.edge_ae import EdgeStreamingAE
        model = EdgeStreamingAE(in_channels=3, input_size=64, latent_dim=32)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 3, 64, 64)
    
    def test_latent_statistics(self, model):
        x = torch.randn(8, 1, 64, 64)
        stats = model.get_latent_statistics(x)
        assert 'mean' in stats
        assert 'std' in stats
        assert len(stats['mean']) == 32


class TestConvVAE:
    @pytest.fixture
    def model(self):
        from core.models.vae import ConvVAE
        return ConvVAE(in_channels=1, input_size=64, latent_dim=32)
    
    def test_forward_returns_tuple(self, model):
        x = torch.randn(4, 1, 64, 64)
        result = model(x)
        assert isinstance(result, tuple)
        assert len(result) == 3
        x_hat, mu, logvar = result
        assert x_hat.shape == (4, 1, 64, 64)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)
    
    def test_encode_deterministic(self, model):
        model.eval()
        x = torch.randn(4, 1, 64, 64)
        z = model.encode(x)
        assert z.shape == (4, 32)
    
    def test_loss_computation(self, model):
        x = torch.randn(4, 1, 64, 64)
        x_hat, mu, logvar = model(x)
        losses = model.compute_loss(x, x_hat, mu, logvar)
        assert 'total_loss' in losses
        assert 'recon_loss' in losses
        assert 'kl_loss' in losses
        assert losses['total_loss'].item() > 0
    
    def test_sampling(self, model):
        samples = model.sample(num_samples=8)
        assert samples.shape == (8, 1, 64, 64)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0
    
    def test_interpolation(self, model):
        x1 = torch.randn(1, 1, 64, 64)
        x2 = torch.randn(1, 1, 64, 64)
        interp = model.interpolate(x1, x2, num_steps=5)
        assert interp.shape == (5, 1, 64, 64)
    
    def test_kl_annealing(self, model):
        x = torch.randn(4, 1, 64, 64)
        
        # At step 0, beta should be near 0
        model._global_step = 0
        _, mu, logvar = model(x)
        losses_early = model.compute_loss(x, model.decode(mu), mu, logvar)
        
        # At step 2000, beta should be full
        model._global_step = 2000
        losses_late = model.compute_loss(x, model.decode(mu), mu, logvar)
        
        assert losses_late['effective_beta'].item() >= losses_early['effective_beta'].item()


class TestTransformerAutoencoder:
    @pytest.fixture
    def model(self):
        from core.models.transformer_ae import TransformerAutoencoder
        return TransformerAutoencoder(
            in_channels=1, input_size=64, latent_dim=32,
            patch_size=8, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=64
        )
    
    def test_forward_shape(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_encode_shape(self, model):
        x = torch.randn(2, 1, 64, 64)
        z = model.encode(x)
        assert z.shape == (2, 32)
    
    def test_decode_shape(self, model):
        z = torch.randn(2, 32)
        x_hat = model.decode(z)
        assert x_hat.shape == (2, 1, 64, 64)
    
    def test_output_range(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    
    def test_diagnostics(self, model):
        x = torch.randn(2, 1, 64, 64)
        diag = model.diagnostics(x)
        assert diag['architecture'] == 'TransformerAutoencoder'
        assert diag['latent_dim'] == 32


class TestModelRegistry:
    def test_list_architectures(self):
        from core.models.registry import ModelRegistry
        archs = ModelRegistry.list_architectures()
        assert 'edge_ae' in archs
        assert 'vae' in archs
        assert 'transformer_ae' in archs
    
    def test_create_edge_ae(self):
        from core.models.registry import ModelRegistry
        model = ModelRegistry.create('edge_ae', in_channels=1, input_size=64, latent_dim=32)
        assert model.latent_dim == 32
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
    
    def test_create_vae(self):
        from core.models.registry import ModelRegistry
        model = ModelRegistry.create('vae', in_channels=1, input_size=64, latent_dim=16)
        assert model.latent_dim == 16
    
    def test_create_invalid(self):
        from core.models.registry import ModelRegistry
        with pytest.raises(ValueError):
            ModelRegistry.create('nonexistent_model')
    
    def test_compare_architectures(self):
        from core.models.registry import ModelRegistry
        results = ModelRegistry.compare_architectures(
            in_channels=1, input_size=64, latent_dim=32,
            batch_size=4, num_warmup=2, num_iterations=5
        )
        assert 'edge_ae' in results
        assert 'parameters' in results['edge_ae']
        assert results['edge_ae']['parameters'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])