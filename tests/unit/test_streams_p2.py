"""
Unit Tests for core.streams (Phase 10)
"""

import pytest
import torch
import numpy as np
import tempfile
import os


class TestBaseUtilities:
    def test_infinite_batches(self):
        from core.streams.base import infinite_batches
        data = torch.arange(20).view(20, 1).float()
        it = infinite_batches(data, batch_size=6)
        for _ in range(5):
            batch = next(it)
            assert batch.size(0) <= 6

    def test_infinite_batches_shuffle(self):
        from core.streams.base import infinite_batches
        data = torch.arange(20).view(20, 1).float()
        it = infinite_batches(data, batch_size=20, shuffle=True)
        batch = next(it)
        assert batch.size(0) == 20
        # Shuffled → not equal to original order (very high probability)
        assert not torch.equal(batch, data)

    def test_normalize_minmax(self):
        from core.streams.base import normalize_batch
        b = torch.randn(4, 1, 8, 8) * 10 + 50
        norm = normalize_batch(b, mode="minmax")
        assert norm.min() >= -1e-6
        assert norm.max() <= 1.0 + 1e-6

    def test_normalize_zscore(self):
        from core.streams.base import normalize_batch
        b = torch.randn(8, 1, 4, 4) * 5 + 3
        norm = normalize_batch(b, mode="zscore")
        assert abs(float(norm.mean())) < 1e-5
        assert abs(float(norm.std()) - 1.0) < 1e-4

    def test_sanitize_batch(self):
        from core.streams.base import sanitize_batch
        b = torch.tensor([[1.0, float('nan'), float('inf')]])
        out = sanitize_batch(b)
        assert torch.isfinite(out).all()


class TestSyntheticStreams:
    def test_gaussian_stream(self):
        from core.streams.synthetic import SyntheticStream
        s = SyntheticStream(batch_size=4, shape=(1, 16, 16), seed=1)
        it = iter(s)
        batch = next(it)
        assert batch.shape == (4, 1, 16, 16)

    def test_sinusoidal_stream(self):
        from core.streams.synthetic import SinusoidalStream
        s = SinusoidalStream(batch_size=8, channels=2, length=32)
        batch = next(iter(s))
        assert batch.shape == (8, 2, 32)

    def test_mixed_signal_stream(self):
        from core.streams.synthetic import MixedSignalStream
        s = MixedSignalStream(batch_size=4, channels=1, length=64)
        batch = next(iter(s))
        assert batch.shape == (4, 1, 64)
        # Mixed signals include spikes → max should be > 2.0 at least sometimes
        assert torch.isfinite(batch).all()

    def test_spike_stream(self):
        from core.streams.synthetic import SpikeStream
        s = SpikeStream(batch_size=4, shape=(1, 16, 16), spike_density=0.1)
        batch = next(iter(s))
        assert batch.shape == (4, 1, 16, 16)
        # Most values should be zero (sparse)
        frac_zero = float((batch == 0).float().mean())
        assert frac_zero > 0.7


class TestLoaders:
    def test_load_npy(self, tmp_path):
        from core.streams.loaders import load_npy_to_tensor
        arr = np.random.rand(5, 32, 32).astype(np.float32)
        path = tmp_path / "test.npy"
        np.save(path, arr)
        t = load_npy_to_tensor(str(path), target_size=32)
        assert t.shape == (5, 1, 32, 32)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_load_npy_1d(self, tmp_path):
        from core.streams.loaders import load_npy_to_tensor
        arr = np.random.rand(40).astype(np.float32)
        path = tmp_path / "1d.npy"
        np.save(path, arr)
        t = load_npy_to_tensor(str(path), target_size=16)
        assert t.shape[0] == 40
        assert t.shape[1] == 1

    def test_load_pt(self, tmp_path):
        from core.streams.loaders import load_pt_to_tensor
        tensor = torch.rand(8, 1, 16, 16)
        path = tmp_path / "test.pt"
        torch.save(tensor, path)
        t = load_pt_to_tensor(str(path), target_size=16)
        assert t.shape == (8, 1, 16, 16)

    def test_load_pt_dict(self, tmp_path):
        from core.streams.loaders import load_pt_to_tensor
        obj = {"data": torch.rand(4, 1, 8, 8)}
        path = tmp_path / "test.pt"
        torch.save(obj, path)
        t = load_pt_to_tensor(str(path), target_size=8)
        assert t.shape == (4, 1, 8, 8)

    def test_load_images_from_pil(self):
        from core.streams.loaders import load_images_to_tensor
        from PIL import Image
        imgs = [Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8))
                for _ in range(3)]
        t = load_images_to_tensor(imgs, target_size=16, channels=1)
        assert t.shape == (3, 1, 16, 16)

    def test_load_csv(self, tmp_path):
        from core.streams.loaders import load_csv_to_tensor
        csv_content = "time,val\n1,0.5\n2,0.7\n3,0.9\n4,1.1\n5,1.3\n"
        path = tmp_path / "test.csv"
        path.write_text(csv_content)
        t = load_csv_to_tensor(str(path), value_columns=["val"])
        assert t.shape[0] == 5
        assert t.shape[1] == 1


class TestSignalStreams:
    def test_audio_waveform_mode(self):
        from core.streams.signals import AudioStream
        s = AudioStream(batch_size=4, mode="waveform", window_length=1024)
        batch = next(iter(s))
        assert batch.shape == (4, 1, 1024)

    def test_audio_spectrogram_mode(self):
        from core.streams.signals import AudioStream
        s = AudioStream(batch_size=2, mode="spectrogram", spectrogram_size=32)
        batch = next(iter(s))
        assert batch.shape == (2, 1, 32, 32)
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0 + 1e-5

    def test_timeseries_stream(self):
        from core.streams.signals import TimeSeriesStream
        s = TimeSeriesStream(
            window_length=32, channels=2, batch_size=4
        )
        batch = next(iter(s))
        assert batch.shape == (4, 2, 32)

    def test_timeseries_1d_input(self):
        from core.streams.signals import TimeSeriesStream
        series = torch.sin(torch.linspace(0, 20, 512))
        s = TimeSeriesStream(series=series, window_length=32, batch_size=4)
        batch = next(iter(s))
        assert batch.shape == (4, 1, 32)


class TestAugmentation:
    def test_gaussian_noise(self):
        from core.streams.augment import AugmentationPipeline
        aug = AugmentationPipeline([(AugmentationPipeline.gaussian_noise(0.1), 1.0)])
        x = torch.zeros(4, 1, 8, 8)
        out = aug(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)

    def test_horizontal_flip_involution(self):
        from core.streams.augment import AugmentationPipeline
        aug = AugmentationPipeline([(AugmentationPipeline.horizontal_flip(), 1.0)])
        x = torch.randn(2, 1, 8, 8)
        out = aug(x)
        assert torch.allclose(out, torch.flip(x, dims=[-1]))

    def test_composite_pipeline(self):
        from core.streams.augment import AugmentationPipeline
        aug = AugmentationPipeline.image_preset(seed=42)
        x = torch.rand(4, 1, 16, 16)
        for _ in range(5):
            out = aug(x)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()

    def test_signal_preset(self):
        from core.streams.augment import AugmentationPipeline
        aug = AugmentationPipeline.signal_preset(seed=42)
        x = torch.randn(4, 2, 32)
        out = aug(x)
        assert out.shape == x.shape


class TestFactory:
    def test_build_stream_synthetic(self):
        from core.streams import build_stream
        s = build_stream('synthetic', batch_size=4, shape=(1, 8, 8))
        assert s.batch_size == 4

    def test_build_stream_sinusoidal(self):
        from core.streams import build_stream
        s = build_stream('sinusoidal', batch_size=4, channels=1, length=16)
        batch = next(iter(s))
        assert batch.shape == (4, 1, 16)

    def test_build_stream_invalid(self):
        from core.streams import build_stream
        with pytest.raises(ValueError):
            build_stream('nonexistent_stream_type')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])