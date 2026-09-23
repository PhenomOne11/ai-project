"""
Model Registry & Factory
Centralized model creation, architecture profiling, and comparison utilities.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Type, Optional, Any, List, Tuple
from core.models.base import BaseModel
from core.models.edge_ae import EdgeStreamingAE
from core.models.vae import ConvVAE
from core.models.transformer_ae import TransformerAutoencoder
from core.math_utils.tensor_ops import count_parameters, model_memory_mb


class ModelRegistry:
    """
    Singleton registry for all available model architectures.
    Supports dynamic registration, instantiation by name,
    and architecture comparison.
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}
    _initialized = False
    
    @classmethod
    def _ensure_init(cls):
        if not cls._initialized:
            cls.register("edge_ae", EdgeStreamingAE)
            cls.register("vae", ConvVAE)
            cls.register("transformer_ae", TransformerAutoencoder)
            cls._initialized = True
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model architecture by name."""
        cls._registry[name.lower()] = model_class
    
    @classmethod
    def create(
        cls,
        architecture: str,
        **kwargs
    ) -> BaseModel:
        """
        Instantiate a model by architecture name.
        
        Args:
            architecture: Registered architecture name
            **kwargs: Constructor arguments forwarded to the model
        
        Returns:
            Instantiated model
        """
        cls._ensure_init()
        key = architecture.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")
        
        return cls._registry[key](**kwargs)
    
    @classmethod
    def list_architectures(cls) -> List[str]:
        """List all registered architecture names."""
        cls._ensure_init()
        return list(cls._registry.keys())
    
    @classmethod
    def compare_architectures(
        cls,
        in_channels: int = 1,
        input_size: int = 64,
        latent_dim: int = 32,
        batch_size: int = 16,
        num_warmup: int = 5,
        num_iterations: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark all registered architectures on identical inputs.
        
        Returns per-architecture metrics: parameters, memory, latency, throughput.
        """
        cls._ensure_init()
        results = {}
        device = "cpu"
        sample_input = torch.randn(batch_size, in_channels, input_size, input_size, device=device)
        
        for name, model_cls in cls._registry.items():
            try:
                # Instantiate
                if name == "transformer_ae":
                    model = model_cls(
                        in_channels=in_channels,
                        input_size=input_size,
                        latent_dim=latent_dim,
                        patch_size=8
                    ).to(device)
                else:
                    model = model_cls(
                        in_channels=in_channels,
                        input_size=input_size,
                        latent_dim=latent_dim
                    ).to(device)
                
                model.eval()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(num_warmup):
                        _ = model(sample_input)
                
                # Benchmark
                latencies = []
                with torch.no_grad():
                    for _ in range(num_iterations):
                        t0 = time.perf_counter()
                        _ = model(sample_input)
                        latencies.append((time.perf_counter() - t0) * 1000)
                
                import numpy as np
                lat_arr = np.array(latencies)
                
                results[name] = {
                    'parameters': count_parameters(model),
                    'memory_mb': model_memory_mb(model),
                    'latency_mean_ms': float(lat_arr.mean()),
                    'latency_p50_ms': float(np.percentile(lat_arr, 50)),
                    'latency_p95_ms': float(np.percentile(lat_arr, 95)),
                    'latency_p99_ms': float(np.percentile(lat_arr, 99)),
                    'throughput_fps': float(1000.0 / lat_arr.mean() * batch_size),
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    @classmethod
    def print_comparison(
        cls,
        in_channels: int = 1,
        input_size: int = 64,
        latent_dim: int = 32
    ) -> None:
        """Pretty-print architecture comparison table."""
        results = cls.compare_architectures(
            in_channels=in_channels,
            input_size=input_size,
            latent_dim=latent_dim
        )
        
        print("=" * 85)
        print("  NEURALSTREAM-CORE: MODEL ARCHITECTURE COMPARISON")
        print("=" * 85)
        print(f"  {'Architecture':<18} {'Params':>10} {'Mem(MB)':>8} {'p50(ms)':>8} {'p95(ms)':>8} {'FPS':>8}")
        print("-" * 85)
        
        for name, metrics in results.items():
            if 'error' in metrics:
                print(f"  {name:<18} ERROR: {metrics['error']}")
            else:
                print(
                    f"  {name:<18} "
                    f"{metrics['parameters']:>10,} "
                    f"{metrics['memory_mb']:>8.3f} "
                    f"{metrics['latency_p50_ms']:>8.2f} "
                    f"{metrics['latency_p95_ms']:>8.2f} "
                    f"{metrics['throughput_fps']:>8.0f}"
                )
        
        print("=" * 85)