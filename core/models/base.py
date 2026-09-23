"""
Abstract Base Model for NeuralStream-Core
Provides standardized lifecycle hooks, activation capture,
parameter freezing, and diagnostic reporting for all architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from core.math_utils.tensor_ops import (
    check_tensor_health, count_parameters, model_memory_mb, flatten_except_batch
)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all NeuralStream-Core models.
    
    Provides:
    - Standardized encode/decode interface
    - Forward hook registration for drift detection
    - Parameter freezing/unfreezing
    - Health diagnostics
    - Latent space dimensionality reporting
    """
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self._latent_dim = latent_dim
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._is_calibrated = False
        self._calibration_diagnostics: Dict[str, Any] = {}
    
    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent bottleneck."""
        return self._latent_dim
    
    @property
    def is_calibrated(self) -> bool:
        """Whether the model has undergone LSUV or other calibration."""
        return self._is_calibrated
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor (batch, channels, height, width) or (batch, features)
        
        Returns:
            Latent tensor of shape (batch, latent_dim)
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent tensor of shape (batch, latent_dim)
        
        Returns:
            Reconstructed tensor matching input shape
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: encode then decode."""
        z = self.encode(x)
        return self.decode(z)
    
    def encode_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode-decode pass returning both latent and reconstruction.
        
        Returns:
            (latent, reconstruction)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat
    
    # ---- Activation Hooks ----
    
    def register_activation_hooks(self, layer_names: Optional[List[str]] = None) -> None:
        """
        Register forward hooks on specified layers to capture activations
        for drift detection and visualization.
        
        Args:
            layer_names: List of module names to hook. If None, hooks all
                        Conv2d and Linear layers.
        """
        self.remove_activation_hooks()
        
        for name, module in self.named_modules():
            if layer_names is not None and name not in layer_names:
                continue
            if layer_names is None and not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if name == '':
                continue
            
            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    self._activation_cache[layer_name] = out.detach()
                return hook_fn
            
            handle = module.register_forward_hook(make_hook(name))
            self._hook_handles.append(handle)
    
    def remove_activation_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._activation_cache.clear()
    
    def get_activations(self, layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Retrieve cached activations from the last forward pass.
        
        Args:
            layer_name: Specific layer to retrieve. If None, returns all.
        """
        if layer_name is not None:
            if layer_name in self._activation_cache:
                return {layer_name: self._activation_cache[layer_name]}
            return {}
        return dict(self._activation_cache)
    
    # ---- Parameter Management ----
    
    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (stop gradient updates)."""
        for name, param in self.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for name, param in self.named_parameters():
            if 'encoder' in name:
                param.requires_grad = True
    
    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters."""
        for name, param in self.named_parameters():
            if 'decoder' in name:
                param.requires_grad = False
    
    def unfreeze_decoder(self) -> None:
        """Unfreeze all decoder parameters."""
        for name, param in self.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
    
    def freeze_all(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_parameter_count(self) -> int:
        """Count parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_frozen_parameter_count(self) -> int:
        """Count parameters with requires_grad=False."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    # ---- Diagnostics ----
    
    def diagnostics(self, sample_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate comprehensive model diagnostics report.
        
        Args:
            sample_input: Optional input tensor for shape/health analysis
        
        Returns:
            Dictionary with model statistics
        """
        report = {
            'architecture': self.__class__.__name__,
            'latent_dim': self._latent_dim,
            'total_parameters': count_parameters(self, trainable_only=False),
            'trainable_parameters': count_parameters(self, trainable_only=True),
            'frozen_parameters': self.get_frozen_parameter_count(),
            'memory_mb': model_memory_mb(self),
            'is_calibrated': self._is_calibrated,
            'num_layers': sum(1 for _ in self.modules()),
            'num_conv_layers': sum(1 for m in self.modules() if isinstance(m, nn.Conv2d)),
            'num_linear_layers': sum(1 for m in self.modules() if isinstance(m, nn.Linear)),
            'num_bn_layers': sum(1 for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))),
        }
        
        if sample_input is not None:
            self.eval()
            with torch.no_grad():
                try:
                    z = self.encode(sample_input)
                    x_hat = self.decode(z)
                    
                    report['input_shape'] = list(sample_input.shape)
                    report['latent_shape'] = list(z.shape)
                    report['output_shape'] = list(x_hat.shape)
                    report['input_health'] = check_tensor_health(sample_input)
                    report['latent_health'] = check_tensor_health(z)
                    report['output_health'] = check_tensor_health(x_hat)
                    report['reconstruction_error'] = float(
                        torch.nn.functional.mse_loss(x_hat, sample_input)
                    )
                except Exception as e:
                    report['forward_pass_error'] = str(e)
        
        return report
    
    def print_diagnostics(self, sample_input: Optional[torch.Tensor] = None) -> None:
        """Pretty-print model diagnostics."""
        report = self.diagnostics(sample_input)
        
        print("=" * 60)
        print(f"  MODEL DIAGNOSTICS: {report['architecture']}")
        print("=" * 60)
        print(f"  Latent Dimension     : {report['latent_dim']}")
        print(f"  Total Parameters     : {report['total_parameters']:,}")
        print(f"  Trainable Parameters : {report['trainable_parameters']:,}")
        print(f"  Frozen Parameters    : {report['frozen_parameters']:,}")
        print(f"  Memory Footprint     : {report['memory_mb']:.4f} MB")
        print(f"  Calibrated           : {report['is_calibrated']}")
        print(f"  Total Layers         : {report['num_layers']}")
        print(f"    Conv2d             : {report['num_conv_layers']}")
        print(f"    Linear             : {report['num_linear_layers']}")
        print(f"    BatchNorm          : {report['num_bn_layers']}")
        
        if 'input_shape' in report:
            print(f"  Input Shape          : {report['input_shape']}")
            print(f"  Latent Shape         : {report['latent_shape']}")
            print(f"  Output Shape         : {report['output_shape']}")
            print(f"  Recon Error (MSE)    : {report['reconstruction_error']:.6f}")
            print(f"  Latent Health        : healthy={report['latent_health']['is_healthy']}")
        
        print("=" * 60)
    
    def mark_calibrated(self, diagnostics: Optional[Dict] = None) -> None:
        """Mark model as calibrated with optional diagnostic data."""
        self._is_calibrated = True
        if diagnostics:
            self._calibration_diagnostics = diagnostics